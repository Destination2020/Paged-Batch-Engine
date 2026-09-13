#include "serving/request_checkpoint.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace serving {

namespace {

constexpr uint64_t kCheckpointRuntimeIncarnation = 1;

struct CheckpointPublicationWire {
  int64_t client_request_id;
  uint64_t revision;
  uint64_t payload_bytes;
  int32_t valid_tokens;
  int32_t committed_tokens;
};

data::Digest256 publication_digest(const CheckpointPublicationWire& value,
                                   const std::string& model_namespace) {
  data::Digest256 result{};
  const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
  for (uint64_t lane = 0; lane < 4; ++lane) {
    uint64_t hash = UINT64_C(1469598103934665603) ^
                    (UINT64_C(0x9e3779b97f4a7c15) * (lane + 1));
    for (size_t i = 0; i < sizeof(value); ++i) hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    for (uint8_t byte : model_namespace) hash = (hash ^ byte) * UINT64_C(1099511628211);
    for (size_t i = 0; i < sizeof(hash); ++i)
      result[lane * sizeof(hash) + i] = static_cast<uint8_t>(hash >> (i * 8));
  }
  return result;
}

std::string publication_checksum(const data::Digest256& digest) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string value(digest.size() * 2, '0');
  for (size_t i = 0; i < digest.size(); ++i) {
    value[i * 2] = kHex[digest[i] >> 4];
    value[i * 2 + 1] = kHex[digest[i] & 15];
  }
  return value;
}

uint64_t publication_capacity(size_t max_records) {
  if (max_records > std::numeric_limits<uint64_t>::max() /
                        sizeof(CheckpointPublicationWire))
    return std::numeric_limits<uint64_t>::max();
  return static_cast<uint64_t>(max_records) * sizeof(CheckpointPublicationWire);
}

bool checked_add(size_t value, size_t* total) {
  if (!total || value > std::numeric_limits<size_t>::max() - *total)
    return false;
  *total += value;
  return true;
}

bool checked_vector_bytes(size_t count, size_t element_bytes, size_t* total) {
  if (element_bytes != 0 &&
      count > std::numeric_limits<size_t>::max() / element_bytes)
    return false;
  return checked_add(count * element_bytes, total);
}

bool checkpoint_bytes(const SequenceState& sequence, size_t kv_bytes,
                      const std::string& model_namespace, size_t* bytes) {
  if (!bytes) return false;
  size_t total = sizeof(RequestCheckpointManifest) + sizeof(SequenceState);
  if (!checked_add(kv_bytes, &total) ||
      !checked_add(model_namespace.size(), &total) ||
      !checked_vector_bytes(sequence.prompt_tokens.size(), sizeof(int32_t), &total) ||
      !checked_vector_bytes(sequence.output_tokens.size(), sizeof(int32_t), &total) ||
      !checked_vector_bytes(sequence.multimodal_positions.size(), sizeof(int32_t), &total) ||
      !checked_add(sequence.multimodal_feature_content.size(), &total) ||
      !checked_add(sequence.multimodal_feature_representation.size(), &total) ||
      !checked_vector_bytes(sequence.token_times.size(),
                            sizeof(std::chrono::steady_clock::time_point), &total) ||
      !checked_add(sequence.finish_reason.size(), &total) ||
      !checked_add(sequence.last_preempt_reason.size(), &total)) return false;
  for (const auto& item : sequence.outbox) {
    if (!checked_add(sizeof(ProcessOutboxItem), &total) ||
        !checked_add(item.text.size(), &total)) return false;
  }
  for (const auto& stop : sequence.generation_config.sampling.stop) {
    if (!checked_add(sizeof(std::string), &total) ||
        !checked_add(stop.size(), &total)) return false;
  }
  *bytes = total;
  return true;
}

}  // namespace

RequestCheckpointStore::RequestCheckpointStore(size_t max_records,
                                               size_t max_payload_bytes)
    : max_records_(max_records), max_payload_bytes_(max_payload_bytes),
      publication_runtime_(kCheckpointRuntimeIncarnation,
                           publication_capacity(max_records), max_records) {
  if (max_records_ == 0) throw std::invalid_argument("checkpoint capacity is zero");
  if (max_payload_bytes_ == 0)
    throw std::invalid_argument("checkpoint byte capacity is zero");
}

RequestCheckpointStore::~RequestCheckpointStore() {
  for (const auto& item : records_) {
    if (item.second.restoring_request_id != -1) std::terminate();
  }
}

bool RequestCheckpointStore::prepare(
    const SequenceState& sequence, const std::string& model_namespace,
    base::KVCacheManager* manager, CheckpointTicket* ticket) {
  if (!manager || !ticket || model_namespace.empty() ||
      sequence.client_request_id < 0 || sequence.scheduled_tokens != 0 ||
      sequence.emitted_cursor > sequence.output_tokens.size() ||
      !manager->is_valid_request(sequence.request_id)) return false;
  if (records_.size() >= max_records_) {
    ++stats_.record_capacity_rejections;
    return false;
  }
  size_t kv_bytes = 0;
  size_t record_bytes = 0;
  if (!manager->snapshot_request_bytes(sequence.request_id, &kv_bytes) ||
      !checkpoint_bytes(sequence, kv_bytes, model_namespace, &record_bytes) ||
      record_bytes > max_payload_bytes_ - payload_bytes_) {
    ++stats_.byte_capacity_rejections;
    return false;
  }
  if (next_revision_ == UINT64_MAX) return false;
  base::KVRequestSnapshot kv;
  if (!manager->snapshot_request(sequence.request_id, &kv) ||
      sequence.computed_tokens > kv.valid_tokens) return false;

  const uint64_t revision = next_revision_ + 1;
  Record record;
  record.sequence = sequence;
  record.kv = std::move(kv);
  record.manifest.model_namespace = model_namespace;
  record.manifest.client_request_id = sequence.client_request_id;
  record.manifest.revision = revision;
  record.manifest.kv_committed_tokens = record.kv.committed_tokens;
  record.manifest.valid_tokens = record.kv.valid_tokens;
  record.manifest.pending_next_token = sequence.next_token;
  record.manifest.sampling_seed = sequence.generation_config.sampling.seed;
  record.manifest.sampling_counter = sequence.sampling_counter;
  record.manifest.emitted_cursor = sequence.emitted_cursor;
  record.manifest.output_enqueued_cursor = sequence.output_enqueued_cursor;
  record.manifest.multimodal_position_values = sequence.multimodal_positions.size();
  record.manifest.multimodal_exact_dependency = sequence.multimodal_exact_dependency;
  record.manifest.state = CheckpointState::kPreparing;
  record.payload_bytes = record_bytes;
  const CheckpointPublicationWire publication{
      sequence.client_request_id, revision, static_cast<uint64_t>(record_bytes),
      record.kv.valid_tokens, record.kv.committed_tokens};
  data::DataReservation reservation;
  reservation.operation = {kCheckpointRuntimeIncarnation, revision};
  reservation.kind = data::DataKind::kCheckpoint;
  reservation.content.digest = publication_digest(publication, model_namespace);
  reservation.representation.digest[0] = 1;  // RequestCheckpointManifest v1.
  reservation.logical_bytes = sizeof(publication);
  if (publication_runtime_.reserve(reservation, &record.publication_handle) !=
      data::DataError::kOk) return false;
  uint8_t* publication_bytes = nullptr;
  size_t publication_size = 0;
  if (publication_runtime_.begin_write(record.publication_handle, &publication_bytes,
                                       &publication_size) != data::DataError::kOk ||
      publication_size != sizeof(publication)) {
    publication_runtime_.release_producer(record.publication_handle);
    return false;
  }
  std::memcpy(publication_bytes, &publication, sizeof(publication));
  records_.emplace(Key{sequence.client_request_id, revision}, std::move(record));
  next_revision_ = revision;
  payload_bytes_ += record_bytes;
  *ticket = {sequence.client_request_id, revision};
  ++stats_.prepared;
  stats_.max_records = std::max(stats_.max_records, records_.size());
  stats_.payload_bytes = payload_bytes_;
  stats_.max_payload_bytes = std::max(stats_.max_payload_bytes, payload_bytes_);
  stats_.revision_metadata_records = current_revision_.size();
  return true;
}

bool RequestCheckpointStore::commit(const CheckpointTicket& ticket) {
  const auto it = records_.find({ticket.client_request_id, ticket.revision});
  if (it == records_.end() || it->second.manifest.state != CheckpointState::kPreparing)
    return false;
  const uint64_t previous = current_revision(ticket.client_request_id);
  if (previous >= ticket.revision) {
    ++stats_.stale_commit_rejections;
    erase_record(it);
    return false;
  }
  data::DataRef publication_ref;
  const CheckpointPublicationWire publication{
      it->second.manifest.client_request_id, it->second.manifest.revision,
      static_cast<uint64_t>(it->second.payload_bytes), it->second.manifest.valid_tokens,
      it->second.manifest.kv_committed_tokens};
  const auto seal_status = publication_runtime_.seal(
      it->second.publication_handle, sizeof(CheckpointPublicationWire),
      publication_checksum(publication_digest(
          publication, it->second.manifest.model_namespace)),
      &publication_ref);
  if (seal_status != data::DataError::kOk) {
    erase_record(it);
    return false;
  }
  it->second.publication_ref = publication_ref;
  it->second.publication_sealed = true;
  it->second.manifest.state = CheckpointState::kReady;
  current_revision_[ticket.client_request_id] = ticket.revision;
  if (previous != 0 && previous != ticket.revision) {
    const auto old = records_.find({ticket.client_request_id, previous});
    if (old != records_.end() && old->second.restoring_request_id == -1)
      erase_record(old);
  }
  ++stats_.committed;
  stats_.revision_metadata_records = current_revision_.size();
  return true;
}

bool RequestCheckpointStore::save(
    const SequenceState& sequence, const std::string& model_namespace,
    base::KVCacheManager* manager, CheckpointTicket* ticket) {
  CheckpointTicket pending;
  if (!prepare(sequence, model_namespace, manager, &pending)) return false;
  if (!commit(pending)) {
    erase(pending);
    return false;
  }
  *ticket = pending;
  return true;
}

bool RequestCheckpointStore::begin_restore(
    const CheckpointTicket& ticket, const std::string& model_namespace,
    base::KVCacheManager* manager) {
  auto it = records_.find({ticket.client_request_id, ticket.revision});
  const bool current = current_revision(ticket.client_request_id) == ticket.revision;
  if (!manager || it == records_.end() || !current ||
      it->second.manifest.state != CheckpointState::kReady ||
      it->second.manifest.model_namespace != model_namespace) {
    ++stats_.stale_restore_rejections;
    return false;
  }
  if (publication_runtime_.acquire(data::DataKind::kCheckpoint,
                                   it->second.publication_ref.content,
                                   it->second.publication_ref.representation,
                                   &it->second.publication_lease) != data::DataError::kOk) {
    ++stats_.stale_restore_rejections;
    return false;
  }
  base::RequestId restored = -1;
  if (!manager->restore_request_snapshot(it->second.kv, &restored)) {
    it->second.publication_lease.reset();
    return false;
  }
  it->second.restoring_request_id = restored;
  it->second.manifest.state = CheckpointState::kRestoring;
  return true;
}

bool RequestCheckpointStore::commit_restore(
    const CheckpointTicket& ticket, base::KVCacheManager* manager,
    SequenceState* restored) {
  auto it = records_.find({ticket.client_request_id, ticket.revision});
  if (!manager || !restored || it == records_.end() ||
      it->second.manifest.state != CheckpointState::kRestoring ||
      it->second.restoring_request_id == -1) return false;
  const bool current = current_revision(ticket.client_request_id) == ticket.revision;
  if (!current) {
    manager->free_request(it->second.restoring_request_id);
    it->second.restoring_request_id = -1;
    it->second.manifest.state = CheckpointState::kReady;
    ++stats_.stale_restore_rejections;
    erase_record(it);
    return false;
  }
  *restored = it->second.sequence;
  restored->request_id = it->second.restoring_request_id;
  restored->scheduled_tokens = 0;
  restored->status = restored->is_prefill() ? SequenceStatus::kWaiting
                                             : SequenceStatus::kRunning;
  it->second.restoring_request_id = -1;
  it->second.publication_lease.reset();
  it->second.manifest.state = CheckpointState::kReady;
  ++stats_.restored;
  return true;
}

bool RequestCheckpointStore::cancel_restore(
    const CheckpointTicket& ticket, base::KVCacheManager* manager) {
  auto it = records_.find({ticket.client_request_id, ticket.revision});
  if (!manager || it == records_.end() ||
      it->second.manifest.state != CheckpointState::kRestoring) return false;
  manager->free_request(it->second.restoring_request_id);
  it->second.restoring_request_id = -1;
  it->second.publication_lease.reset();
  it->second.manifest.state = CheckpointState::kReady;
  ++stats_.cancelled_restores;
  return true;
}

void RequestCheckpointStore::cancel_client(int64_t client_request_id,
                                           base::KVCacheManager* manager) {
  for (auto it = records_.begin(); it != records_.end();) {
    if (it->first.first != client_request_id) {
      ++it;
      continue;
    }
    if (it->second.restoring_request_id != -1) {
      if (manager && manager->is_valid_request(it->second.restoring_request_id))
        manager->free_request(it->second.restoring_request_id);
      it->second.restoring_request_id = -1;
      ++stats_.cancelled_restores;
    }
    auto doomed = it++;
    erase_record(doomed);
  }
  current_revision_.erase(client_request_id);
  stats_.revision_metadata_records = current_revision_.size();
}

bool RequestCheckpointStore::erase(const CheckpointTicket& ticket) {
  const auto it = records_.find({ticket.client_request_id, ticket.revision});
  if (it == records_.end() || it->second.restoring_request_id != -1) return false;
  const auto current = current_revision_.find(ticket.client_request_id);
  if (current != current_revision_.end() && current->second == ticket.revision)
    current_revision_.erase(current);
  erase_record(it);
  stats_.revision_metadata_records = current_revision_.size();
  return true;
}

void RequestCheckpointStore::erase_record(std::map<Key, Record>::iterator it) {
  if (it == records_.end()) return;
  if (it->second.payload_bytes > payload_bytes_) std::terminate();
  it->second.publication_lease.reset();
  if (it->second.publication_sealed)
    publication_runtime_.withdraw(it->second.publication_ref);
  if (it->second.publication_handle.allocation_id != 0)
    publication_runtime_.release_producer(it->second.publication_handle);
  payload_bytes_ -= it->second.payload_bytes;
  records_.erase(it);
  stats_.payload_bytes = payload_bytes_;
}

bool RequestCheckpointStore::manifest(
    const CheckpointTicket& ticket, RequestCheckpointManifest* manifest) const {
  const auto it = records_.find({ticket.client_request_id, ticket.revision});
  if (!manifest || it == records_.end() ||
      it->second.manifest.state == CheckpointState::kPreparing) return false;
  *manifest = it->second.manifest;
  return true;
}

uint64_t RequestCheckpointStore::current_revision(int64_t client_request_id) const {
  const auto it = current_revision_.find(client_request_id);
  return it == current_revision_.end() ? 0 : it->second;
}

}  // namespace serving
