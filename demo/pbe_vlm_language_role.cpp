#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include <nlohmann/json.hpp>

#include "base/alloc.h"
#include "data/data_client.h"
#include "data/kv_snapshot_codec.h"
#include "data/multimodal_prefix_key.h"
#include "data/tensor_bundle.h"
#include "data/wire_io.h"
#include "model/qwen2.h"
#include "serving/node_memory_budget.h"
#include "serving/scheduler.h"

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr uint32_t kPersistentPDKVPayloadMagic = 0x44505650;

class ScopeExit {
 public:
  explicit ScopeExit(std::function<void()> action) : action_(std::move(action)) {}
  ~ScopeExit() { if (active_) action_(); }
  void dismiss() { active_ = false; }
 private:
  std::function<void()> action_;
  bool active_ = true;
};

std::string Hex(const data::Digest256& digest) {
  static const char alphabet[] = "0123456789abcdef";
  std::string result;
  result.reserve(64);
  for (uint8_t byte : digest) {
    result.push_back(alphabet[byte >> 4]);
    result.push_back(alphabet[byte & 15]);
  }
  return result;
}

bool ParseHex(const std::string& text, data::Digest256* digest) {
  if (!digest || text.size() != 64) return false;
  auto nibble = [](char value) -> int {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
  };
  for (size_t i = 0; i < digest->size(); ++i) {
    const int high = nibble(text[2 * i]);
    const int low = nibble(text[2 * i + 1]);
    if (high < 0 || low < 0) return false;
    (*digest)[i] = static_cast<uint8_t>((high << 4) | low);
  }
  return true;
}

const data::TensorComponent* Find(const data::TensorBundleSchema& schema,
                                  const std::string& name) {
  for (const auto& component : schema.components) {
    if (component.name == name) return &component;
  }
  return nullptr;
}

template <class T>
bool ReadComponent(const std::vector<uint8_t>& bytes,
                   const data::TensorComponent* component,
                   std::vector<T>* output) {
  if (!component || !output || component->byte_length % sizeof(T) != 0 ||
      component->byte_offset > bytes.size() ||
      component->byte_length > bytes.size() - component->byte_offset) {
    return false;
  }
  output->resize(component->byte_length / sizeof(T));
  std::memcpy(output->data(), bytes.data() + component->byte_offset,
              component->byte_length);
  return true;
}

base::ExternalKVPoolBinding Binding(
    const data::IpcPoolDescriptor& descriptor,
    const std::shared_ptr<data::CudaIpcPoolImport>& importer,
    const std::vector<int32_t>& slots,
    const std::vector<int32_t>& initial_slots = {}) {
  base::ExternalKVPoolBinding binding;
  binding.base = importer->mapped();
  binding.bytes = descriptor.bytes;
  binding.device = importer->device();
  binding.num_layers = descriptor.num_layers;
  binding.num_blocks = descriptor.num_blocks;
  binding.block_size = descriptor.block_size;
  binding.num_kv_heads = descriptor.num_kv_heads;
  binding.head_size = descriptor.head_size;
  binding.storage_dtype = static_cast<base::DataType>(descriptor.storage_dtype);
  binding.initial_blocks = initial_slots;
  for (int32_t slot : slots) {
    if (std::find(initial_slots.begin(), initial_slots.end(), slot) ==
        initial_slots.end()) binding.allocatable_blocks.push_back(slot);
  }
  binding.capsule = importer;
  return binding;
}

std::vector<int32_t> ParseSlots(const std::string& value) {
  std::vector<int32_t> output;
  size_t begin = 0;
  while (begin < value.size()) {
    const size_t end = value.find(',', begin);
    const std::string item = value.substr(begin, end == std::string::npos
        ? std::string::npos : end - begin);
    if (!item.empty()) output.push_back(std::stoi(item));
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  std::sort(output.begin(), output.end());
  if (std::adjacent_find(output.begin(), output.end()) != output.end())
    throw std::runtime_error("duplicate_initial_slot");
  return output;
}

std::vector<uint8_t> PackPersistentPDKV(
    const base::ExternalKVRequestState& state,
    const data::LeaseToken& pool_lease, int32_t first_token,
    int64_t rope_delta, const std::vector<int32_t>& prompt) {
  std::vector<uint8_t> output;
  data::wire::Put(kPersistentPDKVPayloadMagic, &output);
  data::wire::Put<uint16_t>(1, &output);
  data::wire::Put<uint16_t>(0, &output);
  data::wire::Put(pool_lease.service_incarnation, &output);
  data::wire::Put(pool_lease.consumer_incarnation, &output);
  data::wire::Put(pool_lease.lease_id, &output);
  data::wire::Put<uint32_t>(first_token, &output);
  uint64_t delta_bits = 0;
  std::memcpy(&delta_bits, &rope_delta, sizeof(delta_bits));
  data::wire::Put(delta_bits, &output);
  data::wire::Put<uint32_t>(state.block_size, &output);
  data::wire::Put<uint32_t>(state.num_layers, &output);
  data::wire::Put<uint32_t>(state.valid_tokens, &output);
  data::wire::Put<uint32_t>(state.committed_tokens, &output);
  data::wire::Put<uint64_t>(prompt.size(), &output);
  for (const auto& ids : state.block_ids_per_layer) {
    data::wire::Put<uint32_t>(ids.size(), &output);
    for (int32_t id : ids) data::wire::Put<uint32_t>(id, &output);
  }
  data::wire::PutBytes(prompt.data(), prompt.size() * sizeof(int32_t), &output);
  return output;
}

bool UnpackPersistentPDKV(
    const std::vector<uint8_t>& bytes, base::ExternalKVRequestState* state,
    data::LeaseToken* pool_lease, int32_t* first_token, int64_t* rope_delta,
    std::vector<int32_t>* prompt) {
  const uint8_t* cursor = bytes.data();
  const uint8_t* end = cursor + bytes.size();
  uint32_t magic = 0, first = 0, block = 0, layers = 0, valid = 0, committed = 0;
  uint16_t version = 0, reserved = 0;
  uint64_t delta_bits = 0, prompt_size = 0;
  if (!data::wire::Get(&cursor, end, &magic) ||
      !data::wire::Get(&cursor, end, &version) ||
      !data::wire::Get(&cursor, end, &reserved) ||
      !data::wire::Get(&cursor, end, &pool_lease->service_incarnation) ||
      !data::wire::Get(&cursor, end, &pool_lease->consumer_incarnation) ||
      !data::wire::Get(&cursor, end, &pool_lease->lease_id) ||
      !data::wire::Get(&cursor, end, &first) ||
      !data::wire::Get(&cursor, end, &delta_bits) ||
      !data::wire::Get(&cursor, end, &block) ||
      !data::wire::Get(&cursor, end, &layers) ||
      !data::wire::Get(&cursor, end, &valid) ||
      !data::wire::Get(&cursor, end, &committed) ||
      !data::wire::Get(&cursor, end, &prompt_size) ||
      magic != kPersistentPDKVPayloadMagic || version != 1 || reserved ||
      layers == 0 || layers > 256 || !pool_lease->lease_id) return false;
  state->schema_version = 1;
  state->block_size = block;
  state->num_layers = layers;
  state->valid_tokens = valid;
  state->committed_tokens = committed;
  state->block_ids_per_layer.resize(layers);
  for (auto& ids : state->block_ids_per_layer) {
    uint32_t count = 0;
    if (!data::wire::Get(&cursor, end, &count) || count > 65536) return false;
    ids.resize(count);
    for (auto& id : ids) {
      uint32_t raw = 0;
      if (!data::wire::Get(&cursor, end, &raw) || raw > INT32_MAX) return false;
      id = static_cast<int32_t>(raw);
    }
  }
  if (prompt_size > static_cast<uint64_t>(end - cursor) / sizeof(int32_t) ||
      prompt_size * sizeof(int32_t) != static_cast<uint64_t>(end - cursor)) return false;
  prompt->resize(prompt_size);
  std::memcpy(prompt->data(), cursor, prompt_size * sizeof(int32_t));
  *first_token = static_cast<int32_t>(first);
  std::memcpy(rope_delta, &delta_bits, sizeof(delta_bits));
  return true;
}

struct RequestInput {
  std::string request_id;
  uint64_t generation = 0;
  int32_t max_new_tokens = 0;
  int32_t cancel_after_tokens = 0;
  std::string diagnostic_key;
  std::string diagnostic_mode;
  int32_t diagnostic_dump_logits_step = -1;
  std::string feature_content;
  std::string feature_representation;
  std::vector<int32_t> semantic_tokens;
  int32_t matched_prefix_tokens = 0;
  bool prefix_boundary_observed = false;
  int64_t prefix_lookup_us = 0;
  Clock::time_point deadline;
  data::RemoteDataLease bundle;
  size_t feature_offset = 0;
  size_t feature_bytes = 0;
  std::vector<int32_t> tokens;
  std::vector<int32_t> positions;
  int64_t rope_delta = 0;
  int32_t image_begin = -1;
  int32_t feature_rows = 0;
  int32_t hidden = 0;
  int64_t client_id = -1;
  std::vector<serving::MemoryDemand> demands;
};

bool BuildSemanticTokens(RequestInput* input,
                         const std::string& media_content_text) {
  data::ContentId media_content;
  if (!input || !ParseHex(media_content_text, &media_content.digest)) return false;
  auto image = std::find(input->tokens.begin(), input->tokens.end(), 151655);
  input->image_begin = static_cast<int32_t>(image - input->tokens.begin());
  if (image == input->tokens.end() ||
      input->image_begin + input->feature_rows >
          static_cast<int32_t>(input->tokens.size())) return false;
  std::vector<uint8_t> semantic_wire;
  static constexpr char kAdapterRevision[] = "qwen25-vl-pbe-adapter-v3";
  static constexpr char kModelRevision[] = "qwen25-vl-3b-instruct-pbe-bf16-v1";
  semantic_wire.insert(semantic_wire.end(), kAdapterRevision,
                       kAdapterRevision + sizeof(kAdapterRevision) - 1);
  semantic_wire.insert(semantic_wire.end(), kModelRevision,
                       kModelRevision + sizeof(kModelRevision) - 1);
  semantic_wire.insert(semantic_wire.end(), media_content.digest.begin(),
                       media_content.digest.end());
  semantic_wire.insert(semantic_wire.end(), input->feature_representation.begin(),
                       input->feature_representation.end());
  for (int axis = 0; axis < 3; ++axis) {
    const auto* begin = input->positions.data() +
        axis * input->tokens.size() + input->image_begin;
    semantic_wire.insert(semantic_wire.end(),
        reinterpret_cast<const uint8_t*>(begin),
        reinterpret_cast<const uint8_t*>(begin + input->feature_rows));
  }
  semantic_wire.insert(semantic_wire.end(),
      reinterpret_cast<const uint8_t*>(&input->rope_delta),
      reinterpret_cast<const uint8_t*>(&input->rope_delta) +
          sizeof(input->rope_delta));
  const data::ContentId semantic_media{
      data::DataChecksum(semantic_wire.data(), semantic_wire.size())};
  return data::MakeMultimodalRadixTokens(
      input->tokens,
      {{static_cast<uint32_t>(input->image_begin),
        static_cast<uint32_t>(input->feature_rows), semantic_media}},
      &input->semantic_tokens);
}

struct NumericalSnapshot {
  std::vector<float> logits;
  std::vector<int32_t> prompt_tokens;
  std::vector<int32_t> positions;
  std::vector<int32_t> output_tokens;
  int64_t rope_delta = 0;
  int32_t selected_token = -1;
  std::string mode;
};

std::pair<int32_t, int32_t> Top2(const std::vector<float>& logits) {
  int32_t first = -1;
  int32_t second = -1;
  for (int32_t token = 0; token < static_cast<int32_t>(logits.size()); ++token) {
    if (first < 0 || logits[token] > logits[first] ||
        (logits[token] == logits[first] && token < first)) {
      second = first;
      first = token;
    } else if (second < 0 || logits[token] > logits[second] ||
               (logits[token] == logits[second] && token < second)) {
      second = token;
    }
  }
  return {first, second};
}

class BatchResourceGuard {
 public:
  BatchResourceGuard(data::DataClient* client, serving::NodeMemoryBudget* budget,
                     std::vector<RequestInput>* requests)
      : client_(client), budget_(budget), requests_(requests) {}
  ~BatchResourceGuard() { Release(); }
  void Release() {
    if (!active_) return;
    for (auto& request : *requests_) {
      if (!request.demands.empty()) budget_->release(request.demands);
      if (request.bundle.token.lease_id) client_->release(request.bundle.token);
      request.demands.clear();
      request.bundle.token = {};
    }
    active_ = false;
  }
 private:
  data::DataClient* client_;
  serving::NodeMemoryBudget* budget_;
  std::vector<RequestInput>* requests_;
  bool active_ = true;
};

Json PoolJson(const serving::NodeMemoryBudget& budget,
              serving::MemoryPool pool) {
  const auto state = budget.state(pool);
  return {{"capacity", state.capacity}, {"used", state.used}, {"peak", state.peak}};
}

class LanguageRole {
 public:
  LanguageRole(const std::string& model_path, const std::string& tokenizer,
               const std::string& endpoint, int device, size_t bundle_capacity,
               size_t staging_capacity, size_t requested_kv_slots,
               bool radix_cache_enabled, bool direction_lanes_enabled,
               std::vector<int32_t> initial_slots = {},
               const std::string& weight_mode = "private",
               const std::string& model_sha256 = "",
               const std::string& weight_layout = "")
      : client_(endpoint),
        model_(std::make_unique<model::Qwen2Model>(
            base::TokenizerType::kEncodeBpe, tokenizer, model_path, false)),
        device_(device), initial_slots_(std::move(initial_slots)) {
    ScopeExit rollback([this] {
      model_.reset();
      shared_weight_import_.reset();
      if (shared_weight_lease_active_) {
        client_.release_shared_weight(shared_weight_lease_.token);
        shared_weight_lease_active_ = false;
      }
      importer_.reset();
      if (grant_active_) {
        client_.release_ipc_slots(grant_.token);
        grant_active_ = false;
      }
    });
    const auto pool_error = client_.get_ipc_pool(&descriptor_);
    if (pool_error != data::DataError::kOk) {
      throw std::runtime_error("get_ipc_pool_failed:" +
                               std::to_string(static_cast<int>(pool_error)));
    }
    if (descriptor_.device != device_) {
      throw std::runtime_error("persistent_language_role_requires_same_gpu_ipc_pool");
    }
    const size_t slots = requested_kv_slots == 0
        ? descriptor_.num_blocks : requested_kv_slots;
    if (slots > static_cast<size_t>(descriptor_.num_blocks) ||
        client_.reserve_ipc_slots(slots, &grant_) !=
        data::DataError::kOk) {
      throw std::runtime_error("reserve_requested_ipc_slots_failed");
    }
    grant_active_ = true;
    importer_ = std::make_shared<data::CudaIpcPoolImport>();
    std::string error;
    uint64_t copied = 0;
    double copy_ms = 0;
    std::string path;
    if (!importer_->open(descriptor_, device_, &error) ||
        !importer_->prepare_compute_view(nullptr, &copied, &copy_ms, &path, &error)) {
      client_.release_ipc_slots(grant_.token);
      grant_active_ = false;
      throw std::runtime_error("open_ipc_pool_failed:" + error);
    }
    model_->set_kv_cache_blocks_per_layer(descriptor_.num_blocks);
    model_->set_serving_workspace_token_capacity(512);
    model_->set_radix_cache_enabled(radix_cache_enabled);
    radix_cache_enabled_ = radix_cache_enabled;
    host_cache_bytes_ = size_t{256} << 20;
    model_->set_host_cache_config(
        {true, host_cache_bytes_, 256, 2, direction_lanes_enabled});
    for (int32_t slot : initial_slots_) {
      if (slot < 0 || slot >= descriptor_.num_blocks ||
          std::find(grant_.slots.begin(), grant_.slots.end(), slot) != grant_.slots.end())
        throw std::runtime_error("invalid_or_owned_initial_slot");
    }
    model_->set_external_kv_pool(
        Binding(descriptor_, importer_, grant_.slots, initial_slots_));
    model_file_bytes_ = std::filesystem::file_size(model_path);
    weight_mode_ = weight_mode;
    if (weight_mode_ == "shared") {
      data::Digest256 model_content{};
      if (!ParseHex(model_sha256, &model_content) ||
          weight_layout != data::kQwen2Bf16SharedWeightLayout) {
        throw std::runtime_error("invalid_shared_weight_identity");
      }
      const auto layout_identity = data::DataChecksum(
          reinterpret_cast<const uint8_t*>(weight_layout.data()), weight_layout.size());
      const auto acquire = client_.acquire_shared_weight(
          model_content, layout_identity, base::DataType::kDataTypeBf16,
          model_file_bytes_, &shared_weight_lease_);
      if (acquire != data::DataError::kOk) {
        throw std::runtime_error("acquire_shared_weight_failed:" +
                                 std::to_string(static_cast<int>(acquire)));
      }
      shared_weight_lease_active_ = true;
      shared_weight_import_ = std::make_shared<data::SharedWeightImport>();
      if (!shared_weight_import_->open(shared_weight_lease_.descriptor, device_, &error) ||
          !shared_weight_import_->wait_ready(nullptr, &error)) {
        client_.release_shared_weight(shared_weight_lease_.token);
        shared_weight_lease_active_ = false;
        throw std::runtime_error("open_shared_weight_failed:" + error);
      }
      model::SharedWeightBinding binding;
      binding.device_file_base = shared_weight_import_->mapped();
      binding.file_bytes = shared_weight_lease_.descriptor.bytes;
      binding.dtype = shared_weight_lease_.descriptor.dtype;
      binding.owner_incarnation = shared_weight_lease_.descriptor.owner_incarnation;
      binding.allocation_id = shared_weight_lease_.descriptor.allocation_id;
      binding.generation = shared_weight_lease_.descriptor.generation;
      binding.capsule = shared_weight_import_;
      model_->set_shared_weight_binding(std::move(binding));
    } else if (weight_mode_ != "private") {
      throw std::runtime_error("unsupported_weight_mode");
    }
    if (cudaMemGetInfo(&device_free_before_model_, &device_total_bytes_) != cudaSuccess)
      throw std::runtime_error("cuda_mem_info_before_model_failed");
    const auto status = model_->init(base::DeviceType::kDeviceCUDA, device_);
    if (!status) {
      model_.reset();
      importer_.reset();
      client_.release_ipc_slots(grant_.token);
      grant_active_ = false;
      throw std::runtime_error(status.get_err_msg());
    }
    // Qwen2Model owns one persistent KV request for its legacy single-sequence
    // forward() path.  Record the post-init value so serving recovery is
    // measured against the model's real steady-state baseline rather than an
    // incorrect absolute zero.
    model_kv_request_baseline_ =
        model_->kv_cache_manager()->num_active_requests();
    size_t device_free_after_model = 0;
    size_t device_total_after_model = 0;
    if (cudaMemGetInfo(&device_free_after_model, &device_total_after_model) != cudaSuccess ||
        device_total_after_model != device_total_bytes_)
      throw std::runtime_error("cuda_mem_info_after_model_failed");
    device_free_low_water_ = device_free_after_model;

    const size_t block_bytes = descriptor_.bytes / descriptor_.num_blocks;
    const auto capacity = model_->serving_capacity_info();
    const size_t workspace_bytes = capacity.serving_workspace_reserved_bytes;
    const size_t model_allocation_bytes =
        device_free_before_model_ >= device_free_after_model
            ? device_free_before_model_ - device_free_after_model
            : 0;
    const size_t private_model_nonworkspace_bytes =
        model_allocation_bytes > workspace_bytes
            ? model_allocation_bytes - workspace_bytes : model_allocation_bytes;
    const size_t weight_bytes = weight_mode_ == "shared"
        ? 0 : private_model_nonworkspace_bytes;
    private_model_nonworkspace_bytes_ = private_model_nonworkspace_bytes;
    model_allocation_bytes_ = model_allocation_bytes;
    workspace_bytes_per_token_ = capacity.workspace_bytes_per_token;
    workspace_reserved_bytes_ = workspace_bytes;
    device_admission_limit_bytes_ = device_total_bytes_ > (size_t{256} << 20)
        ? device_total_bytes_ - (size_t{256} << 20) : device_total_bytes_;
    budget_.set_capacity(serving::MemoryPool::kWeights, weight_bytes);
    budget_.set_capacity(serving::MemoryPool::kWorkspaces, workspace_bytes);
    allocatable_kv_bytes_ = grant_.slots.size() * block_bytes;
    budget_.set_capacity(serving::MemoryPool::kKV, allocatable_kv_bytes_);
    budget_.set_capacity(serving::MemoryPool::kBundles, bundle_capacity);
    budget_.set_capacity(serving::MemoryPool::kStaging, staging_capacity);
    budget_.set_capacity(serving::MemoryPool::kQuarantine, block_bytes * 2);
    budget_.set_capacity(serving::MemoryPool::kHostCache, host_cache_bytes_);
    persistent_demands_ = {
        {serving::MemoryPool::kWeights, weight_bytes},
        {serving::MemoryPool::kWorkspaces, workspace_bytes},
    };
    if (!budget_.reserve(persistent_demands_)) {
      throw std::runtime_error("fixed_model_budget_reservation_failed");
    }
    budget_.freeze();
    rollback.dismiss();
  }

  ~LanguageRole() {
    for (const auto& item : pd_holds_) {
      if (model_->kv_cache_manager()->is_valid_request(item.second.request_id))
        model_->kv_cache_manager()->free_request(item.second.request_id);
    }
    pd_holds_.clear();
    cudaDeviceSynchronize();
    model_.reset();
    // Tensor destruction returns allocations to the process-wide caching
    // allocator.  A stopped role has no subsequent workload that can reuse
    // them, so return the cache to CUDA before closing IPC mappings.
    base::CUDADeviceAllocatorFactory::get_instance()->release_all_cached();
    shared_weight_import_.reset();
    if (shared_weight_lease_active_) {
      client_.release_shared_weight(shared_weight_lease_.token);
      shared_weight_lease_active_ = false;
    }
    importer_.reset();
    if (grant_active_) client_.release_ipc_slots(grant_.token);
  }

  bool Cancel(const std::string& request_id, uint64_t generation) {
    if (request_id.empty() || generation == 0) return false;
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    const auto latest = latest_generations_.find(request_id);
    if (latest != latest_generations_.end() && generation < latest->second)
      return false;
    auto& cancelled = cancelled_generations_[request_id];
    cancelled = std::max(cancelled, generation);
    return true;
  }

  bool RequestCheckpoint(const std::string& request_id, uint64_t generation) {
    if (request_id.empty() || generation == 0) return false;
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    const auto latest = latest_generations_.find(request_id);
    if (latest != latest_generations_.end() && generation < latest->second)
      return false;
    auto& requested = checkpoint_generations_[request_id];
    requested = std::max(requested, generation);
    return true;
  }

  void EnsureOwnerAlive() const {
    uint64_t incarnation = 0;
    const auto status = client_.ping(&incarnation);
    if (status != data::DataError::kOk ||
        incarnation != descriptor_.owner_incarnation) {
      throw std::runtime_error("data_owner_unavailable_or_restarted_fail_stop");
    }
  }

  void CancelAll() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    cancel_all_ = true;
  }

  Json DemotePrefixToHost() {
    auto* manager = model_->kv_cache_manager();
    int32_t admitted = 0;
    while (true) {
      const int32_t batch = manager->demote_radix_cache_to_host();
      admitted += batch;
      if (!manager->drain_cache_transfers())
        return {{"ok", false}, {"error", "host_demotion_drain_failed"}};
      if (batch == 0) break;
    }
    const auto& stats = manager->radix_cache_stats();
    return {{"ok", true}, {"event", "prefix_demoted_to_host"},
            {"admitted_pages", admitted},
            {"host_demoted_blocks", stats.host_demoted_blocks}};
  }

  Json DemotePrefixAsync() {
    auto* manager = model_->kv_cache_manager();
    const int32_t admitted = manager->demote_radix_cache_to_host(2);
    const auto* scheduler = manager->transfer_scheduler_stats();
    return {{"ok", true}, {"event", "prefix_demotion_submitted"},
            {"admitted_pages", admitted},
            {"physical_submissions", scheduler ? scheduler->physical_submissions : 0}};
  }

  Json ClearPrefixCache() {
    auto* manager = model_->kv_cache_manager();
    manager->clear_radix_cache();
    return {{"ok", true}, {"event", "prefix_cache_cleared"},
            {"recovery_objects", manager->recovery_dependency_object_count()}};
  }

  Json PDPrefill(const Json& command) {
    EnsureOwnerAlive();
    if (!command.contains("request") || !command["request"].is_object())
      throw std::runtime_error("pd_prefill_request_missing");
    const auto started = Clock::now();
    const auto& item = command["request"];
    RequestInput input;
    input.request_id = item.value("request_id", "");
    input.generation = item.value("generation", uint64_t{0});
    input.max_new_tokens = item.value("max_new_tokens", 16);
    input.feature_content = item.value("feature_content", "");
    input.feature_representation = item.value("feature_representation", "");
    if (input.request_id.empty() || input.generation == 0 ||
        input.max_new_tokens <= 0 || input.max_new_tokens > 256)
      throw std::runtime_error("invalid_pd_prefill_identity");
    const std::string held_key = input.request_id + ":" +
        std::to_string(input.generation);
    if (pd_holds_.count(held_key)) throw std::runtime_error("duplicate_pd_handoff");
    std::string generation_error;
    if (!AdmitGeneration(input.request_id, input.generation, &generation_error))
      throw std::runtime_error(generation_error);

    data::ContentId content;
    data::RepresentationId representation;
    if (!ParseHex(item.value("content", ""), &content.digest))
      throw std::runtime_error("invalid_pd_bundle_content");
    const std::string representation_text = item.value("representation", "");
    if (representation_text.empty())
      representation = data::RepresentationIdFromString("qwen25-vl-request-bundle-v3");
    else if (!ParseHex(representation_text, &representation.digest))
      throw std::runtime_error("invalid_pd_bundle_representation");
    if (client_.acquire(data::DataKind::kTensorBundle, content, representation,
            data::DataLeaseKind::kCompute, &input.bundle) != data::DataError::kOk)
      throw std::runtime_error("pd_bundle_acquire_failed");
    struct BundleRelease {
      data::DataClient* client;
      data::RemoteDataLease* lease;
      ~BundleRelease() { if (lease->token.lease_id) client->release(lease->token); }
    } bundle_release{&client_, &input.bundle};
    data::TensorBundleSchema schema;
    std::string reason;
    if (data::DecodeTensorBundlePayload(input.bundle.bytes.data(),
            input.bundle.bytes.size(), &schema, &reason) != data::DataError::kOk)
      throw std::runtime_error("pd_bundle_preflight_failed:" + reason);
    const auto* features = Find(schema, "image_features");
    std::vector<int64_t> deltas;
    if (!features || features->shape.size() != 2 ||
        !ReadComponent(input.bundle.bytes, Find(schema, "input_ids"), &input.tokens) ||
        !ReadComponent(input.bundle.bytes, Find(schema, "position_ids"), &input.positions) ||
        !ReadComponent(input.bundle.bytes, Find(schema, "rope_delta"), &deltas) ||
        deltas.size() != 1 || input.positions.size() != input.tokens.size() * 3 ||
        features->byte_length != features->shape[0] * features->shape[1] * 2)
      throw std::runtime_error("pd_bundle_shape_or_coverage_mismatch");
    input.rope_delta = deltas[0];
    input.feature_offset = features->byte_offset;
    input.feature_bytes = features->byte_length;
    input.feature_rows = static_cast<int32_t>(features->shape[0]);
    input.hidden = static_cast<int32_t>(features->shape[1]);
    if (input.hidden != hidden_size_ ||
        !BuildSemanticTokens(&input, input.feature_content))
      throw std::runtime_error("pd_semantic_input_invalid");

    const size_t block_bytes = descriptor_.bytes / descriptor_.num_blocks;
    const size_t blocks = (input.tokens.size() + input.max_new_tokens +
        descriptor_.block_size - 1) / descriptor_.block_size;
    input.demands = {{serving::MemoryPool::kKV, blocks * block_bytes},
                     {serving::MemoryPool::kBundles, input.bundle.bytes.size()},
                     {serving::MemoryPool::kStaging, input.feature_bytes}};
    if (!budget_.reserve(input.demands))
      throw std::runtime_error("pd_prefill_budget_exhausted");
    bool demands_owned = true;
    try {
      serving::SchedulerConfig config;
      config.max_num_seqs = 1;
      config.max_num_batched_tokens = 512;
      config.prefill_chunk_cap = 512;
      serving::Scheduler scheduler(config, model_->kv_cache_manager());
      serving::GenerationConfig generation(1);
      generation.sampling.repetition_penalty = 1.05;
      const int64_t client_id = scheduler.add_request(input.semantic_tokens, generation);
      auto output = scheduler.schedule_step();
      if (output.scheduled_seqs.size() != 1 || output.num_prefill_seqs != 1 ||
          output.num_decode_seqs != 0 ||
          output.num_tokens_per_seq[0] != static_cast<int32_t>(input.tokens.size()))
        throw std::runtime_error("pd_prefill_not_single_complete_prompt");
      auto batch = scheduler.build_mixed_batch(
          output, model_->device_context()->compute_queue);
      auto embeddings = model_->embedding(input.tokens);
      auto allocator = base::CUDADeviceAllocatorFactory::get_instance();
      allocator->memcpy(input.bundle.bytes.data() + input.feature_offset,
          embeddings.input_embeddings.ptr<uint16_t>(
              static_cast<size_t>(input.image_begin) * input.hidden),
          input.feature_bytes, base::MemcpyKind::kMemcpyCPU2CUDA,
          model_->device_context()->compute_queue, false);
      tensor::Tensor mrope(base::DataType::kDataTypeInt32, 3, input.tokens.size(),
                           true, allocator);
      allocator->memcpy(input.positions.data(), mrope.ptr<int32_t>(), mrope.byte_size(),
          base::MemcpyKind::kMemcpyCPU2CUDA, model_->device_context()->compute_queue, false);
      batch.input_embeddings_override = embeddings.input_embeddings;
      batch.mrope_positions = mrope;
      const auto forward = model_->forward_mixed_batch(batch);
      if (!forward) throw std::runtime_error(forward.get_err_msg());
      const auto sampled = model_->batch_sample(batch, output);
      if (sampled.size() != 1) throw std::runtime_error("pd_prefill_first_token_missing");
      const int32_t first_token = sampled[0];
      if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("pd_prefill_sync_failed");
      base::ExternalKVRequestState state;
      if (!model_->kv_cache_manager()->export_external_request(
              batch.request_ids[0], &state))
        throw std::runtime_error("pd_prefill_export_failed");

      Json oracle = Json::array();
      const int32_t oracle_steps = command.value("oracle_steps", 0);
      if (oracle_steps > 0) {
        base::RequestId branch = -1;
        base::BranchTokenBoundaries boundaries{state.valid_tokens, 0, 0};
        if (!model_->kv_cache_manager()->fork_request(
                batch.request_ids[0], boundaries, true, first_token, &branch))
          throw std::runtime_error("pd_oracle_fork_failed");
        oracle = RunPDDecode(branch, input.tokens, first_token,
                             input.rope_delta, oracle_steps)["tokens"];
      }
      const auto payload = PackPersistentPDKV(
          state, grant_.token, first_token, input.rope_delta, input.tokens);
      data::DataReservation reservation;
      uint64_t owner = 0;
      if (client_.ping(&owner) != data::DataError::kOk)
        throw std::runtime_error("pd_data_owner_unavailable");
      reservation.operation = {owner, (uint64_t(getpid()) << 32) ^
          static_cast<uint64_t>(Clock::now().time_since_epoch().count())};
      reservation.kind = data::DataKind::kKVPage;
      reservation.content = data::ContentIdFromBytes(payload.data(), payload.size());
      reservation.representation = data::RepresentationIdFromString(
          "pbe-v4-persistent-pd-kv-v1");
      reservation.logical_bytes = payload.size();
      data::AllocationHandle allocation;
      if (client_.reserve(reservation, &allocation) != data::DataError::kOk)
        throw std::runtime_error("pd_handoff_reserve_failed");
      data::DataRef ref;
      const auto sealed = client_.seal(allocation, payload,
          data::DataChecksum(payload.data(), payload.size()), &ref);
      client_.release_producer(allocation);
      if (sealed != data::DataError::kOk)
        throw std::runtime_error("pd_handoff_seal_failed");
      PDHold hold;
      hold.request_id = batch.request_ids[0];
      hold.demands = input.demands;
      input.demands.clear();
      demands_owned = false;
      pd_holds_.emplace(held_key, std::move(hold));
      ++pd_prefill_requests_;
      RecordDeviceMemory();
      return {{"ok", true}, {"worker_pid", getpid()},
              {"request_id", input.request_id}, {"generation", input.generation},
              {"handoff_valid", true}, {"kv_content", Hex(ref.content.digest)},
              {"kv_representation", Hex(reservation.representation.digest)},
              {"pool_grant", grant_.token.lease_id}, {"pool_slots", grant_.slots},
              {"prompt_tokens", input.tokens.size()},
              {"exported_valid_tokens", state.valid_tokens},
              {"first_token", first_token}, {"oracle_tokens", std::move(oracle)},
              {"prefill_ms", std::chrono::duration<double, std::milli>(
                  Clock::now() - started).count()},
              {"scheduler_client_id", client_id}};
    } catch (...) {
      if (demands_owned) budget_.release(input.demands);
      throw;
    }
  }

  Json PDDecode(const Json& command) {
    EnsureOwnerAlive();
    const auto started = Clock::now();
    data::ContentId content;
    data::RepresentationId representation;
    if (!ParseHex(command.value("kv_content", ""), &content.digest) ||
        !ParseHex(command.value("kv_representation", ""), &representation.digest))
      throw std::runtime_error("invalid_pd_handoff_identity");
    data::RemoteDataLease lease;
    if (client_.acquire(data::DataKind::kKVPage, content, representation,
            data::DataLeaseKind::kCompute, &lease) != data::DataError::kOk)
      throw std::runtime_error("pd_handoff_acquire_failed");
    struct LeaseRelease {
      data::DataClient* client;
      data::RemoteDataLease* lease;
      ~LeaseRelease() { if (lease->token.lease_id) client->release(lease->token); }
    } lease_release{&client_, &lease};
    base::ExternalKVRequestState state;
    data::LeaseToken prefix_grant;
    int32_t first_token = -1;
    int64_t rope_delta = 0;
    std::vector<int32_t> prompt;
    if (!UnpackPersistentPDKV(lease.bytes, &state, &prefix_grant,
            &first_token, &rope_delta, &prompt))
      throw std::runtime_error("pd_handoff_decode_failed");
    std::vector<int32_t> pages = state.block_ids_per_layer.empty()
        ? std::vector<int32_t>{} : state.block_ids_per_layer.front();
    for (const auto& layer : state.block_ids_per_layer)
      if (layer != pages) throw std::runtime_error("pd_handoff_layer_ids_diverged");
    for (int32_t page : pages)
      if (!std::binary_search(initial_slots_.begin(), initial_slots_.end(), page))
        throw std::runtime_error("pd_handoff_page_not_in_prefill_grant");
    base::RequestId request_id = -1;
    if (!model_->kv_cache_manager()->restore_external_shared_request(state, &request_id))
      throw std::runtime_error("pd_handoff_restore_failed");
    Json result = RunPDDecode(request_id, prompt, first_token, rope_delta,
                              command.value("max_new_tokens", 16));
    result["ok"] = true;
    result["worker_pid"] = getpid();
    result["request_id"] = command.value("request_id", "");
    result["generation"] = command.value("generation", uint64_t{0});
    result["handoff_valid"] = prefix_grant.lease_id != 0 && !pages.empty();
    result["prefix_grant"] = prefix_grant.lease_id;
    result["private_grant"] = grant_.token.lease_id;
    result["imported_metadata_bytes"] = lease.bytes.size();
    result["prompt_tokens"] = prompt.size();
    result["prefill_tokens_saved"] = state.valid_tokens;
    result["actual_computed_prompt_tokens"] = 0;
    result["decode_ms"] = std::chrono::duration<double, std::milli>(
        Clock::now() - started).count();
    ++pd_decode_requests_;
    RecordDeviceMemory();
    return result;
  }

  Json PDRelease(const Json& command) {
    const std::string key = command.value("request_id", "") + ":" +
        std::to_string(command.value("generation", uint64_t{0}));
    auto found = pd_holds_.find(key);
    if (found == pd_holds_.end())
      return {{"ok", false}, {"error", "pd_handoff_not_found"}};
    model_->kv_cache_manager()->free_request(found->second.request_id);
    budget_.release(found->second.demands);
    pd_holds_.erase(found);
    return {{"ok", true}, {"worker_pid", getpid()},
            {"event", "pd_handoff_released"}, {"remaining_handoffs", pd_holds_.size()}};
  }

  Json Probe(const Json& command) {
    EnsureOwnerAlive();
    if (!command.contains("request") || !command["request"].is_object())
      throw std::runtime_error("probe_request_missing");
    const auto& item = command["request"];
    RequestInput input;
    input.max_new_tokens = item.value("max_new_tokens", 16);
    data::ContentId content;
    data::RepresentationId representation;
    if (!ParseHex(item.value("content", ""), &content.digest))
      throw std::runtime_error("invalid_bundle_content_id");
    const std::string representation_text = item.value("representation", "");
    if (representation_text.empty())
      representation = data::RepresentationIdFromString("qwen25-vl-request-bundle-v3");
    else if (!ParseHex(representation_text, &representation.digest))
      throw std::runtime_error("invalid_bundle_representation_id");
    input.feature_content = item.value("feature_content", "");
    input.feature_representation = item.value("feature_representation", "");
    data::Digest256 feature_representation_digest{};
    if (!ParseHex(input.feature_representation, &feature_representation_digest))
      throw std::runtime_error("invalid_feature_representation_id");
    const auto acquired = client_.acquire(data::DataKind::kTensorBundle, content,
        representation, data::DataLeaseKind::kCompute, &input.bundle);
    if (acquired != data::DataError::kOk)
      throw std::runtime_error("probe_bundle_acquire_failed");
    struct Release {
      data::DataClient* client;
      data::RemoteDataLease* lease;
      ~Release() { if (lease->token.lease_id) client->release(lease->token); }
    } release{&client_, &input.bundle};
    data::TensorBundleSchema schema;
    std::string reason;
    if (data::DecodeTensorBundlePayload(input.bundle.bytes.data(),
        input.bundle.bytes.size(), &schema, &reason) != data::DataError::kOk)
      throw std::runtime_error("probe_bundle_preflight_failed:" + reason);
    const auto* features = Find(schema, "image_features");
    std::vector<int64_t> deltas;
    if (!features || features->shape.size() != 2 ||
        !ReadComponent(input.bundle.bytes, Find(schema, "input_ids"), &input.tokens) ||
        !ReadComponent(input.bundle.bytes, Find(schema, "position_ids"), &input.positions) ||
        !ReadComponent(input.bundle.bytes, Find(schema, "rope_delta"), &deltas) ||
        deltas.size() != 1 || input.positions.size() != input.tokens.size() * 3)
      throw std::runtime_error("probe_bundle_shape_or_coverage_mismatch");
    input.rope_delta = deltas[0];
    input.feature_bytes = features->byte_length;
    input.feature_rows = static_cast<int32_t>(features->shape[0]);
    input.hidden = static_cast<int32_t>(features->shape[1]);
    if (input.hidden != hidden_size_ || !BuildSemanticTokens(&input, input.feature_content))
      throw std::runtime_error("probe_semantic_prefix_failed");
    const size_t block_bytes = descriptor_.bytes / descriptor_.num_blocks;
    const size_t blocks = (input.tokens.size() + input.max_new_tokens +
        descriptor_.block_size - 1) / descriptor_.block_size;
    const size_t required_kv = blocks * block_bytes;
    const auto kv = budget_.state(serving::MemoryPool::kKV);
    const auto bundles = budget_.state(serving::MemoryPool::kBundles);
    const auto staging = budget_.state(serving::MemoryPool::kStaging);
    const size_t kv_free = kv.capacity - kv.used;
    const size_t bundle_free = bundles.capacity - bundles.used;
    const size_t staging_free = staging.capacity - staging.used;
    const auto prefix = model_->kv_cache_manager()->probe_radix_cache(
        input.semantic_tokens);
    return {{"ok", true}, {"worker_pid", getpid()},
            {"admissible", kv_free >= required_kv &&
                 bundle_free >= input.bundle.bytes.size() &&
                 staging_free >= input.feature_bytes},
            {"capacity", {{"kv_free_bytes", kv_free},
                           {"bundle_free_bytes", bundle_free},
                           {"staging_free_bytes", staging_free}}},
            {"required", {{"kv_bytes", required_kv},
                           {"bundle_bytes", input.bundle.bytes.size()},
                           {"staging_bytes", input.feature_bytes}}},
            {"prefix", {{"prompt_tokens", prefix.prompt_tokens},
                         {"matched_tokens", prefix.matched_tokens},
                         {"gpu_pages", prefix.gpu_pages},
                         {"host_pages", prefix.host_pages},
                         {"missing_pages", prefix.missing_pages},
                         {"missing_bytes", prefix.missing_pages * block_bytes},
                         {"fully_serviceable", prefix.fully_serviceable}}}};
  }

  Json Infer(const Json& command) {
    EnsureOwnerAlive();
    const auto started = Clock::now();
    if (!command.contains("requests") || !command["requests"].is_array() ||
        command["requests"].empty() || command["requests"].size() > 8) {
      throw std::runtime_error("requests_must_be_array_of_1_to_8");
    }
    std::vector<RequestInput> requests;
    BatchResourceGuard resource_guard(&client_, &budget_, &requests);
    requests.reserve(command["requests"].size());
    std::map<int64_t, size_t> client_to_index;
    size_t rejected = 0;
    size_t acquired_bundle_bytes = 0;
    size_t admitted_kv_bytes = 0;
    size_t admitted_staging_bytes = 0;
    Json rejection_items = Json::array();
    const size_t block_bytes = descriptor_.bytes / descriptor_.num_blocks;

    for (const auto& item : command["requests"]) {
      RequestInput input;
      input.request_id = item.value("request_id", "");
      input.generation = item.value("generation", uint64_t{0});
      input.max_new_tokens = item.value("max_new_tokens", 16);
      input.cancel_after_tokens = item.value("cancel_after_tokens", 0);
      input.diagnostic_key = item.value("diagnostic_key", "");
      input.diagnostic_mode = item.value("diagnostic_mode", "");
      input.diagnostic_dump_logits_step =
          item.value("diagnostic_dump_logits_step", -1);
      const int timeout_ms = item.value("timeout_ms", 30000);
      if (input.request_id.empty() || input.generation == 0 ||
          input.max_new_tokens <= 0 || input.max_new_tokens > 256 || timeout_ms <= 0 ||
          input.cancel_after_tokens < 0) {
        throw std::runtime_error("invalid_typed_request_fields");
      }
      if (item.contains("deadline_monotonic_ns")) {
        const auto deadline_ns = item["deadline_monotonic_ns"].get<int64_t>();
        input.deadline = Clock::time_point(std::chrono::nanoseconds(deadline_ns));
      } else {
        input.deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
      }
      std::string generation_error;
      if (!AdmitGeneration(input.request_id, input.generation, &generation_error)) {
        rejection_items.push_back({{"request_id", input.request_id},
                                   {"generation", input.generation},
                                   {"reason", generation_error}});
        ++rejected;
        continue;
      }
      data::ContentId content;
      data::RepresentationId representation;
      if (!ParseHex(item.value("content", ""), &content.digest)) {
        throw std::runtime_error("invalid_bundle_content_id");
      }
      const std::string media_content_text = item.value("feature_content", "");
      const std::string representation_text = item.value("representation", "");
      if (representation_text.empty()) {
        representation = data::RepresentationIdFromString("qwen25-vl-request-bundle-v3");
      } else if (!ParseHex(representation_text, &representation.digest)) {
        throw std::runtime_error("invalid_bundle_representation_id");
      }
      input.feature_content = media_content_text;
      input.feature_representation = item.value("feature_representation", "");
      data::Digest256 feature_representation_digest{};
      if (!ParseHex(input.feature_representation, &feature_representation_digest))
        throw std::runtime_error("invalid_feature_representation_id");
      const auto acquire_error = client_.acquire(
          data::DataKind::kTensorBundle, content, representation,
          data::DataLeaseKind::kCompute, &input.bundle);
      if (acquire_error != data::DataError::kOk) {
        throw std::runtime_error("bundle_acquire_failed:" +
                                 std::to_string(static_cast<int>(acquire_error)));
      }
      acquired_bundle_bytes += input.bundle.bytes.size();
      data::TensorBundleSchema schema;
      std::string reason;
      if (data::DecodeTensorBundlePayload(input.bundle.bytes.data(),
                                          input.bundle.bytes.size(), &schema,
                                          &reason) != data::DataError::kOk) {
        client_.release(input.bundle.token);
        throw std::runtime_error("bundle_preflight_failed:" + reason);
      }
      const auto* features = Find(schema, "image_features");
      const auto* token_component = Find(schema, "input_ids");
      const auto* position_component = Find(schema, "position_ids");
      const auto* delta_component = Find(schema, "rope_delta");
      std::vector<int64_t> deltas;
      if (!features || features->shape.size() != 2 ||
          !ReadComponent(input.bundle.bytes, token_component, &input.tokens) ||
          !ReadComponent(input.bundle.bytes, position_component, &input.positions) ||
          !ReadComponent(input.bundle.bytes, delta_component, &deltas) ||
          deltas.size() != 1 || input.positions.size() != input.tokens.size() * 3 ||
          features->byte_length != features->shape[0] * features->shape[1] * 2) {
        client_.release(input.bundle.token);
        throw std::runtime_error("bundle_shape_or_coverage_mismatch");
      }
      input.rope_delta = deltas[0];
      input.feature_offset = features->byte_offset;
      input.feature_bytes = features->byte_length;
      input.feature_rows = static_cast<int32_t>(features->shape[0]);
      input.hidden = static_cast<int32_t>(features->shape[1]);
      if (input.hidden != hidden_size_) {
        client_.release(input.bundle.token);
        throw std::runtime_error("bundle_hidden_size_mismatch");
      }
      data::ContentId media_content;
      if (!ParseHex(media_content_text, &media_content.digest)) {
        client_.release(input.bundle.token);
        throw std::runtime_error("invalid_feature_content_id");
      }
      auto image = std::find(input.tokens.begin(), input.tokens.end(), 151655);
      input.image_begin = static_cast<int32_t>(image - input.tokens.begin());
      if (image == input.tokens.end() || input.image_begin + input.feature_rows >
                                           static_cast<int32_t>(input.tokens.size())) {
        client_.release(input.bundle.token);
        throw std::runtime_error("image_span_out_of_bounds");
      }
      std::vector<uint8_t> semantic_wire;
      static constexpr char kMultimodalAdapterRevision[] =
          "qwen25-vl-pbe-adapter-v3";
      static constexpr char kModelRepresentationRevision[] =
          "qwen25-vl-3b-instruct-pbe-bf16-v1";
      semantic_wire.insert(semantic_wire.end(), kMultimodalAdapterRevision,
                           kMultimodalAdapterRevision +
                               sizeof(kMultimodalAdapterRevision) - 1);
      semantic_wire.insert(semantic_wire.end(), kModelRepresentationRevision,
                           kModelRepresentationRevision +
                               sizeof(kModelRepresentationRevision) - 1);
      semantic_wire.insert(semantic_wire.end(), media_content.digest.begin(),
                           media_content.digest.end());
      semantic_wire.insert(semantic_wire.end(), input.feature_representation.begin(),
                           input.feature_representation.end());
      // Text after the image may differ while the multimodal prefix is reusable.
      // Only the three-axis positions covered by the image dependency belong in
      // the media identity.
      for (int axis = 0; axis < 3; ++axis) {
        const auto* begin = input.positions.data() +
            axis * input.tokens.size() + input.image_begin;
        const auto* end = begin + input.feature_rows;
        semantic_wire.insert(semantic_wire.end(),
                             reinterpret_cast<const uint8_t*>(begin),
                             reinterpret_cast<const uint8_t*>(end));
      }
      semantic_wire.insert(semantic_wire.end(),
                           reinterpret_cast<const uint8_t*>(&input.rope_delta),
                           reinterpret_cast<const uint8_t*>(&input.rope_delta) +
                               sizeof(input.rope_delta));
      const data::ContentId semantic_media{
          data::DataChecksum(semantic_wire.data(), semantic_wire.size())};
      if (!data::MakeMultimodalRadixTokens(
              input.tokens,
              {{static_cast<uint32_t>(input.image_begin),
                static_cast<uint32_t>(input.feature_rows), semantic_media}},
              &input.semantic_tokens)) {
        client_.release(input.bundle.token);
        throw std::runtime_error("semantic_prefix_key_failed");
      }

      const size_t blocks = (input.tokens.size() + input.max_new_tokens +
                             descriptor_.block_size - 1) /
                            descriptor_.block_size;
      input.demands = {
          {serving::MemoryPool::kKV, blocks * block_bytes},
          {serving::MemoryPool::kBundles, input.bundle.bytes.size()},
          {serving::MemoryPool::kStaging, input.feature_bytes},
      };
      if (Clock::now() >= input.deadline || !budget_.reserve(input.demands)) {
        rejection_items.push_back({{"request_id", input.request_id},
                                   {"generation", input.generation},
                                   {"reason", Clock::now() >= input.deadline
                                                  ? "deadline_exceeded"
                                                  : "unified_budget_exhausted"}});
        client_.release(input.bundle.token);
        ++rejected;
        continue;
      }
      size_t current_free = 0;
      size_t current_total = 0;
      if (cudaMemGetInfo(&current_free, &current_total) != cudaSuccess ||
          current_total != device_total_bytes_ ||
          current_total - current_free > device_admission_limit_bytes_) {
        budget_.release(input.demands);
        input.demands.clear();
        rejection_items.push_back({{"request_id", input.request_id},
                                   {"generation", input.generation},
                                   {"reason", "device_memory_admission_exhausted"}});
        client_.release(input.bundle.token);
        input.bundle.token = {};
        ++rejected;
        continue;
      }
      admitted_kv_bytes += blocks * block_bytes;
      admitted_staging_bytes += input.feature_bytes;
      requests.push_back(std::move(input));
    }

    if (requests.empty()) {
      Json result = ResultJson({}, rejected, started, acquired_bundle_bytes,
                               admitted_kv_bytes, admitted_staging_bytes, 0, 0);
      result["rejections"] = std::move(rejection_items);
      return result;
    }

    serving::SchedulerConfig config;
    config.max_num_seqs = static_cast<int32_t>(requests.size());
    config.max_num_batched_tokens = 512;
    config.prefill_chunk_cap = 128;
    serving::Scheduler scheduler(config, model_->kv_cache_manager());
    for (size_t i = 0; i < requests.size(); ++i) {
      serving::GenerationConfig generation(requests[i].max_new_tokens);
      generation.sampling.repetition_penalty = 1.05;
      const auto lookup_started = Clock::now();
      requests[i].client_id = scheduler.add_request(requests[i].semantic_tokens, generation);
      requests[i].prefix_lookup_us = std::chrono::duration_cast<std::chrono::microseconds>(
          Clock::now() - lookup_started).count();
      requests[i].matched_prefix_tokens =
          scheduler.request_computed_tokens(requests[i].client_id);
      if (!scheduler.set_multimodal_checkpoint_state(
              requests[i].client_id, requests[i].positions, requests[i].rope_delta,
              requests[i].image_begin, requests[i].feature_rows, requests[i].hidden,
              requests[i].feature_content, requests[i].feature_representation, true))
        throw std::runtime_error("set_multimodal_checkpoint_state_failed");
      client_to_index[requests[i].client_id] = i;
    }

    auto allocator = base::CUDADeviceAllocatorFactory::get_instance();
    int64_t mixed_steps = 0;
    int64_t mixed_prefill_decode_steps = 0;
    Json numerical_diagnostics = Json::array();
    Json checkpoint_events = Json::array();
    while (scheduler.has_active_requests()) {
      auto output = scheduler.schedule_step();
      if (output.total_tokens <= 0) continue;
      for (auto* sequence : output.scheduled_seqs) {
        auto& request = requests[client_to_index.at(sequence->client_request_id)];
        if (!request.prefix_boundary_observed) {
          request.matched_prefix_tokens = std::min<int32_t>(
              sequence->computed_tokens, request.tokens.size());
          request.prefix_boundary_observed = true;
        }
      }
      auto batch = scheduler.build_mixed_batch(output, model_->device_context()->compute_queue);
      std::vector<int32_t> flat_tokens;
      std::vector<int32_t> flat_positions(3 * output.total_tokens);
      flat_tokens.reserve(output.total_tokens);
      int32_t cursor = 0;
      for (size_t row = 0; row < output.scheduled_seqs.size(); ++row) {
        auto* sequence = output.scheduled_seqs[row];
        const auto found = client_to_index.find(sequence->client_request_id);
        if (found == client_to_index.end()) throw std::runtime_error("unknown_sequence");
        auto& request = requests[found->second];
        const int32_t count = output.num_tokens_per_seq[row];
        if (sequence->is_prefill()) {
          const int32_t begin = sequence->computed_tokens;
          for (int32_t i = 0; i < count; ++i) flat_tokens.push_back(request.tokens[begin + i]);
          for (int axis = 0; axis < 3; ++axis) {
            for (int32_t i = 0; i < count; ++i) {
              flat_positions[axis * output.total_tokens + cursor + i] =
                  request.positions[axis * request.tokens.size() + begin + i];
            }
          }
        } else {
          flat_tokens.push_back(sequence->next_token);
          const int32_t position = static_cast<int32_t>(request.tokens.size()) +
                                   request.rope_delta + sequence->generated_tokens - 1;
          for (int axis = 0; axis < 3; ++axis) {
            flat_positions[axis * output.total_tokens + cursor] = position;
          }
        }
        cursor += count;
      }
      auto embeddings = model_->embedding(flat_tokens);
      cursor = 0;
      size_t feature_bytes_copied = 0;
      for (size_t row = 0; row < output.scheduled_seqs.size(); ++row) {
        auto* sequence = output.scheduled_seqs[row];
        auto& request = requests[client_to_index.at(sequence->client_request_id)];
        const int32_t count = output.num_tokens_per_seq[row];
        if (sequence->is_prefill()) {
          const int32_t begin = sequence->computed_tokens;
          const int32_t overlap_begin = std::max(begin, request.image_begin);
          const int32_t overlap_end = std::min(begin + count,
                                               request.image_begin + request.feature_rows);
          if (overlap_begin < overlap_end) {
            const size_t rows = overlap_end - overlap_begin;
            const size_t bytes = rows * request.hidden * 2;
            const size_t source_offset = request.feature_offset +
                (overlap_begin - request.image_begin) * request.hidden * 2;
            const size_t target_row = cursor + overlap_begin - begin;
            allocator->memcpy(request.bundle.bytes.data() + source_offset,
                              embeddings.input_embeddings.ptr<uint16_t>(target_row * request.hidden),
                              bytes, base::MemcpyKind::kMemcpyCPU2CUDA,
                              model_->device_context()->compute_queue, false);
            feature_bytes_copied += bytes;
          }
        }
        cursor += count;
      }
      tensor::Tensor mrope(base::DataType::kDataTypeInt32, 3, output.total_tokens,
                           true, allocator);
      allocator->memcpy(flat_positions.data(), mrope.ptr<int32_t>(), mrope.byte_size(),
                        base::MemcpyKind::kMemcpyCPU2CUDA,
                        model_->device_context()->compute_queue, false);
      batch.input_embeddings_override = embeddings.input_embeddings;
      batch.mrope_positions = mrope;
      const auto status = model_->forward_mixed_batch(batch);
      if (!status) throw std::runtime_error(status.get_err_msg());
      RecordDeviceMemory();
      const auto sampled = model_->batch_sample(batch, output);
      for (size_t sample_idx = 0; sample_idx < batch.sample_row_to_request.size();
           ++sample_idx) {
        const int32_t row = batch.sample_row_to_request[sample_idx];
        if (row < 0 || row >= static_cast<int32_t>(output.scheduled_seqs.size()) ||
            sample_idx >= batch.logits_row_indices.size() ||
            sample_idx >= static_cast<size_t>(sampled.size()))
          throw std::runtime_error("numerical_diagnostic_row_mismatch");
        auto* sequence = output.scheduled_seqs[row];
        auto& request = requests[client_to_index.at(sequence->client_request_id)];
        if (request.diagnostic_key.empty()) continue;
        std::vector<float> logits;
        if (!model_->copy_logits_row_to_host(batch.logits_row_indices[sample_idx], &logits))
          throw std::runtime_error("numerical_diagnostic_logits_copy_failed");
        const int32_t step = sequence->generated_tokens;
        const auto top2 = Top2(logits);
        if (top2.first < 0 || top2.second < 0)
          throw std::runtime_error("numerical_diagnostic_top2_failed");
        Json item = {{"diagnostic_key", request.diagnostic_key},
                     {"mode", request.diagnostic_mode},
                     {"step", step},
                     {"selected_token", sampled[static_cast<int32_t>(sample_idx)]},
                     {"top2_ids", {top2.first, top2.second}},
                     {"top2_values", {logits[top2.first], logits[top2.second]}},
                     {"top2_margin", logits[top2.first] - logits[top2.second]},
                     {"history_tokens", sequence->output_tokens}};
        if (request.diagnostic_dump_logits_step == step) {
          item["logits"] = logits;
          item["prompt_tokens"] = request.tokens;
          item["positions"] = request.positions;
          item["rope_delta"] = request.rope_delta;
        }
        const auto key = std::make_pair(request.diagnostic_key, step);
        auto baseline = numerical_baselines_.find(key);
        if (baseline == numerical_baselines_.end()) {
          NumericalSnapshot snapshot;
          snapshot.logits = std::move(logits);
          snapshot.prompt_tokens = request.tokens;
          snapshot.positions = request.positions;
          snapshot.output_tokens = sequence->output_tokens;
          snapshot.rope_delta = request.rope_delta;
          snapshot.selected_token = sampled[static_cast<int32_t>(sample_idx)];
          snapshot.mode = request.diagnostic_mode;
          numerical_baselines_.emplace(key, std::move(snapshot));
          item["comparison"] = "baseline_captured";
        } else {
          const auto& expected = baseline->second;
          const bool same_history = expected.prompt_tokens == request.tokens &&
              expected.positions == request.positions &&
              expected.output_tokens == sequence->output_tokens &&
              expected.rope_delta == request.rope_delta;
          if (expected.logits.size() != logits.size())
            throw std::runtime_error("numerical_diagnostic_vocab_mismatch");
          double max_abs = 0.0;
          double sum_abs = 0.0;
          double reference_abs = 0.0;
          for (size_t token = 0; token < logits.size(); ++token) {
            const double error = std::abs(static_cast<double>(logits[token]) -
                                          expected.logits[token]);
            max_abs = std::max(max_abs, error);
            sum_abs += error;
            reference_abs += std::abs(static_cast<double>(expected.logits[token]));
          }
          const auto expected_top2 = Top2(expected.logits);
          item["comparison"] = "against_baseline";
          item["baseline_mode"] = expected.mode;
          item["baseline_selected_token"] = expected.selected_token;
          item["baseline_top2_ids"] = {expected_top2.first, expected_top2.second};
          item["baseline_top2_values"] = {
              expected.logits[expected_top2.first], expected.logits[expected_top2.second]};
          item["baseline_top2_margin"] =
              expected.logits[expected_top2.first] - expected.logits[expected_top2.second];
          item["same_history"] = same_history;
          item["logits_max_abs"] = max_abs;
          item["logits_mean_abs"] = sum_abs / logits.size();
          item["logits_relative_mean"] =
              sum_abs / std::max(std::numeric_limits<double>::min(), reference_abs);
        }
        numerical_diagnostics.push_back(std::move(item));
      }
      scheduler.process_outputs(output, batch, sampled, [](int32_t) { return false; });
      ++mixed_steps;
      if (output.num_decode_seqs && output.num_prefill_seqs) ++mixed_prefill_decode_steps;

      const auto now = Clock::now();
      for (auto& request : requests) {
        const auto it = std::find_if(output.scheduled_seqs.begin(), output.scheduled_seqs.end(),
            [&](const serving::SequenceState* sequence) {
              return sequence->client_request_id == request.client_id;
            });
        if (it == output.scheduled_seqs.end()) continue;
        const auto* sequence = *it;
        if (IsCancelled(request.request_id, request.generation)) {
          scheduler.cancel_request(request.client_id, "cancelled_by_client");
        } else if (now >= request.deadline) {
          scheduler.cancel_request(request.client_id, "deadline_exceeded");
        } else if (request.cancel_after_tokens > 0 &&
                   sequence->generated_tokens >= request.cancel_after_tokens) {
          scheduler.cancel_request(request.client_id, "cancelled_by_coordinator");
        }
      }
      for (auto& request : requests) {
        if (!ConsumeCheckpointRequest(request.request_id, request.generation)) continue;
        uint64_t revision = 0;
        if (!scheduler.suspend_request(request.client_id, &revision)) {
          checkpoint_events.push_back({{"request_id", request.request_id},
                                       {"generation", request.generation},
                                       {"ok", false}, {"error", "checkpoint_save_failed"}});
          continue;
        }
        serving::RequestCheckpointManifest manifest;
        const bool manifest_ok = scheduler.checkpoint_manifest(
            request.client_id, revision, &manifest);
        const bool restored = scheduler.restore_request(request.client_id, revision);
        checkpoint_events.push_back({
            {"request_id", request.request_id}, {"generation", request.generation},
            {"revision", revision}, {"ok", manifest_ok && restored},
            {"restored", restored},
            {"kv_committed_tokens", manifest_ok ? manifest.kv_committed_tokens : 0},
            {"kv_block_size", descriptor_.block_size},
            {"private_tail_tokens", manifest_ok && descriptor_.block_size > 0
                 ? manifest.kv_committed_tokens % descriptor_.block_size : 0},
            {"private_tail_restored", restored && manifest_ok &&
                 descriptor_.block_size > 0 &&
                 manifest.kv_committed_tokens % descriptor_.block_size != 0},
            {"position_values", manifest_ok ? manifest.multimodal_position_values : 0},
            {"exact_multimodal_dependency",
             manifest_ok && manifest.multimodal_exact_dependency},
            {"sampling_counter", manifest_ok ? manifest.sampling_counter : 0},
            {"emitted_cursor", manifest_ok ? manifest.emitted_cursor : 0}});
      }
      total_feature_copy_bytes_ += feature_bytes_copied;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("cuda_sync_failed");
    auto finished = scheduler.pop_finished();
    Json response = ResultJson(finished, rejected, started, acquired_bundle_bytes,
                               admitted_kv_bytes, admitted_staging_bytes,
                               mixed_steps, mixed_prefill_decode_steps);
    response["rejections"] = std::move(rejection_items);
    response["numerical_diagnostics"] = std::move(numerical_diagnostics);
    response["checkpoint_events"] = std::move(checkpoint_events);
    int64_t matched_prefix_tokens = 0;
    int64_t prompt_tokens = 0;
    for (const auto& request : requests) {
      matched_prefix_tokens += std::max(0, request.matched_prefix_tokens);
      prompt_tokens += request.tokens.size();
    }
    response["semantic_prefix"] = {
        {"matched_tokens", matched_prefix_tokens},
        {"actual_computed_prompt_tokens", prompt_tokens - matched_prefix_tokens},
        {"prompt_tokens", prompt_tokens},
        {"lookup_before_compute", true},
        {"identity_includes_media_position_and_representation", true}};
    for (auto& output_item : response["outputs"]) {
      const int64_t client_id = output_item["client_id"].get<int64_t>();
      const auto found = client_to_index.find(client_id);
      if (found != client_to_index.end()) {
        output_item["request_id"] = requests[found->second].request_id;
        output_item["generation"] = requests[found->second].generation;
        output_item["semantic_matched_tokens"] =
            requests[found->second].matched_prefix_tokens;
        output_item["semantic_prompt_tokens"] =
            requests[found->second].tokens.size();
        output_item["semantic_actual_computed_prompt_tokens"] =
            requests[found->second].tokens.size() -
            requests[found->second].matched_prefix_tokens;
        output_item["semantic_lookup_us"] =
            requests[found->second].prefix_lookup_us;
      }
    }
    resource_guard.Release();
    return response;
  }

  Json Status() const {
    data::IpcPoolStats ipc;
    client_.ipc_pool_stats(&ipc);
    const auto& radix = model_->kv_cache_manager()->radix_cache_stats();
    const auto* transfer = model_->kv_cache_manager()->transfer_scheduler_stats();
    const int32_t active_kv_requests =
        model_->kv_cache_manager()->num_active_requests();
    data::SharedWeightStats shared_stats;
    const bool shared_stats_ok = weight_mode_ == "shared" &&
        client_.shared_weight_stats(&shared_stats) == data::DataError::kOk;
    const auto& shared_report = model_->shared_weight_report();
    return {{"ok", true}, {"worker_pid", getpid()},
            {"batches", batches_}, {"requests", total_requests_},
            {"pd", {{"prefill_requests", pd_prefill_requests_},
                     {"decode_requests", pd_decode_requests_},
                     {"active_handoffs", pd_holds_.size()},
                     {"kv_active_requests", active_kv_requests},
                     {"kv_request_baseline", model_kv_request_baseline_},
                     {"serving_active_requests",
                      active_kv_requests - model_kv_request_baseline_},
                     {"initial_prefill_slots", initial_slots_.size()}}},
            {"radix_cache_enabled", radix_cache_enabled_},
            {"feature_copy_bytes", total_feature_copy_bytes_},
            {"weights", {{"mode", weight_mode_},
                         {"layout", weight_mode_ == "shared"
                              ? data::kQwen2Bf16SharedWeightLayout : "process-owned"},
                         {"trusted_read_only_contract", weight_mode_ == "shared"},
                         {"hardware_read_only_enforced", false},
                         {"shared", shared_report.enabled},
                         {"owner_incarnation", shared_weight_lease_.descriptor.owner_incarnation},
                         {"allocation_id", shared_weight_lease_.descriptor.allocation_id},
                         {"generation", shared_weight_lease_.descriptor.generation},
                         {"consumer_incarnation", shared_weight_lease_.token.consumer_incarnation},
                         {"import_lease_id", shared_weight_lease_.token.lease_id},
                         {"imported_logical_bytes", shared_report.imported_logical_bytes},
                         {"physical_bytes", shared_stats_ok ? shared_stats.physical_bytes : 0},
                         {"owner_upload_count", shared_stats_ok ? shared_stats.upload_count : 0},
                         {"owner_allocation_count", shared_stats_ok ? shared_stats.allocation_count : 0},
                         {"owner_active_leases", shared_stats_ok ? shared_stats.active_leases : 0},
                         {"bound_tensor_views", shared_report.bound_tensor_views},
                         {"bound_tensor_logical_bytes", shared_report.bound_tensor_logical_bytes},
                         {"unique_tensor_ranges", shared_report.unique_tensor_ranges},
                         {"unique_tensor_bytes", shared_report.unique_tensor_bytes},
                         {"attention_views_bound", shared_report.attention_views_bound},
                         {"embedding_view_bound", shared_report.embedding_view_bound},
                         {"output_view_bound", shared_report.output_view_bound}}},
            {"pool", {{"total_slots", ipc.total_slots}, {"free_slots", ipc.free_slots},
                       {"active_grants", ipc.active_grants},
                       {"worker_granted_slots", grant_.slots.size()},
                       {"worker_initial_slots", initial_slots_.size()},
                       {"worker_allocatable_kv_bytes", allocatable_kv_bytes_}}},
            {"budget", {{"kv", PoolJson(budget_, serving::MemoryPool::kKV)},
                         {"weights", PoolJson(budget_, serving::MemoryPool::kWeights)},
                         {"workspaces", PoolJson(budget_, serving::MemoryPool::kWorkspaces)},
                         {"bundles", PoolJson(budget_, serving::MemoryPool::kBundles)},
                         {"staging", PoolJson(budget_, serving::MemoryPool::kStaging)}}},
            {"host_cache_budget", PoolJson(budget_, serving::MemoryPool::kHostCache)},
            {"device_memory", {{"total_bytes", device_total_bytes_},
                                {"admission_limit_bytes", device_admission_limit_bytes_},
                                {"preexisting_used_bytes",
                                 device_total_bytes_ - device_free_before_model_},
                                {"model_process_allocation_bytes", model_allocation_bytes_},
                                {"model_file_bytes", model_file_bytes_},
                                {"workspace_bytes_per_token", workspace_bytes_per_token_},
                                {"workspace_reserved_bytes", workspace_reserved_bytes_},
                                {"private_model_nonworkspace_bytes",
                                 private_model_nonworkspace_bytes_},
                                {"external_kv_owner_bytes", descriptor_.bytes},
                                {"peak_observed_used_bytes",
                                 device_total_bytes_ - device_free_low_water_},
                                {"within_admission_limit",
                                 device_total_bytes_ - device_free_low_water_ <=
                                     device_admission_limit_bytes_}}},
            {"radix_cache", {{"lookup_requests", radix.lookup_requests},
                              {"cache_hits", radix.cache_hits},
                              {"cache_misses", radix.cache_misses},
                              {"tokens_reused", radix.tokens_reused},
                              {"published_blocks", radix.published_blocks},
                              {"evicted_blocks", radix.evicted_blocks},
                              {"host_demoted_blocks", radix.host_demoted_blocks},
                              {"host_restore_requests", radix.host_restore_requests},
                              {"host_restored_blocks", radix.host_restored_blocks},
                              {"host_restore_failures", radix.host_restore_failures}}},
            {"recovery_dependency", {
                {"actual_eviction_path", true},
                {"objects", model_->kv_cache_manager()->recovery_dependency_object_count()},
                {"demote_decisions", radix.recovery_graph_demote_decisions},
                {"keep_decisions", radix.recovery_graph_keep_decisions},
                {"committed_demotions", radix.recovery_graph_committed_demotions},
                {"restores", radix.recovery_graph_restores},
                {"publish_rejections", radix.recovery_graph_publish_rejections}}},
            {"transfer_scheduler", transfer ? Json{
                {"physical_submissions", transfer->physical_submissions},
                {"background_completed", transfer->background_completed},
                {"demand_completed", transfer->demand_completed},
                {"max_d2h_active", transfer->max_d2h_active},
                {"max_h2d_active", transfer->max_h2d_active},
                {"oldest_queue_ticks", transfer->oldest_queue_ticks}} : Json::object()},
            {"budget_invariant", budget_.invariant_holds()}};
  }

 private:
  Json RunPDDecode(base::RequestId request_id,
                   const std::vector<int32_t>& prompt, int32_t first_token,
                   int64_t rope_delta, int32_t steps) {
    if (steps <= 0 || steps > 256)
      throw std::runtime_error("invalid_pd_decode_steps");
    serving::SchedulerConfig config;
    config.max_num_seqs = 1;
    config.max_num_batched_tokens = 16;
    serving::Scheduler scheduler(config, model_->kv_cache_manager());
    serving::GenerationConfig generation(steps);
    generation.sampling.repetition_penalty = 1.05;
    const int32_t initial_context =
        model_->kv_cache_manager()->get_context_len(request_id);
    scheduler.add_decode_ready_request(
        request_id, prompt, generation, initial_context, first_token);
    auto allocator = base::CUDADeviceAllocatorFactory::get_instance();
    int32_t produced = 1;
    int64_t scheduled_prefill_tokens = 0;
    int64_t scheduled_decode_tokens = 0;
    while (scheduler.has_active_requests()) {
      auto output = scheduler.schedule_step();
      if (output.total_tokens <= 0) continue;
      scheduled_prefill_tokens += output.total_tokens - output.num_decode_seqs;
      scheduled_decode_tokens += output.num_decode_seqs;
      if (output.num_prefill_seqs != 0 || output.num_decode_seqs != 1)
        throw std::runtime_error("pd_decode_scheduled_prefill");
      auto batch = scheduler.build_decode_batch(
          output, model_->device_context()->compute_queue);
      const int32_t position = initial_context + rope_delta + produced - 1;
      allocator->memcpy(&position, batch.positions.ptr<int32_t>(), sizeof(position),
          base::MemcpyKind::kMemcpyCPU2CUDA,
          model_->device_context()->compute_queue, false);
      const auto forward = model_->forward_decode_batch(batch);
      if (!forward) throw std::runtime_error(forward.get_err_msg());
      const auto sampled = model_->batch_sample(batch, output);
      produced += sampled.size();
      scheduler.process_outputs(output, batch, sampled, [](int32_t) { return false; });
    }
    if (cudaDeviceSynchronize() != cudaSuccess)
      throw std::runtime_error("pd_decode_sync_failed");
    auto finished = scheduler.pop_finished();
    if (finished.size() != 1 || finished[0].failed)
      throw std::runtime_error("pd_decode_did_not_finish");
    return {{"tokens", finished[0].output_tokens},
            {"text", model_->decode(finished[0].output_tokens)},
            {"ttft_ms", finished[0].ttft_ms()},
            {"itl_ms", finished[0].mean_token_gap_ms()},
            {"token_itl_ms", finished[0].token_gaps_ms()},
            {"latency_ms", finished[0].latency_ms()},
            {"scheduled_prefill_tokens", scheduled_prefill_tokens},
            {"scheduled_decode_tokens", scheduled_decode_tokens}};
  }

  struct PDHold {
    base::RequestId request_id = -1;
    std::vector<serving::MemoryDemand> demands;
  };

  bool AdmitGeneration(const std::string& request_id, uint64_t generation,
                       std::string* error) {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    auto latest = latest_generations_.find(request_id);
    if (latest != latest_generations_.end() && generation <= latest->second) {
      if (error) *error = generation < latest->second
          ? "stale_generation" : "duplicate_generation";
      return false;
    }
    latest_generations_[request_id] = generation;
    return true;
  }

  bool IsCancelled(const std::string& request_id, uint64_t generation) const {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    if (cancel_all_) return true;
    const auto it = cancelled_generations_.find(request_id);
    return it != cancelled_generations_.end() && it->second >= generation;
  }

  bool ConsumeCheckpointRequest(const std::string& request_id, uint64_t generation) {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);
    const auto it = checkpoint_generations_.find(request_id);
    if (it == checkpoint_generations_.end() || it->second < generation) return false;
    checkpoint_generations_.erase(it);
    return true;
  }

  void RecordDeviceMemory() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
        total_bytes != device_total_bytes_)
      throw std::runtime_error("cuda_mem_info_sample_failed");
    device_free_low_water_ = std::min(device_free_low_water_, free_bytes);
    if (total_bytes - free_bytes > device_admission_limit_bytes_)
      throw std::runtime_error("device_memory_budget_exceeded");
  }

  Json ResultJson(const std::vector<serving::SequenceState>& finished, size_t rejected,
                  Clock::time_point started, size_t acquired_bundle_bytes,
                  size_t admitted_kv_bytes, size_t admitted_staging_bytes,
                  int64_t mixed_steps, int64_t mixed_prefill_decode_steps) {
    Json outputs = Json::array();
    for (const auto& sequence : finished) {
      outputs.push_back({{"client_id", sequence.client_request_id},
                         {"failed", sequence.failed},
                         {"finish_reason", sequence.finish_reason},
                         {"tokens", sequence.output_tokens},
                         {"text", model_->decode(sequence.output_tokens)},
                         {"ttft_ms", sequence.ttft_ms()},
                         {"itl_ms", sequence.mean_token_gap_ms()},
                         {"token_itl_ms", sequence.token_gaps_ms()},
                         {"latency_ms", sequence.latency_ms()},
                         {"outbox_items", sequence.outbox.size()}});
    }
    ++batches_;
    total_requests_ += finished.size() + rejected;
    const double latency = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    return {{"ok", true}, {"worker_pid", getpid()}, {"outputs", std::move(outputs)},
            {"admitted", finished.size()}, {"rejected", rejected},
            {"batch_latency_ms", latency}, {"mixed_steps", mixed_steps},
            {"mixed_prefill_decode_steps", mixed_prefill_decode_steps},
            {"acquired_bundle_bytes", acquired_bundle_bytes},
            {"admitted_kv_bytes", admitted_kv_bytes},
            {"admitted_staging_bytes", admitted_staging_bytes},
            {"budget_invariant", budget_.invariant_holds()}};
  }

  data::DataClient client_;
  std::unique_ptr<model::Qwen2Model> model_;
  int device_ = 0;
  const int32_t hidden_size_ = 2048;
  data::IpcPoolDescriptor descriptor_;
  data::IpcSlotGrant grant_;
  bool grant_active_ = false;
  std::shared_ptr<data::CudaIpcPoolImport> importer_;
  std::shared_ptr<data::SharedWeightImport> shared_weight_import_;
  data::SharedWeightLease shared_weight_lease_;
  bool shared_weight_lease_active_ = false;
  std::string weight_mode_ = "private";
  std::vector<int32_t> initial_slots_;
  serving::NodeMemoryBudget budget_;
  std::vector<serving::MemoryDemand> persistent_demands_;
  uint64_t batches_ = 0;
  uint64_t total_requests_ = 0;
  uint64_t total_feature_copy_bytes_ = 0;
  uint64_t pd_prefill_requests_ = 0;
  uint64_t pd_decode_requests_ = 0;
  int32_t model_kv_request_baseline_ = 0;
  std::map<std::string, PDHold> pd_holds_;
  std::map<std::pair<std::string, int32_t>, NumericalSnapshot> numerical_baselines_;
  size_t device_total_bytes_ = 0;
  size_t device_free_before_model_ = 0;
  size_t device_free_low_water_ = 0;
  size_t device_admission_limit_bytes_ = 0;
  size_t model_file_bytes_ = 0;
  size_t model_allocation_bytes_ = 0;
  size_t private_model_nonworkspace_bytes_ = 0;
  size_t workspace_bytes_per_token_ = 0;
  size_t workspace_reserved_bytes_ = 0;
  size_t allocatable_kv_bytes_ = 0;
  size_t host_cache_bytes_ = 0;
  mutable std::mutex lifecycle_mutex_;
  std::map<std::string, uint64_t> latest_generations_;
  std::map<std::string, uint64_t> cancelled_generations_;
  std::map<std::string, uint64_t> checkpoint_generations_;
  bool cancel_all_ = false;
  bool radix_cache_enabled_ = true;
};
}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  if (argc < 5 || argc > 14) {
    std::cerr << "usage: pbe_vlm_language_role MODEL TOKENIZER DATA_ENDPOINT DEVICE "
                 "[BUNDLE_CAPACITY] [STAGING_CAPACITY] [KV_SLOTS] [RADIX_CACHE] "
                 "[DIRECTION_LANES] [INITIAL_PREFILL_SLOTS_CSV] "
                 "[WEIGHT_MODE] [MODEL_SHA256] [WEIGHT_LAYOUT]\n";
    return 2;
  }
  try {
    const size_t bundle_capacity = argc > 5 ? std::stoull(argv[5]) : size_t{64} << 20;
    const size_t staging_capacity = argc > 6 ? std::stoull(argv[6]) : size_t{32} << 20;
    const size_t kv_slots = argc > 7 ? std::stoull(argv[7]) : 0;
    const bool radix_cache = argc <= 8 || std::stoi(argv[8]) != 0;
    const bool direction_lanes = argc <= 9 || std::stoi(argv[9]) != 0;
    const std::vector<int32_t> initial_slots = argc > 10
        ? ParseSlots(argv[10]) : std::vector<int32_t>{};
    const std::string weight_mode = argc > 11 ? argv[11] : "private";
    const std::string model_sha256 = argc > 12 ? argv[12] : "";
    const std::string weight_layout = argc > 13 ? argv[13] : "";
    LanguageRole role(argv[1], argv[2], argv[3], std::stoi(argv[4]),
                      bundle_capacity, staging_capacity, kv_slots, radix_cache,
                      direction_lanes, initial_slots, weight_mode,
                      model_sha256, weight_layout);
    std::mutex output_mutex;
    auto emit = [&](Json response) {
      std::lock_guard<std::mutex> lock(output_mutex);
      std::cout << response.dump() << std::endl;
    };
    emit({{"event", "ready"}, {"worker_pid", getpid()},
          {"protocol", "pbe-vlm-language-jsonl-v2"},
          {"async_queue_capacity", 16}, {"external_cancel", true},
          {"absolute_deadline", true}});
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::deque<Json> queue;
    bool input_stopped = false;
    std::thread owner([&] {
      while (true) {
        Json command;
        {
          std::unique_lock<std::mutex> lock(queue_mutex);
          queue_cv.wait(lock, [&] { return input_stopped || !queue.empty(); });
          if (queue.empty()) break;
          command = std::move(queue.front());
          queue.pop_front();
        }
        Json response;
        const auto op_id = command.value("op_id", uint64_t{0});
        try {
          const std::string op = command.value("op", "infer");
          if (op == "infer") response = role.Infer(command);
          else if (op == "status") response = role.Status();
          else if (op == "probe") response = role.Probe(command);
          else if (op == "demote_prefix") response = role.DemotePrefixToHost();
          else if (op == "demote_prefix_async") response = role.DemotePrefixAsync();
          else if (op == "clear_prefix_cache") response = role.ClearPrefixCache();
          else if (op == "pd_prefill") response = role.PDPrefill(command);
          else if (op == "pd_decode") response = role.PDDecode(command);
          else if (op == "pd_release") response = role.PDRelease(command);
          else if (op == "shutdown") {
            response = {{"ok", true}, {"event", "shutdown"}, {"worker_pid", getpid()}};
          } else response = {{"ok", false}, {"error", "unknown_queued_op"}};
        } catch (const std::exception& error) {
          response = {{"ok", false}, {"error", error.what()}, {"worker_pid", getpid()}};
        }
        response["op_id"] = op_id;
        emit(std::move(response));
        if (command.value("op", "infer") == "shutdown") break;
      }
    });
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      try {
        const Json command = Json::parse(line);
        const std::string op = command.value("op", "infer");
        if (op == "cancel" || op == "checkpoint") {
          const bool accepted = op == "cancel"
              ? role.Cancel(command.value("request_id", ""),
                            command.value("generation", uint64_t{0}))
              : role.RequestCheckpoint(command.value("request_id", ""),
                                       command.value("generation", uint64_t{0}));
          emit({{"ok", accepted},
                {"event", op + "_ack"},
                {"request_id", command.value("request_id", "")},
                {"generation", command.value("generation", uint64_t{0})},
                {"op_id", command.value("op_id", uint64_t{0})},
                {"worker_pid", getpid()}});
          continue;
        }
        if (op != "infer" && op != "status" && op != "probe" &&
            op != "demote_prefix" && op != "demote_prefix_async" &&
            op != "clear_prefix_cache" &&
            op != "pd_prefill" && op != "pd_decode" && op != "pd_release" &&
            op != "shutdown") {
          emit({{"ok", false}, {"error", "unknown_op"},
                {"op_id", command.value("op_id", uint64_t{0})}});
          continue;
        }
        bool queued = false;
        {
          std::lock_guard<std::mutex> lock(queue_mutex);
          if (queue.size() < 16) {
            queue.push_back(command);
            queued = true;
          }
        }
        if (!queued) {
          emit({{"ok", false}, {"error", "command_queue_full"},
                {"op_id", command.value("op_id", uint64_t{0})}});
          continue;
        }
        queue_cv.notify_one();
        if (op == "shutdown") break;
      } catch (const std::exception& error) {
        emit({{"ok", false}, {"error", error.what()}, {"worker_pid", getpid()}});
      }
    }
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      input_stopped = true;
    }
    queue_cv.notify_one();
    owner.join();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "PBE_VLM_LANGUAGE_FATAL " << error.what() << "\n";
    return 1;
  }
}
