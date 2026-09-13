#ifndef KUIPER_INCLUDE_SERVING_REQUEST_CHECKPOINT_H_
#define KUIPER_INCLUDE_SERVING_REQUEST_CHECKPOINT_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>

#include "base/kv_cache_manager.h"
#include "data/data_runtime.h"
#include "serving/sequence_state.h"

namespace serving {

enum class CheckpointState { kPreparing, kReady, kRestoring };

struct CheckpointTicket {
  int64_t client_request_id = -1;
  uint64_t revision = 0;
  bool valid() const { return client_request_id >= 0 && revision != 0; }
};

struct RequestCheckpointManifest {
  uint64_t manifest_version = 1;
  std::string model_namespace;
  int64_t client_request_id = -1;
  uint64_t revision = 0;
  int32_t kv_committed_tokens = 0;
  int32_t valid_tokens = 0;
  int32_t pending_next_token = -1;
  uint64_t sampling_seed = 0;
  uint64_t sampling_counter = 0;
  uint64_t emitted_cursor = 0;
  uint64_t output_enqueued_cursor = 0;
  uint64_t multimodal_position_values = 0;
  bool multimodal_exact_dependency = false;
  CheckpointState state = CheckpointState::kPreparing;
};

struct RequestCheckpointStats {
  uint64_t prepared = 0;
  uint64_t committed = 0;
  uint64_t restored = 0;
  uint64_t stale_restore_rejections = 0;
  uint64_t cancelled_restores = 0;
  uint64_t stale_commit_rejections = 0;
  uint64_t record_capacity_rejections = 0;
  uint64_t byte_capacity_rejections = 0;
  size_t max_records = 0;
  size_t payload_bytes = 0;
  size_t max_payload_bytes = 0;
  size_t revision_metadata_records = 0;
};

// Process-local transactional checkpoint store. Preparing records are never
// visible to restore; commit publishes one immutable manifest revision.
class RequestCheckpointStore {
 public:
  explicit RequestCheckpointStore(
      size_t max_records,
      size_t max_payload_bytes = std::numeric_limits<size_t>::max());
  ~RequestCheckpointStore();

  bool prepare(const SequenceState& sequence, const std::string& model_namespace,
               base::KVCacheManager* manager, CheckpointTicket* ticket);
  bool commit(const CheckpointTicket& ticket);
  bool save(const SequenceState& sequence, const std::string& model_namespace,
            base::KVCacheManager* manager, CheckpointTicket* ticket);

  bool begin_restore(const CheckpointTicket& ticket,
                     const std::string& model_namespace,
                     base::KVCacheManager* manager);
  bool commit_restore(const CheckpointTicket& ticket,
                      base::KVCacheManager* manager,
                      SequenceState* restored);
  bool cancel_restore(const CheckpointTicket& ticket,
                      base::KVCacheManager* manager);
  void cancel_client(int64_t client_request_id, base::KVCacheManager* manager);
  bool erase(const CheckpointTicket& ticket);

  bool manifest(const CheckpointTicket& ticket,
                RequestCheckpointManifest* manifest) const;
  uint64_t current_revision(int64_t client_request_id) const;
  size_t size() const { return records_.size(); }
  size_t payload_bytes() const { return payload_bytes_; }
  size_t revision_metadata_size() const { return current_revision_.size(); }
  size_t publication_object_count() const { return publication_runtime_.object_count(); }
  uint64_t publication_bytes() const { return publication_runtime_.bytes_used(); }
  const RequestCheckpointStats& stats() const { return stats_; }

 private:
  using Key = std::pair<int64_t, uint64_t>;
  struct Record {
    RequestCheckpointManifest manifest;
    SequenceState sequence;
    base::KVRequestSnapshot kv;
    base::RequestId restoring_request_id = -1;
    size_t payload_bytes = 0;
    data::AllocationHandle publication_handle;
    data::DataRef publication_ref;
    data::DataLease publication_lease;
    bool publication_sealed = false;
  };

  void erase_record(std::map<Key, Record>::iterator it);

  size_t max_records_;
  size_t max_payload_bytes_;
  size_t payload_bytes_ = 0;
  uint64_t next_revision_ = 0;
  // The original serving checkpoint path publishes typed, immutable metadata
  // through the same runtime contract used by later cross-process objects.
  data::LocalDataRuntime publication_runtime_;
  std::map<Key, Record> records_;
  std::map<int64_t, uint64_t> current_revision_;
  RequestCheckpointStats stats_;
};

}  // namespace serving
#endif
