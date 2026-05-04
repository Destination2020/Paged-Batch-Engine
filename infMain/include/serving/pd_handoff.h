#ifndef KUIPER_INCLUDE_SERVING_PD_HANDOFF_H_
#define KUIPER_INCLUDE_SERVING_PD_HANDOFF_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "base/base.h"
#include "base/kv_cache_manager.h"
#include "base/kv_cache_format.h"

namespace serving {

struct GlobalRequestId {
  std::string value;

  bool empty() const { return value.empty(); }
};

struct HandoffId {
  uint64_t value = 0;

  bool valid() const { return value != 0; }
};

struct KVPoolDescriptor {
  int32_t device_id = 0;
  int32_t layer_num = 0;
  int32_t block_size = 0;
  int32_t kv_head_num = 0;
  int32_t head_size = 0;
  base::DataType dtype = base::DataType::kDataTypeUnknown;
  base::BlockStorageMode storage_mode = base::BlockStorageMode::kPlain;

  int64_t bytes_per_block_per_layer() const;
  bool compatible_with(const KVPoolDescriptor& other) const;
};

struct KVBlockMapping {
  int32_t layer_idx = 0;
  std::vector<int32_t> src_block_ids;
  std::vector<int32_t> dst_block_ids;
};

struct KVBlockManifest {
  GlobalRequestId client_request_id;
  HandoffId handoff_id;
  int32_t prompt_tokens = 0;
  int32_t computed_tokens = 0;
  int32_t first_token = -1;
  KVPoolDescriptor src_pool;
  KVPoolDescriptor dst_pool;
  std::vector<KVBlockMapping> layer_mappings;

  base::Status validate() const;
};

enum class KVTransferState {
  kPending,
  kCompleted,
  kFailed,
  kCancelled,
};

struct KVTransferStatus {
  KVTransferState state = KVTransferState::kPending;
  std::string error;

  static KVTransferStatus Pending();
  static KVTransferStatus Completed();
  static KVTransferStatus Failed(std::string message);
  static KVTransferStatus Cancelled(std::string message);
  bool done() const;
  bool ok() const;
};

class KVTransferConnector {
 public:
  virtual ~KVTransferConnector() = default;

  virtual base::Status submit(const KVBlockManifest& manifest,
                              HandoffId* handle) = 0;
  virtual KVTransferStatus poll(HandoffId handle) = 0;
  virtual void cancel(HandoffId handle, const std::string& reason) = 0;
};

class InProcNoCopyKVTransferConnector final : public KVTransferConnector {
 public:
  base::Status submit(const KVBlockManifest& manifest,
                      HandoffId* handle) override;
  KVTransferStatus poll(HandoffId handle) override;
  void cancel(HandoffId handle, const std::string& reason) override;

 private:
  uint64_t next_handoff_id_ = 1;
  std::vector<std::pair<HandoffId, KVTransferStatus>> transfers_;
};

class InProcKVBlockCopyConnector final : public KVTransferConnector {
 public:
  InProcKVBlockCopyConnector(base::KVCacheManager* src_kv_manager,
                             base::KVCacheManager* dst_kv_manager,
                             void* transfer_queue = nullptr,
                             bool need_sync = true);
  ~InProcKVBlockCopyConnector() override;

  base::Status submit(const KVBlockManifest& manifest,
                      HandoffId* handle) override;
  KVTransferStatus poll(HandoffId handle) override;
  void cancel(HandoffId handle, const std::string& reason) override;

 private:
  base::Status copy_blocks(const KVBlockManifest& manifest);
  KVTransferStatus poll_transfer(size_t index);

  base::KVCacheManager* src_kv_manager_ = nullptr;
  base::KVCacheManager* dst_kv_manager_ = nullptr;
  void* transfer_queue_ = nullptr;
  bool need_sync_ = true;
  uint64_t next_handoff_id_ = 1;
  struct TransferRecord {
    HandoffId handle;
    KVTransferStatus status;
    void* completion_event = nullptr;
  };
  std::vector<TransferRecord> transfers_;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_PD_HANDOFF_H_
