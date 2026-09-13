#ifndef KUIPER_INCLUDE_SERVING_PD_HANDOFF_H_
#define KUIPER_INCLUDE_SERVING_PD_HANDOFF_H_

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "base/base.h"
#include "base/kv_cache_manager.h"
#include "base/kv_cache_format.h"
#include "cache/transfer_plan.h"

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
  virtual KVTransferStatus drain(HandoffId handle) { return poll(handle); }
  virtual void release(HandoffId /*handle*/) {}
};

enum class LayerKVConnectorRole {
  kDisabled,
  kProducer,
  kConsumer,
};

struct LayerKVTransferRequest {
  GlobalRequestId client_request_id;
  HandoffId handoff_id;
  base::RequestId src_request_id = -1;
  base::RequestId dst_request_id = -1;
  int32_t prompt_tokens = 0;
  int32_t computed_tokens = 0;
  KVPoolDescriptor src_pool;
  KVPoolDescriptor dst_pool;
};

class LayerKVTransferConnector {
 public:
  virtual ~LayerKVTransferConnector() = default;

  virtual base::Status prepare(const LayerKVTransferRequest& request) = 0;
  virtual base::Status save_kv_layer(int32_t layer_idx) = 0;
  virtual base::Status wait_for_layer_load(int32_t layer_idx,
                                           void* consumer_queue = nullptr) = 0;
  virtual KVTransferStatus poll_layer(int32_t layer_idx) = 0;
  virtual KVTransferStatus poll() = 0;
  virtual void cancel(const std::string& reason) = 0;
};

class InProcNoCopyKVTransferConnector final : public KVTransferConnector {
 public:
  base::Status submit(const KVBlockManifest& manifest,
                      HandoffId* handle) override;
  KVTransferStatus poll(HandoffId handle) override;
  void cancel(HandoffId handle, const std::string& reason) override;
  KVTransferStatus drain(HandoffId handle) override;
  void release(HandoffId handle) override;

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
  KVTransferStatus drain(HandoffId handle) override;
  void release(HandoffId handle) override;

 private:
  base::Status copy_blocks(const KVBlockManifest& manifest);
  KVTransferStatus poll_transfer(size_t index);

  base::KVCacheManager* src_kv_manager_ = nullptr;
  base::KVCacheManager* dst_kv_manager_ = nullptr;
  void* transfer_queue_ = nullptr;
  bool need_sync_ = true;
  cache::TransferPlanner planner_;
  uint64_t next_handoff_id_ = 1;
  struct TransferRecord {
    HandoffId handle;
    KVTransferStatus status;
    void* completion_event = nullptr;
    bool cancel_requested = false;
    std::string cancel_reason;
  };
  std::vector<TransferRecord> transfers_;
};

class CudaP2PKVTransferConnector final : public KVTransferConnector {
 public:
  CudaP2PKVTransferConnector(base::KVCacheManager* src_kv_manager,
                             base::KVCacheManager* dst_kv_manager,
                             int32_t src_device_id,
                             int32_t dst_device_id,
                             void* transfer_queue = nullptr,
                             bool need_sync = true);
  ~CudaP2PKVTransferConnector() override;

  base::Status submit(const KVBlockManifest& manifest,
                      HandoffId* handle) override;
  KVTransferStatus poll(HandoffId handle) override;
  void cancel(HandoffId handle, const std::string& reason) override;
  KVTransferStatus drain(HandoffId handle) override;
  void release(HandoffId handle) override;

  bool peer_copy_enabled() const { return peer_copy_enabled_; }

 private:
  base::Status copy_blocks(const KVBlockManifest& manifest);
  KVTransferStatus poll_transfer(size_t index);

  base::KVCacheManager* src_kv_manager_ = nullptr;
  base::KVCacheManager* dst_kv_manager_ = nullptr;
  int32_t src_device_id_ = 0;
  int32_t dst_device_id_ = 0;
  void* transfer_queue_ = nullptr;
  bool need_sync_ = true;
  bool peer_copy_enabled_ = false;
  cache::TransferPlanner planner_;
  uint64_t next_handoff_id_ = 1;
  struct TransferRecord {
    HandoffId handle;
    KVTransferStatus status;
    void* completion_event = nullptr;
    bool cancel_requested = false;
    std::string cancel_reason;
  };
  std::vector<TransferRecord> transfers_;
};

class NcclKVBlockTransferConnector final : public KVTransferConnector {
 public:
  NcclKVBlockTransferConnector(base::KVCacheManager* src_kv_manager,
                               base::KVCacheManager* dst_kv_manager,
                               int32_t src_device_id,
                               int32_t dst_device_id,
                               void* transfer_queue = nullptr,
                               bool need_sync = true);
  ~NcclKVBlockTransferConnector() override;

  base::Status submit(const KVBlockManifest& manifest,
                      HandoffId* handle) override;
  KVTransferStatus poll(HandoffId handle) override;
  void cancel(HandoffId handle, const std::string& reason) override;

 private:
  base::Status ensure_comms();
  base::Status copy_blocks(const KVBlockManifest& manifest);
  KVTransferStatus poll_transfer(size_t index);

  base::KVCacheManager* src_kv_manager_ = nullptr;
  base::KVCacheManager* dst_kv_manager_ = nullptr;
  int32_t src_device_id_ = 0;
  int32_t dst_device_id_ = 0;
  void* transfer_queue_ = nullptr;
  bool need_sync_ = true;
  bool comms_initialized_ = false;
  void* src_comm_ = nullptr;
  void* dst_comm_ = nullptr;
  void* src_stream_ = nullptr;
  void* dst_stream_ = nullptr;
  bool owns_src_stream_ = false;
  bool owns_dst_stream_ = false;
  uint64_t next_handoff_id_ = 1;
  struct TransferRecord {
    HandoffId handle;
    KVTransferStatus status;
    void* completion_event = nullptr;
  };
  std::vector<TransferRecord> transfers_;
};

using KVManifestResolver =
    std::function<base::Status(uint64_t logical_page, KVBlockManifest* manifest)>;
std::unique_ptr<cache::FlightExecutor> MakeKVConnectorFlightExecutor(
    KVTransferConnector* connector, KVManifestResolver resolver);

enum class RemoteNcclKVTransferRole {
  kProducer,
  kConsumer,
};

struct RemoteNcclKVTransferOptions {
  RemoteNcclKVTransferRole role = RemoteNcclKVTransferRole::kProducer;
  base::KVCacheManager* kv_manager = nullptr;
  int32_t device_id = 0;
  std::string nccl_unique_id;
  void* stream = nullptr;
  bool need_sync = true;
};

base::Status run_remote_nccl_kv_block_transfer(
    const KVBlockManifest& manifest,
    const RemoteNcclKVTransferOptions& options);

class RemoteNcclLayerKVTransferConnector final : public LayerKVTransferConnector {
 public:
  explicit RemoteNcclLayerKVTransferConnector(
      RemoteNcclKVTransferOptions options);
  ~RemoteNcclLayerKVTransferConnector() override;

  base::Status prepare(const LayerKVTransferRequest& request) override;
  base::Status save_kv_layer(int32_t layer_idx) override;
  base::Status wait_for_layer_load(int32_t layer_idx,
                                   void* consumer_queue = nullptr) override;
  KVTransferStatus poll_layer(int32_t layer_idx) override;
  KVTransferStatus poll() override;
  void cancel(const std::string& reason) override;

 private:
  struct LayerRecord {
    KVTransferStatus status = KVTransferStatus::Pending();
    void* completion_event = nullptr;
    int32_t transferred_tokens = 0;
  };

  base::Status validate_request(const LayerKVTransferRequest& request) const;
  base::Status ensure_comm();
  base::Status copy_layer(int32_t layer_idx, void** completion_event);
  KVTransferStatus poll_layer_unlocked(int32_t layer_idx);
  int32_t required_blocks(int32_t tokens) const;

  RemoteNcclKVTransferOptions options_;
  LayerKVTransferRequest request_;
  bool prepared_ = false;
  bool cancelled_ = false;
  std::string cancel_reason_;
  bool comm_initialized_ = false;
  void* comm_ = nullptr;
  void* stream_ = nullptr;
  bool owns_stream_ = false;
  std::vector<LayerRecord> layer_records_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
};

class NcclLayerKVTransferConnector final : public LayerKVTransferConnector {
 public:
  NcclLayerKVTransferConnector(base::KVCacheManager* src_kv_manager,
                               base::KVCacheManager* dst_kv_manager,
                               int32_t src_device_id,
                               int32_t dst_device_id,
                               void* src_ready_queue = nullptr,
                               void* dst_transfer_queue = nullptr,
                               bool need_sync = false);
  ~NcclLayerKVTransferConnector() override;

  base::Status prepare(const LayerKVTransferRequest& request) override;
  base::Status save_kv_layer(int32_t layer_idx) override;
  base::Status wait_for_layer_load(int32_t layer_idx,
                                   void* consumer_queue = nullptr) override;
  KVTransferStatus poll_layer(int32_t layer_idx) override;
  KVTransferStatus poll() override;
  void cancel(const std::string& reason) override;

 private:
  struct LayerRecord {
    KVTransferStatus status = KVTransferStatus::Pending();
    void* completion_event = nullptr;
    int32_t transferred_tokens = 0;
  };

  base::Status validate_request(const LayerKVTransferRequest& request) const;
  base::Status ensure_comms();
  base::Status enqueue_source_ready_wait();
  base::Status copy_layer(const KVBlockMapping& mapping,
                          void** completion_event);
  KVTransferStatus poll_layer_unlocked(int32_t layer_idx);
  int32_t required_blocks(int32_t tokens) const;

  base::KVCacheManager* src_kv_manager_ = nullptr;
  base::KVCacheManager* dst_kv_manager_ = nullptr;
  int32_t src_device_id_ = 0;
  int32_t dst_device_id_ = 0;
  void* src_ready_queue_ = nullptr;
  void* dst_transfer_queue_ = nullptr;
  bool need_sync_ = false;
  bool comms_initialized_ = false;
  void* src_comm_ = nullptr;
  void* dst_comm_ = nullptr;
  void* src_stream_ = nullptr;
  void* dst_stream_ = nullptr;
  bool owns_src_stream_ = false;
  bool owns_dst_stream_ = false;
  bool prepared_ = false;
  bool cancelled_ = false;
  std::string cancel_reason_;
  LayerKVTransferRequest request_;
  std::vector<LayerRecord> layer_records_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_PD_HANDOFF_H_
