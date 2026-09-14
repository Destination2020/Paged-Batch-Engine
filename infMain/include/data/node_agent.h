#ifndef KUIPER_INCLUDE_DATA_NODE_AGENT_H_
#define KUIPER_INCLUDE_DATA_NODE_AGENT_H_

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <set>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include <mutex>

#include "data/content_registry.h"
#include "data/shared_weight.h"

namespace data {

struct NodeAgentConfig {
  std::string endpoint = kDefaultDataEndpoint;
  uint64_t incarnation = 0;
  uint64_t capacity_bytes = 1ULL << 30;
  size_t capacity_objects = 4096;
  size_t capacity_leases = 16384;
  uint32_t request_timeout_ms = 2000;
  int32_t ipc_pool_device = -1;
  uint64_t ipc_pool_bytes = 0;
  int32_t ipc_pool_num_layers = 0;
  int32_t ipc_pool_num_blocks = 0;
  int32_t ipc_pool_block_size = 0;
  int32_t ipc_pool_num_kv_heads = 0;
  int32_t ipc_pool_head_size = 0;
  uint8_t ipc_pool_storage_dtype = 0;
  std::string shared_weight_path;
  Digest256 shared_weight_content{};
  Digest256 shared_weight_layout{};
  base::DataType shared_weight_dtype = base::DataType::kDataTypeUnknown;
  uint64_t shared_weight_staging_bytes = 64ULL << 20;
};

class NodeAgent {
 public:
  explicit NodeAgent(NodeAgentConfig config);
  ~NodeAgent();
  int run();
  void stop();

 private:
  void serve_connection(int fd);
  NodeAgentConfig config_;
  std::unique_ptr<ContentRegistry> registry_;
  std::unique_ptr<CudaIpcPoolOwner> ipc_pool_;
  std::unique_ptr<SharedWeightOwner> shared_weight_;
  std::mutex ipc_pool_mutex_;
  std::vector<uint64_t> ipc_slot_owners_;
  struct IpcGrantEntry {
    OperationId operation;
    std::vector<int32_t> slots;
    uint64_t attach_refs = 0;
    bool release_requested = false;
  };
  std::map<uint64_t,IpcGrantEntry> ipc_grants_;
  std::map<OperationId,uint64_t> ipc_grant_operations_;
  uint64_t next_ipc_grant_id_=1;
  struct IpcAttachEntry {
    OperationId operation;
    uint64_t source_grant_id = 0;
    AllocationHandle metadata_allocation;
    uint64_t provider_incarnation = 0;
    uint64_t target_incarnation = 0;
    uint32_t valid_tokens = 0;
    std::vector<int32_t> slots;
  };
  std::map<uint64_t, IpcAttachEntry> ipc_attaches_;
  std::map<OperationId, uint64_t> ipc_attach_operations_;
  uint64_t next_ipc_attach_id_ = 1;
  void collect_ipc_grant_locked(uint64_t grant_id);
  std::mutex shared_weight_mutex_;
  struct SharedWeightLeaseEntry { OperationId operation; };
  std::map<uint64_t, SharedWeightLeaseEntry> shared_weight_leases_;
  std::map<OperationId, uint64_t> shared_weight_operations_;
  uint64_t next_shared_weight_lease_id_ = 1;
  std::atomic<bool> stopping_{false};
  int listen_fd_ = -1;
  std::vector<std::thread> workers_;
  std::mutex queue_mutex_;
  std::condition_variable queue_ready_;
  std::deque<int> connection_queue_;
  std::set<int> active_connections_;
};

}  // namespace data
#endif
