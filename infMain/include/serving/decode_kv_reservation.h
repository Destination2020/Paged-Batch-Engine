#ifndef KUIPER_INCLUDE_SERVING_DECODE_KV_RESERVATION_H_
#define KUIPER_INCLUDE_SERVING_DECODE_KV_RESERVATION_H_

#include <cstdint>
#include <vector>

#include "base/base.h"
#include "base/kv_cache_manager.h"
#include "serving/pd_handoff.h"

namespace serving {

struct DecodeKVReservationRequest {
  GlobalRequestId client_request_id;
  HandoffId handoff_id;
  int32_t prompt_tokens = 0;
  int32_t computed_tokens = 0;
  int32_t first_token = -1;
  KVPoolDescriptor src_pool;
  std::vector<std::vector<int32_t>> src_block_ids_per_layer;
};

struct DecodeKVReservation {
  base::RequestId decode_request_id = -1;
  int32_t reserved_tokens = 0;
  std::vector<std::vector<int32_t>> dst_block_ids_per_layer;

  bool valid() const { return decode_request_id >= 0; }
};

class DecodeKVReservationManager {
 public:
  DecodeKVReservationManager(base::KVCacheManager* kv_manager,
                             KVPoolDescriptor dst_pool);

  base::Status reserve(const DecodeKVReservationRequest& request,
                       DecodeKVReservation* reservation,
                       KVBlockManifest* manifest);
  void release(DecodeKVReservation* reservation);

  const KVPoolDescriptor& dst_pool() const { return dst_pool_; }

 private:
  base::Status validate_request(const DecodeKVReservationRequest& request) const;
  int32_t required_blocks(int32_t tokens) const;

  base::KVCacheManager* kv_manager_ = nullptr;
  KVPoolDescriptor dst_pool_;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_DECODE_KV_RESERVATION_H_
