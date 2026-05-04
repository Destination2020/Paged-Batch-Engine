#ifndef KUIPER_INCLUDE_SERVING_PD_HANDOFF_BUILDER_H_
#define KUIPER_INCLUDE_SERVING_PD_HANDOFF_BUILDER_H_

#include <cstdint>

#include "base/base.h"
#include "base/kv_cache_manager.h"
#include "serving/decode_kv_reservation.h"
#include "serving/pd_handoff.h"
#include "serving/scheduler.h"

namespace serving {

struct PrefillHandoffBuildRequest {
  int32_t scheduled_request_index = 0;
  GlobalRequestId client_request_id;
  HandoffId handoff_id;
};

class PDHandoffBuilder final {
 public:
  PDHandoffBuilder(const base::KVCacheManager* kv_manager,
                   KVPoolDescriptor src_pool);

  base::Status build_decode_reservation_request(
      const SchedulerOutput& output,
      const PrefillHandoffBuildRequest& request,
      DecodeKVReservationRequest* reservation_request) const;

 private:
  const base::KVCacheManager* kv_manager_ = nullptr;
  KVPoolDescriptor src_pool_;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_PD_HANDOFF_BUILDER_H_
