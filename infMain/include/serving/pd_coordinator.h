#ifndef KUIPER_INCLUDE_SERVING_PD_COORDINATOR_H_
#define KUIPER_INCLUDE_SERVING_PD_COORDINATOR_H_

#include <string>

#include "base/base.h"
#include "serving/pd_handoff.h"

namespace serving {

enum class PDHandoffPhase {
  kCreated,
  kTransferSubmitted,
  kDecodeReady,
  kFailed,
  kCancelled,
};

struct PDHandoffState {
  HandoffId handoff_id;
  PDHandoffPhase phase = PDHandoffPhase::kCreated;
  KVTransferStatus transfer_status = KVTransferStatus::Pending();
  std::string error;

  bool ready_for_decode() const;
  bool terminal() const;
};

class PDCoordinator {
 public:
  explicit PDCoordinator(KVTransferConnector* connector);

  base::Status start_prefill_handoff(const KVBlockManifest& manifest,
                                     PDHandoffState* state);
  base::Status advance(PDHandoffState* state);
  void cancel(PDHandoffState* state, const std::string& reason);

 private:
  KVTransferConnector* connector_ = nullptr;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_PD_COORDINATOR_H_
