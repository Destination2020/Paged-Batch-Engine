#include "serving/pd_coordinator.h"

#include <glog/logging.h>

#include <utility>

namespace serving {

bool PDHandoffState::ready_for_decode() const {
  return phase == PDHandoffPhase::kDecodeReady;
}

bool PDHandoffState::terminal() const {
  return phase == PDHandoffPhase::kDecodeReady ||
         phase == PDHandoffPhase::kFailed ||
         phase == PDHandoffPhase::kCancelled;
}

PDCoordinator::PDCoordinator(KVTransferConnector* connector)
    : connector_(connector) {
  CHECK_NE(connector_, nullptr);
}

base::Status PDCoordinator::start_prefill_handoff(const KVBlockManifest& manifest,
                                                  PDHandoffState* state) {
  if (state == nullptr) {
    return base::error::InvalidArgument("pd coordinator state is null");
  }
  *state = {};

  HandoffId handle;
  base::Status status = connector_->submit(manifest, &handle);
  if (!status) {
    state->phase = PDHandoffPhase::kFailed;
    state->error = status.get_err_msg();
    state->transfer_status = KVTransferStatus::Failed(status.get_err_msg());
    return status;
  }

  state->handoff_id = handle;
  state->phase = PDHandoffPhase::kTransferSubmitted;
  state->transfer_status = KVTransferStatus::Pending();
  return advance(state);
}

base::Status PDCoordinator::advance(PDHandoffState* state) {
  if (state == nullptr) {
    return base::error::InvalidArgument("pd coordinator state is null");
  }
  if (state->phase == PDHandoffPhase::kDecodeReady) {
    return base::error::Success();
  }
  if (state->phase == PDHandoffPhase::kFailed) {
    return base::error::InternalError(state->error);
  }
  if (state->phase == PDHandoffPhase::kCancelled) {
    return base::error::InternalError(state->error);
  }
  if (!state->handoff_id.valid()) {
    state->phase = PDHandoffPhase::kFailed;
    state->error = "pd coordinator handoff id is invalid";
    state->transfer_status = KVTransferStatus::Failed(state->error);
    return base::error::InvalidArgument(state->error);
  }

  state->transfer_status = connector_->poll(state->handoff_id);
  if (!state->transfer_status.done()) {
    state->phase = PDHandoffPhase::kTransferSubmitted;
    return base::error::Success();
  }
  if (state->transfer_status.ok()) {
    state->phase = PDHandoffPhase::kDecodeReady;
    state->error.clear();
    return base::error::Success();
  }
  if (state->transfer_status.state == KVTransferState::kCancelled) {
    state->phase = PDHandoffPhase::kCancelled;
  } else {
    state->phase = PDHandoffPhase::kFailed;
  }
  state->error = state->transfer_status.error;
  return base::error::InternalError(state->error);
}

void PDCoordinator::cancel(PDHandoffState* state, const std::string& reason) {
  if (state == nullptr || state->terminal()) {
    return;
  }
  if (state->handoff_id.valid()) {
    connector_->cancel(state->handoff_id, reason);
  }
  state->phase = PDHandoffPhase::kCancelled;
  state->error = reason;
  state->transfer_status = KVTransferStatus::Cancelled(reason);
}

}  // namespace serving
