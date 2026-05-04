#include "serving/pd_handoff_builder.h"

#include <glog/logging.h>

#include <utility>

namespace serving {

PDHandoffBuilder::PDHandoffBuilder(const base::KVCacheManager* kv_manager,
                                   KVPoolDescriptor src_pool)
    : kv_manager_(kv_manager), src_pool_(std::move(src_pool)) {
  CHECK_NE(kv_manager_, nullptr);
}

base::Status PDHandoffBuilder::build_decode_reservation_request(
    const SchedulerOutput& output,
    const PrefillHandoffBuildRequest& request,
    DecodeKVReservationRequest* reservation_request) const {
  if (reservation_request == nullptr) {
    return base::error::InvalidArgument("prefill handoff reservation output is null");
  }
  if (request.scheduled_request_index < 0 ||
      request.scheduled_request_index >= static_cast<int32_t>(output.scheduled_seqs.size())) {
    return base::error::InvalidArgument("prefill handoff request index is out of range");
  }
  const SequenceState* seq = output.scheduled_seqs[request.scheduled_request_index];
  if (seq == nullptr) {
    return base::error::InvalidArgument("prefill handoff sequence is null");
  }
  if (seq->is_prefill()) {
    return base::error::InvalidArgument("prefill handoff sequence is not fully computed");
  }
  if (!kv_manager_->is_valid_request(seq->request_id)) {
    return base::error::InvalidArgument("prefill handoff sequence has invalid KV request");
  }
  if (kv_manager_->get_context_len(seq->request_id) != seq->computed_tokens) {
    return base::error::InvalidArgument("prefill handoff KV context length mismatch");
  }

  DecodeKVReservationRequest out;
  out.client_request_id = request.client_request_id;
  out.handoff_id = request.handoff_id;
  out.prompt_tokens = static_cast<int32_t>(seq->prompt_tokens.size());
  out.computed_tokens = seq->computed_tokens;
  out.first_token = seq->next_token;
  out.src_pool = src_pool_;
  out.src_block_ids_per_layer.resize(src_pool_.layer_num);
  for (int32_t layer_idx = 0; layer_idx < src_pool_.layer_num; ++layer_idx) {
    out.src_block_ids_per_layer[layer_idx] =
        kv_manager_->get_block_ids(seq->request_id, layer_idx);
  }
  *reservation_request = std::move(out);
  return base::error::Success();
}

}  // namespace serving
