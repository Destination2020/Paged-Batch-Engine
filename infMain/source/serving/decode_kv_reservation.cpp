#include "serving/decode_kv_reservation.h"

#include <algorithm>

#include <glog/logging.h>

namespace serving {

DecodeKVReservationManager::DecodeKVReservationManager(base::KVCacheManager* kv_manager,
                                                       KVPoolDescriptor dst_pool)
    : kv_manager_(kv_manager), dst_pool_(std::move(dst_pool)) {
  CHECK_NE(kv_manager_, nullptr);
}

base::Status DecodeKVReservationManager::validate_request(
    const DecodeKVReservationRequest& request) const {
  if (request.client_request_id.empty()) {
    return base::error::InvalidArgument("decode kv reservation missing client_request_id");
  }
  if (!request.handoff_id.valid()) {
    return base::error::InvalidArgument("decode kv reservation missing handoff_id");
  }
  if (request.prompt_tokens <= 0) {
    return base::error::InvalidArgument("decode kv reservation prompt_tokens must be positive");
  }
  if (request.computed_tokens <= 0 || request.computed_tokens > request.prompt_tokens) {
    return base::error::InvalidArgument("decode kv reservation computed_tokens is invalid");
  }
  if (!request.src_pool.compatible_with(dst_pool_)) {
    return base::error::InvalidArgument("decode kv reservation source/destination pools are incompatible");
  }
  if (dst_pool_.layer_num != kv_manager_->num_layers()) {
    return base::error::InvalidArgument("decode kv reservation dst layer_num mismatch");
  }
  if (dst_pool_.block_size != kv_manager_->block_size()) {
    return base::error::InvalidArgument("decode kv reservation dst block_size mismatch");
  }
  if (static_cast<int32_t>(request.src_block_ids_per_layer.size()) != dst_pool_.layer_num) {
    return base::error::InvalidArgument("decode kv reservation source layer count mismatch");
  }
  const int32_t blocks = required_blocks(request.computed_tokens);
  for (const auto& layer_blocks : request.src_block_ids_per_layer) {
    if (static_cast<int32_t>(layer_blocks.size()) < blocks) {
      return base::error::InvalidArgument("decode kv reservation source block count is insufficient");
    }
  }
  return base::error::Success();
}

int32_t DecodeKVReservationManager::required_blocks(int32_t tokens) const {
  if (tokens <= 0) {
    return 0;
  }
  return (tokens + dst_pool_.block_size - 1) / dst_pool_.block_size;
}

base::Status DecodeKVReservationManager::reserve(
    const DecodeKVReservationRequest& request,
    DecodeKVReservation* reservation,
    KVBlockManifest* manifest) {
  if (reservation == nullptr) {
    return base::error::InvalidArgument("decode kv reservation output is null");
  }
  if (manifest == nullptr) {
    return base::error::InvalidArgument("decode kv reservation manifest output is null");
  }
  *reservation = {};
  *manifest = {};

  base::Status status = validate_request(request);
  if (!status) {
    return status;
  }

  const int32_t blocks = required_blocks(request.computed_tokens);
  const base::RequestId decode_request_id = kv_manager_->register_request();
  if (!kv_manager_->append_slots(decode_request_id, request.computed_tokens)) {
    kv_manager_->free_request(decode_request_id);
    return base::error::InternalError("decode kv reservation failed to allocate dst KV blocks");
  }

  reservation->decode_request_id = decode_request_id;
  reservation->reserved_tokens = request.computed_tokens;
  reservation->first_token = request.first_token;
  reservation->dst_block_ids_per_layer.resize(dst_pool_.layer_num);
  for (int32_t layer_idx = 0; layer_idx < dst_pool_.layer_num; ++layer_idx) {
    const std::vector<int32_t>& dst_blocks = kv_manager_->get_block_ids(decode_request_id, layer_idx);
    if (static_cast<int32_t>(dst_blocks.size()) < blocks) {
      release(reservation);
      return base::error::InternalError("decode kv reservation dst block count is insufficient");
    }
    reservation->dst_block_ids_per_layer[layer_idx].assign(
        dst_blocks.begin(), dst_blocks.begin() + blocks);
  }

  manifest->client_request_id = request.client_request_id;
  manifest->handoff_id = request.handoff_id;
  manifest->prompt_tokens = request.prompt_tokens;
  manifest->computed_tokens = request.computed_tokens;
  manifest->first_token = request.first_token;
  manifest->src_pool = request.src_pool;
  manifest->dst_pool = dst_pool_;
  manifest->layer_mappings.reserve(dst_pool_.layer_num);
  for (int32_t layer_idx = 0; layer_idx < dst_pool_.layer_num; ++layer_idx) {
    KVBlockMapping mapping;
    mapping.layer_idx = layer_idx;
    mapping.src_block_ids.assign(request.src_block_ids_per_layer[layer_idx].begin(),
                                 request.src_block_ids_per_layer[layer_idx].begin() + blocks);
    mapping.dst_block_ids = reservation->dst_block_ids_per_layer[layer_idx];
    manifest->layer_mappings.push_back(std::move(mapping));
  }

  status = manifest->validate();
  if (!status) {
    release(reservation);
    *manifest = {};
    return status;
  }
  return base::error::Success();
}

void DecodeKVReservationManager::release(DecodeKVReservation* reservation) {
  if (reservation == nullptr || !reservation->valid()) {
    return;
  }
  if (kv_manager_->is_valid_request(reservation->decode_request_id)) {
    kv_manager_->free_request(reservation->decode_request_id);
  }
  *reservation = {};
}

}  // namespace serving
