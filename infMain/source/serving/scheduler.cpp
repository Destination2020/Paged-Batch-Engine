// Continuous batching scheduler implementation
#include "serving/scheduler.h"
#include <glog/logging.h>
#include "base/alloc.h"
#include "model/qwen2.h"

namespace serving {

Scheduler::Scheduler(int32_t max_batch_size, base::KVCacheManager* kv_manager)
    : max_batch_size_(max_batch_size),
      kv_manager_(kv_manager) {
  CHECK_GT(max_batch_size_, 0);
  CHECK_NE(kv_manager_, nullptr);
}

void Scheduler::add_request(std::vector<int32_t> prompt_tokens) {
  SequenceState seq;
  seq.request_id = kv_manager_->register_request();
  seq.prompt_tokens = std::move(prompt_tokens);
  seq.next_token = -1;
  seq.finished = false;
  waiting_.push_back(std::move(seq));
}

DecodeBatchMetadata Scheduler::schedule_step(const model::Qwen2Model& model) {
  // Phase 1: Admit waiting requests (simple blocking prefill)
  while (!waiting_.empty() &&
         static_cast<int32_t>(running_.size()) < max_batch_size_) {
    auto& seq = waiting_.front();

    int32_t prompt_len = static_cast<int32_t>(seq.prompt_tokens.size());
    int32_t blocks_needed = (prompt_len + kv_manager_->block_size() - 1) / kv_manager_->block_size();
    if (kv_manager_->num_free_blocks(0) < blocks_needed + 2) {
      break;
    }

    prefill_sequence(seq, model);
    running_.push_back(std::move(seq));
    waiting_.pop_front();
  }

  if (running_.empty()) {
    return DecodeBatchMetadata{};
  }

  // Phase 2: Append a decode slot for each running sequence
  for (auto& seq : running_) {
    bool ok = kv_manager_->append_slot(seq.request_id);
    CHECK(ok) << "Failed to append slot for request " << seq.request_id;
  }

  // Phase 3: Build GPU batch metadata
  return build_batch();
}

void Scheduler::prefill_sequence(SequenceState& seq, const model::Qwen2Model& model) {
  const auto& tokens = seq.prompt_tokens;
  const int32_t prompt_len = static_cast<int32_t>(tokens.size());
  CHECK_GT(prompt_len, 0);

  // Embed entire prompt once
  const auto& prompt_embedding = model.embedding(
      std::vector<int>(tokens.begin(), tokens.end()));
  tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

  for (int32_t pos = 0; pos < prompt_len; ++pos) {
    // Append a KV slot for this token via kv_cache_manager
    bool ok = kv_manager_->append_slot(seq.request_id);
    CHECK(ok) << "Failed to append slot during prefill";

    pos_tensor.index<int32_t>(0) = pos;

    if (pos < prompt_len - 1) {
      // Prompt phase: fill KV cache, don't sample
      tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, true);
      int next_unused = 0;
      // Use predict_with_request to write KV to the correct request
      model.predict_with_request(input, pos_tensor, seq.request_id, true, next_unused);
    } else {
      // Last token: sample first decode token
      std::vector<int32_t> last_tok = {tokens.back()};
      const auto& last_emb = model.embedding(
          std::vector<int>(last_tok.begin(), last_tok.end()));
      tensor::Tensor input = model.fill_input(pos_tensor, last_emb, false);
      int next = 0;
      model.predict_with_request(input, pos_tensor, seq.request_id, false, next);
      seq.next_token = next;
    }
  }
}

DecodeBatchMetadata Scheduler::build_batch() {
  const int32_t bs = static_cast<int32_t>(running_.size());
  if (bs == 0) return DecodeBatchMetadata{};

  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

  std::vector<int32_t> cpu_token_ids(bs);
  std::vector<int32_t> cpu_positions(bs);
  std::vector<base::RequestId> request_ids(bs);

  for (int32_t i = 0; i < bs; ++i) {
    cpu_token_ids[i] = running_[i].next_token;
    cpu_positions[i] = running_[i].context_len();
    request_ids[i] = running_[i].request_id;
  }

  auto cpu_seq_lens = kv_manager_->build_seq_lens(request_ids);
  auto cpu_slot_mapping = kv_manager_->build_slot_mapping(request_ids, 0);
  auto [cpu_block_tables, max_blocks] = kv_manager_->build_block_tables(request_ids, 0);

  DecodeBatchMetadata batch;
  batch.batch_size = bs;
  batch.max_blocks_per_seq = max_blocks;
  batch.request_ids = request_ids;

  batch.token_ids = tensor::Tensor(base::DataType::kDataTypeInt32, bs, true, alloc_cu);
  batch.positions = tensor::Tensor(base::DataType::kDataTypeInt32, bs, true, alloc_cu);
  batch.seq_lens = tensor::Tensor(base::DataType::kDataTypeInt32, bs, true, alloc_cu);
  batch.slot_mapping = tensor::Tensor(base::DataType::kDataTypeInt32, bs, true, alloc_cu);
  batch.block_tables = tensor::Tensor(base::DataType::kDataTypeInt32, bs * max_blocks, true, alloc_cu);

  batch.token_ids.set_device_type(base::DeviceType::kDeviceCUDA);
  batch.positions.set_device_type(base::DeviceType::kDeviceCUDA);
  batch.seq_lens.set_device_type(base::DeviceType::kDeviceCUDA);
  batch.slot_mapping.set_device_type(base::DeviceType::kDeviceCUDA);
  batch.block_tables.set_device_type(base::DeviceType::kDeviceCUDA);

  alloc_cu->memcpy(cpu_token_ids.data(),
                   const_cast<int32_t*>(batch.token_ids.ptr<int32_t>()),
                   bs * sizeof(int32_t),
                   base::MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
  alloc_cu->memcpy(cpu_positions.data(),
                   const_cast<int32_t*>(batch.positions.ptr<int32_t>()),
                   bs * sizeof(int32_t),
                   base::MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
  alloc_cu->memcpy(cpu_seq_lens.data(),
                   const_cast<int32_t*>(batch.seq_lens.ptr<int32_t>()),
                   bs * sizeof(int32_t),
                   base::MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
  alloc_cu->memcpy(cpu_slot_mapping.data(),
                   const_cast<int32_t*>(batch.slot_mapping.ptr<int32_t>()),
                   bs * sizeof(int32_t),
                   base::MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);
  alloc_cu->memcpy(cpu_block_tables.data(),
                   const_cast<int32_t*>(batch.block_tables.ptr<int32_t>()),
                   bs * max_blocks * sizeof(int32_t),
                   base::MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);

  return batch;
}

void Scheduler::process_outputs(const std::vector<int32_t>& sampled_tokens,
                                const std::function<bool(int32_t)>& is_eos) {
  CHECK_EQ(static_cast<int32_t>(sampled_tokens.size()),
           static_cast<int32_t>(running_.size()));

  auto it = running_.begin();
  int32_t i = 0;
  while (it != running_.end()) {
    int32_t token = sampled_tokens[i];
    it->next_token = token;

    if (is_eos(token)) {
      it->finished = true;
      kv_manager_->free_request(it->request_id);
      finished_.push_back(std::move(*it));
      it = running_.erase(it);
    } else {
      it->output_tokens.push_back(token);
      ++it;
    }
    ++i;
  }
}

std::vector<SequenceState> Scheduler::pop_finished() {
  std::vector<SequenceState> result;
  result.swap(finished_);
  return result;
}

bool Scheduler::has_active_requests() const {
  return !waiting_.empty() || !running_.empty();
}

}  // namespace serving
