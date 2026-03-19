// Updated on March 15, 2026
#ifndef KUIPER_INCLUDE_MODEL_LLAMA_CONFIG_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_CONFIG_H_
namespace model {
struct ModelConfig {
  int32_t dim = 0;
  int32_t hidden_dim = 0;
  int32_t layer_num = 0;
  int32_t head_num = 0;
  int32_t kv_head_num = 0;
  int32_t vocab_size = 0;
  int32_t seq_len = 0;
#ifdef QWEN3_SUPPORT
  int32_t immediate_dim_ = 0;
#endif
};

struct TransformerConfig {
  int32_t kv_dim_ = 0;
  int32_t kv_mul_ = 0;
  int32_t head_size_ = 0; //dim_ / head_num_ number of hidden units per attention head
  int32_t vocab_size_ = 0;

  int32_t dim_ = 0;
  int32_t hidden_dim_ = 0;
  int32_t layer_num_ = 0;
  int32_t head_num_ = 0;
  int32_t kv_head_num_ = 0;
  int32_t seq_len_ = 0;
  bool is_shared_weight_ = false;
#ifdef QWEN3_SUPPORT
  int32_t immediate_dim_ = 0;
#endif

  int32_t moe_expert_num_ = 0;
  int32_t moe_topk_ = 0;
  int32_t moe_shared_expert_num_ = 0;
  int32_t moe_hidden_dim_ = 0;
  int32_t moe_shared_hidden_dim_ = 0; 
  int32_t moe_sparse_step_ = 1;
  int32_t moe_norm_topk_prob_ = 0;
};

inline constexpr int32_t kMoeMagic = 0x4D4F4531; // 'M' 'O' 'E' '1'

struct MoeHeader {
  int32_t magic = 0;             // identify the file type
  int32_t moe_expert_num = 0;    // number of experts in the MoE layer
  int32_t moe_topk = 0;          // number of experts selected for each token(top-k)
  int32_t moe_shared_expert_num = 0; // number of shared experts across MoE layers
  int32_t moe_hidden_dim = 0;    // hidden dimension for the feed-forward networks in MoE layers
  int32_t moe_shared_hidden_dim = 0; // hidden dimension for the shared feed-forward networks
  int32_t moe_sparse_step = 1;   // sparse step for the MoE layer
  int32_t moe_norm_topk_prob = 0; // whether to normalize the top-k probabilities

};


}  // namespace model
#endif  // KUIPER_INCLUDE_MODEL_LLAMA_CONFIG_H_
