#ifndef KUIPER_INCLUDE_BASE_COMPRESSED_RADIX_CACHE_TREE_H_
#define KUIPER_INCLUDE_BASE_COMPRESSED_RADIX_CACHE_TREE_H_

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace base {

// A standalone compressed radix tree for block-aligned prefix-cache experiments.
//
// This class intentionally does not depend on KVCacheManager, BlockAllocator, or
// model execution code. Tests can validate prefix matching, node splitting, and
// path pinning with synthetic token/block-id data before the structure is wired
// into the serving runtime.
template <typename Payload>
class BasicCompressedRadixCacheTree {
 public:
  using BlockKey = std::vector<int32_t>;

  struct Node {
    int64_t id = 0;
    Node* parent = nullptr;
    std::map<BlockKey, std::unique_ptr<Node>> children;
    std::vector<int32_t> segment_tokens;
    std::vector<std::vector<Payload>> segment_block_ids_per_layer;
    int32_t depth_blocks_before = 0;
    int32_t active_ref_count = 0;
    uint64_t last_access_tick = 0;

    int32_t segment_blocks(int32_t block_size) const {
      return static_cast<int32_t>(segment_tokens.size()) / block_size;
    }

    int32_t prefix_blocks(int32_t block_size) const {
      return depth_blocks_before + segment_blocks(block_size);
    }

    int32_t prefix_tokens(int32_t block_size) const {
      return prefix_blocks(block_size) * block_size;
    }

    bool is_root() const { return parent == nullptr; }
  };

  struct MatchResult {
    int32_t matched_tokens = 0;
    int32_t matched_blocks = 0;
    Node* matched_leaf = nullptr;
    std::vector<std::vector<Payload>> block_ids_per_layer;
  };

  struct InsertResult {
    int32_t existing_prefix_tokens = 0;
    int32_t existing_prefix_blocks = 0;
    int32_t inserted_tokens = 0;
    int32_t inserted_blocks = 0;
    Node* leaf = nullptr;
    bool split_performed = false;
  };

  BasicCompressedRadixCacheTree(int32_t block_size, int32_t num_layers)
      : block_size_(block_size), num_layers_(num_layers) {
    if (block_size_ <= 0) {
      throw std::invalid_argument("block_size must be positive");
    }
    if (num_layers_ <= 0) {
      throw std::invalid_argument("num_layers must be positive");
    }
    root_ = std::make_unique<Node>();
    root_->id = next_node_id_++;
    root_->segment_block_ids_per_layer.resize(num_layers_);
  }

  int32_t block_size() const { return block_size_; }
  int32_t num_layers() const { return num_layers_; }
  Node* root() const { return root_.get(); }
  int32_t split_count() const { return split_count_; }
  uint64_t access_tick() const { return access_tick_; }

  // Read-only probe: no split, LRU touch, pin or caller scratch mutation.
  MatchResult probe_prefix(const std::vector<int32_t>& tokens) const {
    MatchResult result{0, 0, root_.get(), empty_block_ids()};
    Node* node = root_.get();
    const int32_t full_blocks = full_blocks_for_tokens(tokens);
    while (result.matched_blocks < full_blocks) {
      auto it = node->children.find(block_key(tokens, result.matched_blocks));
      if (it == node->children.end()) break;
      Node* child = it->second.get();
      const auto count = common_prefix_blocks(child->segment_tokens, tokens,
          result.matched_blocks, full_blocks - result.matched_blocks);
      for (int32_t layer = 0; layer < num_layers_; ++layer) {
        const auto& payload = child->segment_block_ids_per_layer[layer];
        result.block_ids_per_layer[layer].insert(result.block_ids_per_layer[layer].end(),
                                                payload.begin(), payload.begin() + count);
      }
      result.matched_blocks += count;
      result.matched_tokens = result.matched_blocks * block_size_;
      // Only a full segment is a pinnable leaf. Materialize with match_prefix
      // after the caller accepts a prefix ending inside a segment.
      if (count < child->segment_blocks(block_size_)) break;
      node = child;
      result.matched_leaf = node;
    }
    return result;
  }

  MatchResult match_prefix(const std::vector<int32_t>& tokens) {
    const int32_t full_blocks = full_blocks_for_tokens(tokens);
    if (full_blocks == 0) {
      return MatchResult{0, 0, root_.get(), empty_block_ids()};
    }

    Node* node = root_.get();
    int32_t consumed_blocks = 0;
    bool split_performed = false;

    while (consumed_blocks < full_blocks) {
      const BlockKey key = block_key(tokens, consumed_blocks);
      auto it = node->children.find(key);
      if (it == node->children.end()) {
        break;
      }

      Node* child = it->second.get();
      child->last_access_tick = ++access_tick_;
      const int32_t matched_blocks =
          common_prefix_blocks(child->segment_tokens, tokens, consumed_blocks,
                               full_blocks - consumed_blocks);
      if (matched_blocks == 0) {
        break;
      }

      consumed_blocks += matched_blocks;
      if (matched_blocks < child->segment_blocks(block_size_)) {
        node = split_node(child, matched_blocks);
        split_performed = true;
        break;
      }

      node = child;
    }

    MatchResult result;
    result.matched_blocks = consumed_blocks;
    result.matched_tokens = consumed_blocks * block_size_;
    result.matched_leaf = node;
    result.block_ids_per_layer = collect_block_ids(node);
    if (split_performed) {
      result.matched_leaf = node;
      result.block_ids_per_layer = collect_block_ids(node);
    }
    return result;
  }

  InsertResult insert(const std::vector<int32_t>& tokens,
                      const std::vector<std::vector<Payload>>& block_ids_per_layer) {
    const int32_t full_blocks = full_blocks_for_tokens(tokens);
    validate_block_ids(block_ids_per_layer, full_blocks);

    InsertResult result;
    if (full_blocks == 0) {
      result.leaf = root_.get();
      return result;
    }

    Node* node = root_.get();
    int32_t consumed_blocks = 0;

    while (consumed_blocks < full_blocks) {
      const BlockKey key = block_key(tokens, consumed_blocks);
      auto it = node->children.find(key);
      if (it == node->children.end()) {
        break;
      }

      Node* child = it->second.get();
      child->last_access_tick = ++access_tick_;
      const int32_t matched_blocks =
          common_prefix_blocks(child->segment_tokens, tokens, consumed_blocks,
                               full_blocks - consumed_blocks);
      if (matched_blocks == 0) {
        break;
      }

      consumed_blocks += matched_blocks;
      if (matched_blocks < child->segment_blocks(block_size_)) {
        node = split_node(child, matched_blocks);
        result.split_performed = true;
        break;
      }

      node = child;
    }

    result.existing_prefix_blocks = consumed_blocks;
    result.existing_prefix_tokens = consumed_blocks * block_size_;

    if (consumed_blocks < full_blocks) {
      Node* inserted_leaf =
          append_child(node, tokens, block_ids_per_layer, consumed_blocks, full_blocks);
      result.leaf = inserted_leaf;
      result.inserted_blocks = full_blocks - consumed_blocks;
      result.inserted_tokens = result.inserted_blocks * block_size_;
    } else {
      result.leaf = node;
    }

    return result;
  }

  void pin_path(Node* leaf) {
    require_node(leaf);
    if (leaf == root_.get()) {
      throw std::logic_error("cannot pin the radix root as a cached prefix path");
    }
    for (Node* node = leaf; node != nullptr && node != root_.get(); node = node->parent) {
      ++node->active_ref_count;
    }
  }

  void unpin_path(Node* leaf) {
    require_node(leaf);
    if (leaf == root_.get()) {
      throw std::logic_error("cannot unpin the radix root as a cached prefix path");
    }
    for (Node* node = leaf; node != nullptr && node != root_.get(); node = node->parent) {
      if (node->active_ref_count <= 0) {
        throw std::logic_error("attempted to unpin an unpinned radix path");
      }
      --node->active_ref_count;
    }
  }

  std::vector<std::vector<Payload>> collect_block_ids(Node* leaf) const {
    require_node(leaf);
    std::vector<const Node*> path;
    for (const Node* node = leaf; node != nullptr && node != root_.get();
         node = node->parent) {
      path.push_back(node);
    }
    std::reverse(path.begin(), path.end());

    std::vector<std::vector<Payload>> result(num_layers_);
    for (const Node* node : path) {
      for (int32_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
        result[layer_idx].insert(result[layer_idx].end(),
                                 node->segment_block_ids_per_layer[layer_idx].begin(),
                                 node->segment_block_ids_per_layer[layer_idx].end());
      }
    }
    return result;
  }

  std::vector<Node*> collect_evictable_leaves() const {
    std::vector<Node*> leaves;
    collect_evictable_leaves(root_.get(), &leaves);
    std::sort(leaves.begin(), leaves.end(), [](const Node* lhs, const Node* rhs) {
      if (lhs->last_access_tick == rhs->last_access_tick) {
        return lhs->id < rhs->id;
      }
      return lhs->last_access_tick < rhs->last_access_tick;
    });
    return leaves;
  }

  int32_t evictable_blocks() const {
    return evictable_blocks(root_.get());
  }

  std::vector<std::vector<Payload>> erase_leaf(Node* leaf) {
    require_node(leaf);
    if (leaf == root_.get()) {
      throw std::logic_error("cannot erase the radix root");
    }
    if (!leaf->children.empty()) {
      throw std::logic_error("only radix leaves can be erased");
    }
    if (leaf->active_ref_count != 0) {
      throw std::logic_error("cannot erase an active radix path");
    }

    Node* parent = leaf->parent;
    if (parent == nullptr) {
      throw std::logic_error("radix leaf has no parent");
    }

    const BlockKey key = first_segment_block_key(leaf);
    auto it = parent->children.find(key);
    if (it == parent->children.end() || it->second.get() != leaf) {
      throw std::logic_error("parent does not own radix leaf under expected key");
    }

    std::vector<std::vector<Payload>> removed_block_ids =
        leaf->segment_block_ids_per_layer;
    parent->children.erase(it);
    return removed_block_ids;
  }

  std::vector<std::vector<Payload>> clear_and_collect_block_ids() {
    std::vector<std::vector<Payload>> block_ids(num_layers_);
    collect_all_block_ids(root_.get(), &block_ids);
    root_->children.clear();
    root_->last_access_tick = ++access_tick_;
    split_count_ = 0;
    return block_ids;
  }

  int32_t node_count() const { return node_count(root_.get()); }

 private:
  int32_t full_blocks_for_tokens(const std::vector<int32_t>& tokens) const {
    return static_cast<int32_t>(tokens.size()) / block_size_;
  }

  std::vector<std::vector<Payload>> empty_block_ids() const {
    return std::vector<std::vector<Payload>>(num_layers_);
  }

  BlockKey block_key(const std::vector<int32_t>& tokens, int32_t block_idx) const {
    const int32_t begin = block_idx * block_size_;
    return BlockKey(tokens.begin() + begin, tokens.begin() + begin + block_size_);
  }

  BlockKey first_segment_block_key(const Node* node) const {
    if (node == nullptr || node->segment_tokens.size() < static_cast<size_t>(block_size_)) {
      throw std::logic_error("radix node does not contain a full block");
    }
    return BlockKey(node->segment_tokens.begin(), node->segment_tokens.begin() + block_size_);
  }

  int32_t common_prefix_blocks(const std::vector<int32_t>& segment_tokens,
                               const std::vector<int32_t>& query_tokens,
                               int32_t query_start_block,
                               int32_t max_query_blocks) const {
    const int32_t segment_blocks =
        static_cast<int32_t>(segment_tokens.size()) / block_size_;
    const int32_t compare_blocks = std::min(segment_blocks, max_query_blocks);
    int32_t matched_blocks = 0;
    for (; matched_blocks < compare_blocks; ++matched_blocks) {
      const int32_t segment_begin = matched_blocks * block_size_;
      const int32_t query_begin = (query_start_block + matched_blocks) * block_size_;
      for (int32_t offset = 0; offset < block_size_; ++offset) {
        if (segment_tokens[segment_begin + offset] != query_tokens[query_begin + offset]) {
          return matched_blocks;
        }
      }
    }
    return matched_blocks;
  }

  Node* split_node(Node* child, int32_t prefix_blocks) {
    require_node(child);
    const int32_t child_blocks = child->segment_blocks(block_size_);
    if (prefix_blocks <= 0 || prefix_blocks >= child_blocks) {
      throw std::invalid_argument("split_node requires an internal block boundary");
    }

    Node* parent = child->parent;
    if (parent == nullptr) {
      throw std::logic_error("cannot split root node");
    }

    const BlockKey old_child_key = first_segment_block_key(child);
    auto it = parent->children.find(old_child_key);
    if (it == parent->children.end() || it->second.get() != child) {
      throw std::logic_error("parent does not own radix child under expected key");
    }

    std::unique_ptr<Node> suffix_node = std::move(it->second);
    parent->children.erase(it);

    auto prefix_node = std::make_unique<Node>();
    Node* prefix_ptr = prefix_node.get();
    prefix_ptr->id = next_node_id_++;
    prefix_ptr->parent = parent;
    prefix_ptr->depth_blocks_before = suffix_node->depth_blocks_before;
    prefix_ptr->active_ref_count = suffix_node->active_ref_count;
    prefix_ptr->last_access_tick = ++access_tick_;

    const int32_t prefix_tokens = prefix_blocks * block_size_;
    prefix_ptr->segment_tokens.assign(suffix_node->segment_tokens.begin(),
                                      suffix_node->segment_tokens.begin() + prefix_tokens);
    suffix_node->segment_tokens.erase(suffix_node->segment_tokens.begin(),
                                      suffix_node->segment_tokens.begin() + prefix_tokens);

    prefix_ptr->segment_block_ids_per_layer.resize(num_layers_);
    for (int32_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
      auto& suffix_blocks = suffix_node->segment_block_ids_per_layer[layer_idx];
      prefix_ptr->segment_block_ids_per_layer[layer_idx].assign(
          suffix_blocks.begin(), suffix_blocks.begin() + prefix_blocks);
      suffix_blocks.erase(suffix_blocks.begin(), suffix_blocks.begin() + prefix_blocks);
    }

    suffix_node->parent = prefix_ptr;
    suffix_node->depth_blocks_before = prefix_ptr->prefix_blocks(block_size_);

    const BlockKey suffix_key = first_segment_block_key(suffix_node.get());
    prefix_ptr->children.emplace(suffix_key, std::move(suffix_node));

    const BlockKey prefix_key = first_segment_block_key(prefix_ptr);
    Node* raw_prefix = prefix_ptr;
    parent->children.emplace(prefix_key, std::move(prefix_node));
    ++split_count_;
    return raw_prefix;
  }

  Node* append_child(Node* parent,
                     const std::vector<int32_t>& tokens,
                     const std::vector<std::vector<Payload>>& block_ids_per_layer,
                     int32_t start_block,
                     int32_t end_block) {
    require_node(parent);
    if (start_block >= end_block) {
      throw std::invalid_argument("append_child requires a non-empty block range");
    }

    auto node = std::make_unique<Node>();
    Node* raw = node.get();
    raw->id = next_node_id_++;
    raw->parent = parent;
    raw->depth_blocks_before = parent->prefix_blocks(block_size_);
    raw->last_access_tick = ++access_tick_;

    const int32_t token_begin = start_block * block_size_;
    const int32_t token_end = end_block * block_size_;
    raw->segment_tokens.assign(tokens.begin() + token_begin, tokens.begin() + token_end);

    raw->segment_block_ids_per_layer.resize(num_layers_);
    for (int32_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
      raw->segment_block_ids_per_layer[layer_idx].assign(
          block_ids_per_layer[layer_idx].begin() + start_block,
          block_ids_per_layer[layer_idx].begin() + end_block);
    }

    const BlockKey key = first_segment_block_key(raw);
    auto [it, inserted] = parent->children.emplace(key, std::move(node));
    if (!inserted) {
      throw std::logic_error("radix child already exists for appended segment");
    }
    return raw;
  }

  void validate_block_ids(const std::vector<std::vector<Payload>>& block_ids_per_layer,
                          int32_t required_blocks) const {
    if (static_cast<int32_t>(block_ids_per_layer.size()) != num_layers_) {
      throw std::invalid_argument("block_ids_per_layer layer count mismatch");
    }
    for (int32_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
      if (static_cast<int32_t>(block_ids_per_layer[layer_idx].size()) < required_blocks) {
        throw std::invalid_argument("block_ids_per_layer does not cover full blocks");
      }
    }
  }

  void require_node(const Node* node) const {
    if (node == nullptr) {
      throw std::invalid_argument("radix node must not be null");
    }
  }

  int32_t node_count(const Node* node) const {
    int32_t count = 1;
    for (const auto& [key, child] : node->children) {
      (void)key;
      count += node_count(child.get());
    }
    return count;
  }

  void collect_evictable_leaves(Node* node, std::vector<Node*>* leaves) const {
    if (node == nullptr || leaves == nullptr) {
      throw std::invalid_argument("collect_evictable_leaves received null input");
    }
    if (node != root_.get() && node->children.empty() && node->active_ref_count == 0) {
      leaves->push_back(node);
      return;
    }
    for (const auto& [key, child] : node->children) {
      (void)key;
      collect_evictable_leaves(child.get(), leaves);
    }
  }

  int32_t evictable_blocks(const Node* node) const {
    if (node == nullptr) {
      throw std::invalid_argument("evictable_blocks received null node");
    }
    int32_t blocks = 0;
    if (node != root_.get() && node->active_ref_count == 0) {
      blocks += node->segment_blocks(block_size_);
    }
    for (const auto& [key, child] : node->children) {
      (void)key;
      blocks += evictable_blocks(child.get());
    }
    return blocks;
  }

  void collect_all_block_ids(
      const Node* node, std::vector<std::vector<Payload>>* block_ids) const {
    if (node == nullptr || block_ids == nullptr) {
      throw std::invalid_argument("collect_all_block_ids received null input");
    }
    for (int32_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
      (*block_ids)[layer_idx].insert((*block_ids)[layer_idx].end(),
                                     node->segment_block_ids_per_layer[layer_idx].begin(),
                                     node->segment_block_ids_per_layer[layer_idx].end());
    }
    for (const auto& [key, child] : node->children) {
      (void)key;
      collect_all_block_ids(child.get(), block_ids);
    }
  }

  int32_t block_size_ = 0;
  int32_t num_layers_ = 0;
  std::unique_ptr<Node> root_;
  int64_t next_node_id_ = 0;
  uint64_t access_tick_ = 0;
  int32_t split_count_ = 0;
};

using CompressedRadixCacheTree = BasicCompressedRadixCacheTree<int32_t>;
using LogicalRadixCacheTree = BasicCompressedRadixCacheTree<uint64_t>;

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_COMPRESSED_RADIX_CACHE_TREE_H_
