#ifndef KUIPER_INCLUDE_CACHE_TRANSFER_PLAN_H_
#define KUIPER_INCLUDE_CACHE_TRANSFER_PLAN_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/base.h"
#include "cache/layout_codec.h"
#include "cache/page_schema.h"

namespace cache {

enum class TransferPlacement : uint8_t {
  kExactDirect = 0,
  kSenderPack = 1,
  kReceiverUnpack = 2,
};

struct LogicalCopyOperation {
  // The covered logical slice. source_component and destination_component
  // describe the physical representation fragments containing it.
  ComponentDescriptor component;
  ComponentDescriptor source_component;
  ComponentDescriptor destination_component;
};

struct TransferPlan {
  std::string logical_fingerprint;
  std::string source_representation_fingerprint;
  std::string destination_representation_fingerprint;
  // Kept for source compatibility with exact-layout callers.
  std::string schema_fingerprint;
  TransferPlacement placement = TransferPlacement::kExactDirect;
  std::vector<LogicalCopyOperation> operations;
  size_t template_bytes = 0;
};

class TransferPlanner {
 public:
  explicit TransferPlanner(size_t max_entries = 128,
                           size_t max_template_bytes = size_t{1} << 20);

  base::Status compile(const PageSchema& source,
                       const PageSchema& destination,
                       TransferPlacement placement,
                       std::shared_ptr<const TransferPlan>* transfer_plan);
  base::Status plan(const PageSchema& source,
                    const PageSchema& destination,
                    TransferPlan* transfer_plan);

  size_t cache_hits() const { return cache_hits_; }
  size_t cache_entries() const { return plan_cache_.size(); }
  size_t cache_bytes() const { return cache_bytes_; }
  size_t cache_evictions() const { return cache_evictions_; }

 private:
  struct CacheEntry {
    std::shared_ptr<const TransferPlan> plan;
    uint64_t last_use = 0;
  };
  size_t max_entries_;
  size_t max_template_bytes_;
  size_t cache_bytes_ = 0;
  size_t cache_evictions_ = 0;
  uint64_t use_clock_ = 0;
  std::unordered_map<std::string, CacheEntry> plan_cache_;
  size_t cache_hits_ = 0;
};

struct ConstEndpoint {
  const uint8_t* data = nullptr;
  size_t bytes = 0;
  const LayoutDescriptor* layout = nullptr;
};

struct MutableEndpoint {
  uint8_t* data = nullptr;
  size_t bytes = 0;
  const LayoutDescriptor* layout = nullptr;
};

// Each span belongs to its own allocation; layout offsets remain logical packing
// metadata and are never used to derive pointers between separate allocations.
template <typename Pointer>
struct ComponentSpan {
  ComponentDescriptor component;
  Pointer data = nullptr;
  size_t bytes = 0;
};

struct ConstComponentEndpoint {
  std::vector<ComponentSpan<const uint8_t*>> components;
  const LayoutDescriptor* layout = nullptr;
};

struct MutableComponentEndpoint {
  std::vector<ComponentSpan<uint8_t*>> components;
  const LayoutDescriptor* layout = nullptr;
};

struct BoundCopyOperation {
  ComponentDescriptor component;
  const uint8_t* source = nullptr;
  uint8_t* destination = nullptr;
  size_t bytes = 0;
  int64_t source_virtual_block_id = -1;
  int64_t destination_virtual_block_id = -1;
};

class BoundTransfer {
 public:
  static base::Status Bind(const TransferPlan& plan,
                           const PageSchema& schema,
                           const ConstEndpoint& source,
                           const MutableEndpoint& destination,
                           BoundTransfer* bound_transfer);

  static base::Status Bind(const TransferPlan& plan,
                           const PageSchema& source_schema,
                           const PageSchema& destination_schema,
                           const ConstEndpoint& source,
                           const MutableEndpoint& destination,
                           BoundTransfer* bound_transfer);

  static base::Status BindComponents(const TransferPlan& plan,
                                     const PageSchema& schema,
                                     const ConstComponentEndpoint& source,
                                     const MutableComponentEndpoint& destination,
                                     BoundTransfer* bound_transfer);

  static base::Status BindComponents(
      const TransferPlan& plan,
      const PageSchema& source_schema,
      const PageSchema& destination_schema,
      const ConstComponentEndpoint& source,
      const MutableComponentEndpoint& destination,
      BoundTransfer* bound_transfer);

  const std::vector<BoundCopyOperation>& operations() const { return operations_; }
  uint64_t source_epoch() const { return source_epoch_; }
  uint64_t destination_epoch() const { return destination_epoch_; }
  TransferPlacement placement() const { return placement_; }
  size_t staging_bytes() const { return staging_bytes_; }

 private:
  std::vector<BoundCopyOperation> operations_;
  uint64_t source_epoch_ = 0;
  uint64_t destination_epoch_ = 0;
  TransferPlacement placement_ = TransferPlacement::kExactDirect;
  size_t staging_bytes_ = 0;
};

base::Status ExecuteCpuTransfer(const BoundTransfer& transfer);
base::Status ExecuteCpuTransferStaged(const BoundTransfer& transfer,
                                      uint8_t* staging,
                                      size_t staging_bytes);

}  // namespace cache

#endif  // KUIPER_INCLUDE_CACHE_TRANSFER_PLAN_H_
