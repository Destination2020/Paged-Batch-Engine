#include "cache/transfer_plan.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace cache {

namespace {

bool SameStorage(const PageSchema& source, const PageSchema& destination) {
  return source.storage.storage_mode == destination.storage.storage_mode &&
         source.storage.logical_dtype == destination.storage.logical_dtype &&
         source.storage.storage_dtype == destination.storage.storage_dtype &&
         source.storage.scale_dtype == destination.storage.scale_dtype;
}

size_t TemplateBytes(const TransferPlan& plan) {
  return sizeof(TransferPlan) + plan.logical_fingerprint.size() +
         plan.source_representation_fingerprint.size() +
         plan.destination_representation_fingerprint.size() +
         plan.schema_fingerprint.size() +
         plan.operations.size() * sizeof(LogicalCopyOperation);
}

std::string CacheKey(const PageSchema& source, const PageSchema& destination,
                     TransferPlacement placement) {
  // codec/topology versions are explicit constants until runtime topology is
  // configurable; addresses and allocation generations are deliberately absent.
  return "codec=1|topology=1|" + source.fingerprint() + "->" +
         destination.fingerprint() + "|placement=" +
         std::to_string(static_cast<int>(placement));
}

}  // namespace

TransferPlanner::TransferPlanner(size_t max_entries, size_t max_template_bytes)
    : max_entries_(max_entries), max_template_bytes_(max_template_bytes) {
  if (max_entries_ == 0 || max_template_bytes_ == 0)
    throw std::invalid_argument("transfer plan cache capacity is zero");
}

base::Status TransferPlanner::compile(
    const PageSchema& source, const PageSchema& destination,
    TransferPlacement placement,
    std::shared_ptr<const TransferPlan>* transfer_plan) {
  if (transfer_plan == nullptr) {
    return base::error::InvalidArgument("Transfer plan output is null.");
  }
  base::Status source_status = source.validate();
  if (!source_status) {
    return source_status;
  }
  base::Status destination_status = destination.validate();
  if (!destination_status) {
    return destination_status;
  }
  const std::string source_fingerprint = source.fingerprint();
  const std::string destination_fingerprint = destination.fingerprint();
  if (source.logical_fingerprint() != destination.logical_fingerprint() ||
      !SameStorage(source, destination)) {
    return base::error::InvalidArgument(
        "Source and destination schemas require an unsupported conversion.");
  }

  const std::string cache_key = CacheKey(source, destination, placement);
  const auto cached = plan_cache_.find(cache_key);
  if (cached != plan_cache_.end()) {
    ++cache_hits_;
    cached->second.last_use = ++use_clock_;
    *transfer_plan = cached->second.plan;
    return base::error::Success();
  }

  TransferPlan candidate;
  candidate.logical_fingerprint = source.logical_fingerprint();
  candidate.source_representation_fingerprint = source_fingerprint;
  candidate.destination_representation_fingerprint = destination_fingerprint;
  candidate.schema_fingerprint = source_fingerprint;
  candidate.placement = placement;
  for (const auto& destination_component : destination.components) {
    for (const auto& source_component : source.components) {
      if (source_component.layer != destination_component.layer ||
          source_component.kind != destination_component.kind) continue;
      const int32_t begin = std::max(source_component.head_begin,
                                     destination_component.head_begin);
      const int32_t end = std::min(
          source_component.head_begin + source_component.head_count,
          destination_component.head_begin + destination_component.head_count);
      if (begin >= end) continue;
      ComponentDescriptor slice = destination_component;
      slice.head_begin = begin;
      slice.head_count = end - begin;
      candidate.operations.push_back(
          {slice, source_component, destination_component});
    }
  }
  std::sort(candidate.operations.begin(), candidate.operations.end(),
            [](const LogicalCopyOperation& left,
               const LogicalCopyOperation& right) {
    return left.component.semantic_key() < right.component.semantic_key();
  });
  size_t covered_bytes = 0;
  for (const auto& operation : candidate.operations) {
    if (operation.component.byte_size() >
        std::numeric_limits<size_t>::max() - covered_bytes)
      return base::error::InvalidArgument("Transfer plan byte size overflows.");
    covered_bytes += operation.component.byte_size();
  }
  size_t destination_bytes = 0;
  for (const auto& component : destination.components) {
    if (component.byte_size() >
        std::numeric_limits<size_t>::max() - destination_bytes)
      return base::error::InvalidArgument("Destination representation overflows.");
    destination_bytes += component.byte_size();
  }
  if (covered_bytes != destination_bytes) {
    return base::error::InvalidArgument(
        "Transfer plan does not cover the destination representation.");
  }
  candidate.template_bytes = TemplateBytes(candidate);
  auto immutable = std::make_shared<const TransferPlan>(std::move(candidate));
  if (immutable->template_bytes <= max_template_bytes_) {
    while (!plan_cache_.empty() &&
           (plan_cache_.size() >= max_entries_ ||
            immutable->template_bytes > max_template_bytes_ - cache_bytes_)) {
      auto victim = std::min_element(
          plan_cache_.begin(), plan_cache_.end(), [](const auto& left,
                                                     const auto& right) {
            return left.second.last_use < right.second.last_use;
          });
      cache_bytes_ -= victim->second.plan->template_bytes;
      plan_cache_.erase(victim);
      ++cache_evictions_;
    }
    cache_bytes_ += immutable->template_bytes;
    plan_cache_.emplace(cache_key, CacheEntry{immutable, ++use_clock_});
  }
  *transfer_plan = std::move(immutable);
  return base::error::Success();
}

base::Status TransferPlanner::plan(const PageSchema& source,
                                   const PageSchema& destination,
                                   TransferPlan* transfer_plan) {
  if (!transfer_plan)
    return base::error::InvalidArgument("Transfer plan output is null.");
  std::shared_ptr<const TransferPlan> immutable;
  auto status = compile(source, destination, TransferPlacement::kExactDirect,
                        &immutable);
  if (!status) return status;
  *transfer_plan = *immutable;
  return base::error::Success();
}

namespace {
template <typename Endpoint>
base::Status ValidateSpans(const Endpoint& endpoint, const PageSchema& schema) {
  if (endpoint.layout == nullptr) {
    return base::error::InvalidArgument("Component endpoint layout is null.");
  }
  auto status = endpoint.layout->validate(schema);
  if (!status) return status;
  if (endpoint.components.size() != schema.components.size()) {
    return base::error::InvalidArgument("Component endpoint coverage is incomplete.");
  }
  std::unordered_map<std::string, bool> seen;
  std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
  for (const auto& span : endpoint.components) {
    const auto expected = std::find_if(schema.components.begin(), schema.components.end(),
        [&](const auto& item) { return item == span.component; });
    if (expected == schema.components.end() ||
        !seen.emplace(span.component.semantic_key(), true).second ||
        span.data == nullptr || span.bytes < expected->byte_size()) {
      return base::error::InvalidArgument("Invalid, duplicate or undersized component span.");
    }
    const auto begin = reinterpret_cast<uintptr_t>(span.data);
    const auto bytes = expected->byte_size();
    if (bytes > std::numeric_limits<uintptr_t>::max() - begin) {
      return base::error::InvalidArgument("Component address range overflows.");
    }
    ranges.emplace_back(begin, begin + bytes);
  }
  std::sort(ranges.begin(), ranges.end());
  for (size_t i = 1; i < ranges.size(); ++i) {
    if (ranges[i].first < ranges[i - 1].second) {
      return base::error::InvalidArgument("Component address ranges overlap.");
    }
  }
  return base::error::Success();
}
}  // namespace

base::Status BoundTransfer::Bind(const TransferPlan& plan,
                                 const PageSchema& schema,
                                 const ConstEndpoint& source,
                                 const MutableEndpoint& destination,
                                 BoundTransfer* bound_transfer) {
  return Bind(plan, schema, schema, source, destination, bound_transfer);
}

base::Status BoundTransfer::Bind(const TransferPlan& plan,
                                 const PageSchema& source_schema,
                                 const PageSchema& destination_schema,
                                 const ConstEndpoint& source,
                                 const MutableEndpoint& destination,
                                 BoundTransfer* bound_transfer) {
  if (bound_transfer == nullptr || source.layout == nullptr || destination.layout == nullptr) {
    return base::error::InvalidArgument("Bound transfer endpoint or output is null.");
  }
  auto status = source.layout->validate(source_schema);
  if (!status) return status;
  status = destination.layout->validate(destination_schema);
  if (!status) return status;
  if (source.bytes < source.layout->total_bytes || destination.bytes < destination.layout->total_bytes ||
      source.data == nullptr || destination.data == nullptr) {
    return base::error::InvalidArgument("Transfer endpoint buffer is null or too small.");
  }
  ConstComponentEndpoint scattered_source{{}, source.layout};
  MutableComponentEndpoint scattered_destination{{}, destination.layout};
  for (const auto& entry : source.layout->components) {
    scattered_source.components.push_back(
        {entry.component, source.data + entry.offset_bytes, entry.size_bytes});
  }
  for (const auto& entry : destination.layout->components) {
    scattered_destination.components.push_back(
        {entry.component, destination.data + entry.offset_bytes, entry.size_bytes});
  }
  return BindComponents(plan, source_schema, destination_schema, scattered_source,
                        scattered_destination, bound_transfer);
}

base::Status BoundTransfer::BindComponents(const TransferPlan& plan,
                                            const PageSchema& schema,
                                            const ConstComponentEndpoint& source,
                                            const MutableComponentEndpoint& destination,
                                            BoundTransfer* bound_transfer) {
  return BindComponents(plan, schema, schema, source, destination,
                        bound_transfer);
}

base::Status BoundTransfer::BindComponents(
    const TransferPlan& plan, const PageSchema& source_schema,
    const PageSchema& destination_schema, const ConstComponentEndpoint& source,
    const MutableComponentEndpoint& destination,
    BoundTransfer* bound_transfer) {
  if (bound_transfer == nullptr) {
    return base::error::InvalidArgument("Bound transfer output is null.");
  }
  auto status = source_schema.validate();
  if (!status) return status;
  status = destination_schema.validate();
  if (!status) return status;
  if (plan.logical_fingerprint != source_schema.logical_fingerprint() ||
      plan.logical_fingerprint != destination_schema.logical_fingerprint() ||
      plan.source_representation_fingerprint != source_schema.fingerprint() ||
      plan.destination_representation_fingerprint !=
          destination_schema.fingerprint()) {
    return base::error::InvalidArgument("Transfer plan does not match the page schema.");
  }
  status = ValidateSpans(source, source_schema);
  if (!status) return status;
  status = ValidateSpans(destination, destination_schema);
  if (!status) return status;

  BoundTransfer candidate;
  candidate.source_epoch_ = source.layout->epoch;
  candidate.destination_epoch_ = destination.layout->epoch;
  candidate.placement_ = plan.placement;
  std::vector<std::pair<uintptr_t, uintptr_t>> destination_ranges;
  size_t copied_bytes = 0;
  for (const auto& operation : plan.operations) {
    const auto src = std::find_if(source.components.begin(), source.components.end(),
        [&](const auto& item) {
          return item.component == operation.source_component;
        });
    const auto dst = std::find_if(destination.components.begin(), destination.components.end(),
        [&](const auto& item) {
          return item.component == operation.destination_component;
        });
    const auto& slice = operation.component;
    const auto& source_component = operation.source_component;
    const auto& destination_component = operation.destination_component;
    if (src == source.components.end() || dst == destination.components.end() ||
        slice.layer != source_component.layer ||
        slice.layer != destination_component.layer ||
        slice.kind != source_component.kind ||
        slice.kind != destination_component.kind ||
        slice.token_count != source_component.token_count ||
        slice.token_count != destination_component.token_count ||
        slice.elements_per_head != source_component.elements_per_head ||
        slice.elements_per_head != destination_component.elements_per_head ||
        slice.dtype != source_component.dtype ||
        slice.dtype != destination_component.dtype ||
        slice.head_begin < source_component.head_begin ||
        slice.head_count <= 0 ||
        slice.head_begin + slice.head_count >
            source_component.head_begin + source_component.head_count ||
        slice.head_begin < destination_component.head_begin ||
        slice.head_begin + slice.head_count >
            destination_component.head_begin + destination_component.head_count) {
      return base::error::InvalidArgument(
          "Transfer plan has an invalid representation slice.");
    }
    const auto src_layout = std::find_if(
        source.layout->components.begin(), source.layout->components.end(),
        [&](const auto& item) { return item.component == source_component; });
    const auto dst_layout = std::find_if(
        destination.layout->components.begin(),
        destination.layout->components.end(),
        [&](const auto& item) { return item.component == destination_component; });
    if (src_layout == source.layout->components.end() ||
        dst_layout == destination.layout->components.end()) {
      return base::error::InvalidArgument(
          "Transfer plan component is absent from endpoint layout.");
    }

    const size_t element_bytes = base::DataTypeSize(slice.dtype);
    const size_t head_bytes =
        static_cast<size_t>(slice.elements_per_head) * element_bytes;
    const size_t slice_bytes = static_cast<size_t>(slice.head_count) * head_bytes;
    const size_t source_stride =
        static_cast<size_t>(source_component.head_count) * head_bytes;
    const size_t destination_stride =
        static_cast<size_t>(destination_component.head_count) * head_bytes;
    const size_t source_head_offset =
        static_cast<size_t>(slice.head_begin - source_component.head_begin) *
        head_bytes;
    const size_t destination_head_offset =
        static_cast<size_t>(slice.head_begin - destination_component.head_begin) *
        head_bytes;
    const bool one_contiguous_copy =
        slice.head_begin == source_component.head_begin &&
        slice.head_count == source_component.head_count &&
        slice.head_begin == destination_component.head_begin &&
        slice.head_count == destination_component.head_count;
    const int32_t rows = one_contiguous_copy ? 1 : slice.token_count;
    const size_t row_bytes = one_contiguous_copy ? slice.byte_size() : slice_bytes;
    for (int32_t token = 0; token < rows; ++token) {
      const size_t source_offset = one_contiguous_copy
          ? 0
          : static_cast<size_t>(token) * source_stride + source_head_offset;
      const size_t destination_offset = one_contiguous_copy
          ? 0
          : static_cast<size_t>(token) * destination_stride +
                destination_head_offset;
      if (source_offset > src->bytes || row_bytes > src->bytes - source_offset ||
          destination_offset > dst->bytes ||
          row_bytes > dst->bytes - destination_offset ||
          row_bytes > std::numeric_limits<uintptr_t>::max() -
                          reinterpret_cast<uintptr_t>(dst->data + destination_offset) ||
          row_bytes > std::numeric_limits<size_t>::max() - copied_bytes) {
        return base::error::InvalidArgument(
            "Transfer plan slice exceeds endpoint bounds.");
      }
      ComponentDescriptor bound_component = slice;
      if (!one_contiguous_copy) bound_component.token_count = 1;
      candidate.operations_.push_back(
          {bound_component, src->data + source_offset,
           dst->data + destination_offset, row_bytes,
           src_layout->virtual_block_id, dst_layout->virtual_block_id});
      const uintptr_t destination_begin =
          reinterpret_cast<uintptr_t>(dst->data + destination_offset);
      destination_ranges.emplace_back(destination_begin,
                                      destination_begin + row_bytes);
      copied_bytes += row_bytes;
    }
  }
  size_t expected_bytes = 0;
  for (const auto& component : destination_schema.components) {
    if (component.byte_size() >
        std::numeric_limits<size_t>::max() - expected_bytes)
      return base::error::InvalidArgument("Destination byte size overflows.");
    expected_bytes += component.byte_size();
  }
  std::sort(destination_ranges.begin(), destination_ranges.end());
  for (size_t index = 1; index < destination_ranges.size(); ++index) {
    if (destination_ranges[index].first < destination_ranges[index - 1].second)
      return base::error::InvalidArgument(
          "Transfer plan writes a destination byte more than once.");
  }
  if (copied_bytes != expected_bytes)
    return base::error::InvalidArgument(
        "Transfer plan leaves destination bytes uncovered.");
  candidate.staging_bytes_ = copied_bytes;
  *bound_transfer = std::move(candidate);
  return base::error::Success();
}

base::Status ExecuteCpuTransfer(const BoundTransfer& transfer) {
  for (const auto& operation : transfer.operations()) {
    if (operation.source == nullptr || operation.destination == nullptr || operation.bytes == 0 ||
        operation.bytes != operation.component.byte_size()) {
      return base::error::InvalidArgument("Bound transfer contains an invalid copy operation.");
    }
  }
  for (const auto& operation : transfer.operations()) {
    std::memcpy(operation.destination, operation.source, operation.bytes);
  }
  return base::error::Success();
}

base::Status ExecuteCpuTransferStaged(const BoundTransfer& transfer,
                                      uint8_t* staging,
                                      size_t staging_bytes) {
  if (!staging || staging_bytes < transfer.staging_bytes())
    return base::error::InvalidArgument("Transfer staging buffer is too small.");
  size_t offset = 0;
  for (const auto& operation : transfer.operations()) {
    if (!operation.source || !operation.destination || operation.bytes == 0 ||
        operation.bytes != operation.component.byte_size() ||
        operation.bytes > staging_bytes - offset)
      return base::error::InvalidArgument(
          "Bound transfer contains an invalid staged copy operation.");
    std::memcpy(staging + offset, operation.source, operation.bytes);
    offset += operation.bytes;
  }
  offset = 0;
  for (const auto& operation : transfer.operations()) {
    std::memcpy(operation.destination, staging + offset, operation.bytes);
    offset += operation.bytes;
  }
  return base::error::Success();
}

}  // namespace cache
