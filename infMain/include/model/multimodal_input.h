#ifndef KUIPER_INCLUDE_MODEL_MULTIMODAL_INPUT_H_
#define KUIPER_INCLUDE_MODEL_MULTIMODAL_INPUT_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace model {

inline constexpr uint32_t kMultimodalSequencePlanVersion = 1;
inline constexpr uint32_t kTensorBundleSchemaVersion = 1;

enum class MultimodalInputError : uint32_t {
  kOk = 0,
  kUnsupportedVersion = 1,
  kPositionCountMismatch = 2,
  kInvalidSpan = 3,
  kOverlappingSpan = 4,
  kFeatureLengthMismatch = 5,
  kFeatureRefOutOfRange = 6,
  kInvalidGrid = 7,
  kInvalidBundleComponent = 8,
};

struct VisionFeatureRef {
  // Stable textual IDs cross the wire. No address or C++ object layout is part
  // of the protocol; M2 maps these IDs to typed DataRef handles.
  std::string content_id;
  std::string representation_id;
  uint64_t rows = 0;
  uint64_t width = 0;
  uint64_t bytes = 0;
  std::string checksum_sha256;
};

struct MediaSpan {
  uint32_t token_begin = 0;  // inclusive index in expanded_token_ids
  uint32_t token_end = 0;    // exclusive index
  uint32_t feature_ref_index = 0;
  std::array<uint32_t, 3> grid_thw{0, 0, 0};
};

struct DecodePositionState {
  // Qwen2.5-VL advances decode positions by token offset + rope_delta.  The
  // offset remains separate so cache/chunk offsets are never mistaken for an
  // mRoPE coordinate.
  uint64_t next_token_offset = 0;
  int64_t rope_delta = 0;
};

struct MultimodalSequencePlan {
  uint32_t version = kMultimodalSequencePlanVersion;
  std::vector<int32_t> expanded_token_ids;
  std::vector<MediaSpan> media_spans;
  std::vector<VisionFeatureRef> feature_refs;
  // Axis-major [temporal, height, width], each axis has one entry per token.
  std::array<std::vector<int32_t>, 3> position_ids;
  DecodePositionState decode_position;
};

struct TensorBundleComponent {
  std::string name;
  uint32_t dtype = 0;
  std::vector<uint64_t> shape;
  uint64_t byte_offset = 0;
  uint64_t byte_length = 0;
  std::string checksum_sha256;
};

struct TensorBundleSchema {
  uint32_t version = kTensorBundleSchemaVersion;
  uint64_t total_bytes = 0;
  std::vector<TensorBundleComponent> components;
};

inline MultimodalInputError ValidateMultimodalSequencePlan(
    const MultimodalSequencePlan& plan, std::string* reason = nullptr) {
  auto fail = [reason](MultimodalInputError error, const char* text) {
    if (reason != nullptr) {
      *reason = text;
    }
    return error;
  };
  if (plan.version != kMultimodalSequencePlanVersion) {
    return fail(MultimodalInputError::kUnsupportedVersion, "unsupported sequence plan version");
  }
  const size_t tokens = plan.expanded_token_ids.size();
  for (const auto& axis : plan.position_ids) {
    if (axis.size() != tokens) {
      return fail(MultimodalInputError::kPositionCountMismatch,
                  "every mRoPE axis must contain one value per expanded token");
    }
  }
  uint32_t previous_end = 0;
  for (const MediaSpan& span : plan.media_spans) {
    if (span.token_begin >= span.token_end || span.token_end > tokens) {
      return fail(MultimodalInputError::kInvalidSpan, "media span is outside the token sequence");
    }
    if (span.token_begin < previous_end) {
      return fail(MultimodalInputError::kOverlappingSpan, "media spans overlap or are unsorted");
    }
    if (span.feature_ref_index >= plan.feature_refs.size()) {
      return fail(MultimodalInputError::kFeatureRefOutOfRange,
                  "media span references a missing feature bundle");
    }
    if (span.grid_thw[0] == 0 || span.grid_thw[1] == 0 || span.grid_thw[2] == 0) {
      return fail(MultimodalInputError::kInvalidGrid, "media grid dimensions must be nonzero");
    }
    const uint64_t span_rows = span.token_end - span.token_begin;
    if (plan.feature_refs[span.feature_ref_index].rows != span_rows) {
      return fail(MultimodalInputError::kFeatureLengthMismatch,
                  "media span length differs from merged feature rows");
    }
    previous_end = span.token_end;
  }
  return MultimodalInputError::kOk;
}

inline MultimodalInputError ValidateTensorBundleSchema(const TensorBundleSchema& bundle,
                                                       std::string* reason = nullptr) {
  auto fail = [reason](MultimodalInputError error, const char* text) {
    if (reason != nullptr) {
      *reason = text;
    }
    return error;
  };
  if (bundle.version != kTensorBundleSchemaVersion) {
    return fail(MultimodalInputError::kUnsupportedVersion, "unsupported tensor bundle version");
  }
  uint64_t previous_end = 0;
  for (const TensorBundleComponent& component : bundle.components) {
    if (component.name.empty() || component.shape.empty() || component.byte_length == 0 ||
        component.byte_offset < previous_end ||
        component.byte_offset > bundle.total_bytes ||
        component.byte_length > bundle.total_bytes - component.byte_offset) {
      return fail(MultimodalInputError::kInvalidBundleComponent,
                  "bundle component has invalid name, shape, size, or byte range");
    }
    previous_end = component.byte_offset + component.byte_length;
  }
  return MultimodalInputError::kOk;
}

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_MULTIMODAL_INPUT_H_
