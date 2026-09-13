#ifndef KUIPER_INCLUDE_DATA_MULTIMODAL_PREFIX_KEY_H_
#define KUIPER_INCLUDE_DATA_MULTIMODAL_PREFIX_KEY_H_

#include <cstdint>
#include <vector>

#include "data/data_client.h"

namespace data {

struct MediaPrefixBinding {
  uint32_t token_offset = 0;
  uint32_t token_length = 0;
  ContentId content;
};

inline ContentId MakeMultimodalPrefixContentId(
    const std::vector<int32_t>& tokens,
    const std::vector<MediaPrefixBinding>& media) {
  std::vector<uint8_t> wire;
  const auto put32 = [&wire](uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8)
      wire.push_back(static_cast<uint8_t>(value >> shift));
  };
  wire.insert(wire.end(), {'m','m','-','k','v','-','v','1'});
  put32(static_cast<uint32_t>(tokens.size()));
  for (int32_t token : tokens) put32(static_cast<uint32_t>(token));
  put32(static_cast<uint32_t>(media.size()));
  for (const auto& binding : media) {
    put32(binding.token_offset);
    put32(binding.token_length);
    wire.insert(wire.end(), binding.content.digest.begin(), binding.content.digest.end());
  }
  return {DataChecksum(wire.data(), wire.size())};
}

// Produces a page-aligned radix key with the same token count as the model
// input. Eight placeholder positions carry all 256 content-id bits, so image
// identity participates in KV lookup without changing physical page mapping.
inline bool MakeMultimodalRadixTokens(
    const std::vector<int32_t>& tokens,
    const std::vector<MediaPrefixBinding>& media,
    std::vector<int32_t>* keyed_tokens) {
  if (!keyed_tokens) return false;
  *keyed_tokens = tokens;
  for (const auto& binding : media) {
    if (binding.token_length < 8 || binding.token_offset > tokens.size() ||
        binding.token_length > tokens.size() - binding.token_offset) return false;
    for (uint32_t word = 0; word < 8; ++word) {
      uint32_t value = 0;
      for (uint32_t byte = 0; byte < 4; ++byte)
        value |= static_cast<uint32_t>(binding.content.digest[word * 4 + byte]) << (byte * 8);
      (*keyed_tokens)[binding.token_offset + word] =
          static_cast<int32_t>(value ^ (0x9e3779b9u + word));
    }
  }
  return true;
}

}  // namespace data
#endif
