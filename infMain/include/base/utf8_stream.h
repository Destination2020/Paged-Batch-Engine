#pragma once
#include <cstdint>
#include <string>

namespace base {
// BPE tokens can end in the middle of a code point. Keep that suffix until the
// next token arrives; emit replacement characters only for malformed/final data.
inline std::string drain_utf8(std::string* pending, bool final = false) {
  std::string output;
  size_t i = 0;
  while (i < pending->size()) {
    const auto first = static_cast<unsigned char>((*pending)[i]);
    int length = first < 0x80 ? 1 : first >= 0xc2 && first <= 0xdf ? 2 :
                 first >= 0xe0 && first <= 0xef ? 3 : first >= 0xf0 && first <= 0xf4 ? 4 : 0;
    if (!length) { output += "\xef\xbf\xbd"; ++i; continue; }
    bool valid = true;
    const size_t available = pending->size() - i;
    for (size_t j = 1; j < static_cast<size_t>(length) && j < available; ++j) {
      const auto byte = static_cast<unsigned char>((*pending)[i + j]);
      if (byte < 0x80 || byte > 0xbf ||
          (j == 1 && ((first == 0xe0 && byte < 0xa0) || (first == 0xed && byte > 0x9f) ||
                      (first == 0xf0 && byte < 0x90) || (first == 0xf4 && byte > 0x8f)))) valid = false;
    }
    if (!valid) { output += "\xef\xbf\xbd"; ++i; continue; }
    if (available < static_cast<size_t>(length)) {
      if (!final) break;
      output += "\xef\xbf\xbd"; i = pending->size(); break;
    }
    output.append(*pending, i, length); i += length;
  }
  pending->erase(0, i);
  return output;
}
}  // namespace base
