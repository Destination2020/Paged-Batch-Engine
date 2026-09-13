#ifndef KUIPER_INCLUDE_DATA_WIRE_IO_H_
#define KUIPER_INCLUDE_DATA_WIRE_IO_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "data/data_ref.h"

namespace data {
namespace wire {

template <typename T>
inline void Put(T value, std::vector<uint8_t>* out) {
  for (size_t i = 0; i < sizeof(T); ++i)
    out->push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xff));
}
inline void PutBytes(const void* bytes, size_t size, std::vector<uint8_t>* out) {
  const auto* first = static_cast<const uint8_t*>(bytes);
  out->insert(out->end(), first, first + size);
}
template <typename T>
inline bool Get(const uint8_t** cursor, const uint8_t* end, T* value) {
  if (!cursor || !*cursor || !value || static_cast<size_t>(end - *cursor) < sizeof(T))
    return false;
  *value = 0;
  for (size_t i = 0; i < sizeof(T); ++i)
    *value |= static_cast<T>((*cursor)[i]) << (8 * i);
  *cursor += sizeof(T);
  return true;
}
inline bool GetBytes(const uint8_t** cursor, const uint8_t* end, void* output,
                     size_t size) {
  if (!cursor || !*cursor || !output || static_cast<size_t>(end - *cursor) < size)
    return false;
  std::memcpy(output, *cursor, size);
  *cursor += size;
  return true;
}
inline void PutHandle(const AllocationHandle& h, std::vector<uint8_t>* out) {
  Put<uint64_t>(h.owner_incarnation, out); Put<uint64_t>(h.allocation_id, out);
  Put<uint32_t>(h.generation, out);
}
inline bool GetHandle(const uint8_t** p, const uint8_t* e, AllocationHandle* h) {
  return Get<uint64_t>(p, e, &h->owner_incarnation) &&
         Get<uint64_t>(p, e, &h->allocation_id) && Get<uint32_t>(p, e, &h->generation);
}

}  // namespace wire
}  // namespace data
#endif
