#include "data/tensor_bundle.h"

#include <limits>
#include <set>
#include <cstring>
#include "data/data_client.h"
#include "data/wire_io.h"

namespace data {
namespace {

uint64_t DTypeBytes(TensorDType dtype) {
  switch (dtype) {
    case TensorDType::kFloat32:
    case TensorDType::kInt32:
      return 4;
    case TensorDType::kFloat16:
    case TensorDType::kBFloat16:
      return 2;
    case TensorDType::kInt64:
      return 8;
    case TensorDType::kUint8:
      return 1;
    case TensorDType::kUnknown:
      return 0;
  }
  return 0;
}

}  // namespace

DataError DecodeTensorBundlePayload(const uint8_t* bytes, size_t size,
                                    TensorBundleSchema* schema, std::string* reason) {
  auto fail = [reason](DataError error, const char* message) {
    if (reason) *reason = message; return error;
  };
  if (!bytes || !schema || size < 24) return fail(DataError::kInvalidArgument, "short bundle header");
  const uint8_t* cursor = bytes; const uint8_t* end = bytes + size;
  uint32_t magic=0, alignment=0, reserved=0; uint16_t version=0, count=0; uint64_t total=0;
  if (!wire::Get(&cursor,end,&magic) || !wire::Get(&cursor,end,&version) ||
      !wire::Get(&cursor,end,&count) || !wire::Get(&cursor,end,&total) ||
      !wire::Get(&cursor,end,&alignment) || !wire::Get(&cursor,end,&reserved) ||
      magic != 0x4e424250 || version != 1 || reserved != 0 || total != size || count == 0)
    return fail(DataError::kInvalidArgument, "invalid bundle header");
  TensorBundleSchema value; value.alignment=alignment; value.total_bytes=total;
  value.components.reserve(count);
  for (uint16_t i=0;i<count;++i) {
    uint16_t name_size=0,dtype=0,rank=0,pad=0; TensorComponent c;
    if (!wire::Get(&cursor,end,&name_size)||!wire::Get(&cursor,end,&dtype)||
        !wire::Get(&cursor,end,&rank)||!wire::Get(&cursor,end,&pad)||
        !wire::Get(&cursor,end,&c.byte_offset)||!wire::Get(&cursor,end,&c.byte_length)||
        !wire::GetBytes(&cursor,end,c.checksum.data(),32)||pad||name_size==0||rank==0||rank>16||
        static_cast<size_t>(end-cursor)<name_size)
      return fail(DataError::kInvalidArgument,"invalid component metadata");
    c.dtype=static_cast<TensorDType>(dtype);c.name.assign(reinterpret_cast<const char*>(cursor),name_size);cursor+=name_size;c.shape.resize(rank);
    for(auto&dim:c.shape)if(!wire::Get(&cursor,end,&dim))return fail(DataError::kInvalidArgument,"short component shape");
    value.components.push_back(std::move(c));
  }
  auto status=ValidateTensorBundle(value,reason);if(status!=DataError::kOk)return status;
  for(const auto&c:value.components)if(DataChecksum(bytes+c.byte_offset,c.byte_length)!=c.checksum)
    return fail(DataError::kChecksumMismatch,"component checksum mismatch");
  *schema=std::move(value);return DataError::kOk;
}

DataError ValidateTensorBundle(const TensorBundleSchema& schema,
                               std::string* reason) {
  auto fail = [reason](DataError error, const char* message) {
    if (reason != nullptr) *reason = message;
    return error;
  };
  if (schema.version != kTensorBundleSchemaVersion)
    return fail(DataError::kUnsupportedVersion, "unsupported tensor bundle version");
  if (schema.total_bytes == 0 || schema.components.empty() || schema.alignment == 0 ||
      (schema.alignment & (schema.alignment - 1)) != 0)
    return fail(DataError::kInvalidArgument, "invalid bundle size, component count, or alignment");

  std::set<std::string> names;
  uint64_t previous_end = 0;
  for (const auto& component : schema.components) {
    const uint64_t element_bytes = DTypeBytes(component.dtype);
    if (component.name.empty() || !names.emplace(component.name).second ||
        element_bytes == 0 || component.shape.empty() || component.byte_length == 0)
      return fail(DataError::kInvalidArgument, "invalid or duplicate tensor component");
    if (component.byte_offset % schema.alignment != 0 ||
        component.byte_offset < previous_end || component.byte_offset > schema.total_bytes ||
        component.byte_length > schema.total_bytes - component.byte_offset)
      return fail(DataError::kCoverageMismatch, "tensor component range is invalid");

    uint64_t expected = element_bytes;
    for (uint64_t dimension : component.shape) {
      if (dimension == 0 || expected > std::numeric_limits<uint64_t>::max() / dimension)
        return fail(DataError::kCoverageMismatch, "tensor shape is empty or overflows");
      expected *= dimension;
    }
    if (expected != component.byte_length)
      return fail(DataError::kCoverageMismatch, "tensor shape does not match its byte length");
    previous_end = component.byte_offset + component.byte_length;
  }
  return DataError::kOk;
}

}  // namespace data
