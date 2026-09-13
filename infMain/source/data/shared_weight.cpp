#include "data/shared_weight.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

#include <cuda.h>
#include <openssl/evp.h>

#include "data/wire_io.h"

namespace data {
namespace {

bool CudaOk(cudaError_t status, const char* operation, std::string* error) {
  if (status == cudaSuccess) return true;
  if (error) {
    std::ostringstream stream;
    stream << operation << ": " << cudaGetErrorString(status);
    *error = stream.str();
  }
  return false;
}

bool ValidState(SharedWeightState state) {
  return state >= SharedWeightState::kAbsent &&
         state <= SharedWeightState::kDraining;
}

}  // namespace

bool EncodeSharedWeightDescriptor(const SharedWeightDescriptor& d,
                                  std::vector<uint8_t>* output) {
  if (!output || !output->empty() || d.version != kSharedWeightVersion ||
      d.state != SharedWeightState::kReady || d.owner_incarnation == 0 ||
      d.allocation_id == 0 || d.generation == 0 || d.device < 0 || d.bytes == 0 ||
      d.dtype == base::DataType::kDataTypeUnknown) {
    return false;
  }
  wire::Put(kSharedWeightMagic, output);
  wire::Put(d.version, output);
  output->push_back(static_cast<uint8_t>(d.state));
  output->push_back(0);
  wire::Put(d.owner_incarnation, output);
  wire::Put(d.allocation_id, output);
  wire::Put(d.generation, output);
  wire::Put<uint32_t>(d.device, output);
  wire::Put(d.bytes, output);
  output->push_back(static_cast<uint8_t>(d.dtype));
  for (int i = 0; i < 7; ++i) output->push_back(0);
  wire::PutBytes(d.model_content.data(), d.model_content.size(), output);
  wire::PutBytes(d.layout_identity.data(), d.layout_identity.size(), output);
  wire::PutBytes(&d.device_uuid, sizeof(d.device_uuid), output);
  wire::PutBytes(&d.memory, sizeof(d.memory), output);
  wire::PutBytes(&d.ready, sizeof(d.ready), output);
  return true;
}

bool DecodeSharedWeightDescriptor(const uint8_t* bytes, size_t size,
                                  SharedWeightDescriptor* output) {
  if (!bytes || !output) return false;
  const uint8_t* cursor = bytes;
  const uint8_t* end = bytes + size;
  SharedWeightDescriptor d;
  uint32_t magic = 0;
  uint32_t device = 0;
  uint8_t state = 0;
  uint8_t reserved = 0;
  uint8_t dtype = 0;
  uint8_t padding[7]{};
  if (!wire::Get(&cursor, end, &magic) ||
      !wire::Get(&cursor, end, &d.version) ||
      !wire::Get(&cursor, end, &state) ||
      !wire::Get(&cursor, end, &reserved) ||
      !wire::Get(&cursor, end, &d.owner_incarnation) ||
      !wire::Get(&cursor, end, &d.allocation_id) ||
      !wire::Get(&cursor, end, &d.generation) ||
      !wire::Get(&cursor, end, &device) ||
      !wire::Get(&cursor, end, &d.bytes) ||
      !wire::Get(&cursor, end, &dtype) ||
      !wire::GetBytes(&cursor, end, padding, sizeof(padding)) ||
      !wire::GetBytes(&cursor, end, d.model_content.data(), d.model_content.size()) ||
      !wire::GetBytes(&cursor, end, d.layout_identity.data(), d.layout_identity.size()) ||
      !wire::GetBytes(&cursor, end, &d.device_uuid, sizeof(d.device_uuid)) ||
      !wire::GetBytes(&cursor, end, &d.memory, sizeof(d.memory)) ||
      !wire::GetBytes(&cursor, end, &d.ready, sizeof(d.ready)) || cursor != end) {
    return false;
  }
  d.state = static_cast<SharedWeightState>(state);
  d.device = static_cast<int32_t>(device);
  d.dtype = static_cast<base::DataType>(dtype);
  if (magic != kSharedWeightMagic || d.version != kSharedWeightVersion ||
      !ValidState(d.state) || d.state != SharedWeightState::kReady || reserved != 0 ||
      std::any_of(std::begin(padding), std::end(padding), [](uint8_t v) { return v != 0; }) ||
      d.owner_incarnation == 0 || d.allocation_id == 0 || d.generation == 0 ||
      d.device < 0 || d.bytes == 0 || d.dtype == base::DataType::kDataTypeUnknown) {
    return false;
  }
  *output = d;
  return true;
}

SharedWeightOwner::~SharedWeightOwner() { reset(); }

void SharedWeightOwner::reset() {
  if (descriptor_.device >= 0) cudaSetDevice(descriptor_.device);
  if (ready_) cudaEventDestroy(ready_);
  if (memory_) cudaFree(memory_);
  ready_ = nullptr;
  memory_ = nullptr;
}

bool SharedWeightOwner::load_file(const std::string& path, int32_t device,
                                  uint64_t owner_incarnation,
                                  const Digest256& model_content,
                                  const Digest256& layout_identity,
                                  base::DataType dtype, uint64_t staging_bytes,
                                  std::string* error) {
  if (memory_ || path.empty() || device < 0 || owner_incarnation == 0 ||
      dtype == base::DataType::kDataTypeUnknown || staging_bytes == 0 ||
      staging_bytes > (1ULL << 30)) {
    if (error) *error = "invalid shared weight load configuration";
    return false;
  }
  stats_.state = SharedWeightState::kLoading;
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    stats_.state = SharedWeightState::kFailed;
    if (error) *error = "cannot open shared weight model file";
    return false;
  }
  const auto end = stream.tellg();
  if (end <= 0 || static_cast<uint64_t>(end) >
                      static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    stats_.state = SharedWeightState::kFailed;
    if (error) *error = "invalid shared weight model file size";
    return false;
  }
  const uint64_t bytes = static_cast<uint64_t>(end);
  stream.seekg(0);
  if (!CudaOk(cudaSetDevice(device), "cudaSetDevice", error) ||
      !CudaOk(cudaMalloc(&memory_, bytes), "cudaMalloc shared weights", error) ||
      !CudaOk(cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming |
                                                   cudaEventInterprocess),
              "create shared weight ready event", error)) {
    stats_.state = SharedWeightState::kFailed;
    reset();
    return false;
  }
  const auto begin = std::chrono::steady_clock::now();
  std::vector<uint8_t> staging(static_cast<size_t>(std::min(staging_bytes, bytes)));
  EVP_MD_CTX* digest_context = EVP_MD_CTX_new();
  if (digest_context == nullptr ||
      EVP_DigestInit_ex(digest_context, EVP_sha256(), nullptr) != 1) {
    if (digest_context) EVP_MD_CTX_free(digest_context);
    stats_.state = SharedWeightState::kFailed;
    if (error) *error = "initialize model SHA-256 failed";
    reset();
    return false;
  }
  uint64_t offset = 0;
  while (offset < bytes) {
    const size_t chunk = static_cast<size_t>(std::min<uint64_t>(staging.size(), bytes - offset));
    stream.read(reinterpret_cast<char*>(staging.data()), chunk);
    if (stream.gcount() != static_cast<std::streamsize>(chunk) ||
        EVP_DigestUpdate(digest_context, staging.data(), chunk) != 1 ||
        !CudaOk(cudaMemcpy(static_cast<uint8_t*>(memory_) + offset, staging.data(), chunk,
                           cudaMemcpyHostToDevice),
                "upload shared weight chunk", error)) {
      stats_.state = SharedWeightState::kFailed;
      EVP_MD_CTX_free(digest_context);
      reset();
      return false;
    }
    offset += chunk;
  }
  Digest256 actual_content{};
  unsigned int actual_content_bytes = 0;
  const bool digest_ok =
      EVP_DigestFinal_ex(digest_context, actual_content.data(),
                         &actual_content_bytes) == 1;
  EVP_MD_CTX_free(digest_context);
  if (!digest_ok || actual_content_bytes != actual_content.size() ||
      actual_content != model_content) {
    stats_.state = SharedWeightState::kFailed;
    if (error) *error = "shared weight model SHA-256 mismatch";
    reset();
    return false;
  }
  if (!CudaOk(cudaEventRecord(ready_), "record shared weight ready event", error) ||
      !CudaOk(cudaEventSynchronize(ready_), "synchronize shared weight upload", error) ||
      !CudaOk(cudaIpcGetMemHandle(&descriptor_.memory, memory_),
              "export shared weight memory", error) ||
      !CudaOk(cudaIpcGetEventHandle(&descriptor_.ready, ready_),
              "export shared weight ready event", error)) {
    stats_.state = SharedWeightState::kFailed;
    reset();
    return false;
  }
  CUdevice driver_device = 0;
  CUuuid driver_uuid{};
  if (cuInit(0) != CUDA_SUCCESS ||
      cuDeviceGet(&driver_device, device) != CUDA_SUCCESS ||
      cuDeviceGetUuid_v2(&driver_uuid, driver_device) != CUDA_SUCCESS) {
    stats_.state = SharedWeightState::kFailed;
    if (error) *error = "query shared weight GPU UUID failed";
    reset();
    return false;
  }
  static_assert(sizeof(driver_uuid.bytes) == sizeof(descriptor_.device_uuid.bytes),
                "CUDA UUID ABI mismatch");
  std::memcpy(descriptor_.device_uuid.bytes, driver_uuid.bytes,
              sizeof(driver_uuid.bytes));
  descriptor_.state = SharedWeightState::kReady;
  descriptor_.owner_incarnation = owner_incarnation;
  descriptor_.allocation_id = 1;
  descriptor_.generation = 1;
  descriptor_.device = device;
  descriptor_.bytes = bytes;
  descriptor_.dtype = dtype;
  descriptor_.model_content = model_content;
  descriptor_.layout_identity = layout_identity;
  stats_.state = SharedWeightState::kReady;
  stats_.owner_incarnation = owner_incarnation;
  stats_.allocation_id = 1;
  stats_.generation = 1;
  stats_.physical_bytes = bytes;
  stats_.upload_bytes = bytes;
  stats_.upload_count = 1;
  stats_.allocation_count = 1;
  stats_.staging_peak_bytes = staging.size();
  stats_.upload_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - begin).count();
  return true;
}

bool SharedWeightOwner::descriptor(SharedWeightDescriptor* output,
                                   std::string* error) const {
  if (!output || descriptor_.state != SharedWeightState::kReady || !memory_) {
    if (error) *error = "shared weights are not ready";
    return false;
  }
  *output = descriptor_;
  return true;
}

SharedWeightImport::~SharedWeightImport() {
  if (device_ >= 0) cudaSetDevice(device_);
  if (ready_) cudaEventDestroy(ready_);
  if (mapped_) cudaIpcCloseMemHandle(mapped_);
}

bool SharedWeightImport::open(const SharedWeightDescriptor& descriptor,
                              int32_t device, std::string* error) {
  if (mapped_ || descriptor.state != SharedWeightState::kReady ||
      descriptor.device != device) {
    if (error) *error = "shared weight mapping requires the owner's physical GPU";
    return false;
  }
  CUdevice driver_device = 0;
  CUuuid driver_uuid{};
  if (cuInit(0) != CUDA_SUCCESS ||
      cuDeviceGet(&driver_device, device) != CUDA_SUCCESS ||
      cuDeviceGetUuid_v2(&driver_uuid, driver_device) != CUDA_SUCCESS ||
      std::memcmp(driver_uuid.bytes, descriptor.device_uuid.bytes,
                  sizeof(driver_uuid.bytes)) != 0) {
    if (error) *error = "shared weight physical GPU UUID mismatch";
    return false;
  }
  descriptor_ = descriptor;
  device_ = device;
  return CudaOk(cudaSetDevice(device), "cudaSetDevice", error) &&
         CudaOk(cudaIpcOpenMemHandle(&mapped_, descriptor.memory,
                                    cudaIpcMemLazyEnablePeerAccess),
                "open shared weight memory", error) &&
         CudaOk(cudaIpcOpenEventHandle(&ready_, descriptor.ready),
                "open shared weight ready event", error);
}

bool SharedWeightImport::wait_ready(void* stream, std::string* error) {
  if (!mapped_ || !ready_) return false;
  return CudaOk(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream), ready_),
                                   "wait shared weight ready", error);
}

}  // namespace data
