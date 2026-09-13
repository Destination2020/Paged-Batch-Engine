#ifndef KUIPER_INCLUDE_DATA_IPC_POOL_H_
#define KUIPER_INCLUDE_DATA_IPC_POOL_H_

#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>

namespace data {

inline constexpr uint32_t kIpcPoolMagic = 0x50494250;  // "PBIP"
inline constexpr uint16_t kIpcPoolVersion = 2;

struct IpcPoolDescriptor {
  uint64_t owner_incarnation = 0;
  int32_t device = -1;
  uint64_t bytes = 0;
  int32_t num_layers = 0;
  int32_t num_blocks = 0;
  int32_t block_size = 0;
  int32_t num_kv_heads = 0;
  int32_t head_size = 0;
  uint8_t storage_dtype = 0;
  cudaIpcMemHandle_t memory{};
  cudaIpcEventHandle_t ready{};
  cudaIpcEventHandle_t consumed{};
};

bool EncodeIpcPoolDescriptor(const IpcPoolDescriptor& descriptor,
                             std::vector<uint8_t>* output);
bool DecodeIpcPoolDescriptor(const uint8_t* bytes, size_t size,
                             IpcPoolDescriptor* output);

class CudaIpcPoolOwner {
 public:
  ~CudaIpcPoolOwner();
  bool create(int device, uint64_t bytes, uint8_t pattern, std::string* error);
  bool descriptor(uint64_t incarnation, IpcPoolDescriptor* output,
                  std::string* error);
  bool set_kv_layout(int32_t num_layers,int32_t num_blocks,int32_t block_size,
                     int32_t num_kv_heads,int32_t head_size,uint8_t storage_dtype,
                     std::string* error);
  bool wait_consumed(std::string* error);
 private:
  int device_ = -1; void* memory_ = nullptr; uint64_t bytes_ = 0;
  cudaEvent_t ready_ = nullptr; cudaEvent_t consumed_ = nullptr;
  int32_t num_layers_=0,num_blocks_=0,block_size_=0,num_kv_heads_=0,head_size_=0;
  uint8_t storage_dtype_=0;
};

class CudaIpcPoolImport {
 public:
  ~CudaIpcPoolImport();
  bool open(const IpcPoolDescriptor& descriptor, int device, std::string* error);
  bool prepare_compute_view(void* stream, uint64_t* transferred_bytes,
                            double* elapsed_ms, std::string* path,
                            std::string* error);
  // Cross-device page-granular replica. The allocation preserves the source
  // address layout, but only the listed logical block slots are transferred.
  bool prepare_compute_view_slots(const std::vector<int32_t>& slots, void* stream,
                                  uint64_t* transferred_bytes,
                                  double* elapsed_ms, std::string* path,
                                  std::string* error);
  bool validate_and_signal(uint8_t pattern, uint64_t* transferred_bytes,
                           double* elapsed_ms, std::string* path, std::string* error);
  bool wait_ready(void* stream, std::string* error);
  void* mapped() const { return compute_view_; }
  int device() const { return device_; }
  const IpcPoolDescriptor& descriptor() const { return descriptor_; }
 private:
  IpcPoolDescriptor descriptor_{}; int device_ = -1; void* ipc_mapped_ = nullptr;
  void* compute_view_ = nullptr;
  cudaEvent_t ready_ = nullptr; cudaEvent_t consumed_ = nullptr;
};

}  // namespace data
#endif
