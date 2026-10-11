// Real exec-spawned clients: CUDA IPC mapping, page-major payload, deferred
// reclamation, and timestamp intervals for cross-process GPU overlap.
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include "base/shared_kv_arena.h"
#include "base/block_allocator.h"
namespace {
void check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
__device__ unsigned long long timer() {
  unsigned long long result;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(result));
  return result;
}
__global__ void probe(unsigned long long* times, unsigned long long cycles) {
  if (threadIdx.x) return;
  times[0] = timer();
  const auto begin = clock64();
  while (clock64() - begin < cycles) {}
  times[1] = timer();
}
}
int main(int argc, char** argv) {
  if (argc != 4) { std::cerr << "usage: mps_kv_probe ENDPOINT MODEL_ID BARRIER_FILE\n"; return 2; }
  try {
    check(cudaSetDevice(0));
    cudaStream_t stream{};
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto arena = std::make_shared<base::SharedKVArenaClient>(argv[1], argv[2], 8ull << 20, 0);
    nlohmann::json result;
    result["pid"] = getpid();
    // Two different geometries, including all-layer page sizes not powers of 2.
    const int layers = std::string(argv[2]) == "probe-a" ? 24 : 28;
    const int head = layers == 24 ? 64 : 128;
    auto group = std::make_shared<base::SharedKVPageGroup>(arena, layers, 16 * 2 * head * 2, 2ull << 20, stream);
    std::vector<std::unique_ptr<base::BlockAllocator>> allocators;
    std::vector<int> ids;
    for (int l = 0; l < layers; ++l) {
      allocators.emplace_back(new base::BlockAllocator(16, 2, head, group, l));
      ids.push_back(allocators.back()->allocate());
      if (ids.back() < 0) throw std::runtime_error("no probe page");
      if (ids.back() != ids.front()) throw std::runtime_error("layer block IDs differ");
      auto ptrs = allocators.back()->get_block_payload_ptrs(ids.back());
      check(cudaMemsetAsync(ptrs.key, 0x30 + l, ptrs.key_value_bytes, stream));
      check(cudaMemsetAsync(ptrs.value, 0x70 + l, ptrs.key_value_bytes, stream));
    }
    check(cudaStreamSynchronize(stream));
    for (int l = 0; l < layers; ++l) {
      auto ptrs = allocators[l]->get_block_payload_ptrs(ids[l]);
      std::vector<unsigned char> host(ptrs.key_value_bytes);
      check(cudaMemcpy(host.data(), ptrs.key, host.size(), cudaMemcpyDeviceToHost));
      for (auto byte : host) if (byte != 0x30 + l) throw std::runtime_error("K layer corruption");
      check(cudaMemcpy(host.data(), ptrs.value, host.size(), cudaMemcpyDeviceToHost));
      for (auto byte : host) if (byte != 0x70 + l) throw std::runtime_error("V layer corruption");
    }
    result["cuda_ipc_memory"] = true;
    result["page_major_layers"] = layers;
    result["kv_bytes_verified"] = true;
    // V4 compute/I/O pins must retain Chunk pages after request ownership ends.
    std::vector<base::BlockHandle> handles;
    std::vector<base::BlockLease> compute(layers), io(layers);
    for (int l = 0; l < layers; ++l) {
      auto& allocator = *allocators[l];
      handles.push_back(allocator.handle(ids[l]));
      if (!allocator.acquire_lease(handles.back(), base::BlockLeaseKind::kCompute, &compute[l]) ||
          !allocator.acquire_lease(handles.back(), base::BlockLeaseKind::kIO, &io[l]))
        throw std::runtime_error("V4 lease acquisition failed for Chunk page");
      allocator.free(ids[l]);
      compute[l].reset();
      if (allocator.ref_count(ids[l]) != 1 || allocator.io_pins(ids[l]) != 1)
        throw std::runtime_error("I/O lease lost ownership of Chunk page");
    }
    base::BlockLease wrong_pool;
    if (allocators[1]->acquire_lease(handles[0], base::BlockLeaseKind::kIO, &wrong_pool))
      throw std::runtime_error("cross-layer handle accepted");
    group->collect();
    if (!group->leased_bytes()) throw std::runtime_error("pinned chunk reclaimed");
    io.clear();
    check(cudaStreamSynchronize(stream));
    group->collect();
    if (group->leased_bytes()) throw std::runtime_error("unpinned chunk not reclaimed");
    for (int l = 0; l < layers; ++l) {
      ids[l] = allocators[l]->allocate();
      if (ids[l] < 0 || ids[l] != ids[0]) throw std::runtime_error("page reallocation failed");
      base::BlockLease stale;
      if (allocators[l]->is_current(handles[l]) ||
          allocators[l]->acquire_lease(handles[l], base::BlockLeaseKind::kIO, &stale))
        throw std::runtime_error("stale Chunk page handle accepted after reallocation");
    }
    result["v4_compute_io_leases"] = true;
    result["v4_stale_handles_rejected"] = true;
    unsigned long long* times = nullptr;
    check(cudaMalloc(&times, 2 * sizeof(*times)));
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0));
    std::ofstream(std::string(argv[3]) + "." + argv[2]) << getpid();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!std::ifstream(argv[3]).good()) {
      if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("barrier timeout");
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    probe<<<1, 32, 0, stream>>>(times, static_cast<unsigned long long>(prop.clockRate) * 300);
    check(cudaGetLastError());
    cudaEvent_t fence{};
    check(cudaEventCreateWithFlags(&fence, cudaEventDisableTiming));
    check(cudaEventRecord(fence, stream));
    const int retired_id = ids[0];
    allocators[0]->free(retired_id);
    const int replacement_id = allocators[0]->allocate();
    if (replacement_id < 0) throw std::runtime_error("replacement page unavailable");
    const auto fence_status = cudaEventQuery(fence);
    if (fence_status != cudaSuccess && fence_status != cudaErrorNotReady) check(fence_status);
    if (fence_status == cudaErrorNotReady && replacement_id == retired_id)
      throw std::runtime_error("retired GPU page was reused before fence");
    result["deferred_retirement_verified"] = fence_status == cudaErrorNotReady && replacement_id != retired_id;
    ids[0] = replacement_id;
    unsigned long long host_times[2];
    check(cudaMemcpyAsync(host_times, times, sizeof(host_times), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    result["gpu_start_ns"] = host_times[0]; result["gpu_end_ns"] = host_times[1];
    check(cudaEventDestroy(fence));
    for (int l = 0; l < layers; ++l) allocators[l]->free(ids[l]);
    check(cudaStreamSynchronize(stream));
    group->collect();
    result["chunks_reclaimed"] = true;
    allocators.clear(); group.reset(); arena.reset();
    check(cudaFree(times)); check(cudaStreamDestroy(stream));
    std::cout << result.dump() << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "PROBE_ERROR " << e.what() << '\n'; return 1;
  }
}
