// Diagnostic only: retain device-side progress samples, not just kernel lifetimes.
// Each record is little-endian {globaltimer:u64, smid:u32, accumulator_bits:u32}.
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <unistd.h>

struct Sample { uint64_t time; uint32_t sm; uint32_t value; };
static_assert(sizeof(Sample) == 16);

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

__global__ void sample_sm_activity(Sample* samples, float* results, int count) {
  float value = 1.0f + threadIdx.x * .001f;
  for (int index = 0; index < count; ++index) {
    #pragma unroll
    for (int step = 0; step < 16; ++step) value = fmaf(value, .999999f, .00001f);
    if (threadIdx.x == 0) {
      uint64_t now;
      uint32_t sm;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
      asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
      samples[static_cast<size_t>(blockIdx.x) * count + index] = {now, sm, __float_as_uint(value)};
    }
  }
  results[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: mps_sm_activity_probe BARRIER_PATH OUTPUT_PREFIX\n";
    return 2;
  }
  try {
    constexpr int blocks = 4, threads = 32, count = 250000;
    check(cudaSetDevice(0));
    Sample* device = nullptr;
    float* results = nullptr;
    check(cudaMalloc(&device, sizeof(Sample) * blocks * count));
    check(cudaMalloc(&results, sizeof(float) * blocks * threads));
    sample_sm_activity<<<blocks, threads>>>(device, results, 1);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    const std::string prefix = argv[2];
    std::ofstream(prefix + ".ready") << getpid() << '\n';
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (!std::ifstream(argv[1]).good()) {
      if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("barrier timeout");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    sample_sm_activity<<<blocks, threads>>>(device, results, count);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    std::vector<Sample> host(blocks * count);
    check(cudaMemcpy(host.data(), device, host.size() * sizeof(Sample), cudaMemcpyDeviceToHost));
    std::vector<float> sums(blocks * threads);
    check(cudaMemcpy(sums.data(), results, sums.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (float value : sums) if (!std::isfinite(value) || value <= 1.0f)
      throw std::runtime_error("arithmetic result invalid");
    std::ofstream output(prefix + ".bin", std::ios::binary);
    output.write(reinterpret_cast<const char*>(host.data()), host.size() * sizeof(Sample));
    output.close();
    if (!output) throw std::runtime_error("sample output failed");
    check(cudaFree(results));
    check(cudaFree(device));
    std::cout << "{\"pid\":" << getpid() << ",\"blocks\":" << blocks
              << ",\"threads_per_block\":" << threads << ",\"samples_per_block\":" << count
              << ",\"arithmetic_verified\":true}\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
