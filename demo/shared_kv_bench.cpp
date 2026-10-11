// Host wall time microbenchmark; GPU writes are a synthetic payload, not attention.
#include <cuda_runtime_api.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include "base/block_allocator.h"
#include "base/shared_kv_arena.h"

using Clock = std::chrono::steady_clock;
static void check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
static double us(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}
int main(int argc, char** argv) {
  if (argc != 6) return 2; // endpoint or private, layers, head size, held pages, iterations
  try {
    const int layers = std::stoi(argv[2]), head = std::stoi(argv[3]);
    const int held = std::stoi(argv[4]), iterations = std::stoi(argv[5]);
    const uint64_t half = 16ull * 2 * head * 2;
    const uint64_t bytes = 84ull << 20;
    const int capacity = bytes / (2 * layers * half);
    if (held < 0 || held > capacity || iterations < 1) throw std::invalid_argument("invalid workload");
    check(cudaSetDevice(0));
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::shared_ptr<base::SharedKVArenaClient> arena;
    std::shared_ptr<base::SharedKVPageGroup> group;
    if (std::string(argv[1]) != "private") {
      arena = std::make_shared<base::SharedKVArenaClient>(argv[1],
          "bench-" + std::to_string(getpid()), bytes, 0);
      group = std::make_shared<base::SharedKVPageGroup>(arena, layers, half, 2ull << 20, stream);
    }
    std::vector<std::unique_ptr<base::BlockAllocator>> pools;
    for (int l = 0; l < layers; ++l) {
      if (group) pools.emplace_back(new base::BlockAllocator(16, 2, head, group, l));
      else pools.emplace_back(new base::BlockAllocator(capacity, 16, 2, head,
          base::DataType::kDataTypeBf16, base::DeviceType::kDeviceCUDA));
    }
    auto allocate = [&]() {
      std::vector<int> row;
      for (auto& pool : pools) {
        int id = pool->allocate();
        if (id < 0) throw std::runtime_error("unexpected exhaustion");
        row.push_back(id);
      }
      return row;
    };
    auto release = [&](const std::vector<int>& row) {
      for (int l = 0; l < layers; ++l) pools[l]->free(row[l]);
    };
    auto drain = [&]() { check(cudaStreamSynchronize(stream)); if (group) group->collect(); };
    std::vector<std::vector<int>> resident;
    for (int i = 0; i < held; ++i) resident.push_back(allocate());
    drain();
    nlohmann::json samples = nlohmann::json::array();
    for (int round = -3; round < iterations; ++round) {
      auto start = Clock::now();
      if (held == capacity) {
        if (pools[0]->allocate() != -1) throw std::runtime_error("full pool accepted allocation");
        double denied = us(start);
        if (round >= 0) samples.push_back({{"denied_us", denied}});
        continue;
      }
      std::vector<std::vector<int>> rows;
      for (int i = 0; i < 4; ++i) rows.push_back(allocate());
      double alloc = us(start);
      start = Clock::now();
      for (const auto& row : rows) for (int l = 0; l < layers; ++l) {
        auto payload = pools[l]->get_block_payload_ptrs(row[l]);
        check(cudaMemsetAsync(payload.key, 17, payload.key_value_bytes, stream));
        check(cudaMemsetAsync(payload.value, 29, payload.key_value_bytes, stream));
      }
      check(cudaStreamSynchronize(stream));
      double write = us(start);
      start = Clock::now();
      for (const auto& row : rows) release(row);
      double release_cpu = us(start);
      drain();
      double release_complete = us(start);
      if (round >= 0) samples.push_back({{"allocate_us", alloc}, {"write_us", write},
          {"release_cpu_us", release_cpu}, {"release_complete_us", release_complete}});
    }
    for (const auto& row : resident) release(row);
    drain();
    if (group && group->leased_bytes()) throw std::runtime_error("leaked chunks");
    std::cout << nlohmann::json{{"shared", bool(group)}, {"layers", layers},
        {"held_pages", held}, {"capacity_pages", capacity}, {"batch_pages", 4},
        {"samples", samples}}.dump() << std::endl;
    pools.clear(); group.reset(); arena.reset();
    check(cudaStreamDestroy(stream));
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl; return 1;
  }
}
