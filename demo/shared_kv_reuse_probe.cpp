// A command-driven real CUDA IPC client for deterministic cross-model reuse tests.
#include <cuda_runtime_api.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include <nlohmann/json.hpp>
#include "base/block_allocator.h"
#include "base/shared_kv_arena.h"

namespace {
void check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
void CUDART_CB gate(void* pointer) {
  auto* open = static_cast<std::atomic<bool>*>(pointer);
  while (!open->load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
}
}

int main(int argc, char** argv) {
  if (argc != 5) return 2; // endpoint, model identity, layers, head size
  try {
    check(cudaSetDevice(0));
    cudaStream_t stream{};
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const int layers = std::stoi(argv[3]), head = std::stoi(argv[4]);
    const uint64_t half = 16ull * 2 * head * 2, page_bytes = half * 2 * layers;
    constexpr uint64_t arena_bytes = 21ull << 20; // common multiple of both Qwen page sizes
    auto arena = std::make_shared<base::SharedKVArenaClient>(argv[1], argv[2], arena_bytes, 0);
    auto group = std::make_shared<base::SharedKVPageGroup>(arena, layers, half, 2ull << 20, stream);
    std::vector<std::unique_ptr<base::BlockAllocator>> pools;
    for (int l = 0; l < layers; ++l)
      pools.emplace_back(new base::BlockAllocator(16, 2, head, group, l));
    std::vector<int> ids;
    std::vector<base::BlockLease> pins;
    std::atomic<bool> open{true};
    bool owned = false;
    bool first_layer_released = false;
    std::cout << nlohmann::json{{"ok", true}, {"ready", true}}.dump() << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
      const auto request = nlohmann::json::parse(line);
      const auto command = request.at("command").get<std::string>();
      nlohmann::json result{{"ok", true}};
      if (command == "fill") {
        if (owned || !ids.empty() || !pins.empty()) throw std::logic_error("fill while live");
        for (;;) {
          int id = pools[0]->allocate();
          if (id < 0) break;
          for (int l = 1; l < layers; ++l)
            if (pools[l]->allocate() != id) throw std::logic_error("layer ID mismatch");
          ids.push_back(id);
        }
        owned = !ids.empty();
        std::vector<uint64_t> offsets;
        for (int id : ids) offsets.push_back(static_cast<uint64_t>(id) * half);
        std::sort(offsets.begin(), offsets.end());
        result["page_offsets"] = offsets;
        result["page_bytes"] = page_bytes;
        result["pages"] = ids.size();
        result["owned_bytes"] = ids.size() * page_bytes;
        if (!ids.empty()) {
          if (ids.size() * page_bytes != arena_bytes) throw std::logic_error("unexpected partial fill");
          for (size_t i = 0; i < offsets.size(); ++i)
            if (offsets[i] != i * page_bytes) throw std::logic_error("gap/overlap in physical pages");
          const int pattern = request.at("pattern");
          for (int id : ids) for (int l = 0; l < layers; ++l) {
            auto payload = pools[l]->get_block_payload_ptrs(id);
            check(cudaMemsetAsync(payload.key, pattern + l, payload.key_value_bytes, stream));
            check(cudaMemsetAsync(payload.value, pattern + 64 + l, payload.key_value_bytes, stream));
          }
          check(cudaStreamSynchronize(stream));
          std::vector<unsigned char> host(arena_bytes);
          check(cudaMemcpy(host.data(), arena->data(), host.size(), cudaMemcpyDeviceToHost));
          for (uint64_t offset : offsets) for (int l = 0; l < layers; ++l)
            for (int plane = 0; plane < 2; ++plane) {
              auto begin = host.begin() + offset + (2 * l + plane) * half;
              const auto expected = static_cast<unsigned char>(pattern + plane * 64 + l);
              if (!std::all_of(begin, begin + half, [expected](unsigned char b) { return b == expected; }))
                throw std::logic_error("GPU KV payload mismatch after reuse");
            }
          result["all_gpu_bytes_verified"] = true;
        }
      } else if (command == "release_first_layer") {
        if (!owned || first_layer_released) throw std::logic_error("invalid partial release");
        for (int id : ids) pools[0]->free(id);
        first_layer_released = true;
      } else if (command == "rollback") {
        if (owned || !ids.empty() || !pins.empty()) throw std::logic_error("rollback while live");
        int id = pools[0]->allocate();
        if (id < 0) throw std::logic_error("rollback allocation failed");
        pools[0]->free(id); // other layers were reserved but never consumed
        check(cudaStreamSynchronize(stream));
      } else if (command == "pin_and_release") {
        if (!owned) throw std::logic_error("no request pages");
        if (first_layer_released) throw std::logic_error("partial release before pin");
        for (int id : ids) for (int l = 0; l < layers; ++l) {
          for (auto kind : {base::BlockLeaseKind::kCompute, base::BlockLeaseKind::kIO}) {
            base::BlockLease pin;
            if (!pools[l]->acquire_lease(pools[l]->handle(id), kind, &pin))
              throw std::logic_error("lease acquisition failed");
            pins.push_back(std::move(pin));
          }
          pools[l]->free(id);
        }
        owned = false;
      } else if (command == "unpin_deferred") {
        if (pins.empty()) throw std::logic_error("no pins");
        open.store(false);
        check(cudaLaunchHostFunc(stream, gate, &open));
        pins.clear(); // last references retire behind the blocked stream's fence
        ids.clear();
        if (cudaStreamQuery(stream) != cudaErrorNotReady) throw std::logic_error("stream must be pending");
        result["gpu_fence_pending"] = true;
      } else if (command == "drain" || command == "release") {
        open.store(true);
        if (owned) for (int id : ids) for (int l = first_layer_released ? 1 : 0; l < layers; ++l) pools[l]->free(id);
        first_layer_released = false;
        owned = false;
        pins.clear(); ids.clear();
        check(cudaStreamSynchronize(stream));
      } else if (command == "quit") {
        open.store(true);
        if (owned || !pins.empty()) throw std::logic_error("quit with live pages");
        break;
      } else throw std::invalid_argument("unknown command");
      group->collect();
      result["leased_bytes"] = group->leased_bytes();
      std::cout << result.dump() << std::endl;
    }
    open.store(true);
    pools.clear(); group.reset(); arena.reset();
    check(cudaStreamDestroy(stream));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "REUSE_PROBE_ERROR " << e.what() << std::endl;
    return 1;
  }
}
