#include "base/shared_kv_arena.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <deque>
#include <limits>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include "base/zmq_transport.h"

#include "shared_kv_protocol.h"
namespace base {
using namespace shared_kv_detail;
int run_shared_kv_owner(const std::string& endpoint, uint64_t bytes, int device) {
  cuda_check(cudaSetDevice(device));
  void* data = nullptr;
  cuda_check(cudaMalloc(&data, bytes)); // dedicated allocation, never allocator-cached
  try {
    cudaIpcMemHandle_t handle{};
    cuda_check(cudaIpcGetMemHandle(&handle, data));
    const auto epoch = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    GlobalChunkAllocator allocator(bytes, epoch);
    std::unique_ptr<transport::ZmqSocket> socket;
    auto status = transport::make_zmq_rep_socket({endpoint, 1000}, &socket);
    if (!status) throw std::runtime_error(status.get_err_msg());
    const auto uuid = uuid_for_device(device);
    const char* trace_env = std::getenv("PBE_SHARED_KV_TRACE");
    const bool trace = trace_env && std::string(trace_env) == "1";
    for (;;) {
      Json request;
      status = socket->recv_json(&request);
      if (!status) {
        if (status == StatusCode::kInvalidArgument)
          socket->send_json({{"ok", false}, {"error", "invalid_request_json"}});
        continue;
      }
      Json response;
      try {
        const std::string type = request.at("type");
        if (type == "shutdown") {
          if (request.at("epoch").get<uint64_t>() != epoch)
            throw std::invalid_argument("stale_arena_epoch");
          if (allocator.lease_count()) throw std::runtime_error("arena_has_live_leases");
          cuda_check(cudaDeviceSynchronize());
          cuda_check(cudaFree(data));
          data = nullptr;
          socket->set_linger(1000);
          status = socket->send_json({{"ok", true}, {"gpu_synchronized", true}, {"arena_freed", true}});
          if (!status) throw std::runtime_error(status.get_err_msg());
          return 0;
        }
        if (type == "health" || type == "metrics") {
          size_t free_bytes = 0, total_bytes = 0;
          cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
          response = {{"gpu_free_bytes", free_bytes}, {"gpu_total_bytes", total_bytes},
              {"ok", true}, {"bytes", bytes}, {"epoch", epoch}, {"gpu_uuid", uuid},
              {"free_bytes", allocator.free_bytes()}, {"lease_count", allocator.lease_count()},
              {"memory_handle", transport::binary_to_hex_json(std::string(reinterpret_cast<char*>(&handle), sizeof(handle)))}};
        } else {
          if (request.at("epoch").get<uint64_t>() != epoch) throw std::invalid_argument("stale_arena_epoch");
          const std::string model = request.at("model");
          if (type == "register") allocator.register_model(model, request.at("hard_limit"));
          else if (type == "acquire") {
            auto lease = allocator.acquire(model, request.at("layout"), request.at("bytes"), request.at("alignment"));
            response["lease"] = lease_json(lease);
            if (trace) std::cout << Json{{"event", "acquire"}, {"lease", lease_json(lease)}}.dump() << std::endl;
          } else if (type == "release") {
            auto lease = parse_lease(request.at("lease"));
            if (lease.model != model || !request.value("gpu_complete", false))
              throw std::invalid_argument("release_without_owner_identity_or_gpu_fence");
            allocator.release(lease);
            if (trace) std::cout << Json{{"event", "release"}, {"lease", lease_json(lease)}}.dump() << std::endl;
          } else throw std::invalid_argument("unknown_owner_operation");
          response["ok"] = true;
        }
      } catch (const ChunkAllocationError& e) {
        response = {{"ok", false}, {"error", e.what()}, {"allocation_error", static_cast<int>(e.code())}};
      } catch (const std::exception& e) { response = {{"ok", false}, {"error", e.what()}}; }
      status = socket->send_json(response);
      if (!status) throw std::runtime_error(status.get_err_msg());
    }
  } catch (...) { cudaFree(data); throw; }
}
}  // namespace base
