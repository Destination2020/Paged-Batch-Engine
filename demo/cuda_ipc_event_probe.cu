#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <cstring>
#include <vector>
#include <nlohmann/json.hpp>
#include "serving/serving_zmq_rpc.h"
namespace {
void check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
__global__ void delayed_write(unsigned char* data, unsigned long long cycles) {
  const auto begin = clock64();
  while (clock64() - begin < cycles) {}
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) data[i] = static_cast<unsigned char>(i % 251);
}
int exporter(const std::string& endpoint) {
  unsigned char* data{}; cudaStream_t stream{}; cudaEvent_t event{};
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  check(cudaMalloc(&data, 4096));
  check(cudaMemset(data, 0, 4096));
  check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming | cudaEventInterprocess));
  cudaIpcMemHandle_t memory_handle{}; cudaIpcEventHandle_t event_handle{};
  check(cudaIpcGetMemHandle(&memory_handle, data));
  check(cudaIpcGetEventHandle(&event_handle, event));
  std::unique_ptr<serving::ZmqSocket> socket;
  auto status = serving::make_zmq_rep_socket({endpoint, 10000}, &socket);
  if (!status) throw std::runtime_error(status.get_err_msg());
  cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, 0));
  // Publish only after record has been submitted; an unrecorded event is complete.
  delayed_write<<<1, 128, 0, stream>>>(data, static_cast<unsigned long long>(prop.clockRate) * 1500);
  check(cudaGetLastError()); check(cudaEventRecord(event, stream));
  for (;;) {
    nlohmann::json request;
    status = socket->recv_json(&request);
    if (!status) throw std::runtime_error(status.get_err_msg());
    if (request.value("type", "") == "done") {
      status = socket->send_json({{"ok", true}});
      if (!status) throw std::runtime_error(status.get_err_msg());
      break;
    }
    status = socket->send_json({{"ok", true},
        {"memory", serving::binary_to_hex_json(std::string(reinterpret_cast<char*>(&memory_handle), sizeof(memory_handle)))},
        {"event", serving::binary_to_hex_json(std::string(reinterpret_cast<char*>(&event_handle), sizeof(event_handle)))}});
    if (!status) throw std::runtime_error(status.get_err_msg());
  }
  check(cudaStreamSynchronize(stream));
  check(cudaEventDestroy(event)); check(cudaFree(data)); check(cudaStreamDestroy(stream));
  return 0;
}
int importer(const std::string& endpoint) {
  nlohmann::json descriptor;
  base::Status status;
  for (int i = 0; i < 30; ++i) {
    status = serving::zmq_request_response({endpoint, 500}, {{"type", "health"}}, &descriptor);
    if (status) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  if (!status) throw std::runtime_error(status.get_err_msg());
  cudaIpcMemHandle_t memory_handle{}; cudaIpcEventHandle_t event_handle{};
  auto memory = serving::hex_json_to_binary(descriptor.at("memory"));
  auto event_bytes = serving::hex_json_to_binary(descriptor.at("event"));
  if (memory.size() != sizeof(memory_handle) || event_bytes.size() != sizeof(event_handle))
    throw std::runtime_error("invalid IPC handle length");
  std::memcpy(&memory_handle, memory.data(), sizeof(memory_handle));
  std::memcpy(&event_handle, event_bytes.data(), sizeof(event_handle));
  void* data{}; cudaEvent_t event{}; cudaStream_t stream{};
  check(cudaIpcOpenMemHandle(&data, memory_handle, cudaIpcMemLazyEnablePeerAccess));
  check(cudaIpcOpenEventHandle(&event, event_handle));
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const auto before_wait = cudaEventQuery(event);
  if (before_wait != cudaSuccess && before_wait != cudaErrorNotReady) check(before_wait);
  check(cudaStreamWaitEvent(stream, event, 0));
  unsigned char* host{}; check(cudaMallocHost(&host, 4096));
  check(cudaMemcpyAsync(host, data, 4096, cudaMemcpyDeviceToHost, stream));
  check(cudaStreamSynchronize(stream));
  for (int i = 0; i < 4096; ++i)
    if (host[i] != static_cast<unsigned char>(i % 251)) throw std::runtime_error("IPC event ordering/data corruption");
  check(cudaFreeHost(host)); check(cudaStreamDestroy(stream)); check(cudaEventDestroy(event));
  check(cudaIpcCloseMemHandle(data));
  nlohmann::json response;
  status = serving::zmq_request_response({endpoint, 5000}, {{"type", "done"}}, &response);
  if (!status || !response.value("ok", false)) throw std::runtime_error("exporter acknowledgement failed");
  std::cout << nlohmann::json({{"cuda_ipc_memory_event", true}, {"bytes_verified", 4096},
                               {"event_pending_on_import", before_wait == cudaErrorNotReady}}).dump() << '\n';
  return 0;
}
}
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  try { check(cudaSetDevice(0)); return std::string(argv[1]) == "owner" ? exporter(argv[2]) : importer(argv[2]); }
  catch (const std::exception& e) { std::cerr << "IPC_EVENT_PROBE_ERROR " << e.what() << '\n'; return 1; }
}
