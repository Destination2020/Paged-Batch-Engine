#pragma once
#include "base/shared_kv_arena.h"
#include "base/zmq_transport.h"
#include <nlohmann/json.hpp>
#include <limits>
namespace base::shared_kv_detail {
using Json = nlohmann::json;
inline void cuda_check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
inline std::string uuid_for_device(int device) {
  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, device));
  static const char digits[] = "0123456789abcdef";
  std::string uuid;
  for (unsigned char byte : prop.uuid.bytes) {
    uuid += digits[byte >> 4]; uuid += digits[byte & 15];
  }
  return uuid;
}
inline Json lease_json(const ChunkLease& lease) {
  return {{"epoch", lease.epoch}, {"id", lease.id}, {"generation", lease.generation},
          {"offset", lease.offset}, {"bytes", lease.bytes},
          {"model", lease.model}, {"layout", lease.layout}};
}
inline ChunkLease parse_lease(const Json& j) {
  return {j.at("epoch"), j.at("id"), j.at("generation"), j.at("offset"),
          j.at("bytes"), j.at("model"), j.at("layout")};
}
inline uint64_t round_up(uint64_t bytes, uint64_t alignment) {
  if (bytes > std::numeric_limits<uint64_t>::max() - alignment)
    throw std::invalid_argument("KV size overflow");
  return (bytes + alignment - 1) / alignment * alignment;
}
}
