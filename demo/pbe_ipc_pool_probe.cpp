#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <thread>
#include <unistd.h>
#include "data/ipc_pool.h"

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: pbe_ipc_pool_probe export|import DESCRIPTOR DEVICE BYTES_OR_ZERO PATTERN\n";
    return 2;
  }
  const std::string mode = argv[1], file = argv[2];
  const int device = std::stoi(argv[3]);
  const uint8_t pattern = static_cast<uint8_t>(std::stoi(argv[5]));
  std::string error;
  if (mode == "export") {
    const uint64_t bytes = std::stoull(argv[4]);
    data::CudaIpcPoolOwner owner;
    if (!owner.create(device, bytes, pattern, &error)) { std::cerr << error << "\n"; return 3; }
    data::IpcPoolDescriptor descriptor;
    const uint64_t incarnation = (static_cast<uint64_t>(getpid()) << 32) ^
        std::chrono::steady_clock::now().time_since_epoch().count();
    if (!owner.descriptor(incarnation, &descriptor, &error)) { std::cerr << error << "\n"; return 3; }
    std::vector<uint8_t> encoded; data::EncodeIpcPoolDescriptor(descriptor, &encoded);
    const std::string temporary = file + ".tmp." + std::to_string(getpid());
    { std::ofstream out(temporary, std::ios::binary);
      out.write(reinterpret_cast<const char*>(encoded.data()), encoded.size()); }
    if (::rename(temporary.c_str(), file.c_str()) != 0) return 4;
    std::cout << "PBE_IPC_EXPORT pid=" << getpid() << " device=" << device
              << " bytes=" << bytes << "\n" << std::flush;
    bool acknowledged = false;
    for (int i = 0; i < 3000 && !acknowledged; ++i) {
      std::ifstream done(file + ".done"); acknowledged = done.good();
      if (!acknowledged) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!acknowledged) { std::cerr << "consumer acknowledgement timeout\n"; return 5; }
    if (!owner.wait_consumed(&error)) { std::cerr << error << "\n"; return 5; }
    std::cout << "PBE_IPC_CONSUMED pid=" << getpid() << "\n"; return 0;
  }
  if (mode == "import") {
    std::vector<uint8_t> encoded;
    for (int i = 0; i < 200 && encoded.empty(); ++i) {
      std::ifstream in(file, std::ios::binary);
      encoded.assign(std::istreambuf_iterator<char>(in), {});
      if (encoded.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    data::IpcPoolDescriptor descriptor;
    if (!data::DecodeIpcPoolDescriptor(encoded.data(), encoded.size(), &descriptor)) return 4;
    data::CudaIpcPoolImport importer;
    if (!importer.open(descriptor, device, &error)) { std::cerr << error << "\n"; return 5; }
    uint64_t transferred = 0; double elapsed_ms = 0; std::string path;
    if (!importer.validate_and_signal(pattern, &transferred, &elapsed_ms, &path, &error)) {
      std::cerr << error << "\n"; return 6;
    }
    { std::ofstream done(file + ".done"); done << getpid() << "\n"; }
    std::cout << "PBE_IPC_IMPORT pid=" << getpid()
              << " exporter_device=" << descriptor.device << " consumer_device=" << device
              << " pool_bytes=" << descriptor.bytes << " transfer_bytes=" << transferred
              << " elapsed_ms=" << elapsed_ms << " path=" << path << "\n"; return 0;
  }
  return 2;
}
