#include <iostream>
#include "base/shared_kv_arena.h"
int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "usage: shared_kv_owner ENDPOINT ARENA_MIB [DEVICE]\n";
    return 2;
  }
  try {
    const auto mib = std::stoull(argv[2]);
    if (!mib || mib > (1ull << 20)) throw std::invalid_argument("invalid arena MiB");
    return base::run_shared_kv_owner(argv[1], mib << 20, argc == 4 ? std::stoi(argv[3]) : 0);
  } catch (const std::exception& e) { std::cerr << "KV_OWNER_ERROR " << e.what() << '\n'; return 1; }
}
