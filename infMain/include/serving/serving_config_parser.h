#pragma once
#include <string_view>
#include "serving/serving_config.h"
namespace serving {
BenchConfig parse_bench_config(int argc, char* argv[], int prompt_start_index);
inline bool is_cli_flag(std::string_view value) { return value.substr(0, 2) == "--"; }
}
