#ifndef KUIPER_INCLUDE_SERVING_SERVING_ZMQ_ENGINE_CORE_H_
#define KUIPER_INCLUDE_SERVING_SERVING_ZMQ_ENGINE_CORE_H_

#include "base/base.h"
#include "serving/serving_config.h"

namespace serving {

class ServingBenchmarkApp;

base::Status run_zmq_engine_core_server(ServingBenchmarkApp* app,
                                        const BenchConfig& config);

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_SERVING_ZMQ_ENGINE_CORE_H_
