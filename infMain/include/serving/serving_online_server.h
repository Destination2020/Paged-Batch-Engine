// Online HTTP/SSE serving entrypoint for continuous batching.
#ifndef KUIPER_INCLUDE_SERVING_SERVING_ONLINE_SERVER_H_
#define KUIPER_INCLUDE_SERVING_SERVING_ONLINE_SERVER_H_

namespace serving {

class ServingBenchmarkApp;
struct BenchConfig;

int run_online_server(ServingBenchmarkApp* app);
int run_online_server(ServingBenchmarkApp* app, const BenchConfig& config);

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_SERVING_ONLINE_SERVER_H_
