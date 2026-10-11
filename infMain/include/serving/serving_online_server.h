// Online HTTP/SSE serving entrypoint for continuous batching.
#ifndef KUIPER_INCLUDE_SERVING_SERVING_ONLINE_SERVER_H_
#define KUIPER_INCLUDE_SERVING_SERVING_ONLINE_SERVER_H_

namespace serving {

class ServingRuntime;
struct BenchConfig;

int run_online_server(ServingRuntime* app);
int run_online_server(ServingRuntime* app, const BenchConfig& config);

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_SERVING_ONLINE_SERVER_H_
