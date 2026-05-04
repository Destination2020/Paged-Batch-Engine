// Online HTTP/SSE serving entrypoint for continuous batching.
#ifndef KUIPER_INCLUDE_SERVING_SERVING_ONLINE_SERVER_H_
#define KUIPER_INCLUDE_SERVING_SERVING_ONLINE_SERVER_H_

namespace serving {

class ServingBenchmarkApp;

int run_online_server(ServingBenchmarkApp* app);

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_SERVING_ONLINE_SERVER_H_
