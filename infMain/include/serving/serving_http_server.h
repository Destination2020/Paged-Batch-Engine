#ifndef KUIPER_INCLUDE_SERVING_SERVING_HTTP_SERVER_H_
#define KUIPER_INCLUDE_SERVING_SERVING_HTTP_SERVER_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace serving {

struct HttpServerConfig {
  std::string host = "127.0.0.1";
  int32_t port = 8080;
  int32_t listen_backlog = 1024;
  int32_t worker_threads = 0;  // 0 = auto
};

class EpollHttpServer {
 public:
  using ClientHandler = std::function<void(int)>;
  using ReadyCallback = std::function<void()>;

  EpollHttpServer(HttpServerConfig config, ClientHandler handler,
                  ReadyCallback ready_callback = nullptr);
  ~EpollHttpServer();

  EpollHttpServer(const EpollHttpServer&) = delete;
  EpollHttpServer& operator=(const EpollHttpServer&) = delete;

  int run();
  void stop();

 private:
  void start_workers();
  void worker_loop();
  void enqueue_client(int client_fd);
  bool create_and_bind();
  bool create_epoll();
  void accept_ready_clients();

  HttpServerConfig config_;
  ClientHandler handler_;
  ReadyCallback ready_callback_;
  int listen_fd_ = -1;
  int epoll_fd_ = -1;
  std::atomic<bool> stopping_{false};
  std::vector<std::thread> workers_;
  std::mutex queue_mu_;
  std::condition_variable queue_cv_;
  std::deque<int> client_fds_;
};

}  // namespace serving

#endif  // KUIPER_INCLUDE_SERVING_SERVING_HTTP_SERVER_H_
