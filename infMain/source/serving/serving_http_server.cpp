#include "serving/serving_http_server.h"

#include <glog/logging.h>
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <iostream>

namespace serving {
namespace {

int close_fd(int fd) {
  if (fd >= 0) {
    return ::close(fd);
  }
  return 0;
}

bool set_nonblocking(int fd, bool nonblocking) {
  const int flags = ::fcntl(fd, F_GETFL, 0);
  if (flags < 0) {
    return false;
  }
  const int new_flags = nonblocking ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
  return ::fcntl(fd, F_SETFL, new_flags) == 0;
}

int32_t resolve_worker_threads(int32_t requested) {
  if (requested > 0) {
    return requested;
  }
  const unsigned int hardware = std::thread::hardware_concurrency();
  if (hardware == 0) {
    return 8;
  }
  return std::max<int32_t>(4, std::min<int32_t>(32, static_cast<int32_t>(hardware)));
}

}  // namespace

EpollHttpServer::EpollHttpServer(HttpServerConfig config, ClientHandler handler,
                                   ReadyCallback ready_callback)
    : config_(std::move(config)),
      handler_(std::move(handler)),
      ready_callback_(std::move(ready_callback)) {}

EpollHttpServer::~EpollHttpServer() { stop(); }

int EpollHttpServer::run() {
  CHECK(handler_ != nullptr);
  if (!create_and_bind()) {
    return -1;
  }
  if (!create_epoll()) {
    close_fd(listen_fd_);
    listen_fd_ = -1;
    return -1;
  }
  start_workers();
  if (ready_callback_) {
    ready_callback_();
  }

  constexpr int kMaxEvents = 64;
  epoll_event events[kMaxEvents];
  while (!stopping_.load()) {
    const int n = ::epoll_wait(epoll_fd_, events, kMaxEvents, 1000);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      PLOG(ERROR) << "epoll_wait failed";
      break;
    }
    for (int i = 0; i < n; ++i) {
      if (events[i].data.fd == listen_fd_) {
        accept_ready_clients();
      }
    }
  }
  stop();
  return 0;
}

void EpollHttpServer::stop() {
  const bool was_stopping = stopping_.exchange(true);
  if (!was_stopping) {
    queue_cv_.notify_all();
  }
  close_fd(epoll_fd_);
  epoll_fd_ = -1;
  close_fd(listen_fd_);
  listen_fd_ = -1;
  for (auto& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  workers_.clear();

  std::deque<int> pending;
  {
    std::lock_guard<std::mutex> lock(queue_mu_);
    pending.swap(client_fds_);
  }
  for (int fd : pending) {
    close_fd(fd);
  }
}

void EpollHttpServer::start_workers() {
  const int32_t worker_count = resolve_worker_threads(config_.worker_threads);
  workers_.reserve(worker_count);
  for (int32_t i = 0; i < worker_count; ++i) {
    workers_.emplace_back([this]() { worker_loop(); });
  }
  LOG(INFO) << "HTTP worker thread pool started: workers=" << worker_count;
  std::cout << "HTTP_WORKER_POOL_READY workers=" << worker_count
            << " listen_backlog=" << config_.listen_backlog << std::endl;
}

void EpollHttpServer::worker_loop() {
  while (true) {
    int client_fd = -1;
    {
      std::unique_lock<std::mutex> lock(queue_mu_);
      queue_cv_.wait(lock, [&]() { return stopping_.load() || !client_fds_.empty(); });
      if (client_fds_.empty()) {
        if (stopping_.load()) {
          return;
        }
        continue;
      }
      client_fd = client_fds_.front();
      client_fds_.pop_front();
    }
    handler_(client_fd);
    close_fd(client_fd);
  }
}

void EpollHttpServer::enqueue_client(int client_fd) {
  {
    std::lock_guard<std::mutex> lock(queue_mu_);
    client_fds_.push_back(client_fd);
  }
  queue_cv_.notify_one();
}

bool EpollHttpServer::create_and_bind() {
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
  if (listen_fd_ < 0) {
    PLOG(ERROR) << "socket failed";
    return false;
  }
  int reuse = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(config_.port));
  if (::inet_pton(AF_INET, config_.host.c_str(), &addr.sin_addr) != 1) {
    LOG(ERROR) << "Invalid listen host: " << config_.host;
    close_fd(listen_fd_);
    listen_fd_ = -1;
    return false;
  }
  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    PLOG(ERROR) << "bind failed";
    close_fd(listen_fd_);
    listen_fd_ = -1;
    return false;
  }
  const int backlog = std::max<int32_t>(1, config_.listen_backlog);
  if (::listen(listen_fd_, backlog) != 0) {
    PLOG(ERROR) << "listen failed";
    close_fd(listen_fd_);
    listen_fd_ = -1;
    return false;
  }
  return true;
}

bool EpollHttpServer::create_epoll() {
  epoll_fd_ = ::epoll_create1(EPOLL_CLOEXEC);
  if (epoll_fd_ < 0) {
    PLOG(ERROR) << "epoll_create1 failed";
    return false;
  }
  epoll_event event{};
  event.events = EPOLLIN;
  event.data.fd = listen_fd_;
  if (::epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, listen_fd_, &event) != 0) {
    PLOG(ERROR) << "epoll_ctl add listen fd failed";
    return false;
  }
  return true;
}

void EpollHttpServer::accept_ready_clients() {
  while (!stopping_.load()) {
    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);
    const int client_fd = ::accept4(listen_fd_, reinterpret_cast<sockaddr*>(&client_addr),
                                    &client_len, SOCK_CLOEXEC | SOCK_NONBLOCK);
    if (client_fd < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        return;
      }
      if (errno == EINTR) {
        continue;
      }
      PLOG(ERROR) << "accept4 failed";
      return;
    }
    if (!set_nonblocking(client_fd, false)) {
      PLOG(WARNING) << "failed to set client fd blocking mode";
      close_fd(client_fd);
      continue;
    }
    enqueue_client(client_fd);
  }
}

}  // namespace serving
