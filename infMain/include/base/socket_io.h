#pragma once

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <poll.h>
#include <sys/socket.h>

namespace base::socket_io {
using Clock = std::chrono::steady_clock;
using Deadline = Clock::time_point;

// A transaction uses one absolute deadline, including partial reads/writes.
inline bool wait(int fd, short events, Deadline deadline) {
  for (;;) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - Clock::now()).count();
    if (remaining <= 0) { errno = ETIMEDOUT; return false; }
    pollfd descriptor{fd, events, 0};
    const int result = ::poll(&descriptor, 1, static_cast<int>(std::min<int64_t>(remaining, INT_MAX)));
    if (result < 0 && errno == EINTR) continue;
    if (result == 0) errno = ETIMEDOUT;
    return result > 0 && (descriptor.revents & (events | POLLHUP | POLLERR));
  }
}

inline bool transfer(int fd, void* data, size_t bytes, bool sending, Deadline deadline) {
  auto* cursor = static_cast<char*>(data);
  while (bytes) {
    if (!wait(fd, sending ? POLLOUT : POLLIN, deadline)) return false;
    const ssize_t n = sending ? ::send(fd, cursor, bytes, MSG_NOSIGNAL | MSG_DONTWAIT)
                              : ::recv(fd, cursor, bytes, MSG_DONTWAIT);
    if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
    if (n <= 0) return false;
    cursor += n;
    bytes -= static_cast<size_t>(n);
  }
  return true;
}
inline bool send(int fd, const void* data, size_t bytes, Deadline deadline) {
  return transfer(fd, const_cast<void*>(data), bytes, true, deadline);
}
inline bool receive(int fd, void* data, size_t bytes, Deadline deadline) {
  return transfer(fd, data, bytes, false, deadline);
}
}  // namespace base::socket_io
