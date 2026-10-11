#include <gtest/gtest.h>
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>
#include "serving/serving_http_protocol.h"

namespace {
class HttpProtocolTest : public testing::Test {
 protected:
  int fds[2]{-1, -1};
  void SetUp() override {
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);
    serving::configure_http_io(30);
  }
  void TearDown() override { for (int fd : fds) if (fd >= 0) close(fd); }
  int parse(const std::string& wire, std::string* result) {
    EXPECT_EQ(::send(fds[0], wire.data(), wire.size(), MSG_NOSIGNAL), wire.size());
    return serving::read_http_request(fds[1], result);
  }
};
TEST_F(HttpProtocolTest, ReadsOnlyDeclaredBody) {
  std::string request;
  EXPECT_EQ(parse("POST /generate HTTP/1.1\r\ncontent-length: 2\r\n\r\n{}extra", &request), 200);
  EXPECT_EQ(request.substr(request.size() - 2), "{}");
}
TEST_F(HttpProtocolTest, InvalidLengthIsRequestError) {
  std::string request;
  EXPECT_EQ(parse("POST / HTTP/1.1\r\nContent-Length: x\r\n\r\n", &request), 400);
}
TEST_F(HttpProtocolTest, RejectsConflictingFraming) {
  std::string request;
  EXPECT_EQ(parse("POST / HTTP/1.1\r\nContent-Length: 0\r\nContent-Length: 1\r\n\r\n", &request), 400);
}
TEST_F(HttpProtocolTest, RejectsOversizedBodyBeforeReadingIt) {
  std::string request;
  EXPECT_EQ(parse("POST / HTTP/1.1\r\nContent-Length: 8388609\r\n\r\n", &request), 413);
}
TEST_F(HttpProtocolTest, IncompleteHeaderHasDeadline) {
  std::string request;
  const auto start = std::chrono::steady_clock::now();
  EXPECT_EQ(parse("GET / HTTP/1.1\r\n", &request), 408);
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(1));
}
TEST_F(HttpProtocolTest, PeerDisconnectDoesNotRaiseSignal) {
  close(fds[0]); fds[0] = -1;
  EXPECT_FALSE(serving::send_all(fds[1], "response"));
}
}
