#include <gtest/gtest.h>

#include <chrono>
#include <atomic>
#include <cstring>
#include <future>
#include <memory>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "data/node_agent.h"
#include "data/wire_io.h"

namespace {
using namespace std::chrono_literals;

std::string Endpoint(const char* suffix) {
  return "/tmp/pbe-agent-test-" + std::to_string(::getpid()) + "-" + suffix + ".sock";
}

int Connect(const std::string& endpoint) {
  int fd=::socket(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0);
  if(fd<0)return -1;sockaddr_un addr{};addr.sun_family=AF_UNIX;
  std::memcpy(addr.sun_path,endpoint.c_str(),endpoint.size()+1);
  if(::connect(fd,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))!=0){::close(fd);return -1;}
  return fd;
}

std::vector<uint8_t> RequestHeader(data::DataServiceOp op, uint64_t payload_size) {
  std::vector<uint8_t> header;
  data::wire::Put<uint32_t>(data::kDataServiceWireMagic,&header);
  data::wire::Put<uint16_t>(data::kDataServiceProtocolVersion,&header);
  data::wire::Put<uint16_t>(static_cast<uint16_t>(op),&header);
  data::wire::Put<uint16_t>(0,&header);data::wire::Put<uint16_t>(0,&header);
  data::wire::Put<uint64_t>(88,&header);data::wire::Put(payload_size,&header);
  data::wire::Put<uint32_t>(0,&header);
  return header;
}

void DrainUntilClosed(int fd) {
  timeval timeout{0,500000};
  ASSERT_EQ(::setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout)),0);
  uint8_t bytes[64];
  while (::recv(fd,bytes,sizeof(bytes),0)>0) {}
}

void WaitReady(data::DataClient* client) {
  for(int i=0;i<200;++i){uint64_t owner=0;if(client->ping(&owner)==data::DataError::kOk)return;std::this_thread::sleep_for(5ms);}
  FAIL()<<"node agent did not become ready";
}

data::DataReservation Reservation(uint64_t owner,const std::vector<uint8_t>& bytes) {
  data::DataReservation r;r.operation={owner,1};r.kind=data::DataKind::kTensorBundle;
  r.content=data::ContentIdFromBytes(bytes.data(),bytes.size());
  r.representation=data::RepresentationIdFromString("agent-restart-v2");r.logical_bytes=bytes.size();return r;
}

void Publish(data::DataClient* client,const data::DataReservation& r,const std::vector<uint8_t>& bytes) {
  data::AllocationHandle h;ASSERT_EQ(client->reserve(r,&h),data::DataError::kOk);
  data::DataRef ref;ASSERT_EQ(client->seal(h,bytes,data::DataChecksum(bytes.data(),bytes.size()),&ref),data::DataError::kOk);
  ASSERT_EQ(client->release_producer(h),data::DataError::kOk);
}

void SendAcquireAndDrop(const std::string& endpoint,const data::DataReservation& r,
                        data::OperationId operation) {
  std::vector<uint8_t> payload;
  data::wire::Put(operation.owner_incarnation,&payload);data::wire::Put(operation.sequence,&payload);
  payload.push_back(static_cast<uint8_t>(r.kind));data::wire::PutBytes(r.content.digest.data(),32,&payload);
  data::wire::PutBytes(r.representation.digest.data(),32,&payload);payload.push_back(static_cast<uint8_t>(data::DataLeaseKind::kRead));
  std::vector<uint8_t> header;data::wire::Put<uint32_t>(data::kDataServiceWireMagic,&header);data::wire::Put<uint16_t>(data::kDataServiceProtocolVersion,&header);data::wire::Put<uint16_t>(static_cast<uint16_t>(data::DataServiceOp::kAcquire),&header);data::wire::Put<uint16_t>(0,&header);data::wire::Put<uint16_t>(0,&header);data::wire::Put<uint64_t>(77,&header);data::wire::Put<uint64_t>(payload.size(),&header);data::wire::Put<uint32_t>(0,&header);
  int fd=Connect(endpoint);ASSERT_GE(fd,0);ASSERT_EQ(::send(fd,header.data(),header.size(),MSG_NOSIGNAL),static_cast<ssize_t>(header.size()));ASSERT_EQ(::send(fd,payload.data(),payload.size(),MSG_NOSIGNAL),static_cast<ssize_t>(payload.size()));::close(fd);
}
}

TEST(NodeAgentTest, SlowPartialHeadersCannotOccupyWorkersIndefinitely) {
  data::NodeAgentConfig config;config.endpoint=Endpoint("slow");config.incarnation=301;config.request_timeout_ms=100;
  data::NodeAgent agent(config);std::thread server([&]{EXPECT_EQ(agent.run(),0);});
  data::DataClient client(config.endpoint);WaitReady(&client);
  std::vector<int> slow;
  for(int i=0;i<8;++i){int fd=Connect(config.endpoint);ASSERT_GE(fd,0);const uint8_t byte=0x50;ASSERT_EQ(::send(fd,&byte,1,MSG_NOSIGNAL),1);slow.push_back(fd);}
  auto shutdown=std::async(std::launch::async,[&]{return client.shutdown();});
  EXPECT_EQ(shutdown.wait_for(1500ms),std::future_status::ready);
  if(shutdown.wait_for(0ms)!=std::future_status::ready)agent.stop();
  else EXPECT_EQ(shutdown.get(),data::DataError::kOk);
  for(int fd:slow)::close(fd);agent.stop();server.join();
}

TEST(NodeAgentTest, PartialBodyUsesTheSameAbsoluteRequestDeadline) {
  data::NodeAgentConfig config;config.endpoint=Endpoint("partial-body");config.incarnation=302;config.request_timeout_ms=100;
  data::NodeAgent agent(config);std::thread server([&]{EXPECT_EQ(agent.run(),0);});
  data::DataClient client(config.endpoint);WaitReady(&client);
  int fd=Connect(config.endpoint);ASSERT_GE(fd,0);auto header=RequestHeader(data::DataServiceOp::kAcquire,74);
  ASSERT_EQ(::send(fd,header.data(),header.size(),MSG_NOSIGNAL),static_cast<ssize_t>(header.size()));
  const uint8_t byte=1;ASSERT_EQ(::send(fd,&byte,1,MSG_NOSIGNAL),1);
  const auto started=std::chrono::steady_clock::now();DrainUntilClosed(fd);
  EXPECT_LT(std::chrono::steady_clock::now()-started,400ms);
  ::close(fd);EXPECT_EQ(client.shutdown(),data::DataError::kOk);server.join();
}

TEST(NodeAgentTest, HeaderDripCannotRenewTheAbsoluteRequestDeadline) {
  data::NodeAgentConfig config;config.endpoint=Endpoint("header-drip");config.incarnation=303;config.request_timeout_ms=100;
  data::NodeAgent agent(config);std::thread server([&]{EXPECT_EQ(agent.run(),0);});
  data::DataClient client(config.endpoint);WaitReady(&client);
  int fd=Connect(config.endpoint);ASSERT_GE(fd,0);auto header=RequestHeader(data::DataServiceOp::kPing,0);
  std::atomic<bool> done{false};const auto started=std::chrono::steady_clock::now();
  std::thread dripper([&]{for(uint8_t byte:header){if(done||::send(fd,&byte,1,MSG_NOSIGNAL)!=1)break;std::this_thread::sleep_for(20ms);}});
  DrainUntilClosed(fd);done=true;dripper.join();
  EXPECT_LT(std::chrono::steady_clock::now()-started,400ms);
  ::close(fd);EXPECT_EQ(client.shutdown(),data::DataError::kOk);server.join();
}

TEST(NodeAgentTest, StaleWireTokenCannotReleaseLeaseAfterServiceRestart) {
  const auto endpoint=Endpoint("restart");const std::vector<uint8_t> bytes{3,1,4};
  data::LeaseToken stale;
  {
    data::NodeAgentConfig config;config.endpoint=endpoint;config.incarnation=401;
    data::NodeAgent agent(config);std::thread server([&]{EXPECT_EQ(agent.run(),0);});data::DataClient client(endpoint);WaitReady(&client);
    auto r=Reservation(config.incarnation,bytes);Publish(&client,r,bytes);data::RemoteDataLease lease;
    ASSERT_EQ(client.acquire(r.kind,r.content,r.representation,data::DataLeaseKind::kRead,&lease),data::DataError::kOk);stale=lease.token;
    ASSERT_EQ(client.shutdown(),data::DataError::kOk);server.join();
  }
  {
    data::NodeAgentConfig config;config.endpoint=endpoint;config.incarnation=402;
    data::NodeAgent agent(config);std::thread server([&]{EXPECT_EQ(agent.run(),0);});data::DataClient client(endpoint);WaitReady(&client);
    auto r=Reservation(config.incarnation,bytes);Publish(&client,r,bytes);data::RemoteDataLease current;
    ASSERT_EQ(client.acquire(r.kind,r.content,r.representation,data::DataLeaseKind::kRead,&current),data::DataError::kOk);
    EXPECT_EQ(client.release(stale),data::DataError::kOwnerRestarted);data::DataServiceStats stats;ASSERT_EQ(client.stats(&stats),data::DataError::kOk);EXPECT_EQ(stats.active_leases,1u);
    EXPECT_EQ(client.release(current.token),data::DataError::kOk);EXPECT_EQ(client.shutdown(),data::DataError::kOk);server.join();
  }
}

TEST(NodeAgentTest, ReplyLostAcquireCanReplayWithoutAddingALease) {
  data::NodeAgentConfig config;config.endpoint=Endpoint("replay");config.incarnation=501;
  data::NodeAgent agent(config);std::thread server([&]{EXPECT_EQ(agent.run(),0);});data::DataClient client(config.endpoint);WaitReady(&client);
  const std::vector<uint8_t> bytes{2,7,1,8};auto r=Reservation(config.incarnation,bytes);Publish(&client,r,bytes);
  const data::OperationId operation{123456,99};SendAcquireAndDrop(config.endpoint,r,operation);
  data::DataServiceStats stats;
  for(int i=0;i<100;++i){ASSERT_EQ(client.stats(&stats),data::DataError::kOk);if(stats.active_leases==1)break;std::this_thread::sleep_for(2ms);}
  ASSERT_EQ(stats.active_leases,1u);data::RemoteDataLease replay;
  ASSERT_EQ(client.acquire(r.kind,r.content,r.representation,data::DataLeaseKind::kRead,&replay,operation),data::DataError::kOk);
  EXPECT_EQ(replay.token.consumer_incarnation,operation.owner_incarnation);ASSERT_EQ(client.stats(&stats),data::DataError::kOk);EXPECT_EQ(stats.active_leases,1u);
  EXPECT_EQ(client.release(replay.token),data::DataError::kOk);EXPECT_EQ(client.shutdown(),data::DataError::kOk);server.join();
}
