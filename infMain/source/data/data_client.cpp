#include "data/data_client.h"

#include <atomic>
#include <climits>
#include <cerrno>
#include <cstring>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <chrono>

#include "data/protocol.h"
#include "data/wire_io.h"

namespace data {
namespace {
constexpr size_t kHeaderBytes = 32;
std::atomic<uint64_t> next_request{1};
bool WriteAll(int fd, const uint8_t* p, size_t n) {
  while (n) { ssize_t k = ::send(fd, p, n, MSG_NOSIGNAL); if (k <= 0) return false;
    p += k; n -= static_cast<size_t>(k); } return true;
}
bool ReadAll(int fd, uint8_t* p, size_t n) {
  while (n) { ssize_t k = ::recv(fd, p, n, 0); if (k <= 0) return false;
    p += k; n -= static_cast<size_t>(k); } return true;
}
void PutIdentity(DataKind kind, const ContentId& c, const RepresentationId& r,
                 std::vector<uint8_t>* out) {
  out->push_back(static_cast<uint8_t>(kind));
  wire::PutBytes(c.digest.data(), c.digest.size(), out);
  wire::PutBytes(r.digest.data(), r.digest.size(), out);
}
bool GetStats(const std::vector<uint8_t>& v, DataServiceStats* s) {
  const uint8_t* p=v.data(); const uint8_t* e=p+v.size();
  return wire::Get(&p,e,&s->owner_incarnation) && wire::Get(&p,e,&s->bytes_used) &&
    wire::Get(&p,e,&s->object_count) && wire::Get(&p,e,&s->active_leases) &&
    wire::Get(&p,e,&s->bytes_received) && wire::Get(&p,e,&s->bytes_sent) &&
    wire::Get(&p,e,&s->requests) && p==e;
}
}

DataClient::DataClient(std::string endpoint) : endpoint_(std::move(endpoint)) {
  consumer_incarnation_ = static_cast<uint64_t>(::getpid()) << 32;
  consumer_incarnation_ ^=
      static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
  if (consumer_incarnation_ == 0) consumer_incarnation_ = 1;
}

DataError DataClient::transact(DataServiceOp op, const std::vector<uint8_t>& request,
                               std::vector<uint8_t>* response) const {
  if (!response || request.size() > kMaxDataServicePayloadBytes) return DataError::kInvalidArgument;
  int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd < 0) return DataError::kNotReady;
  sockaddr_un addr{}; addr.sun_family = AF_UNIX;
  if (endpoint_.size() >= sizeof(addr.sun_path)) { ::close(fd); return DataError::kInvalidArgument; }
  std::memcpy(addr.sun_path, endpoint_.c_str(), endpoint_.size()+1);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd); return DataError::kNotReady;
  }
  const uint64_t request_id = next_request.fetch_add(1);
  std::vector<uint8_t> h; h.reserve(kHeaderBytes);
  wire::Put<uint32_t>(kDataServiceWireMagic,&h); wire::Put<uint16_t>(kDataServiceProtocolVersion,&h);
  wire::Put<uint16_t>(static_cast<uint16_t>(op),&h); wire::Put<uint16_t>(0,&h);
  wire::Put<uint16_t>(0,&h); wire::Put<uint64_t>(request_id,&h);
  wire::Put<uint64_t>(request.size(),&h); wire::Put<uint32_t>(0,&h);
  if (!WriteAll(fd,h.data(),h.size()) || (!request.empty() && !WriteAll(fd,request.data(),request.size()))) {
    ::close(fd); return DataError::kNotReady;
  }
  std::vector<uint8_t> rh(kHeaderBytes);
  if (!ReadAll(fd,rh.data(),rh.size())) { ::close(fd); return DataError::kNotReady; }
  const uint8_t* p=rh.data(); const uint8_t* e=p+rh.size(); uint32_t magic=0,pad=0;
  uint16_t version=0,rop=0,error=0,reserved=0; uint64_t rid=0,size=0;
  bool ok=wire::Get(&p,e,&magic)&&wire::Get(&p,e,&version)&&wire::Get(&p,e,&rop)&&
    wire::Get(&p,e,&error)&&wire::Get(&p,e,&reserved)&&wire::Get(&p,e,&rid)&&
    wire::Get(&p,e,&size)&&wire::Get(&p,e,&pad);
  if (!ok || magic!=kDataServiceWireMagic || version!=kDataServiceProtocolVersion ||
      rop!=static_cast<uint16_t>(op) || rid!=request_id || size>kMaxDataServicePayloadBytes) {
    ::close(fd); return DataError::kUnsupportedVersion;
  }
  response->resize(static_cast<size_t>(size));
  if (size && !ReadAll(fd,response->data(),response->size())) { ::close(fd); return DataError::kNotReady; }
  ::close(fd); return static_cast<DataError>(error);
}

DataError DataClient::ping(uint64_t* owner) const {
  if(!owner) return DataError::kInvalidArgument; std::vector<uint8_t> r;
  auto s=transact(DataServiceOp::kPing,{},&r); if(s!=DataError::kOk)return s;
  const uint8_t*p=r.data(),*e=p+r.size(); return wire::Get(&p,e,owner)&&p==e?s:DataError::kInvalidArgument;
}
DataError DataClient::reserve(const DataReservation& x, AllocationHandle* h) const {
  if(!h)return DataError::kInvalidArgument; std::vector<uint8_t> q,r;
  wire::Put<uint64_t>(x.operation.owner_incarnation,&q); wire::Put<uint64_t>(x.operation.sequence,&q);
  PutIdentity(x.kind,x.content,x.representation,&q); wire::Put<uint64_t>(x.logical_bytes,&q);
  auto s=transact(DataServiceOp::kReserve,q,&r); if(s!=DataError::kOk)return s;
  const uint8_t*p=r.data(),*e=p+r.size(); return wire::GetHandle(&p,e,h)&&p==e?s:DataError::kInvalidArgument;
}
DataError DataClient::seal(const AllocationHandle& h,const std::vector<uint8_t>& bytes,
                           const Digest256& sum,DataRef* ref) const {
  if(!ref)return DataError::kInvalidArgument; std::vector<uint8_t> q,r,encoded;
  wire::PutHandle(h,&q); wire::PutBytes(sum.data(),sum.size(),&q); wire::Put<uint64_t>(bytes.size(),&q);
  wire::PutBytes(bytes.data(),bytes.size(),&q); auto s=transact(DataServiceOp::kSeal,q,&r);
  if(s!=DataError::kOk)return s; return DecodeDataRef(r.data(),r.size(),ref);
}
DataError DataClient::acquire(DataKind k,const ContentId& c,const RepresentationId& rep,
                              DataLeaseKind lk,RemoteDataLease* lease,
                              OperationId operation) const {
  if(!lease)return DataError::kInvalidArgument;
  if(operation.owner_incarnation==0) operation.owner_incarnation=consumer_incarnation_;
  if(operation.sequence==0) operation.sequence=next_operation_.fetch_add(1);
  std::vector<uint8_t> q,r;
  wire::Put(operation.owner_incarnation,&q); wire::Put(operation.sequence,&q);
  PutIdentity(k,c,rep,&q); q.push_back(static_cast<uint8_t>(lk));
  auto s=transact(DataServiceOp::kAcquire,q,&r);
  if(s==DataError::kNotReady) s=transact(DataServiceOp::kAcquire,q,&r);
  if(s!=DataError::kOk)return s; const uint8_t*p=r.data(),*e=p+r.size(); uint64_t ref_size=0,data_size=0;
  if(!wire::Get(&p,e,&lease->token.service_incarnation)||
     !wire::Get(&p,e,&lease->token.consumer_incarnation)||
     !wire::Get(&p,e,&lease->token.lease_id)||
     !wire::Get(&p,e,&ref_size)||ref_size!=kDataRefWireBytes||
     static_cast<size_t>(e-p)<ref_size)return DataError::kInvalidArgument;
  if(DecodeDataRef(p,ref_size,&lease->ref)!=DataError::kOk)return DataError::kInvalidArgument; p+=ref_size;
  if(!wire::Get(&p,e,&data_size)||data_size>static_cast<uint64_t>(e-p)||data_size!=lease->ref.logical_bytes)
    return DataError::kInvalidArgument; lease->bytes.assign(p,p+data_size); p+=data_size;
  return p==e?DataError::kOk:DataError::kInvalidArgument;
}
DataError DataClient::release(const LeaseToken& token) const {std::vector<uint8_t>q,r;wire::Put(token.service_incarnation,&q);wire::Put(token.consumer_incarnation,&q);wire::Put(token.lease_id,&q);return transact(DataServiceOp::kRelease,q,&r);}
DataError DataClient::withdraw(const DataRef& ref) const {std::vector<uint8_t>q,r;if(EncodeDataRef(ref,&q)!=DataError::kOk)return DataError::kInvalidArgument;return transact(DataServiceOp::kWithdraw,q,&r);}
DataError DataClient::release_producer(const AllocationHandle& h) const {std::vector<uint8_t>q,r;wire::PutHandle(h,&q);return transact(DataServiceOp::kReleaseProducer,q,&r);}
DataError DataClient::stats(DataServiceStats* s) const {if(!s)return DataError::kInvalidArgument;std::vector<uint8_t>r;auto x=transact(DataServiceOp::kStats,{},&r);return x==DataError::kOk&&GetStats(r,s)?x:(x==DataError::kOk?DataError::kInvalidArgument:x);}
DataError DataClient::shutdown() const {std::vector<uint8_t>r;return transact(DataServiceOp::kShutdown,{},&r);}
DataError DataClient::get_ipc_pool(IpcPoolDescriptor* descriptor) const {
  if(!descriptor)return DataError::kInvalidArgument;std::vector<uint8_t>r;
  auto status=transact(DataServiceOp::kGetIpcPool,{},&r);
  if(status!=DataError::kOk)return status;
  return DecodeIpcPoolDescriptor(r.data(),r.size(),descriptor)
      ? DataError::kOk:DataError::kInvalidArgument;
}
DataError DataClient::reserve_ipc_slots(uint32_t count,IpcSlotGrant* grant,
                                        OperationId operation) const {
  if(!grant||!grant->slots.empty()||count==0)return DataError::kInvalidArgument;
  if(operation.owner_incarnation==0)operation.owner_incarnation=consumer_incarnation_;
  if(operation.sequence==0)operation.sequence=next_operation_.fetch_add(1);
  std::vector<uint8_t>q,r;wire::Put(operation.owner_incarnation,&q);
  wire::Put(operation.sequence,&q);wire::Put(count,&q);
  auto status=transact(DataServiceOp::kReserveIpcSlots,q,&r);
  if(status==DataError::kNotReady)status=transact(DataServiceOp::kReserveIpcSlots,q,&r);
  if(status!=DataError::kOk)return status;const uint8_t*p=r.data(),*e=p+r.size();uint32_t n=0;
  if(!wire::Get(&p,e,&grant->token.service_incarnation)||
     !wire::Get(&p,e,&grant->token.consumer_incarnation)||
     !wire::Get(&p,e,&grant->token.lease_id)||!wire::Get(&p,e,&n)||n!=count)return DataError::kInvalidArgument;
  grant->slots.resize(n);for(auto&slot:grant->slots){uint32_t value=0;if(!wire::Get(&p,e,&value)||value>INT32_MAX)return DataError::kInvalidArgument;slot=value;}
  return p==e?DataError::kOk:DataError::kInvalidArgument;
}
DataError DataClient::release_ipc_slots(const LeaseToken& token) const {std::vector<uint8_t>q,r;wire::Put(token.service_incarnation,&q);wire::Put(token.consumer_incarnation,&q);wire::Put(token.lease_id,&q);return transact(DataServiceOp::kReleaseIpcSlots,q,&r);}
DataError DataClient::ipc_pool_stats(IpcPoolStats*stats)const{if(!stats)return DataError::kInvalidArgument;std::vector<uint8_t>r;auto status=transact(DataServiceOp::kIpcPoolStats,{},&r);if(status!=DataError::kOk)return status;const uint8_t*p=r.data(),*e=p+r.size();return wire::Get(&p,e,&stats->total_slots)&&wire::Get(&p,e,&stats->free_slots)&&wire::Get(&p,e,&stats->active_grants)&&p==e?DataError::kOk:DataError::kInvalidArgument;}

DataError DataClient::acquire_shared_weight(const Digest256& model_content,
                                            const Digest256& layout_identity,
                                            base::DataType dtype,
                                            uint64_t expected_bytes,
                                            SharedWeightLease* lease,
                                            OperationId operation) const {
  if (!lease || dtype == base::DataType::kDataTypeUnknown || expected_bytes == 0)
    return DataError::kInvalidArgument;
  if (operation.owner_incarnation == 0) operation.owner_incarnation = consumer_incarnation_;
  if (operation.sequence == 0) operation.sequence = next_operation_.fetch_add(1);
  std::vector<uint8_t> request, response;
  wire::Put(operation.owner_incarnation, &request);
  wire::Put(operation.sequence, &request);
  wire::PutBytes(model_content.data(), model_content.size(), &request);
  wire::PutBytes(layout_identity.data(), layout_identity.size(), &request);
  request.push_back(static_cast<uint8_t>(dtype));
  wire::Put(expected_bytes, &request);
  auto status = transact(DataServiceOp::kAcquireSharedWeight, request, &response);
  if (status == DataError::kNotReady)
    status = transact(DataServiceOp::kAcquireSharedWeight, request, &response);
  if (status != DataError::kOk) return status;
  const uint8_t* cursor = response.data();
  const uint8_t* end = cursor + response.size();
  uint64_t descriptor_bytes = 0;
  if (!wire::Get(&cursor, end, &lease->token.service_incarnation) ||
      !wire::Get(&cursor, end, &lease->token.consumer_incarnation) ||
      !wire::Get(&cursor, end, &lease->token.lease_id) ||
      !wire::Get(&cursor, end, &descriptor_bytes) ||
      descriptor_bytes > static_cast<uint64_t>(end - cursor) ||
      !DecodeSharedWeightDescriptor(cursor, descriptor_bytes, &lease->descriptor)) {
    return DataError::kInvalidArgument;
  }
  cursor += descriptor_bytes;
  return cursor == end ? DataError::kOk : DataError::kInvalidArgument;
}

DataError DataClient::release_shared_weight(const LeaseToken& token) const {
  std::vector<uint8_t> request, response;
  wire::Put(token.service_incarnation, &request);
  wire::Put(token.consumer_incarnation, &request);
  wire::Put(token.lease_id, &request);
  return transact(DataServiceOp::kReleaseSharedWeight, request, &response);
}

DataError DataClient::shared_weight_stats(SharedWeightStats* stats) const {
  if (!stats) return DataError::kInvalidArgument;
  std::vector<uint8_t> response;
  auto status = transact(DataServiceOp::kSharedWeightStats, {}, &response);
  if (status != DataError::kOk) return status;
  const uint8_t* cursor = response.data();
  const uint8_t* end = cursor + response.size();
  uint8_t state = 0;
  uint64_t upload_us = 0;
  if (!wire::Get(&cursor, end, &state) ||
      !wire::Get(&cursor, end, &stats->owner_incarnation) ||
      !wire::Get(&cursor, end, &stats->allocation_id) ||
      !wire::Get(&cursor, end, &stats->generation) ||
      !wire::Get(&cursor, end, &stats->physical_bytes) ||
      !wire::Get(&cursor, end, &stats->upload_bytes) ||
      !wire::Get(&cursor, end, &stats->upload_count) ||
      !wire::Get(&cursor, end, &stats->active_leases) ||
      !wire::Get(&cursor, end, &stats->allocation_count) ||
      !wire::Get(&cursor, end, &stats->staging_peak_bytes) ||
      !wire::Get(&cursor, end, &upload_us) || cursor != end ||
      state > static_cast<uint8_t>(SharedWeightState::kDraining)) {
    return DataError::kInvalidArgument;
  }
  stats->state = static_cast<SharedWeightState>(state);
  stats->upload_ms = static_cast<double>(upload_us) / 1000.0;
  return DataError::kOk;
}

Digest256 DataChecksum(const uint8_t* bytes,size_t size) {
  Digest256 d{}; uint64_t h[4]={1469598103934665603ULL,1099511628211ULL,0x9e3779b97f4a7c15ULL,0xd6e8feb86659fd93ULL};
  for(size_t i=0;i<size;++i)for(int j=0;j<4;++j){h[j]^=static_cast<uint64_t>(bytes[i]+j*41);h[j]*=1099511628211ULL+(j*2);h[j]^=h[j]>>29;}
  std::memcpy(d.data(),h,sizeof(h)); return d;
}
ContentId ContentIdFromBytes(const uint8_t*b,size_t n){ContentId c;c.digest=DataChecksum(b,n);return c;}
RepresentationId RepresentationIdFromString(const std::string&s){RepresentationId r;r.digest=DataChecksum(reinterpret_cast<const uint8_t*>(s.data()),s.size());return r;}
}  // namespace data
