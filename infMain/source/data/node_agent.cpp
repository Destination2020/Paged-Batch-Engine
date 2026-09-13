#include "data/node_agent.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include "data/protocol.h"
#include "data/wire_io.h"

namespace data {
namespace {
constexpr size_t kHeaderBytes = 32;
using Deadline = std::chrono::steady_clock::time_point;
bool WaitFor(int fd, short events, Deadline deadline) {
  for (;;) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) return false;
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline-now);
    pollfd descriptor{fd,events,0};
    const int rc=::poll(&descriptor,1,static_cast<int>(std::max<int64_t>(1,remaining.count())));
    if(rc>0)return (descriptor.revents&events)!=0;
    if(rc==0)return false;
    if(errno!=EINTR)return false;
  }
}
bool ReadAll(int fd,uint8_t*p,size_t n,Deadline deadline){while(n){if(!WaitFor(fd,POLLIN,deadline))return false;ssize_t k=::recv(fd,p,n,MSG_DONTWAIT);if(k<0&&errno==EINTR)continue;if(k<=0)return false;p+=k;n-=static_cast<size_t>(k);}return true;}
bool WriteAll(int fd,const uint8_t*p,size_t n,Deadline deadline){while(n){if(!WaitFor(fd,POLLOUT,deadline))return false;ssize_t k=::send(fd,p,n,MSG_NOSIGNAL|MSG_DONTWAIT);if(k<0&&errno==EINTR)continue;if(k<=0)return false;p+=k;n-=static_cast<size_t>(k);}return true;}
bool ValidKind(uint8_t v){return v>=static_cast<uint8_t>(DataKind::kKVPage)&&v<=static_cast<uint8_t>(DataKind::kCheckpoint);}
bool GetIdentity(const uint8_t**p,const uint8_t*e,DataKind*k,ContentId*c,RepresentationId*r){uint8_t v=0;if(!wire::Get(p,e,&v)||!ValidKind(v))return false;*k=static_cast<DataKind>(v);return wire::GetBytes(p,e,c->digest.data(),32)&&wire::GetBytes(p,e,r->digest.data(),32);}
void PutStats(const DataServiceStats&s,std::vector<uint8_t>*o){wire::Put(s.owner_incarnation,o);wire::Put(s.bytes_used,o);wire::Put(s.object_count,o);wire::Put(s.active_leases,o);wire::Put(s.bytes_received,o);wire::Put(s.bytes_sent,o);wire::Put(s.requests,o);}
}

NodeAgent::NodeAgent(NodeAgentConfig c):config_(std::move(c)){
  if(config_.incarnation==0){config_.incarnation=static_cast<uint64_t>(::getpid())<<32;
    config_.incarnation^=static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    if(config_.incarnation==0)config_.incarnation=1;}
  registry_=std::make_unique<ContentRegistry>(config_.incarnation,config_.capacity_bytes,
                                              config_.capacity_objects,config_.capacity_leases);
}
NodeAgent::~NodeAgent(){stop();for(auto&x:workers_)if(x.joinable())x.join();::unlink(config_.endpoint.c_str());}
void NodeAgent::stop(){
  stopping_=true;
  std::lock_guard<std::mutex> lock(queue_mutex_);
  if(listen_fd_>=0)::shutdown(listen_fd_,SHUT_RDWR);
  for(int fd:active_connections_)::shutdown(fd,SHUT_RDWR);
  for(int fd:connection_queue_){active_connections_.erase(fd);::close(fd);}
  connection_queue_.clear();
  queue_ready_.notify_all();
}

int NodeAgent::run(){
  sockaddr_un addr{};
  if(config_.endpoint.empty()||config_.endpoint.size()>=sizeof(addr.sun_path))return 2;
  if(config_.ipc_pool_bytes){
    if(config_.ipc_pool_device<0)return 6;
    ipc_pool_=std::make_unique<CudaIpcPoolOwner>();std::string error;
    if(!ipc_pool_->create(config_.ipc_pool_device,config_.ipc_pool_bytes,0,&error)){
      std::cerr<<"failed to create Agent-owned IPC pool: "<<error<<std::endl;return 6;
    }
    if(!ipc_pool_->set_kv_layout(config_.ipc_pool_num_layers,config_.ipc_pool_num_blocks,
        config_.ipc_pool_block_size,config_.ipc_pool_num_kv_heads,
        config_.ipc_pool_head_size,config_.ipc_pool_storage_dtype,&error)){
      std::cerr<<"invalid Agent-owned KV layout: "<<error<<std::endl;return 6;
    }
    ipc_slot_owners_.resize(config_.ipc_pool_num_blocks,0);
  }
  if (!config_.shared_weight_path.empty()) {
    if (config_.ipc_pool_device < 0) return 7;
    shared_weight_ = std::make_unique<SharedWeightOwner>();
    std::string error;
    if (!shared_weight_->load_file(config_.shared_weight_path,
                                   config_.ipc_pool_device,
                                   config_.incarnation,
                                   config_.shared_weight_content,
                                   config_.shared_weight_layout,
                                   config_.shared_weight_dtype,
                                   config_.shared_weight_staging_bytes,
                                   &error)) {
      std::cerr << "failed to load Agent-owned shared weights: " << error << std::endl;
      return 7;
    }
  }
  ::unlink(config_.endpoint.c_str()); listen_fd_=::socket(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0);
  if(listen_fd_<0)return 3; addr.sun_family=AF_UNIX;
  std::memcpy(addr.sun_path,config_.endpoint.c_str(),config_.endpoint.size()+1);
  if(::bind(listen_fd_,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))!=0||::listen(listen_fd_,128)!=0)return 4;
  ::chmod(config_.endpoint.c_str(),0600);
  std::cout<<"PBE_DATA_SERVICE_READY pid="<<::getpid()<<" endpoint="<<config_.endpoint
           <<" incarnation="<<config_.incarnation<<" protocol="<<kDataServiceProtocolVersion
           <<" transport=bounded_host_copy capacity_bytes="<<config_.capacity_bytes
           <<" ipc_pool_bytes="<<config_.ipc_pool_bytes
           <<" ipc_pool_device="<<config_.ipc_pool_device
           <<" shared_weight_bytes="
           <<(shared_weight_?shared_weight_->stats().physical_bytes:0)
           <<" shared_weight_uploads="
           <<(shared_weight_?shared_weight_->stats().upload_count:0)<<std::endl;
  for(int i=0;i<8;++i)workers_.emplace_back([this]{for(;;){int fd=-1;{std::unique_lock<std::mutex> lock(queue_mutex_);queue_ready_.wait(lock,[this]{return stopping_||!connection_queue_.empty();});if(connection_queue_.empty()){if(stopping_)return;continue;}fd=connection_queue_.front();connection_queue_.pop_front();}serve_connection(fd);}});
  while(!stopping_){int fd=::accept4(listen_fd_,nullptr,nullptr,SOCK_CLOEXEC);if(fd<0){if(stopping_)break;if(errno==EINTR)continue;return 5;}
    {std::lock_guard<std::mutex> lock(queue_mutex_);if(stopping_||connection_queue_.size()>=256){::close(fd);continue;}active_connections_.insert(fd);connection_queue_.push_back(fd);}queue_ready_.notify_one();}
  if(listen_fd_>=0){::close(listen_fd_);listen_fd_=-1;}
  queue_ready_.notify_all();
  for(auto&x:workers_)if(x.joinable())x.join();workers_.clear();return 0;
}

void NodeAgent::serve_connection(int fd){
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(config_.request_timeout_ms);
  std::vector<uint8_t> h(kHeaderBytes),q,out; DataError status=DataError::kInvalidArgument;
  uint16_t opv=0;uint64_t rid=0,payload_size=0;bool parsed=false,shutdown_after_reply=false;
  if(ReadAll(fd,h.data(),h.size(),deadline)){
    const uint8_t*p=h.data(),*e=p+h.size();uint32_t magic=0,pad=0;uint16_t version=0,error=0,res=0;
    parsed=wire::Get(&p,e,&magic)&&wire::Get(&p,e,&version)&&wire::Get(&p,e,&opv)&&wire::Get(&p,e,&error)&&wire::Get(&p,e,&res)&&wire::Get(&p,e,&rid)&&wire::Get(&p,e,&payload_size)&&wire::Get(&p,e,&pad)&&magic==kDataServiceWireMagic&&version==kDataServiceProtocolVersion&&payload_size<=kMaxDataServicePayloadBytes;
    if(parsed){q.resize(payload_size);parsed=!payload_size||ReadAll(fd,q.data(),q.size(),deadline);}
  }
  if(parsed){
    registry_->add_request();registry_->add_transport_bytes(kHeaderBytes+q.size(),0);
    const uint8_t*p=q.data(),*e=p+q.size();const auto op=static_cast<DataServiceOp>(opv);
    if(op==DataServiceOp::kPing&&p==e){auto s=registry_->stats();wire::Put(s.owner_incarnation,&out);status=DataError::kOk;}
    else if(op==DataServiceOp::kReserve){DataReservation r;uint8_t kind=0;
      if(wire::Get(&p,e,&r.operation.owner_incarnation)&&wire::Get(&p,e,&r.operation.sequence)&&wire::Get(&p,e,&kind)&&ValidKind(kind)&&wire::GetBytes(&p,e,r.content.digest.data(),32)&&wire::GetBytes(&p,e,r.representation.digest.data(),32)&&wire::Get(&p,e,&r.logical_bytes)&&p==e){r.kind=static_cast<DataKind>(kind);AllocationHandle x;status=registry_->reserve(r,&x);if(status==DataError::kOk)wire::PutHandle(x,&out);}}
    else if(op==DataServiceOp::kSeal){AllocationHandle x;Digest256 sum{};uint64_t n=0;
      if(wire::GetHandle(&p,e,&x)&&wire::GetBytes(&p,e,sum.data(),32)&&wire::Get(&p,e,&n)&&n==static_cast<uint64_t>(e-p)){std::vector<uint8_t>b(p,e);DataRef ref;status=registry_->seal(x,b,sum,&ref);if(status==DataError::kOk)status=EncodeDataRef(ref,&out);}}
    else if(op==DataServiceOp::kAcquire){DataKind k;ContentId c;RepresentationId r;uint8_t lk=0;OperationId operation;
      if(wire::Get(&p,e,&operation.owner_incarnation)&&wire::Get(&p,e,&operation.sequence)&&GetIdentity(&p,e,&k,&c,&r)&&wire::Get(&p,e,&lk)&&p==e&&lk<=static_cast<uint8_t>(DataLeaseKind::kIO)){RemoteDataLease lease;status=registry_->acquire(k,c,r,static_cast<DataLeaseKind>(lk),operation,&lease);if(status==DataError::kOk){std::vector<uint8_t>ref;EncodeDataRef(lease.ref,&ref);wire::Put(lease.token.service_incarnation,&out);wire::Put(lease.token.consumer_incarnation,&out);wire::Put(lease.token.lease_id,&out);wire::Put<uint64_t>(ref.size(),&out);wire::PutBytes(ref.data(),ref.size(),&out);wire::Put<uint64_t>(lease.bytes.size(),&out);wire::PutBytes(lease.bytes.data(),lease.bytes.size(),&out);}}}
    else if(op==DataServiceOp::kRelease){LeaseToken token;if(wire::Get(&p,e,&token.service_incarnation)&&wire::Get(&p,e,&token.consumer_incarnation)&&wire::Get(&p,e,&token.lease_id)&&p==e)status=registry_->release(token);}
    else if(op==DataServiceOp::kWithdraw){DataRef ref;if(DecodeDataRef(q.data(),q.size(),&ref)==DataError::kOk)status=registry_->withdraw(ref);}
    else if(op==DataServiceOp::kReleaseProducer){AllocationHandle x;if(wire::GetHandle(&p,e,&x)&&p==e)status=registry_->release_producer(x);}
    else if(op==DataServiceOp::kStats&&p==e){PutStats(registry_->stats(),&out);status=DataError::kOk;}
    else if(op==DataServiceOp::kShutdown&&p==e){status=DataError::kOk;shutdown_after_reply=true;}
    else if(op==DataServiceOp::kGetIpcPool&&p==e&&ipc_pool_){IpcPoolDescriptor descriptor;std::string error;if(ipc_pool_->descriptor(config_.incarnation,&descriptor,&error)&&EncodeIpcPoolDescriptor(descriptor,&out))status=DataError::kOk;}
    else if(op==DataServiceOp::kReserveIpcSlots&&ipc_pool_){OperationId operation;uint32_t count=0;if(wire::Get(&p,e,&operation.owner_incarnation)&&wire::Get(&p,e,&operation.sequence)&&wire::Get(&p,e,&count)&&p==e&&operation.owner_incarnation&&operation.sequence&&count){std::lock_guard<std::mutex> lock(ipc_pool_mutex_);auto replay=ipc_grant_operations_.find(operation);uint64_t id=0;if(replay!=ipc_grant_operations_.end()){id=replay->second;if(ipc_grants_[id].slots.size()!=count)status=DataError::kInvalidArgument;}else{size_t free=std::count(ipc_slot_owners_.begin(),ipc_slot_owners_.end(),0);if(free<count)status=DataError::kCapacityExhausted;else{id=next_ipc_grant_id_++;IpcGrantEntry entry;entry.operation=operation;for(size_t slot=0;slot<ipc_slot_owners_.size()&&entry.slots.size()<count;++slot)if(ipc_slot_owners_[slot]==0){ipc_slot_owners_[slot]=id;entry.slots.push_back(slot);}ipc_grants_[id]=entry;ipc_grant_operations_[operation]=id;status=DataError::kOk;}}if(status==DataError::kOk){const auto&entry=ipc_grants_[id];wire::Put(config_.incarnation,&out);wire::Put(operation.owner_incarnation,&out);wire::Put(id,&out);wire::Put<uint32_t>(entry.slots.size(),&out);for(int32_t slot:entry.slots)wire::Put<uint32_t>(slot,&out);}}}
    else if(op==DataServiceOp::kReleaseIpcSlots&&ipc_pool_){LeaseToken token;if(wire::Get(&p,e,&token.service_incarnation)&&wire::Get(&p,e,&token.consumer_incarnation)&&wire::Get(&p,e,&token.lease_id)&&p==e){std::lock_guard<std::mutex> lock(ipc_pool_mutex_);if(token.service_incarnation!=config_.incarnation)status=DataError::kOwnerRestarted;else{auto grant=ipc_grants_.find(token.lease_id);if(grant==ipc_grants_.end())status=DataError::kOk;else if(grant->second.operation.owner_incarnation!=token.consumer_incarnation)status=DataError::kInvalidArgument;else{for(int32_t slot:grant->second.slots)if(ipc_slot_owners_[slot]==token.lease_id)ipc_slot_owners_[slot]=0;ipc_grant_operations_.erase(grant->second.operation);ipc_grants_.erase(grant);status=DataError::kOk;}}}}
    else if(op==DataServiceOp::kIpcPoolStats&&p==e&&ipc_pool_){std::lock_guard<std::mutex> lock(ipc_pool_mutex_);wire::Put<uint64_t>(ipc_slot_owners_.size(),&out);wire::Put<uint64_t>(std::count(ipc_slot_owners_.begin(),ipc_slot_owners_.end(),0),&out);wire::Put<uint64_t>(ipc_grants_.size(),&out);status=DataError::kOk;}
    else if (op == DataServiceOp::kAcquireSharedWeight && shared_weight_) {
      OperationId operation;
      Digest256 content{}, layout{};
      uint8_t dtype = 0;
      uint64_t expected_bytes = 0;
      if (wire::Get(&p, e, &operation.owner_incarnation) &&
          wire::Get(&p, e, &operation.sequence) &&
          wire::GetBytes(&p, e, content.data(), content.size()) &&
          wire::GetBytes(&p, e, layout.data(), layout.size()) &&
          wire::Get(&p, e, &dtype) && wire::Get(&p, e, &expected_bytes) && p == e &&
          operation.owner_incarnation != 0 && operation.sequence != 0) {
        SharedWeightDescriptor descriptor;
        std::string error;
        if (!shared_weight_->descriptor(&descriptor, &error)) {
          status = DataError::kNotReady;
        } else if (descriptor.model_content != content ||
                   descriptor.layout_identity != layout ||
                   descriptor.dtype != static_cast<base::DataType>(dtype) ||
                   descriptor.bytes != expected_bytes) {
          status = DataError::kCoverageMismatch;
        } else {
          std::lock_guard<std::mutex> lock(shared_weight_mutex_);
          uint64_t lease_id = 0;
          const auto replay = shared_weight_operations_.find(operation);
          if (replay != shared_weight_operations_.end()) {
            lease_id = replay->second;
          } else {
            lease_id = next_shared_weight_lease_id_++;
            shared_weight_leases_[lease_id] = {operation};
            shared_weight_operations_[operation] = lease_id;
          }
          shared_weight_->set_active_leases(shared_weight_leases_.size());
          std::vector<uint8_t> encoded;
          if (EncodeSharedWeightDescriptor(descriptor, &encoded)) {
            wire::Put(config_.incarnation, &out);
            wire::Put(operation.owner_incarnation, &out);
            wire::Put(lease_id, &out);
            wire::Put<uint64_t>(encoded.size(), &out);
            wire::PutBytes(encoded.data(), encoded.size(), &out);
            status = DataError::kOk;
          }
        }
      }
    }
    else if (op == DataServiceOp::kReleaseSharedWeight && shared_weight_) {
      LeaseToken token;
      if (wire::Get(&p, e, &token.service_incarnation) &&
          wire::Get(&p, e, &token.consumer_incarnation) &&
          wire::Get(&p, e, &token.lease_id) && p == e) {
        std::lock_guard<std::mutex> lock(shared_weight_mutex_);
        if (token.service_incarnation != config_.incarnation) {
          status = DataError::kOwnerRestarted;
        } else {
          const auto found = shared_weight_leases_.find(token.lease_id);
          if (found == shared_weight_leases_.end()) {
            status = DataError::kOk;
          } else if (found->second.operation.owner_incarnation !=
                     token.consumer_incarnation) {
            status = DataError::kInvalidArgument;
          } else {
            shared_weight_operations_.erase(found->second.operation);
            shared_weight_leases_.erase(found);
            shared_weight_->set_active_leases(shared_weight_leases_.size());
            status = DataError::kOk;
          }
        }
      }
    }
    else if (op == DataServiceOp::kSharedWeightStats && p == e && shared_weight_) {
      std::lock_guard<std::mutex> lock(shared_weight_mutex_);
      const auto& stats = shared_weight_->stats();
      out.push_back(static_cast<uint8_t>(stats.state));
      wire::Put(stats.owner_incarnation, &out);
      wire::Put(stats.allocation_id, &out);
      wire::Put(stats.generation, &out);
      wire::Put(stats.physical_bytes, &out);
      wire::Put(stats.upload_bytes, &out);
      wire::Put(stats.upload_count, &out);
      wire::Put(stats.active_leases, &out);
      wire::Put(stats.allocation_count, &out);
      wire::Put(stats.staging_peak_bytes, &out);
      wire::Put<uint64_t>(stats.upload_ms * 1000.0, &out);
      status = DataError::kOk;
    }
  } else status=DataError::kUnsupportedVersion;
  std::vector<uint8_t>rh;rh.reserve(kHeaderBytes);wire::Put<uint32_t>(kDataServiceWireMagic,&rh);wire::Put<uint16_t>(kDataServiceProtocolVersion,&rh);wire::Put<uint16_t>(opv,&rh);wire::Put<uint16_t>(static_cast<uint16_t>(status),&rh);wire::Put<uint16_t>(0,&rh);wire::Put<uint64_t>(rid,&rh);wire::Put<uint64_t>(out.size(),&rh);wire::Put<uint32_t>(0,&rh);
  WriteAll(fd,rh.data(),rh.size(),deadline);if(!out.empty())WriteAll(fd,out.data(),out.size(),deadline);
  registry_->add_transport_bytes(0,rh.size()+out.size());
  if(shutdown_after_reply)stop();
  {std::lock_guard<std::mutex> lock(queue_mutex_);active_connections_.erase(fd);}
  ::close(fd);
}
}  // namespace data
