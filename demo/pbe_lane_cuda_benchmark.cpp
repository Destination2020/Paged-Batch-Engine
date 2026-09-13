#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <nlohmann/json.hpp>
#include "cache/transfer_scheduler.h"

namespace {
using Clock = std::chrono::steady_clock;
void Check(cudaError_t s, const char* op) { if (s != cudaSuccess) throw std::runtime_error(std::string(op)+":"+cudaGetErrorString(s)); }
double Ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double,std::milli>(b-a).count(); }

class CudaExecutor final : public cache::FlightExecutor {
 public:
  explicit CudaExecutor(size_t bytes):bytes_(bytes) {
    for(auto& s:streams_) Check(cudaStreamCreate(&s),"stream");
    for(int page=1;page<=3;++page){void* h=nullptr;void* d=nullptr;Check(cudaMallocHost(&h,bytes_),"host");Check(cudaMalloc(&d,bytes_),"device");buffers_[page]={static_cast<uint8_t*>(h),static_cast<uint8_t*>(d)};const uint8_t p=0x30+page;std::fill_n(buffers_[page].host,bytes_,p);Check(cudaMemset(buffers_[page].device,p,bytes_),"init");}
  }
  ~CudaExecutor() override { for(auto& [_,b]:buffers_){cudaFree(b.device);cudaFreeHost(b.host);}for(auto s:streams_)cudaStreamDestroy(s); }
  cache::FlightExecutionResult submit(cache::FlightId id,const cache::FlightKey& key) noexcept override {
    try { Job j; j.key=key;j.submitted=Clock::now();Check(cudaEventCreateWithFlags(&j.done,cudaEventDisableTiming),"event");auto stream=streams_[(key.page-1)%2];auto& b=buffers_.at(key.page);
      if(key.target==cache::TransferTarget::kHost){std::fill_n(b.host,bytes_,uint8_t{0});Check(cudaMemcpyAsync(b.host,b.device,bytes_,cudaMemcpyDeviceToHost,stream),"d2h");}
      else {Check(cudaMemsetAsync(b.device,0,bytes_,stream),"clear");Check(cudaMemcpyAsync(b.device,b.host,bytes_,cudaMemcpyHostToDevice,stream),"h2d");}
      Check(cudaEventRecord(j.done,stream),"record");jobs_.emplace(id,j);submission_times_[id]=j.submitted;timeline_.push_back({{"flight",id},{"page",key.page},{"target",key.target==cache::TransferTarget::kHost?"host":"gpu"},{"event","physical_submit"},{"at_ms",Ms(epoch_,j.submitted)}});return cache::FlightExecutionResult::kPending;
    } catch(...) { return cache::FlightExecutionResult::kFailedSafe; }
  }
  cache::FlightExecutionResult poll(cache::FlightId id) noexcept override {auto it=jobs_.find(id);if(it==jobs_.end())return cache::FlightExecutionResult::kFailedSafe;auto s=cudaEventQuery(it->second.done);if(s==cudaErrorNotReady)return cache::FlightExecutionResult::kPending;if(s!=cudaSuccess)return cache::FlightExecutionResult::kFailedSafe;timeline_.push_back({{"flight",id},{"page",it->second.key.page},{"event","fence_complete"},{"at_ms",Ms(epoch_,Clock::now())}});return cache::FlightExecutionResult::kSucceeded;}
  bool cancel(cache::FlightId) noexcept override {return false;}
  cache::FlightExecutionResult drain(cache::FlightId id) noexcept override {auto it=jobs_.find(id);return it!=jobs_.end()&&cudaEventSynchronize(it->second.done)==cudaSuccess?cache::FlightExecutionResult::kSucceeded:cache::FlightExecutionResult::kFailedSafe;}
  void release(cache::FlightId id) noexcept override {auto it=jobs_.find(id);if(it!=jobs_.end()){cudaEventDestroy(it->second.done);jobs_.erase(it);}}
  bool validate(){Check(cudaDeviceSynchronize(),"sync");std::vector<uint8_t> copy(bytes_);for(int page=1;page<=3;++page){const uint8_t p=0x30+page;auto& b=buffers_.at(page);if(page<=2){Check(cudaMemcpy(copy.data(),b.device,bytes_,cudaMemcpyDeviceToHost),"validate");if(!std::all_of(copy.begin(),copy.end(),[p](uint8_t x){return x==p;}))return false;}else if(!std::all_of(b.host,b.host+bytes_,[p](uint8_t x){return x==p;}))return false;}return true;}
  Clock::time_point submitted(cache::FlightId id) const {return submission_times_.at(id);}
  const nlohmann::json& timeline() const{return timeline_;}
 private:
  struct Buffer{uint8_t* host;uint8_t* device;};struct Job{cache::FlightKey key;cudaEvent_t done=nullptr;Clock::time_point submitted;};size_t bytes_;cudaStream_t streams_[2]{};std::map<int,Buffer> buffers_;std::map<cache::FlightId,Job> jobs_;std::map<cache::FlightId,Clock::time_point> submission_times_;Clock::time_point epoch_=Clock::now();nlohmann::json timeline_=nlohmann::json::array();
};

nlohmann::json Run(bool lanes,size_t bytes){auto executor=std::make_unique<CudaExecutor>(bytes);auto* cuda=executor.get();cache::TransferSchedulerConfig cfg{8,8,2,1,1,1,100};cfg.direction_lanes_enabled=lanes;cache::TransferScheduler scheduler(cfg,std::move(executor));cache::WaiterTicket a,b,demand;if(!scheduler.submit({1,cache::TransferTarget::kGpu,0},cache::TransferPriority::kBackground,{},&a)||!scheduler.submit({2,cache::TransferTarget::kGpu,0},cache::TransferPriority::kBackground,{},&b))throw std::runtime_error("background admission failed");auto queued=Clock::now();if(!scheduler.submit({3,cache::TransferTarget::kHost,-1},cache::TransferPriority::kDecode,{},&demand))throw std::runtime_error("demand admission failed");cache::WaiterState state=cache::WaiterState::kPending;while(!scheduler.state(demand.waiter,&state)||state==cache::WaiterState::kPending){scheduler.poll();std::this_thread::yield();}auto completed=Clock::now();double queue=Ms(queued,cuda->submitted(demand.flight));if(state!=cache::WaiterState::kSucceeded||!scheduler.drain()||!cuda->validate())throw std::runtime_error("transfer or full payload validation failed");auto stats=scheduler.stats();return{{"mode",lanes?"lanes_on":"lanes_off"},{"demand_queue_ms",queue},{"demand_completion_ms",Ms(queued,completed)},{"full_payload_validated",true},{"timeline",cuda->timeline()},{"scheduler",{{"physical_submissions",stats.physical_submissions},{"demand_completed",stats.demand_completed},{"background_completed",stats.background_completed},{"max_d2h_active",stats.max_d2h_active},{"max_h2d_active",stats.max_h2d_active}}}};}
}
int main(int argc,char** argv){try{int repeats=argc>1?std::stoi(argv[1]):5;size_t bytes=argc>2?std::stoull(argv[2]):size_t{256}<<20;std::vector<bool> order;for(int i=0;i<repeats;++i){order.push_back(false);order.push_back(true);}std::mt19937 gen(20260913);std::shuffle(order.begin(),order.end(),gen);nlohmann::json records=nlohmann::json::array();for(bool lanes:order)records.push_back(Run(lanes,bytes));std::cout<<nlohmann::json({{"ok",true},{"repeats",repeats},{"benchmark","pbe_transfer_scheduler_cuda_payload"},{"only_ablation","TransferScheduler.direction_lanes_enabled"},{"bytes_per_transfer",bytes},{"streams",2},{"max_total_active",2},{"records",records}}).dump()<<"\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
