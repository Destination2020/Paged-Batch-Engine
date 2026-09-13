#include "data/ipc_pool.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <sstream>

#include "data/wire_io.h"

namespace data {
namespace {
bool CudaOk(cudaError_t e,const char* what,std::string*error){if(e==cudaSuccess)return true;if(error){std::ostringstream o;o<<what<<": "<<cudaGetErrorString(e);*error=o.str();}return false;}
}
bool EncodeIpcPoolDescriptor(const IpcPoolDescriptor& d,std::vector<uint8_t>*o){
  if(!o||!o->empty()||d.owner_incarnation==0||d.device<0||d.bytes==0)return false;
  wire::Put<uint32_t>(kIpcPoolMagic,o);wire::Put<uint16_t>(kIpcPoolVersion,o);wire::Put<uint16_t>(0,o);
  wire::Put(d.owner_incarnation,o);wire::Put<uint32_t>(d.device,o);wire::Put(d.bytes,o);
  wire::Put<uint32_t>(d.num_layers,o);wire::Put<uint32_t>(d.num_blocks,o);
  wire::Put<uint32_t>(d.block_size,o);wire::Put<uint32_t>(d.num_kv_heads,o);
  wire::Put<uint32_t>(d.head_size,o);o->push_back(d.storage_dtype);
  for(int i=0;i<7;++i)o->push_back(0);
  wire::PutBytes(&d.memory,sizeof(d.memory),o);wire::PutBytes(&d.ready,sizeof(d.ready),o);wire::PutBytes(&d.consumed,sizeof(d.consumed),o);return true;
}
bool DecodeIpcPoolDescriptor(const uint8_t*b,size_t n,IpcPoolDescriptor*d){if(!b||!d)return false;const uint8_t*p=b,*e=b+n;uint32_t magic=0,dev=0,layers=0,blocks=0,block=0,heads=0,head_size=0;uint16_t version=0,res=0;uint8_t dtype=0,pad[7]{};if(!wire::Get(&p,e,&magic)||!wire::Get(&p,e,&version)||!wire::Get(&p,e,&res)||!wire::Get(&p,e,&d->owner_incarnation)||!wire::Get(&p,e,&dev)||!wire::Get(&p,e,&d->bytes)||!wire::Get(&p,e,&layers)||!wire::Get(&p,e,&blocks)||!wire::Get(&p,e,&block)||!wire::Get(&p,e,&heads)||!wire::Get(&p,e,&head_size)||!wire::Get(&p,e,&dtype)||!wire::GetBytes(&p,e,pad,sizeof(pad))||!wire::GetBytes(&p,e,&d->memory,sizeof(d->memory))||!wire::GetBytes(&p,e,&d->ready,sizeof(d->ready))||!wire::GetBytes(&p,e,&d->consumed,sizeof(d->consumed))||p!=e||magic!=kIpcPoolMagic||version!=kIpcPoolVersion||res!=0||d->owner_incarnation==0||d->bytes==0)return false;for(auto x:pad)if(x)return false;d->device=static_cast<int32_t>(dev);d->num_layers=layers;d->num_blocks=blocks;d->block_size=block;d->num_kv_heads=heads;d->head_size=head_size;d->storage_dtype=dtype;return true;}
CudaIpcPoolOwner::~CudaIpcPoolOwner(){if(device_>=0)cudaSetDevice(device_);if(consumed_)cudaEventDestroy(consumed_);if(ready_)cudaEventDestroy(ready_);if(memory_)cudaFree(memory_);}
bool CudaIpcPoolOwner::create(int device,uint64_t bytes,uint8_t pattern,std::string*error){if(memory_||bytes==0)return false;device_=device;bytes_=bytes;if(!CudaOk(cudaSetDevice(device),"cudaSetDevice",error)||!CudaOk(cudaMalloc(&memory_,bytes),"cudaMalloc",error)||!CudaOk(cudaEventCreateWithFlags(&ready_,cudaEventDisableTiming|cudaEventInterprocess),"create ready event",error)||!CudaOk(cudaEventCreateWithFlags(&consumed_,cudaEventDisableTiming|cudaEventInterprocess),"create consumed event",error)||!CudaOk(cudaMemset(memory_,pattern,bytes),"cudaMemset",error)||!CudaOk(cudaEventRecord(ready_),"record ready",error))return false;return true;}
bool CudaIpcPoolOwner::set_kv_layout(int32_t layers,int32_t blocks,int32_t block,int32_t heads,int32_t head_size,uint8_t dtype,std::string*error){if(!memory_||layers<=0||blocks<=0||block<=0||heads<=0||head_size<=0||dtype==0)return false;uint64_t elements=static_cast<uint64_t>(layers)*2*blocks*block*heads*head_size;uint64_t type_bytes=dtype==4?2:(dtype==1?4:0);if(!type_bytes||elements>UINT64_MAX/type_bytes||elements*type_bytes!=bytes_){if(error)*error="KV layout byte size does not match IPC allocation";return false;}num_layers_=layers;num_blocks_=blocks;block_size_=block;num_kv_heads_=heads;head_size_=head_size;storage_dtype_=dtype;return true;}
bool CudaIpcPoolOwner::descriptor(uint64_t incarnation,IpcPoolDescriptor*out,std::string*error){if(!out||!memory_)return false;out->owner_incarnation=incarnation;out->device=device_;out->bytes=bytes_;out->num_layers=num_layers_;out->num_blocks=num_blocks_;out->block_size=block_size_;out->num_kv_heads=num_kv_heads_;out->head_size=head_size_;out->storage_dtype=storage_dtype_;return CudaOk(cudaIpcGetMemHandle(&out->memory,memory_),"get memory handle",error)&&CudaOk(cudaIpcGetEventHandle(&out->ready,ready_),"get ready event handle",error)&&CudaOk(cudaIpcGetEventHandle(&out->consumed,consumed_),"get consumed event handle",error);}
bool CudaIpcPoolOwner::wait_consumed(std::string*error){return CudaOk(cudaEventSynchronize(consumed_),"wait consumed event",error);}
CudaIpcPoolImport::~CudaIpcPoolImport(){if(device_>=0)cudaSetDevice(device_);if(consumed_)cudaEventDestroy(consumed_);if(ready_)cudaEventDestroy(ready_);if(compute_view_&&compute_view_!=ipc_mapped_)cudaFree(compute_view_);if(ipc_mapped_)cudaIpcCloseMemHandle(ipc_mapped_);}
bool CudaIpcPoolImport::open(const IpcPoolDescriptor&d,int device,std::string*error){descriptor_=d;device_=device;if(!CudaOk(cudaSetDevice(device),"cudaSetDevice",error)||!CudaOk(cudaIpcOpenMemHandle(&ipc_mapped_,d.memory,cudaIpcMemLazyEnablePeerAccess),"open memory handle",error)||!CudaOk(cudaIpcOpenEventHandle(&ready_,d.ready),"open ready event",error)||!CudaOk(cudaIpcOpenEventHandle(&consumed_,d.consumed),"open consumed event",error))return false;if(device==d.device)compute_view_=ipc_mapped_;return true;}
bool CudaIpcPoolImport::prepare_compute_view(void* stream,uint64_t*tx,double*ms,std::string*path,std::string*error){
  if(!ipc_mapped_||!tx||!ms||!path)return false;
  auto cuda_stream=static_cast<cudaStream_t>(stream);
  auto begin=std::chrono::steady_clock::now();
  if(!CudaOk(cudaStreamWaitEvent(cuda_stream,ready_),"wait IPC pool ready",error))return false;
  if(device_==descriptor_.device){compute_view_=ipc_mapped_;*tx=0;*path="same_gpu_ipc_map";}
  else{
    int can=0;
    if(!CudaOk(cudaDeviceCanAccessPeer(&can,device_,descriptor_.device),"query peer access",error)||!can){if(error)*error="peer access unsupported";return false;}
    auto peer=cudaDeviceEnablePeerAccess(descriptor_.device,0);
    if(peer!=cudaSuccess&&peer!=cudaErrorPeerAccessAlreadyEnabled)return CudaOk(peer,"enable peer access",error);
    cudaGetLastError();
    if(!compute_view_&&!CudaOk(cudaMalloc(&compute_view_,descriptor_.bytes),"allocate compute replica",error))return false;
    if(!CudaOk(cudaMemcpyPeerAsync(compute_view_,device_,ipc_mapped_,descriptor_.device,descriptor_.bytes,cuda_stream),"copy compute replica",error)||
       !CudaOk(cudaStreamSynchronize(cuda_stream),"synchronize compute replica",error))return false;
    *tx=descriptor_.bytes;*path="cross_gpu_p2p_replica";
  }
  *ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
  return true;
}
bool CudaIpcPoolImport::prepare_compute_view_slots(const std::vector<int32_t>& requested,
    void* stream,uint64_t*tx,double*ms,std::string*path,std::string*error){
  if(device_==descriptor_.device)return prepare_compute_view(stream,tx,ms,path,error);
  if(!ipc_mapped_||!tx||!ms||!path||descriptor_.num_layers<=0||
     descriptor_.num_blocks<=0||descriptor_.block_size<=0||
     descriptor_.num_kv_heads<=0||descriptor_.head_size<=0)return false;
  std::vector<int32_t> slots=requested;
  std::sort(slots.begin(),slots.end());
  slots.erase(std::unique(slots.begin(),slots.end()),slots.end());
  if(slots.empty()||slots.front()<0||slots.back()>=descriptor_.num_blocks){
    if(error)*error="invalid or empty sparse replica slots";return false;
  }
  auto cuda_stream=static_cast<cudaStream_t>(stream);
  const auto begin=std::chrono::steady_clock::now();
  if(!CudaOk(cudaStreamWaitEvent(cuda_stream,ready_),"wait IPC pool ready",error))return false;
  int can=0;
  if(!CudaOk(cudaDeviceCanAccessPeer(&can,device_,descriptor_.device),"query peer access",error)||!can){
    if(error)*error="peer access unsupported";return false;
  }
  auto peer=cudaDeviceEnablePeerAccess(descriptor_.device,0);
  if(peer!=cudaSuccess&&peer!=cudaErrorPeerAccessAlreadyEnabled)
    return CudaOk(peer,"enable peer access",error);
  cudaGetLastError();
  if(!compute_view_&&!CudaOk(cudaMalloc(&compute_view_,descriptor_.bytes),
                             "allocate sparse compute replica",error))return false;
  const uint64_t dtype_bytes=descriptor_.storage_dtype==4?2:
      (descriptor_.storage_dtype==1?4:0);
  const uint64_t block_bytes=static_cast<uint64_t>(descriptor_.block_size)*
      descriptor_.num_kv_heads*descriptor_.head_size*dtype_bytes;
  if(!dtype_bytes||block_bytes==0){if(error)*error="invalid sparse replica dtype";return false;}
  for(int layer=0;layer<descriptor_.num_layers;++layer){
    for(int kv=0;kv<2;++kv){
      for(int32_t slot:slots){
        const uint64_t offset=(static_cast<uint64_t>(layer*2+kv)*
                               descriptor_.num_blocks+slot)*block_bytes;
        auto* destination=static_cast<uint8_t*>(compute_view_)+offset;
        auto* source=static_cast<uint8_t*>(ipc_mapped_)+offset;
        if(!CudaOk(cudaMemcpyPeerAsync(destination,device_,source,descriptor_.device,
                                       block_bytes,cuda_stream),
                   "copy sparse compute page",error))return false;
      }
    }
  }
  if(!CudaOk(cudaStreamSynchronize(cuda_stream),"synchronize sparse replica",error))return false;
  *tx=slots.size()*descriptor_.num_layers*2*block_bytes;
  *path="cross_gpu_p2p_page_subset";
  *ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
  return true;
}
bool CudaIpcPoolImport::wait_ready(void* stream,std::string*error){
  if(!ipc_mapped_||!ready_)return false;
  return CudaOk(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream),ready_),"wait IPC pool ready",error);
}
bool CudaIpcPoolImport::validate_and_signal(uint8_t pattern,uint64_t*tx,double*ms,std::string*path,std::string*error){if(!prepare_compute_view(nullptr,tx,ms,path,error))return false;std::vector<uint8_t>host(descriptor_.bytes);if(!CudaOk(cudaMemcpy(host.data(),compute_view_,host.size(),cudaMemcpyDeviceToHost),"validation copy",error))return false;for(uint8_t v:host)if(v!=pattern){if(error)*error="payload checksum mismatch";return false;}if(!CudaOk(cudaEventRecord(consumed_),"record consumed",error)||!CudaOk(cudaEventSynchronize(consumed_),"sync consumed",error))return false;return true;}
}  // namespace data
