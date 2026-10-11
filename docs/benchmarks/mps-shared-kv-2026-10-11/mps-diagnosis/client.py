import ctypes,json,os,time
from pathlib import Path
cuda=ctypes.CDLL('libcuda.so.1')
results={'pid':os.getpid(),'pipe_directory':os.environ.get('CUDA_MPS_PIPE_DIRECTORY'),'active_thread_percentage':os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')}
def check(name,*args):
 rc=getattr(cuda,name)(*args); results[name]=rc
 if rc: raise RuntimeError(f'{name}: {rc}')
check('cuInit',0)
context=ctypes.c_void_p();check('cuCtxCreate_v2',ctypes.byref(context),0,0)
ptr=ctypes.c_uint64();check('cuMemAlloc_v2',ctypes.byref(ptr),ctypes.c_size_t(4096))
sm=ctypes.c_int();check('cuDeviceGetAttribute',ctypes.byref(sm),16,0);results['multiprocessor_count']=sm.value
Path(os.environ['PBE_DIAG_READY']).write_text(json.dumps(results,indent=2))
try:
 deadline=time.monotonic()+15
 while not Path(os.environ['PBE_DIAG_RELEASE']).exists() and time.monotonic()<deadline:time.sleep(.05)
finally:
 check('cuMemFree_v2',ptr);check('cuCtxDestroy_v2',context)
print(json.dumps(results,indent=2))
