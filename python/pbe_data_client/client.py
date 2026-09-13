"""Dependency-free PBE data-service protocol client."""
from __future__ import annotations
import enum, os, socket, struct, time
from dataclasses import dataclass

MAGIC=0x53534250; VERSION=2; HEADER=struct.Struct("<IHHHHQQI")
REF_MAGIC=0x52444250; REF_SIZE=100
class DataError(enum.IntEnum):
    OK=0; INVALID_ARGUMENT=1; UNSUPPORTED_VERSION=2; UNKNOWN_OBJECT=3
    STALE_GENERATION=4; NOT_READY=5; ALREADY_SEALED=6; CAPACITY_EXHAUSTED=7
    COVERAGE_MISMATCH=8; CHECKSUM_MISMATCH=9; OWNER_RESTARTED=10
class DataKind(enum.IntEnum): KV_PAGE=1; TENSOR_BUNDLE=2; CHECKPOINT=3
@dataclass(frozen=True)
class LeaseToken: service_incarnation:int; consumer_incarnation:int; lease_id:int
@dataclass
class Lease:
    token:LeaseToken; ref:bytes; payload:bytes
    @property
    def lease_id(self)->int:return self.token.lease_id

def checksum256(value:bytes)->bytes:
    mask=(1<<64)-1; h=[1469598103934665603,1099511628211,0x9e3779b97f4a7c15,0xd6e8feb86659fd93]
    for byte in value:
        for j in range(4):
            h[j]^=byte+j*41; h[j]=(h[j]*(1099511628211+j*2))&mask; h[j]^=h[j]>>29
    return struct.pack("<QQQQ",*h)

class DataClient:
    def __init__(self,endpoint:str): self.endpoint=endpoint; self._seq=0; self.client_incarnation=(os.getpid()<<32)^time.monotonic_ns()
    def _call(self,op:int,payload:bytes=b"")->bytes:
        self._seq+=1; rid=(self.client_incarnation+self._seq)&((1<<64)-1)
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
            s.connect(self.endpoint); s.sendall(HEADER.pack(MAGIC,VERSION,op,0,0,rid,len(payload),0)+payload)
            header=b""
            while len(header)<HEADER.size:
                part=s.recv(HEADER.size-len(header));
                if not part: raise ConnectionError("short response header")
                header+=part
            magic,version,rop,error,_,rrid,size,pad=HEADER.unpack(header)
            if (magic,version,rop,rrid,pad)!=(MAGIC,VERSION,op,rid,0): raise ValueError("invalid response header")
            result=b""
            while len(result)<size:
                part=s.recv(size-len(result));
                if not part: raise ConnectionError("short response payload")
                result+=part
            if error: raise RuntimeError(DataError(error))
            return result
    def ping(self)->int: return struct.unpack("<Q",self._call(1))[0]
    def reserve(self,kind:DataKind,content:bytes,representation:bytes,size:int,operation:int|None=None)->tuple[int,int,int]:
        if len(content)!=32 or len(representation)!=32 or size<=0: raise ValueError("invalid reservation")
        owner=self.ping(); seq=operation or ((self.client_incarnation+self._seq+1)&((1<<64)-1))
        response=self._call(2,struct.pack("<QQB",owner,seq,int(kind))+content+representation+struct.pack("<Q",size))
        return struct.unpack("<QQI",response)
    def seal(self,handle:tuple[int,int,int],payload:bytes)->bytes:
        response=self._call(3,struct.pack("<QQI",*handle)+checksum256(payload)+struct.pack("<Q",len(payload))+payload)
        if len(response)!=REF_SIZE:return (_ for _ in ()).throw(ValueError("invalid DataRef"))
        return response
    def acquire(self,kind:DataKind,content:bytes,representation:bytes,lease_kind:int=1)->Lease:
        self._seq+=1; operation=(self.client_incarnation,self._seq)
        request=struct.pack("<QQ",*operation)+bytes([kind])+content+representation+bytes([lease_kind])
        try:r=self._call(4,request)
        except (ConnectionError,OSError):r=self._call(4,request)
        service,consumer,lease_id,ref_size=struct.unpack_from("<QQQQ",r)
        if ref_size!=REF_SIZE: raise ValueError("invalid DataRef size")
        ref=r[32:32+ref_size]; size=struct.unpack_from("<Q",r,32+ref_size)[0]; payload=r[40+ref_size:]
        if len(payload)!=size: raise ValueError("invalid lease payload")
        return Lease(LeaseToken(service,consumer,lease_id),ref,payload)
    def release(self,token:LeaseToken)->None:self._call(5,struct.pack("<QQQ",token.service_incarnation,token.consumer_incarnation,token.lease_id))
    def release_producer(self,handle:tuple[int,int,int])->None:self._call(7,struct.pack("<QQI",*handle))
    def stats(self)->tuple[int,...]:return struct.unpack("<QQQQQQQ",self._call(8))
    def shutdown(self)->None:self._call(9)
