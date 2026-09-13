from __future__ import annotations
import struct
from .client import checksum256
MAGIC=0x4e424250; VERSION=1
DTYPE={"float32":1,"float16":2,"bfloat16":3,"int32":4,"int64":5,"uint8":6}
def encode(components:list[tuple[str,str,tuple[int,...],bytes]],alignment:int=64)->bytes:
    metadata=24+sum(56+len(name.encode())+8*len(shape) for name,_,shape,_ in components)
    cursor=(metadata+alignment-1)//alignment*alignment; records=[]
    for name,dtype,shape,value in components:
        cursor=(cursor+alignment-1)//alignment*alignment; records.append((name,dtype,shape,cursor,value));cursor+=len(value)
    out=bytearray(struct.pack("<IHHQII",MAGIC,VERSION,len(records),cursor,alignment,0))
    for name,dtype,shape,offset,value in records:
        encoded=name.encode();out+=struct.pack("<HHHHQQ",len(encoded),DTYPE[dtype],len(shape),0,offset,len(value));out+=checksum256(value);out+=encoded
        for dim in shape:out+=struct.pack("<Q",dim)
    out+=bytes((alignment-len(out)%alignment)%alignment)
    for _,_,_,offset,value in records:
        out+=bytes(offset-len(out));out+=value
    assert len(out)==cursor
    return bytes(out)
def decode(value:bytes)->list[tuple[str,int,tuple[int,...],bytes]]:
    magic,version,count,total,alignment,reserved=struct.unpack_from("<IHHQII",value);assert (magic,version,total,reserved)==(MAGIC,VERSION,len(value),0)
    cursor=24;result=[]
    for _ in range(count):
        name_len,dtype,rank,res,offset,size=struct.unpack_from("<HHHHQQ",value,cursor);cursor+=24;digest=value[cursor:cursor+32];cursor+=32;name=value[cursor:cursor+name_len].decode();cursor+=name_len;shape=struct.unpack_from("<"+"Q"*rank,value,cursor);cursor+=8*rank;payload=value[offset:offset+size];assert res==0 and checksum256(payload)==digest;result.append((name,dtype,shape,payload))
    return result
