#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/"python"))
from pbe_data_client import DataClient,DataKind
from pbe_data_client.tensor_bundle import decode
from pbe_roles.vision.identity import cache_identity
def main():
 p=argparse.ArgumentParser();p.add_argument("--endpoint",required=True);p.add_argument("--model",type=Path,required=True);p.add_argument("--image",type=Path,required=True);p.add_argument("--size",type=int,default=224);p.add_argument("--cancel",action="store_true");a=p.parse_args();content,rep,_=cache_identity(a.image.read_bytes(),a.size,a.model);client=DataClient(a.endpoint);lease=client.acquire(DataKind.TENSOR_BUNDLE,content,rep);parts=decode(lease.payload);client.release(lease.token);state="CANCELLED" if a.cancel else "CONSUME";print(f"PBE_VISION_{state} pid={os.getpid()} lease={lease.lease_id} bytes={len(lease.payload)} components={[(x[0],x[2],len(x[3])) for x in parts]}")
if __name__=="__main__":main()
