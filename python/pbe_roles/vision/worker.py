#!/usr/bin/env python3
"""One-shot real Qwen2.5-VL vision role with data-service publication."""
from __future__ import annotations
import argparse, fcntl, hashlib, itertools, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/"python"))
from pbe_data_client import DataClient, DataKind
from pbe_data_client.client import checksum256
from pbe_data_client.tensor_bundle import encode, decode
from pbe_roles.vision.identity import cache_identity, singleflight_lock_path

REVISION="66285546d2b821cf421d4f5eb2576359d3770cd3"
def load_visual(model_path:Path,device:str):
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
    config=AutoConfig.from_pretrained(model_path,local_files_only=True).vision_config
    visual=Qwen2_5_VisionTransformerPretrainedModel(config).to(dtype=torch.bfloat16)
    index=json.loads((model_path/"model.safetensors.index.json").read_text())["weight_map"]
    state={}
    for shard in sorted({v for k,v in index.items() if k.startswith("visual.")}):
        with safe_open(model_path/shard,framework="pt",device="cpu") as source:
            for key in source.keys():
                if key.startswith("visual."):state[key.removeprefix("visual.")]=source.get_tensor(key)
    missing,unexpected=visual.load_state_dict(state,strict=False)
    if missing or unexpected:raise RuntimeError(f"visual state mismatch missing={missing} unexpected={unexpected}")
    return visual.eval().to(device)

def sequence_metadata(inputs,vision_config):
    """Build request-derived Qwen2.5-VL mRoPE metadata without loading the LLM."""
    import torch
    token_ids=inputs["input_ids"][0].cpu();attention=inputs["attention_mask"][0].cpu().bool()
    types=inputs.get("mm_token_type_ids")
    if types is None:
        base=attention.long().cumsum(-1)-1;base.masked_fill_(~attention,0)
        return token_ids,base.unsqueeze(0).expand(3,-1).contiguous(),0
    types=types[0].cpu();grid_iter=iter(inputs["image_grid_thw"].cpu());current=0;parts=[]
    for modality,group in itertools.groupby(enumerate(types[attention].tolist()),lambda item:item[1]):
        group=list(group);length=len(group)
        if modality==0:
            parts.append(torch.arange(length).view(1,-1).expand(3,-1)+current);current+=length
        elif modality==1:
            t,h,w=map(int,next(grid_iter));merge=int(vision_config.spatial_merge_size);h//=merge;w//=merge
            temporal=current*int(getattr(vision_config,"tokens_per_second",1))
            parts.append(torch.stack([torch.full((t*h*w,),temporal),torch.arange(current,current+h).repeat_interleave(w).repeat(t),torch.arange(current,current+w).repeat(h*t)]));current+=max(h,w)
        else:raise ValueError("video inputs are outside the V4 image-only scope")
    compact=torch.cat(parts,dim=1);positions=torch.zeros((3,len(token_ids)),dtype=torch.int64);positions[:,attention]=compact
    return token_ids,positions,int(compact.max().item()+1-attention.sum().item())
def main():
    p=argparse.ArgumentParser();p.add_argument("--endpoint",required=True);p.add_argument("--model",type=Path,required=True);p.add_argument("--image",type=Path,required=True);p.add_argument("--size",type=int,default=224);p.add_argument("--device",default="cuda:0");p.add_argument("--reference",type=Path);a=p.parse_args()
    if not 28 <= a.size <= 1024 or a.image.stat().st_size > 64 << 20:
        raise ValueError("image size or encoded bytes exceed the Vision role limit")
    image_bytes=a.image.read_bytes();identity,representation,manifest=cache_identity(image_bytes,a.size,a.model)
    client=DataClient(a.endpoint)
    service_incarnation=client.ping()
    try:
        lease=client.acquire(DataKind.TENSOR_BUNDLE,identity,representation)
        components=decode(lease.payload);client.release(lease.token)
        print(f"PBE_VISION_CACHE_HIT pid={os.getpid()} content={identity.hex()} bytes={len(lease.payload)} components={len(components)} forward_count=0")
        return
    except RuntimeError as error:
        if str(error) != str(int(5)): raise
    lock_path=singleflight_lock_path(a.endpoint,service_incarnation,identity)
    lock_fd=os.open(lock_path,os.O_CREAT|os.O_RDWR,0o600);leader=False
    deadline=time.monotonic()+120
    while not leader:
        try:fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB);leader=True
        except BlockingIOError:
            try:
                lease=client.acquire(DataKind.TENSOR_BUNDLE,identity,representation);components=decode(lease.payload);client.release(lease.token)
                print(f"PBE_VISION_SINGLEFLIGHT_FOLLOWER pid={os.getpid()} content={identity.hex()} bytes={len(lease.payload)} components={len(components)} forward_count=0")
                os.close(lock_fd)
                return
            except RuntimeError as error:
                if str(error)!=str(int(5)):raise
            if time.monotonic()>=deadline:
                os.close(lock_fd)
                raise TimeoutError("vision singleflight publication timeout")
            time.sleep(.1)
    handle=None
    try:
        try:
            lease=client.acquire(DataKind.TENSOR_BUNDLE,identity,representation);components=decode(lease.payload);client.release(lease.token)
            print(f"PBE_VISION_SINGLEFLIGHT_FOLLOWER pid={os.getpid()} content={identity.hex()} bytes={len(lease.payload)} components={len(components)} forward_count=0")
            return
        except RuntimeError as error:
            if str(error)!=str(int(5)):raise
        import numpy as np, torch
        from PIL import Image
        from transformers import AutoConfig, AutoProcessor
        processor=AutoProcessor.from_pretrained(a.model,local_files_only=True,use_fast=False)
        image=Image.open(a.image).convert("RGB").resize((a.size,a.size),Image.Resampling.LANCZOS)
        messages=[{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":"Describe the image briefly and name its main subject."}]}]
        inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt")
        grid_cpu=inputs["image_grid_thw"].cpu();vision_config=AutoConfig.from_pretrained(a.model,local_files_only=True).vision_config
        token_ids,positions,rope_delta=sequence_metadata(inputs,vision_config)
        merge_size=int(getattr(vision_config,"spatial_merge_size",2));tokens=int(grid_cpu.prod(dim=1).sum().item())//(merge_size*merge_size)
        hidden=int(getattr(vision_config,"out_hidden_size",getattr(vision_config,"hidden_size",0)))
        if tokens<=0 or tokens>4096 or hidden<=0:raise ValueError("visual output dimensions exceed the Vision role limit")
        metadata=[("image_grid_thw","int64",tuple(grid_cpu.shape),grid_cpu.numpy().astype("<i8",copy=False).tobytes()),("input_ids","int32",tuple(token_ids.shape),token_ids.numpy().astype("<i4",copy=False).tobytes()),("position_ids","int32",tuple(positions.shape),positions.numpy().astype("<i4",copy=False).tobytes()),("rope_delta","int64",(1,),int(rope_delta).to_bytes(8,"little",signed=True))]
        predicted=encode([("image_features","bfloat16",(tokens,hidden),b"\0"*(tokens*hidden*2)),*metadata])
        if len(predicted)>32<<20:raise ValueError("visual bundle byte limit exceeded")
        handle=client.reserve(DataKind.TENSOR_BUNDLE,identity,representation,len(predicted))
        visual=load_visual(a.model,a.device);pixel=inputs["pixel_values"].to(a.device,dtype=torch.bfloat16);grid=grid_cpu.to(a.device)
        started=time.perf_counter()
        with torch.inference_mode(): features=visual(pixel,grid_thw=grid,return_dict=True).pooler_output.to(torch.bfloat16).cpu().contiguous()
        elapsed=(time.perf_counter()-started)*1000
        feature_bytes=features.view(torch.uint16).numpy().astype("<u2",copy=False).tobytes();grid_bytes=grid_cpu.numpy().astype("<i8",copy=False).tobytes()
        bundle=encode([("image_features","bfloat16",tuple(features.shape),feature_bytes),*metadata])
        if len(bundle)!=len(predicted):raise ValueError("vision output did not match the admitted allocation")
        client.seal(handle,bundle);client.release_producer(handle);handle=None
    except Exception:
        if handle is not None:
            try:client.release_producer(handle)
            except Exception:pass
        raise
    finally:
        fcntl.flock(lock_fd,fcntl.LOCK_UN);os.close(lock_fd)
    max_abs=mean_abs=float("nan")
    if a.reference:
        expected=np.load(a.reference)["image_features"].reshape(features.shape);actual=features.float().numpy();max_abs=float(np.max(np.abs(actual-expected)));mean_abs=float(np.mean(np.abs(actual-expected)))
    print(f"PBE_VISION_PUBLISHED pid={os.getpid()} device={a.device} content={identity.hex()} manifest={manifest.hex()} feature_shape={list(features.shape)} grid={grid_cpu.tolist()} bytes={len(bundle)} forward_count=1 forward_ms={elapsed:.3f} reference_max_abs={max_abs} reference_mean_abs={mean_abs}")
if __name__=="__main__":main()
