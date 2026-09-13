#!/usr/bin/env python3
"""Real variable-length two-image Vision batch with pre-forward admission."""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/"python"))
from pbe_data_client import DataClient,DataKind
from pbe_data_client.tensor_bundle import encode
from pbe_roles.vision.identity import cache_identity
from pbe_roles.vision.worker import load_visual

def main():
    p=argparse.ArgumentParser();p.add_argument("--endpoint",required=True);p.add_argument("--model",type=Path,required=True);p.add_argument("--image",type=Path,required=True);p.add_argument("--sizes",type=int,nargs=2,default=[252,308]);p.add_argument("--device",default="cuda:0");a=p.parse_args()
    if any(s<28 or s>1024 for s in a.sizes) or a.image.stat().st_size>64<<20:raise ValueError("input exceeds Vision batch limits")
    import torch
    from PIL import Image
    from transformers import AutoConfig,AutoProcessor
    raw=a.image.read_bytes();source=Image.open(a.image).convert("RGB");images=[source.resize((s,s),Image.Resampling.LANCZOS) for s in a.sizes];text="Describe the image briefly and name its main subject."
    conversations=[[{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":text}]}] for image in images]
    processor=AutoProcessor.from_pretrained(a.model,local_files_only=True,use_fast=False);inputs=processor.apply_chat_template(conversations,tokenize=True,add_generation_prompt=True,padding=True,return_dict=True,return_tensors="pt")
    grid_cpu=inputs["image_grid_thw"].cpu();config=AutoConfig.from_pretrained(a.model,local_files_only=True).vision_config
    merge=int(getattr(config,"spatial_merge_size",2));hidden=int(getattr(config,"out_hidden_size",getattr(config,"hidden_size",0)))
    splits=(grid_cpu.prod(-1)//(merge**2)).tolist();client=DataClient(a.endpoint);admitted=[]
    try:
        for size,tokens,item_grid in zip(a.sizes,splits,grid_cpu):
            content,rep,_=cache_identity(raw,size,a.model);grid_bytes=item_grid.reshape(1,3).numpy().astype("<i8",copy=False).tobytes()
            predicted=encode([("image_features","bfloat16",(tokens,hidden),b"\0"*(tokens*hidden*2)),("image_grid_thw","int64",(1,3),grid_bytes)])
            admitted.append((size,content,rep,client.reserve(DataKind.TENSOR_BUNDLE,content,rep,len(predicted)),len(predicted)))
        visual=load_visual(a.model,a.device);grid=grid_cpu.to(a.device);started=time.perf_counter()
        with torch.inference_mode():packed=visual(inputs["pixel_values"].to(a.device,dtype=torch.bfloat16),grid_thw=grid,return_dict=True).pooler_output.to(torch.bfloat16).cpu()
        features=torch.split(packed,splits);published=[]
        for item,feature,item_grid in zip(admitted,features,grid_cpu):
            size,content,rep,handle,predicted_size=item;fb=feature.contiguous().view(torch.uint16).numpy().astype("<u2",copy=False).tobytes();gb=item_grid.reshape(1,3).numpy().astype("<i8",copy=False).tobytes();bundle=encode([("image_features","bfloat16",tuple(feature.shape),fb),("image_grid_thw","int64",(1,3),gb)])
            if len(bundle)!=predicted_size:raise ValueError("vision batch output did not match admission")
            client.seal(handle,bundle);client.release_producer(handle);published.append((size,list(feature.shape),len(bundle),content.hex()))
        admitted.clear();print(f"PBE_VISION_MIXED_BATCH pid={os.getpid()} batch=2 split_tokens={splits} elapsed_ms={(time.perf_counter()-started)*1000:.3f} published={published}")
    finally:
        for _,_,_,handle,_ in admitted:
            try:client.release_producer(handle)
            except Exception:pass

if __name__=="__main__":main()
