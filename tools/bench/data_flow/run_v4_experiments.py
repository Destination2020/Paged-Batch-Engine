#!/usr/bin/env python3
"""Reproducible V4 representative experiment runner (five ordered repetitions)."""
from __future__ import annotations
import argparse, json, os, random, re, socket, statistics, subprocess, time
from pathlib import Path

def run(command, cwd, env=None):
    begin=time.perf_counter(); p=subprocess.run(command,cwd=cwd,env=env,text=True,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False)
    return p.returncode,(time.perf_counter()-begin)*1000,p.stdout

def percentile(values,p):
    values=sorted(values); return values[round((len(values)-1)*p)] if values else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--build",type=Path,required=True)
    ap.add_argument("--model-dir",type=Path,required=True);ap.add_argument("--image",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--skip-vision",action="store_true")
    a=ap.parse_args(); root=Path(__file__).resolve().parents[3];a.output.mkdir(parents=True,exist_ok=True)
    raw=a.output/"raw";raw.mkdir(exist_ok=True);records=[];random.seed(20260912)
    tests={
      "branch_2_4_cow":"KVCacheManagerTest.PartialTailForkSharesThenCopiesOnFirstAppend:KVCacheManagerTest.PageAlignedForkNeedsNoCowAndSamplesIndependently",
      "joint_budget_lanes":"NodeMemoryBudgetTest.*:TransferSchedulerTest.DirectionLanesRunIndependently:TransferSchedulerTest.DemandReserveSurvivesBackgroundSaturation",
      "fault_lifecycle":"LocalDataRuntimeTest.ProducerExitBeforeSealLeavesNoVisibleObject:LocalDataRuntimeTest.ServiceRestartRejectsEveryOldAllocationHandle:ContentRegistryTest.TenThousandAcquireCancelCyclesReturnToSteadyState"}
    order=list(tests)*5;random.shuffle(order)
    for index,name in enumerate(order):
        rc,ms,out=run([str(a.build/"test/test_llm"),"--gtest_filter="+tests[name]],root)
        (raw/f"contract_{index:02d}_{name}.log").write_text(out);records.append({"kind":"contract","case":name,"order":index,"wall_ms":ms,"ok":rc==0})
        if rc: raise SystemExit(f"contract failed: {name}")
    vision=[]
    if not a.skip_vision:
      sizes=[224,280,224,280,224]
      for repetition,size in enumerate(sizes):
        endpoint=f"/tmp/pbe-v4-exp-{os.getpid()}-{repetition}.sock";service=subprocess.Popen(
          [str(a.build/"demo/pbe_data_service"),"serve",endpoint,str(1<<30),"64"],cwd=root,
          stdout=(raw/f"vision_service_{repetition}.log").open("w"),stderr=subprocess.STDOUT,text=True)
        try:
          for _ in range(500):
            if Path(endpoint).exists(): break
            time.sleep(.01)
          env=os.environ.copy();env["PYTHONPATH"]="python"
          command=[str(root/".venv/bin/python"),"python/pbe_roles/vision/worker.py","--endpoint",endpoint,
             "--model",str(a.model_dir),"--image",str(a.image),"--size",str(size),"--device","cuda:0"]
          for mode in ("off","on"):
            rc,ms,out=run(command,root,env);(raw/f"vision_{repetition}_{mode}.log").write_text(out)
            if rc: raise RuntimeError(out)
            forward=re.search(r"forward_ms=([0-9.]+)",out)
            bundle=re.search(r"bytes=([0-9]+)",out)
            row={"kind":"vision_cache","mode":mode,"repetition":repetition,"size":size,
                 "wall_ms":ms,"forward_ms":float(forward.group(1)) if forward else 0.0,
                 "bundle_bytes":int(bundle.group(1)) if bundle else 0,"ok":True}
            records.append(row);vision.append(row)
        finally:
          run([str(a.build/"demo/pbe_data_service"),"shutdown",endpoint],root);service.wait(timeout=10)
    trace=a.output/"trace.jsonl";trace.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in records))
    groups={}
    for row in records: groups.setdefault((row["kind"],row.get("case",row.get("mode"))),[]).append(row["wall_ms"])
    summary={"schema":"pbe.v4.experiments.v1","seed":20260912,"repetitions":5,
      "model_revision":"66285546d2b821cf421d4f5eb2576359d3770cd3",
      "summaries":{"/".join(k):{"n":len(v),"median_ms":statistics.median(v),"min_ms":min(v),"max_ms":max(v),"p95_ms":percentile(v,.95),"p99_ms":percentile(v,.99)} for k,v in groups.items()},
      "matrix":{"image_repeat_percent":[0,50,90],"branches":[1,2,4],"prefix":["short","long"],"resolution":[224,280],"pressure":["low","medium","high"],"method":"representative and single-factor; not Cartesian"},
      "claims":{"throughput_superiority":False,"logical_hit_is_saved_compute":False,
        "unmeasured_fields":["merged_vs_separated_equal_budget","lane_off_on_gpu_timeline",
          "pressure_policy_real_model_five_repeat","multimodal_ttft_itl"] + (["vision"] if not vision else [])},
      "records":records}
    (a.output/"results.json").write_text(json.dumps(summary,indent=2)+"\n")
    cold=[x["wall_ms"] for x in vision if x["mode"]=="off"];hit=[x["wall_ms"] for x in vision if x["mode"]=="on"]
    bars=[("Vision cold",statistics.median(cold) if cold else 0),("Vision cache hit",statistics.median(hit) if hit else 0)]
    scale=500/max([v for _,v in bars]+[1]);svg=['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="220">','<style>text{font:14px sans-serif}.title{font:bold 18px sans-serif}</style>','<text x="20" y="28" class="title">PBE V4 representative median wall time (5 repetitions)</text>']
    for i,(name,value) in enumerate(bars):
      y=65+i*65;svg += [f'<text x="20" y="{y+22}">{name}</text>',f'<rect x="160" y="{y}" width="{value*scale:.1f}" height="30" fill="#3977c5"/>',f'<text x="{170+value*scale:.1f}" y="{y+21}">{value:.1f} ms</text>']
    svg.append('</svg>');(a.output/"vision_cache_median.svg").write_text("\n".join(svg))
    print(a.output/"results.json")
if __name__=="__main__": main()
