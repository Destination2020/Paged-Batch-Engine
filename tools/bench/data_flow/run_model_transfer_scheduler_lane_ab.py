#!/usr/bin/env python3
"""Real VLM waiter on PBE scheduler demand while background D2H is active."""
import argparse,json,random,statistics,time
from pathlib import Path
from pbe_roles.coordinator import LanguageProcess,encode

def make_item(name,image):
    return {"request_id":name,"generation":1,"parts":[{"type":"image","path":str(image),"size":224},{"type":"text","text":"Describe this image briefly."}],"max_new_tokens":8,"timeout_ms":120000,"_deadline_monotonic_ns":time.monotonic_ns()+120_000_000_000}
def wire(item,e):
    return {"request_id":item["request_id"],"generation":1,"content":e["content"],"representation":e["representation"],"feature_content":e["feature_content"],"feature_representation":e["feature_representation"],"max_new_tokens":item["max_new_tokens"],"timeout_ms":120000}
def main():
    p=argparse.ArgumentParser();p.add_argument('--vision-endpoint',required=True);p.add_argument('--language-binary',type=Path,required=True);p.add_argument('--model-bin',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--data-endpoint',required=True);p.add_argument('--image-a',type=Path,required=True);p.add_argument('--image-b',type=Path,required=True);p.add_argument('--device',type=int,default=0);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=5);a=p.parse_args()
    ia,ib=make_item('a',a.image_a),make_item('b',a.image_b);_,ea=encode(a.vision_endpoint,ia);_,eb=encode(a.vision_endpoint,ib)
    def worker(lanes): return LanguageProcess([str(a.language_binary),str(a.model_bin),str(a.tokenizer),a.data_endpoint,str(a.device),str(64<<20),str(32<<20),'64','1','1' if lanes else '0'])
    workers={'lanes_off':worker(False),'lanes_on':worker(True)};counts={k:0 for k in workers};order=[m for _ in range(a.repeats) for m in workers];random.Random(20260913).shuffle(order);records=[]
    try:
      for order_index,mode in enumerate(order):
        rep=counts[mode];counts[mode]+=1;w=workers[mode]
        cleared=w.call({'op':'clear_prefix_cache'},timeout=180)
        if not cleared.get('ok') or cleared.get('recovery_objects') != 1:
          raise RuntimeError(f'{mode} cache clear failed: {cleared}')
        def infer(label,e):
          item=make_item(f'{mode}-{rep}-{label}',a.image_a if label=='a' else a.image_b);return w.call({'op':'infer','requests':[wire(item,e)]},timeout=180)
        first=infer('a',ea);demoted=w.call({'op':'demote_prefix'},timeout=180);second=infer('b',eb);submitted=w.call({'op':'demote_prefix_async'})
        started=time.perf_counter();result=infer('a-demand',ea);client_ms=(time.perf_counter()-started)*1000
        if not all(x.get('ok') for x in (first,demoted,second,submitted,result)) or not result.get('outputs'): raise RuntimeError(f'{mode} failed')
        out=result['outputs'][0];records.append({'mode':mode,'repetition':rep,'order':order_index,'client_ms':client_ms,'ttft_ms':out['ttft_ms'],'itl_ms':out['itl_ms'],'token_itl_ms':out['token_itl_ms'],'tokens':out['tokens'],'background_pages_submitted':submitted['admitted_pages'],'semantic_matched_tokens':out['semantic_matched_tokens']})
      status={m:w.call({'op':'status'}) for m,w in workers.items()}
    finally:
      for w in workers.values():w.close()
    summaries={m:{'repeats':a.repeats,'median_client_ms':statistics.median(r['client_ms'] for r in records if r['mode']==m),'median_ttft_ms':statistics.median(r['ttft_ms'] for r in records if r['mode']==m),'median_itl_ms':statistics.median(r['itl_ms'] for r in records if r['mode']==m)} for m in workers}
    result={'ok':all(r['background_pages_submitted']>0 and r['semantic_matched_tokens']>0 for r in records),'schema':'pbe.v4.model-transfer-scheduler-lane-ab.v1','only_ablation':'TransferScheduler.direction_lanes_enabled','same_stream_active_bytes':True,'repeats_each':a.repeats,'summaries':summaries,'records':records,'worker_status':status}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,sort_keys=True))
    if not result['ok']:raise SystemExit(1)
if __name__=='__main__':main()
