"""Repeat actual HTTP sampling across fresh engines; preserve raw responses."""
import argparse
import concurrent.futures
import json
import pathlib
import os
import subprocess
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--binary', required=True)
p.add_argument('--model', required=True)
p.add_argument('--tokenizer', required=True)
p.add_argument('--output', required=True)
p.add_argument('--radix-cache', choices=['on','off'], default='on')
p.add_argument('--cpu-sampling', action='store_true')
p.add_argument('--trace-sampling', action='store_true')
p.add_argument('--port', type=int, default=18972)
a = p.parse_args()
out = pathlib.Path(a.output)
out.mkdir(parents=True, exist_ok=True)
base = f'http://127.0.0.1:{a.port}'
records = []

def request(seed, prompt='Explain KV cache in one sentence.'):
    payload = dict(prompt=prompt, max_new_tokens=24, ignore_eos=True,
                   temperature=0.8, top_k=16, top_p=0.9, seed=seed)
    start = time.monotonic()
    req = urllib.request.Request(base + '/generate',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        body = json.load(response)
    return dict(payload=payload, response=body, wall_ms=(time.monotonic()-start)*1000)

for run in range(3):
    command = [a.binary, a.model, a.tokenizer, '--online-server=1',
        '--listen-host=127.0.0.1', f'--listen-port={a.port}',
        '--max-batched-tokens=128', '--kv-cache-memory-utilization=0.02',
        '--warmup-rounds=0', '--quiet=1', f'--radix-cache={a.radix_cache}']
    with (out / f'server_{run}.log').open('w') as log:
        child_env = os.environ.copy()
        child_env['GLOG_logtostderr'] = '1'
        child_env['KUIPER_BATCH_SAMPLE_CPU_FALLBACK'] = '1' if a.cpu_sampling else '0'
        child_env['KUIPER_TRACE_REQUEST_SAMPLING'] = '1' if a.trace_sampling else '0'
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=child_env)
        try:
            deadline = time.monotonic() + 90
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'engine exited: {proc.returncode}')
                try:
                    with urllib.request.urlopen(base + '/health', timeout=1) as r:
                        if r.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('engine readiness')
                time.sleep(0.2)
            single = [request(123456789012345), request(123456789012345)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                batched = list(pool.map(request, [123456789012345, 987654321, 123456789012345]))
            records.append(dict(run=run, command=command, cpu_sampling=a.cpu_sampling, single=single, concurrent=batched,
                repeat_equal=single[0]['response'] == single[1]['response'],
                concurrent_equal=all(batched[i]['response']==single[0]['response'] for i in (0,2))))
            (out / 'results.json').write_text(json.dumps(records, indent=2))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
print(json.dumps([dict(run=r['run'], repeat_equal=r['repeat_equal'],
    concurrent_equal=r['concurrent_equal']) for r in records], indent=2))
if not all(r['repeat_equal'] for r in records):
    raise SystemExit('same-seed sequential reproducibility failed')
