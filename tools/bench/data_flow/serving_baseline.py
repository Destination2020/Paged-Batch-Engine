"""Three fresh-process cold/warm serving measurements with raw logs."""
import argparse
import json
import os
import pathlib
import statistics
import subprocess
p = argparse.ArgumentParser()
p.add_argument('--binary', required=True)
p.add_argument('--model', required=True)
p.add_argument('--tokenizer', required=True)
p.add_argument('--output', required=True)
p.add_argument('--sampling', action='store_true')
a = p.parse_args()
out = pathlib.Path(a.output)
out.mkdir(parents=True, exist_ok=True)
records = []
env = os.environ.copy()
env['KUIPER_TRACE_REQUEST_SAMPLING'] = '0'
env['KUIPER_BATCH_SAMPLE_CPU_FALLBACK'] = '0'
for warmup in (0, 1):
    for run in range(3):
        cmd = [a.binary, a.model, a.tokenizer, 'Explain KV cache in one sentence.',
               'What is continuous batching?', '--max-new-tokens=24',
               '--max-batched-tokens=128', '--kv-cache-memory-utilization=0.02',
               f'--warmup-rounds={warmup}', '--quiet=1', '--final-summary=1']
        if a.sampling:
            cmd += ['--temperature=0.8', '--top-k=16', '--top-p=0.9',
                    '--seed=123456789012345', '--ignore-eos=1']
        result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, timeout=120)
        logfile = out / f'warmup{warmup}_run{run}.log'
        logfile.write_text(result.stdout)
        result.check_returncode()
        line = next(x for x in result.stdout.splitlines() if x.startswith('FINAL_SUMMARY '))
        stats = {k:float(v) for k,v in (item.split('=',1) for item in line.split()[1:])}
        assert stats['completed_requests']==2 and stats['failed_requests']==0
        assert stats['token_gap_count']==46
        records.append(dict(warmup=warmup, run=run, command=cmd, log=str(logfile), metrics=stats))
summary = {}
for warmup in (0,1):
    summary[str(warmup)] = {}
    for key in ('throughput_tps','ttft_ms','token_gap_p95_ms','wall_ms'):
        values = [r['metrics'][key] for r in records if r['warmup']==warmup]
        summary[str(warmup)][key] = dict(mean=statistics.mean(values),
                stdev=statistics.stdev(values), min=min(values), max=max(values))
(out / 'results.json').write_text(json.dumps(dict(records=records,summary=summary),indent=2))
print(json.dumps(summary,indent=2))
