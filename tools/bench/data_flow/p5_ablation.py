"""Run fixed-budget P5 ablations and retain every raw process log."""
import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--serving-binary', required=True)
parser.add_argument('--test-binary', required=True)
parser.add_argument('--qwen-test-binary', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--tokenizer', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--repetitions', type=int, default=3)
args = parser.parse_args()
out = pathlib.Path(args.output)
out.mkdir(parents=True, exist_ok=True)

def run(command, log_name, env=None, timeout=180):
    result = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=timeout)
    (out / log_name).write_text(result.stdout)
    result.check_returncode()
    return result.stdout

def fields(line):
    values = {}
    for key, value in re.findall(r'(\w+)=([^\s]+)', line):
        try:
            values[key] = float(value)
        except ValueError:
            values[key] = value
    return values

micro_text = run([args.test_binary,
                  '--gtest_filter=P5AblationTest.FixedBudgetReportsLayoutTransactionAndSingleFlightCosts'],
                 'microbench.log')
micro = fields(next(line for line in micro_text.splitlines()
                    if line.startswith('P5_ABLATION ')))

checkpoint_env = os.environ.copy()
checkpoint_env['PBE_QWEN2_TEST_TOKENIZER'] = args.tokenizer
checkpoint_env['PBE_QWEN2_TEST_MODEL'] = args.model
checkpoint_text = run(
    [args.qwen_test_binary,
     '--gtest_filter=Qwen2DeviceInitTest.CheckpointResumeMatchesUninterruptedSampledTokens',
     f'--gtest_repeat={args.repetitions + 1}'], 'checkpoint_model.log',
    checkpoint_env, timeout=300)
checkpoint_rows = [fields(line) for line in checkpoint_text.splitlines()
                   if line.startswith('P5_CHECKPOINT ')]
assert len(checkpoint_rows) == args.repetitions + 1
checkpoint_warmup = checkpoint_rows.pop(0)

prompt = ('Paged KV cache stores attention keys and values in fixed pages. '
          'A scheduler reuses immutable prefix pages and restores cold pages. ') * 4
modes = {
    'recompute': ['--radix-cache=off'],
    'gpu_warm': ['--radix-cache=on'],
    'host_restore': ['--radix-cache=on', '--host-cache=1',
                     '--host-cache-bytes=67108864', '--host-cache-pages=128',
                     '--host-cache-inflight-pages=2',
                     '--host-demote-after-warmup=1'],
}
serving_rows = {mode: [] for mode in modes}
for mode, options in modes.items():
    for repetition in range(args.repetitions):
        command = [args.serving_binary, args.model, args.tokenizer, prompt, prompt,
                   '--max-new-tokens=8', '--ignore-eos=1',
                   '--max-batched-tokens=128', '--prefill-chunk-cap=64',
                   '--kv-cache-memory-utilization=0.02', '--warmup-rounds=1',
                   '--quiet=1', '--final-summary=1'] + options
        text = run(command, f'{mode}_{repetition}.log', timeout=180)
        row = fields(next(line for line in text.splitlines()
                          if line.startswith('FINAL_SUMMARY ')))
        assert row['completed_requests'] == 2 and row['failed_requests'] == 0
        serving_rows[mode].append(row)

metric_names = ['throughput_tps', 'ttft_ms', 'token_gap_p95_ms', 'wall_ms',
                'radix_cache_tokens_reused', 'host_restored_blocks']
serving_summary = {}
for mode, rows in serving_rows.items():
    serving_summary[mode] = {}
    for metric in metric_names:
        values = [row.get(metric, 0.0) for row in rows]
        serving_summary[mode][metric] = {
            'mean': statistics.mean(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values), 'max': max(values),
        }

baseline = serving_summary['recompute']
for mode in ('gpu_warm', 'host_restore'):
    serving_summary[mode]['delta_vs_recompute_percent'] = {
        metric: ((serving_summary[mode][metric]['mean'] /
                  baseline[metric]['mean'] - 1.0) * 100.0)
        for metric in ('throughput_tps', 'ttft_ms', 'token_gap_p95_ms', 'wall_ms')
        if baseline[metric]['mean'] != 0
    }

checkpoint_summary = {}
for metric in ('baseline_ms', 'resumed_ms', 'checkpoint_ms'):
    values = [row[metric] for row in checkpoint_rows]
    checkpoint_summary[metric] = {
        'mean': statistics.mean(values),
        'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
        'min': min(values), 'max': max(values),
    }
checkpoint_summary['resumed_delta_percent'] = (
    checkpoint_summary['resumed_ms']['mean'] /
    checkpoint_summary['baseline_ms']['mean'] - 1.0) * 100.0

result = {
    'fixed_budget': {
        'repetitions': args.repetitions,
        'max_new_tokens': 8,
        'max_batched_tokens': 128,
        'prefill_chunk_cap': 64,
        'kv_cache_memory_utilization': 0.02,
        'host_cache_bytes': 67108864,
        'host_cache_pages': 128,
        'host_cache_inflight_pages': 2,
        'transfer_aging_priority_lane_quanta': 4,
    },
    'microbench': micro,
    'checkpoint_model': {'warmup': checkpoint_warmup, 'runs': checkpoint_rows,
                         'summary': checkpoint_summary},
    'serving': {'runs': serving_rows, 'summary': serving_summary},
}
(out / 'results.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
