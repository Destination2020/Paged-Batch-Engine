"""Verify actual sampling inputs and report first output divergence by handle."""
import json
import pathlib
import re
import sys
from collections import defaultdict
root = pathlib.Path(sys.argv[1])
report = []
for log in sorted(root.glob('server_*.log')):
    groups = defaultdict(list)
    for line in log.read_text().splitlines():
        if 'REQUEST_SAMPLE ' not in line:
            continue
        values = dict(re.findall(r'(\w+)=([\w]+)', line.split('REQUEST_SAMPLE ',1)[1]))
        groups[int(values['handle'])].append({k: (v if k=='backend' else int(v)) for k,v in values.items()})
    assert len(groups) == 5, (log, len(groups))
    reference = {}
    outputs = {}
    comparisons = []
    for handle, rows in groups.items():
        assert [r['counter'] for r in rows] == list(range(24)), (log,handle)
        key = rows[0]['seed']
        stream = [r['uniform24'] for r in rows]
        tokens = [r['token'] for r in rows]
        if key in reference:
            assert stream == reference[key], (log,handle,'random input changed')
            divergence = next((i for i,(a,b) in enumerate(zip(tokens, outputs[key])) if a!=b), None)
            comparisons.append(dict(handle=handle, first_token_divergence=divergence,
                                    batch_sizes=sorted(set(r['batch'] for r in rows))))
        else:
            reference[key] = stream
            outputs[key] = tokens
    report.append(dict(log=str(log), requests=len(groups), random_inputs_equal=True,
                       comparisons=comparisons))
assert len(report)==3
(root / 'trace_analysis.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
