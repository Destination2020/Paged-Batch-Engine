"""Capture local model/build provenance without network or environment secrets."""
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parents[3]
model = pathlib.Path('/tmp/Paged-Batch-Engine-models/Qwen2-0.5B-Instruct')
build = pathlib.Path('/tmp/Paged-Batch-Engine-build-qwen05')
def command(args):
    p = subprocess.run(args, capture_output=True, text=True)
    return dict(command=args, exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
def fingerprint(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())
packages = {}
for name in ['torch', 'transformers', 'accelerate', 'safetensors', 'tokenizers', 'numpy']:
    try:
        d = importlib.metadata.distribution(name)
        packages[name] = dict(version=d.version, location=str(d.locate_file('')))
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
files = [model / n for n in ['config.json','tokenizer.json','tokenizer_config.json',
                             'generation_config.json','model.safetensors']]
files.append(model.parent / 'Qwen2-0.5B-Instruct.bf16.bin')
cache = (build / 'CMakeCache.txt').read_text().splitlines()
selected = [line for line in cache if not line.startswith(('#','//')) and any(
    key in line.split('=',1)[0] for key in ['CMAKE_CXX_COMPILER','CMAKE_CUDA_',
      'CMAKE_BUILD_TYPE','QWEN2_SUPPORT','KUIPER_','USE_CPM','_SOURCE_DIR'])]
source_paths = subprocess.check_output(
    ['git','ls-files','--cached','--others','--exclude-standard','infMain','test','demo',
     'tools/bench/data_flow','CMakeLists.txt','cmake'], cwd=root).decode().splitlines()
source_fingerprints = [fingerprint(root / p) for p in sorted(set(source_paths)) if (root / p).is_file()]
dependencies = {}
for source in sorted((build / '_deps').glob('*-src')):
    if (source / '.git').exists():
        dependencies[source.name] = command(['git','-C',str(source),'rev-parse','HEAD'])
    elif (source / 'CMakeLists.txt').exists():
        dependencies[source.name] = dict(cmake=fingerprint(source / 'CMakeLists.txt'),
            version_lines=[line for line in (source / 'CMakeLists.txt').read_text().splitlines()
                           if 'VERSION' in line][:20])
result = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
    packages=packages, cpp_dependencies=dependencies, source_files=source_fingerprints,
    build_artifacts=[fingerprint(build / p) for p in ["lib/libllama.so", "demo/serving_qwen"]],
    model_files=[fingerprint(p) for p in files],
    model_revision='Unknown upstream revision; exact local bytes identified by SHA256',
    cmake_cache=selected, source_head=command(['git','rev-parse','HEAD']),
    source_status=command(['git','status','--short']),
    compiler=command(['g++','--version']), cuda=command(['nvcc','--version']),
    gpu=command(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv']),
    topology=command(['nvidia-smi','topo','-m']))
(root / 'docs/data_flow_evidence/environment_manifest.json').write_text(json.dumps(result, indent=2))
