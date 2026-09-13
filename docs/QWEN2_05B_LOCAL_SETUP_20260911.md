# Qwen2-0.5B-Instruct 本地环境与冒烟测试

## 已验证环境

- Python: 3.12.3
- PyTorch: 2.9.1+cu128
- Transformers: 5.3.0
- Accelerate: 1.12.0
- CUDA Toolkit: 12.8
- GPU: NVIDIA H20-3e
- Hugging Face 模型目录: `/tmp/Paged-Batch-Engine-models/Qwen2-0.5B-Instruct`
- C++ BF16 权重: `/tmp/Paged-Batch-Engine-models/Qwen2-0.5B-Instruct.bf16.bin`
- CMake 构建目录: `/tmp/Paged-Batch-Engine-build-qwen05`

模型和构建产物放在本机 NVMe `/tmp` 下，避免占用代码仓库和共享文件系统空间；机器重启或清理 `/tmp` 后需要重新生成。

## 激活 Python 环境

```bash
cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hldy-nlp/MMA/cairuwei/chengbingsen/omni_flow_tts/Paged-Batch-Engine
source .venv/bin/activate
```

当前 `.venv` 使用 `--system-site-packages`，复用机器已有的 CUDA PyTorch 和 Transformers，只在虚拟环境内补装了 Accelerate：

```bash
uv venv .venv --clear --system-site-packages --python python3.12
uv pip install --python .venv/bin/python --no-deps accelerate==1.12.0
```

## 下载与导出

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2-0.5B-Instruct",
    local_dir="/tmp/Paged-Batch-Engine-models/Qwen2-0.5B-Instruct",
)
PY

.venv/bin/python tools/export_qwen2.py \
  /tmp/Paged-Batch-Engine-models/Qwen2-0.5B-Instruct.bf16.bin \
  --hf=/tmp/Paged-Batch-Engine-models/Qwen2-0.5B-Instruct \
  --dtype=bf16
```

## 构建 C++ 推理程序

```bash
cmake -S . -B /tmp/Paged-Batch-Engine-build-qwen05 \
  -DQWEN2_SUPPORT=ON \
  -DUSE_CPM=ON \
  -DKUIPER_BUILD_TESTS=OFF \
  -DKUIPER_BUILD_DEMOS=ON \
  -DKUIPER_ENABLE_NCCL=OFF \
  -DKUIPER_ENABLE_ZMQ=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -Dnlohmann_json_DIR=/tmp/Paged-Batch-Engine-build-qwen05/_deps/nlohmann_json-build

cmake --build /tmp/Paged-Batch-Engine-build-qwen05 \
  --target qwen_instruct_infer -j 16
```

## C++ 冒烟测试

```bash
CUDA_VISIBLE_DEVICES=0 \
  /tmp/Paged-Batch-Engine-build-qwen05/demo/qwen_instruct_infer \
  /tmp/Paged-Batch-Engine-models/Qwen2-0.5B-Instruct.bf16.bin \
  /tmp/Paged-Batch-Engine-models/Qwen2-0.5B-Instruct/tokenizer.json \
  '用一句话解释什么是KV cache。'
```

2026-09-11 实测结果：

```text
Assistant: KV cache是一种用于存储和检索键值对的缓存技术，它通过将键值对存储在内存中，然后在需要时从内存中读取，从而提高查询速度。
steps:76
duration:0.218239
steps/s:348.242
```

`qwen_instruct_infer` 现在逐 token 建立 prompt embedding。这样 prompt 长度不再受 serving workspace 的 active-token 容量限制，同时仍沿用原有逐 token prefill 路径。
