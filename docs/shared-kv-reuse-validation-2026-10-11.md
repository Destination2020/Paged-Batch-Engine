# 非 MPS 多模型共享 KV 验证（2026-10-11）

在本地 `feat/v4-cuda-mps-shared-kv` 分支、V4 集成提交 `ea98b1a` 基础上验证。结论：普通多进程 CUDA IPC 路径可以分配、使用、释放共享 KV 容量，并由另一模型重新使用同一显存区间；引用、compute/I/O 租约及 CUDA fence 会阻止提前回收。本次未启用 CUDA MPS。

环境为 WSL2、单张 RTX 4060 Laptop 8 GiB、CUDA 12.6，两个真实模型为本地 BF16 Qwen2.5-0.5B-Instruct 和 Qwen2.5-1.5B-Instruct。测试结束后已关闭本次启动的服务进程。

## 1. 占满、归还、跨模型覆盖写入

新增 `shared_kv_reuse_probe` 和 `tools/serving/validate_kv_reuse.py`，两个 exec 子进程通过 CUDA IPC 映射同一个 21 MiB Arena，进程在全部轮次中保持存活。这里使用真实模型的 KV 布局参数，但不加载模型权重。

| 项目 | 0.5B 布局 | 1.5B 布局 |
| --- | --- | --- |
| 层数 / KV heads / head size | 24 / 2 / 64 | 28 / 2 / 128 |
| 每个全层页字节数（16 tokens、BF16） | 196,608 | 458,752 |
| 占满 21 MiB 的页数 | 112 | 48 |

连续 3 轮完成以下检查：

1. A 占满 Arena，逐层写入 K/V 并回读校验全部 21 MiB；B 分配失败。
2. A 获取 compute/I/O 租约后释放请求引用，B 仍不能分配。
3. A 释放租约，但通过 stream host callback 阻止退休 fence 完成，B 仍不能分配。
4. 解除阻塞并等待 CUDA stream 完成，收集退休页；Arena 空闲字节恢复为 21 MiB。
5. B 使用自己的异构页布局占满同一 Arena，覆盖写入不同 K/V 字节并逐字节回读校验；最后全部释放。

两种布局的页偏移分别完整覆盖 `[0, 21 MiB)`，因此验证的是同一显存范围的复用。Owner 日志记录 72 次 acquire 和 72 次 release，无同时存活的重叠区间，最终无租约。60 次 acquire 使用了曾由另一布局使用过的区间；该数字是事件数，不是独立物理容量。

证据：[capacity-reuse.json](benchmarks/shared-kv-reuse-2026-10-11/capacity-reuse.json)、[capacity-owner.jsonl](benchmarks/shared-kv-reuse-2026-10-11/capacity-owner.jsonl)。

## 2. 两个真实模型的生成及交替复用

使用两个常驻模型 worker 和一个 512 MiB Arena，普通 `process` 后端，每模型同时处理一个请求，关闭 radix cache，BF16 KV。

- 6 个并行提交请求（每模型 3 个，中英文及长提示词）与相同配置的独立 KV 基线逐字一致。
- 随后交替调用 A、B 共 5 轮、10 个请求，输出仍匹配独立基线；每个请求后均确认 `free_bytes=536870912`、`lease_count=0`。
- 全部 25 次 Chunk acquire 对应 25 次 release，活跃区间无重叠；17 次 acquire 复用了曾由另一模型使用过的区间。
- 日志尾部 lease 16–25 明确按 A→B→A→B 交替使用 `offset=0`：A 每次租用 2,162,688 bytes，B 每次租用 2,293,760 bytes，租约 ID/generation 递增。
- 中文 UTF-8 SSE、取消路由隔离、未知模型拒绝，以及杀死一个 worker 后整个 GPU domain 退出均通过。

证据：[real-model-validation.json](benchmarks/shared-kv-reuse-2026-10-11/real-model-validation.json)、[real-model-owner.jsonl](benchmarks/shared-kv-reuse-2026-10-11/real-model-owner.jsonl)、[real-model-reuse-audit.json](benchmarks/shared-kv-reuse-2026-10-11/real-model-reuse-audit.json)。

### 显存压力导致的对照配置差异

首次使用 128-token 请求配置时，5/6 请求匹配，1.5B 的中文短回答不同；串行提交也复现。日志确认独立基线实际 workspace 为 128 token，而共享部署因当时空闲显存不足，被既有安全逻辑缩到 16 token。这不是同一 prefill 分块条件下的对照。

将两边的 `max_batched_tokens` 和 `prefill_chunk_cap` 都显式设为 16 后，日志确认四个模型实例的实际 workspace 均为 16，上述全部对照及重复复用通过。结果支持该差异与分块配置有关，不将不同分块大小下的逐字一致性作为已验证能力，也未通过修改输出断言来绕过失败。

本机可用显存约 6086 MiB，验证配置将 reserve 从默认 768 改为 384 MiB；模型私有硬预算仍为 1536/3584 MiB，共享 Arena 仍为 512 MiB。仓库默认配置未改。成功配置：[config.json](benchmarks/shared-kv-reuse-2026-10-11/config.json)。失败的并发/串行输出及两边 workspace 日志也保存在同一证据目录，文件名以 `unmatched-workspace-` 开头。

## 3. 其他回归

| 验证 | 本次结果 |
| --- | --- |
| C++ `test_llm --gtest_filter='*-test_*.*'`，提供本地 checkpoint 模型 | 212 项：200 通过，12 跳过，0 失败 |
| Python 部署合同 / V4 数据层 | 4 / 5 项通过 |
| 独立 CUDA IPC memory/event、延迟回收、V4 租约及旧句柄拒绝探针 | 通过，Arena 完整归还 |
| V4 `serving_external_kv.sh` | 2 请求成功，最终 64/64 页空闲、0 grant、0 lease |

跳过项依赖双 GPU、NCCL、VLM fixtures 或 Python golden 数据。未宣称完整硬件/多模态验收，也未重新验证独立 `test/cache_core` 构建目标。本次未做 Transformers 数值对照或吞吐性能结论。

## 4. 本地复现

从仓库根目录执行，输出目录需使用新的路径；配置中的绝对模型路径需与本机一致：

```bash
cmake -S . -B build-mps
cmake --build build-mps --target shared_kv_reuse_probe shared_kv_owner serving_qwen -j4
python3 tools/serving/validate_kv_reuse.py --rounds 3 --output-dir /tmp/pbe-reuse-repeat
GLOG_logbufsecs=0 PBE_SHARED_KV_TRACE=1 python3 tools/serving/validate_multi_model.py \
  --config docs/benchmarks/shared-kv-reuse-2026-10-11/config.json \
  --backend process --reuse-rounds 5 --test-worker-failure \
  --output-dir /tmp/pbe-real-reuse-repeat
python3 tools/serving/validate_kv_reuse.py \
  --audit-trace /tmp/pbe-real-reuse-repeat/domain/kv-owner.log
```

`PBE_SHARED_KV_TRACE=1` 是本次新增的可选 Owner acquire/release JSONL 日志，默认关闭，不输出 CUDA IPC handle。复用审计检查单一 owner epoch、租约身份匹配、活跃区间不重叠及最终零租约。

这里的“释放”是归还给共享 Arena，另一模型无需卸载权重或重启进程即可使用。Owner 仍持有整块 cudaMalloc，因此请求结束后 `nvidia-smi` 不会因此减少整个 Arena 的占用；Owner 退出才释放底层分配。不同模型共享的是显存容量，各自重新计算 KV 内容，不能直接将 A 的 KV 数值用于 B。本次关闭 radix cache，未验证保留前缀缓存时的自动回收策略，也未验证故障后继续服务的恢复机制。
