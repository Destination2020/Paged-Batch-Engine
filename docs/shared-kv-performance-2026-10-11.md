# 共享 KV 分配与回收性能优化（2026-10-11）

本次在本地 V4 集成分支完成非 MPS 对照测试及优化。起点为 `ea98b1a` 加上一轮验证工具和可选 trace；优化前的 `libllama.so` 单独保留，利用 `LD_LIBRARY_PATH` 让同一基准程序加载旧库或新库，Owner 也加载对应库。两份库的 SHA-256 见 [libraries.json](benchmarks/shared-kv-performance-2026-10-11/libraries.json)。

环境仍是 WSL2、RTX 4060 Laptop 8 GiB、CUDA 12.6。测试关闭 MPS，性能测试关闭 `PBE_SHARED_KV_TRACE`。这些结果描述共享池容量压力，不代表将驱动可用显存耗尽、触发系统级迁移时的行为。

## 改动

- `SharedKVArenaClient` 复用 REQ socket 和 ZMQ context，通过已有互斥锁串行化完整请求/应答。传输失败后连接失效并报错，不能自动重放可能已提交的 acquire/release。
- 为页建立 ID 索引和有序空闲集合；后续层领取已预留页时直接定位。维护 Chunk 空页数和租用字节计数，避免分配、退休和容量查询反复扫描所有页。
- 待回收队列仅包含已经没有任何层引用的页。它们在同一 CUDA stream 上按记录顺序查询 event，首个未完成事件之后无需继续查询。
- 全层页在最后一个层引用释放后记录一个 fence，替代每层一个 fence；只回收已完成的事件并放入事件池复用。部分层仍活跃时整个页不能复用。此规则依赖现有单 worker/group stream 和 compute/I/O 租约生命周期合同。
- 本地硬配额不足一个全层页时直接返回失败；不足目标 Chunk 但足够单页时直接申请单页，省去已知会失败的 RPC。

空 Chunk 仍及时归还 Owner，保持另一模型可立即申请的能力。没有引入后台显存驱逐、CPU swap、跨模型 KV 数值复用或 MPS 并发假设。

## 分配器微基准

新增 `demo/shared_kv_bench.cpp` 和 `tools/serving/benchmark_shared_kv.py`。同样的 84 MiB 容量、BF16、16-token 页，两种 Qwen 布局；每个样本分配 4 个全层页，提交相同 K/V memset 并等待完成，再释放及完成回收。初始化不计时。每个条件 3 次独立进程运行，每次丢弃 3 个预热样本，记录 30 个样本，总计 90 个。各轮交换 private/shared 顺序。

下表为中位数，单位 µs；“回收”包含释放调用、stream 等待和 collect，不只是将引用计数减一。

| 布局 / 已占用页 | 分配：旧 → 新 | 完整回收：旧 → 新 |
| --- | ---: | ---: |
| 0.5B / 0（空池，每轮重新租 Chunk） | 753.78 → 190.31 | 803.08 → 229.28 |
| 1.5B / 0（空池，每轮重新租 Chunk） | 686.12 → 194.44 | 776.59 → 233.06 |
| 0.5B / 403 of 448（约 90% 页占用） | 473.76 → 5.24 | 533.53 → 8.04 |
| 1.5B / 172 of 192（约 90% 页占用） | 997.78 → 172.01 | 1091.91 → 222.32 |

两个高占用条件的差别由 Chunk 边界解释：0.5B 每 Chunk 11 页，403 个常驻页后已有 Chunk 恰好还剩 4 个空页；1.5B 每 Chunk 5 页，172 个常驻页后只剩 3 页，4 页请求仍需一次新 Chunk RPC。因此不能把约 5 µs 的本地路径当作所有共享分配的耗时。

空池分配下降约 72–75%，完整回收下降约 70–71%。满池且本模型配额也已耗尽时，失败返回从约 1.3–1.4 ms 降到不足 1 µs，省掉了两次必然失败的 RPC；不对接近计时器分辨率的结果宣称精确倍数。这不代表其他模型占满全局池时也能在本地判定。

独立池在这些条件下分配/释放仍约 1–3 µs，显著快于需要 Owner RPC 的共享路径。优化后共享池依然有管理成本。合成 GPU 写入耗时有波动，且包含大量 CUDA memset 提交，不作为 attention 内核性能或端到端吞吐提升的证据。

完整 p50/p95 和原始样本：[micro-before.json](benchmarks/shared-kv-performance-2026-10-11/micro-before.json)、[micro-after.json](benchmarks/shared-kv-performance-2026-10-11/micro-after.json)、[micro-comparison.json](benchmarks/shared-kv-performance-2026-10-11/micro-comparison.json)。

## 真实模型请求延迟

相同英文提示词、greedy、最多 24 个输出 token，统一 16-token workspace/prefill 配置。每个模型/条件丢弃 1 次预热，记录 10 次串行 SSE 请求；共享部署中两个模型同时驻留。独立 KV 基线每次只加载一个模型，因此不能把它与共享部署的差异全部归因于 allocator。

高占用条件由一个不访问 GPU 的容量持有者租用 512 MiB Arena 中的 480 MiB（93.75%），为真实模型留下 32 MiB；这隔离了容量占用影响，不包含另一个模型竞争计算资源或等待显存释放的时间。计时样本均与各自独立基线逐字对照通过，测量后 Arena 全部归还。

下表为中位数，单位 ms。“首文本”是客户端收到第一个非空 UTF-8 SSE 文本的时间，不是内核级首 token 计时。

| 模型 / 条件 | 首文本：旧 → 新 | 整个响应：旧 → 新 |
| --- | ---: | ---: |
| 0.5B / 独立 KV 控制组 | 26.73 → 27.86 | 207.89 → 230.30 |
| 0.5B / 共享空池 | 37.51 → 36.29 | 245.98 → 243.56 |
| 0.5B / 共享池 93.75% 被持有 | 36.84 → 33.79 | 243.48 → 241.21 |
| 1.5B / 独立 KV 控制组 | 67.43 → 84.46 | 541.72 → 681.46 |
| 1.5B / 共享空池 | 70.54 → 64.35 | 505.15 → 475.58 |
| 1.5B / 共享池 93.75% 被持有 | 72.15 → 63.68 | 525.21 → 482.00 |

本次样本中，只要剩余容量够用，接近满池没有带来明显额外延迟。但独立控制组前后也有明显波动，样本数较少，不能将整个响应的变化都归因于此次优化，不能据此承诺稳定的端到端加速百分比。本次没有测量标准逐 token ITL，也没有把 SSE 文本事件数当作 token 吞吐量。

原始样本和 p50/p95：[real-before.json](benchmarks/shared-kv-performance-2026-10-11/real-before.json)、[real-after.json](benchmarks/shared-kv-performance-2026-10-11/real-after.json)、[real-comparison.json](benchmarks/shared-kv-performance-2026-10-11/real-comparison.json)。10 个样本采用 nearest-rank p95，即本组最大值，不能代表长期尾延迟。

## 安全性验证

21 MiB 两进程探针增加了“仅释放第一层，其余层仍活跃”及“只分配 layer 0 后回滚未消费的层预留”检查。5 轮异构布局占满、覆盖写入、逐字节校验、归还及跨模型复用通过：122 次 acquire 对应 122 次 release，最终无租约，活跃区间不重叠。持有 compute/I/O 租约或 CUDA fence 未完成时均不能提前复用。

证据：[reuse.json](benchmarks/shared-kv-performance-2026-10-11/reuse.json)、[reuse-owner.jsonl](benchmarks/shared-kv-performance-2026-10-11/reuse-owner.jsonl)。

新库真实模型还通过了 6 个并行提交请求、5 轮共 10 次交替请求的输出对照和完全归还检查，以及 UTF-8 流式输出、取消隔离、未知模型拒绝、杀死一个 worker 后整个 GPU domain 退出检查。

重新运行 C++ 主回归：212 项中 200 通过、12 跳过、0 失败，包含本地真实模型 checkpoint 恢复。Python 部署合同 4 项及 V4 数据层 5 项通过。独立 CUDA IPC memory/event、V4 compute/I/O 租约、旧代次/跨层句柄拒绝及延迟回收探针通过。12 个跳过项依赖双 GPU、NCCL、VLM fixtures 或 Python golden 数据。本次测试启动的进程均已停止。

## 复现

```bash
cmake -S . -B build-mps
cmake --build build-mps --target shared_kv_bench shared_kv_owner shared_kv_reuse_probe serving_qwen -j4
python3 tools/serving/benchmark_shared_kv.py --output-dir /tmp/pbe-kv-bench-new
python3 tools/serving/validate_kv_reuse.py --rounds 5 --output-dir /tmp/pbe-kv-safe-new
GLOG_logbufsecs=0 PBE_SHARED_KV_TRACE=0 python3 tools/serving/validate_multi_model.py \
  --config docs/benchmarks/shared-kv-reuse-2026-10-11/config.json \
  --backend process --timing-rounds 10 --reuse-rounds 5 --test-worker-failure \
  --output-dir /tmp/pbe-kv-real-new
```

与旧库对照时，给微基准传入 `--library-dir /path/to/before/lib`；真实模型工具使用对应 `LD_LIBRARY_PATH`。旧库必须在优化前保存或从旧源码独立构建，不能将当前库复制一份充当旧库。输出目录使用新路径，模型路径按本机配置调整。两种真实模型统一使用 16-token workspace/prefill 配置，避免显存压力触发不同分块形状而混淆结果。
