# V4 E4 同卡跨角色只读权重共享完成报告

日期：2026-09-13

## 结论

V4 第 19 节 E4-A 至 E4-F 已在真实 Qwen2.5-VL-3B BF16、真实 CUDA、真实独立常驻 Prefill/Decode 进程和真实 KV handoff 主路径上完成，E4 验收为 **1/1**，扩展 E1–E4 合计为 **4/4**。历史 M0–M9/E1–E3 证据未覆盖或重写；E4 开始前的 dirty workspace 已单独冻结并保留。

共享模式由 Data service 持有一个 6,173,974,564-byte CUDA IPC slab。两个 language worker 在首个 compute 前分别获取完整身份绑定的 lease，导入同一 allocation，并将 Qwen2 的 embedding、attention/MLP、norm、bias 和输出头直接绑定为借用的 BF16 GPU view。worker 不加载私有完整 GPU 权重。CUDA IPC 不提供硬件只读保护，本阶段兑现的是可信 worker 的只读合同与访问路径审查。

## 基线与清单

- 基线冻结：`data_flow_evidence/v4/E4/baseline_manifest.json`、`baseline_source_manifest.json`、`build_options.log`、`gpu_baseline.log`、`cuda_baseline.log`、`model_hashes.log`、`pd_baseline_validation.json`。
- 模型内容 SHA-256：`526ed2ed568a8d639211d9eb3cd91aaba56f67bbe89017ec2d2bf2bed6a0bf52`。
- 权重清单：436 个物理区间、437 个逻辑 view、435 个不可变算子 view、1 个显式 tied alias；文件覆盖完整、无越界、无未声明重叠；物理 allocation 数为 1。详见 `data_flow_evidence/v4/E4/weight_manifest.json`。
- 本地参考只记录 checkout/hash，没有复制 Python monkey patch；private 模式继续作为 oracle 和显式回退模式。shared 模式失败不会静默回退。

## 实现与真实绑定

- `data/shared_weight.*` 实现 absent/loading/ready/failed/draining 状态、分块上传、上传过程 SHA-256、producer event/fence、CUDA driver GPU UUID 校验、IPC 导入和析构。
- DataClient/NodeAgent 提供 acquire/release/stats，身份包含 service incarnation、allocation id/generation、模型内容、布局、dtype、bytes、GPU UUID、consumer incarnation、lease id 和 operation id；重试幂等。
- owner 用 64 MiB 有界 staging 上传一次，全部内容和布局校验完成后才发布 ready；共享权重与 KV allocation 分离。
- `model/shared_weight_binding.h`、Layer/Matmul/Qwen2 路径把 CPU mmap 的稳定文件 offset 映射到 imported GPU base，435 个真实算子直接读取共享 view；模型 capsule 在 view 生命周期内持有 import。
- worker 状态区分 physical weight bytes、logical/imported bytes、私有 workspace/激活、KV、Vision/staging、CUDA context/other 和实测设备占用。共享 worker 的物理权重预算为 0，owner 节点账本只收费一次。
- 正常退出顺序为停止 compute、同步、销毁模型 view、清空无后续用途的 CUDA allocator cache、关闭 import、释放权重 lease、释放 KV grant。owner 崩溃后存活 worker 在新 compute 前 fail-stop，不能继续使用旧代。

## 门禁结果

| 门禁 | 结果 | 证据 |
| --- | --- | --- |
| E4-A 基线/manifest | 通过 | `baseline_manifest.json`、`weight_manifest.json` |
| E4-B owner/协议 | 通过 | 单 allocation、单 upload、完整身份/幂等、ready fence |
| E4-C 真实算子绑定 | 通过 | 435 个 immutable operator view；embedding/attention/output 均为 shared；worker 无完整私有副本 |
| E4-D 生命周期/故障 | 通过 | `faults/results.json`：14 个事件、12 组检查全真 |
| E4-E 物理计费/准入 | 通过 | `results.json`、`memory_timeline.jsonl`、`checks.json` |
| E4-F 数值/回归/sanitizer/实验 | 通过 | 20-case、233-test 回归、3-worker memcheck、2×2×5 实验 |

故障矩阵覆盖：同路径内容变化、上传中强杀、content/dtype/bytes/layout/GPU 不匹配、reply-lost acquire 重放、重复 release、双 follower 并发、consumer SIGKILL、旧代迟到 release、owner crash fail-stop、新代恢复以及真实设备容量不足。最终状态无活跃共享权重 lease 或 KV grant，设备占用回到允许的基线差额内。

## 数值证据

private/shared 各自的冻结 20-case M5 门禁均通过，共 40 个模式输出。两模式 33/40 输出逐 token 相同；分叉保留在 case 05、09、15、19，没有将它们笼统解释为 tie。

首次相关分叉按相同 prompt、位置、RoPE 和输出历史重新采集完整 logits：

| case/step | max abs | mean abs | private/shared margin | top-2 union |
| --- | ---: | ---: | --- | --- |
| 05/2 | 0.187500 | 0.029490 | 0.000 / 0.125 | 374, 7952 |
| 09/2 | 0.171875 | 0.031168 | 0.000 / 0.000 | 374, 7952 |
| 15/4 | 0.187500 | 0.028747 | 0.125 / 0.000 | 6396, 11682 |
| 19/4 | 0.218750 | 0.037641 | 0.125 / 0.125 | 6396, 11682 |

全部满足冻结规则：same-history、max abs < 0.75、mean abs < 0.20、top-2 margin ≤ 0.25，分叉选择位于相同 top-2 union。机器可读证据为 `data_flow_evidence/v4/E4/numerics/checks.json` 和 `cross-validation.json`。

## 两组五次实验

每个 cell 5 次，固定随机顺序种子 20260913；每次启动并清理完整 Data/Vision/P/D/new-importer 拓扑。同一请求由不同 P/D PID 完成，handoff 有效，Decode 实际计算 prompt token 为 0，输出与 Prefill oracle 一致。

| 对照 | 模式 | 稳态/启动峰值 p50 MiB | 吞吐 req/s | 准入/提供 | 错误率 |
| --- | --- | ---: | ---: | ---: | ---: |
| 固定 KV | private | 16872 / 16872 | 0.671351 | 5/5 | 0% |
| 固定 KV | shared | 9608 / 9608 | 0.626692 | 5/5 | 0% |
| 固定总预算 | private | 16836 / 16836 | 0.287033 | 5/10 | 50% |
| 固定总预算 | shared | 14144 / 14144 | 0.714255 | 10/10 | 0% |

固定 KV 的共享吞吐下降约 6.7%，该负收益原样保留。固定总预算中，shared 将可用 KV 从 37,748,736 bytes 提升到 4,831,838,208 bytes，增量 4,794,089,472 bytes，不超过单份权重节省量；新增页由请求实际访问，准入由 5/10 提升到 10/10。原始结果见 `data_flow_evidence/v4/E4/results.json`，独立门禁见 `checks.json`。

## 回归与 sanitizer

- E4 单测：4/4 通过，覆盖 descriptor、损坏/不完整 payload、失败加载不可发布、借用 view/边界拒绝。
- 完整 `test_llm`：244 项中 233 通过、11 个条件性 skip、0 失败。原先依赖 `./tmp/test.bin` 的 3 项测试改为在 gtest 临时目录生成自包含夹具并实际通过。
- 条件性 skip：2 个 NCCL connector、3 个 Qwen2.5-VL golden/model、5 个需外部模型的 Qwen2 device/NCCL/checkpoint、1 个 Python bundle golden；它们不是 E4 通过项，E4 的真实模型/CUDA 路径由独立数值、故障、实验和 sanitizer 覆盖。
- 纯文本接口通过：GPU0 Prefill，GPU0/GPU1 Decode tokens 均与 oracle 相同，并完成跨 GPU IPC pool 导入。
- 最终二进制 compute-sanitizer：3 个 language worker 均为 `ERROR SUMMARY: 0 errors`、`LEAK SUMMARY: 0 bytes leaked`。使用 `--report-api-errors no` 仅排除 cuBLAS 初始化内部的 `cuCtxGetLimit` 能力探测；内存访问检查、full leak check 和 `--error-exitcode 99` 保持启用。

详见 `data_flow_evidence/v4/E4/regression/` 与 `data_flow_evidence/v4/E4/sanitizer/`。

## 重放入口

```bash
PYTHONPATH=python .venv/bin/python tools/bench/data_flow/run_e4_shared_weight_ab.py --build build-v3 --model-bin MODEL --tokenizer TOKENIZER --model-dir MODEL_DIR --image IMAGE --device 0 --repeats 5 --output docs/data_flow_evidence/v4/E4/results.json
python3 tools/bench/data_flow/validate_e4_shared_weight.py --evidence docs/data_flow_evidence/v4/E4 --model MODEL --output docs/data_flow_evidence/v4/E4/checks.json --required-repeats 5
PYTHONPATH=python .venv/bin/python tools/bench/data_flow/run_e4_fault_matrix.py --build build-v3 --model-bin MODEL --tokenizer TOKENIZER --model-sha256 SHA256 --device 0 --output docs/data_flow_evidence/v4/E4/faults/results.json
PYTHONPATH=python .venv/bin/python tools/bench/data_flow/run_e4_compute_sanitizer.py --build build-v3 --model-bin MODEL --tokenizer TOKENIZER --model-dir MODEL_DIR --image IMAGE --device 0 --output docs/data_flow_evidence/v4/E4/sanitizer/results.json
build-v3/test/test_llm --gtest_color=no
```

最终源码、二进制、模型/配置/tokenizer 和十个验收文件的 SHA-256 固定在 `data_flow_evidence/v4/E4/completion_manifest.json`。明确后置且未宣称完成的范围仍包括跨机 RDMA、多 provider slice 聚合、异构 TP 布局转换、逐层传输/计算重叠、流式角色图/背压，以及视频/音频/DiT/Draft-Verify。
