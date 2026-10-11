# 单 GPU 2P1D 实机验收：Qwen2.5-1.5B

2026-10-12，在 Linux / RTX 5090 / NVIDIA 610.57.04 上运行真实 BF16 权重。CUDA 12.8 构建，NCCL 关闭。两个 Prefill 与一个 Decode 使用同一个 1.5B 模型、同一个 2112 MiB CUDA IPC KV arena。三个计算进程各持有一份私有权重。

[部署与设计](../../single-gpu-2p1d-2026-10-12.md) · [完整结果](validation.json) · [回归结果](verification.json) · [退出记录](shutdown.json)

## 结果

- 三个实际输入的 greedy 输出与私有 KV 单进程基线逐字相同。
- **7040 / 5040 / 41-token 输入各生成恰好 8192 个 token**，混合负载中 D batch 峰值为 **4**。`ignore_eos=true` 用于持续生成压力，不评价长输出文本质量。
- 期间三个短请求全部在三条极长输出完成前结束。
- 长队列接受 33 个请求（含执行中请求），拒绝 5 个过载请求；短队列仍完成推理，首段文本延迟 22.45 ms。
- 排队及 Prefill 中取消通过；另一个请求在 Decode 生成 20 个 token 后取消，未计为推理失败。
- 正确性、混合负载、队列取消、Decode 取消之后，都恢复 **2112 MiB free / 0 leases**。D、P、owner 按依赖顺序退出，`graceful=true`。
- 默认 CTest **10/10 组通过**，包括共享 KV 跨布局复用、历史测试、新增准入隔离与配置测试；[MPS 注册及畸形 RPC 检查](rpc-validation.json)也通过。

## HTTP 响应时间

![实测请求时间线](request-timeline.svg)

| 极长输出请求 | 首段文本 / s | 请求完成 / s |
| --- | ---: | ---: |
| 7040-token 输入 → 8192-token 输出 | 24.991 | 255.296 |
| 5040-token 输入 → 8192-token 输出 | 8.110 | 231.158 |
| 41-token 输入 → 8192-token 输出 | 0.040 | 184.786 |

| 插入的短请求 | 首段文本 / ms | 请求完成 / ms |
| --- | ---: | ---: |
| 短请求 1 | 25.15 | 83.68 |
| 短请求 2 | 70.27 | 375.37 |
| 短请求 3 | 114.60 | 164.58 |

这些是一次混合负载的客户端单调时钟测量，包含模板编码、排队、Prefill、KV 交接、Decode 和传输。短请求输出上限 24 tokens，其中自然 EOS 可提前结束。三个短样本不能代表 P95/P99，也不是与 1P1D 或关闭 MPS 的性能对照。图中的首段文本时间可能包含 UTF-8 缓冲；它不等同于 GPU 首 token 采样时间。

## 进程与内存证据

MPS server **27827**；prefill-short: 27838 / prefill-long: 27857 / decode: 27876；共享池 owner **27823**。控制端原始 client list 在 `rpc-validation.json` 中，四个 CUDA 客户端均属于同一个 server。

混合负载累计 GPU KV 复制 **411041792 bytes**，Host KV 复制 **0 bytes**。P 源页在 D2D 完成后才释放；D 独立持有续写页。该路径使用全层 page 及连续 page 合并复制，不是零复制页接管。物理字节池可归还和重新分配；不同请求不能因此共享不兼容的 KV 内容。

本图是 HTTP 请求时间线。设备级跨进程同时执行证据来自[独立 MPS 实验](../mps-kernel-evidence-2026-10-12/README.md)，本次没有重新采集 2P1D kernel 时间线，不能据此声称三个模型 kernel 全程同时占用 SM。

## 重放与证据范围

按[部署文档](../../single-gpu-2p1d-2026-10-12.md)运行 `validate_split_pd.py --extreme-output-tokens 8192`。绘图需要 matplotlib：

```bash
python3 tools/serving/plot_split_pd.py \
  --report docs/benchmarks/single-gpu-2p1d-2026-10-12/validation.json \
  --output /tmp/2p1d-request-timeline.svg
```

远端最终原始记录：`/workspace/pbe-validation/split-pd-final/`。本地原始记录：`docs/data_flow_evidence/single-gpu-2p1d-2026-10-12/`，按仓库规则不提交。公开 JSON 移除了 CUDA IPC memory handle；未修改测量值。`source-hashes.json` 记录本次主要实现、验收与绘图文件，`artifact-manifest.json` 记录发布文件及原始归档哈希；哈希用于完整性校验，不代替复跑。

开发中的 4096-token 测试通过；第一次 8192-token 测试完成了混合负载，但验收脚本错误地将 RPC ID 与内部请求 ID 比较，在后续取消断言失败。修正后重新启动整套测试，本目录仅发布最终完整通过的一轮。旧日志保留在原始归档中，没有将失败轮次写成完整通过。
