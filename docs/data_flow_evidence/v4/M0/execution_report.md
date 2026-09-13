# V4 M0 基线复验报告

日期：2026-09-12

## Checkout 与冻结证据

- HEAD：`f3597a763b6cd5cf60b826baa2dc94395c3d02fd`
- 继承的 tracked binary diff SHA-256：`71942cb3b21367fdd3171139f86d9c84b8befcc44df59ccc9f68c41e0477532b`
- 任务开始前源码清单摘要 SHA-256：`13bdac6c6cf0e7b6aebe6482824900177a8f6ad2cbbeb1b44eb70421e02a828c`
- 逐文件源码、模型、工具链、GPU 与拓扑清单：`manifest.json`
- 所有命令、退出码、耗时与日志索引：`checks.json`

原有 `demo/main_qwen2Instruct.cpp` 及全部 V3 dirty/untracked 文件均保留，没有回退或覆盖。

## 复验结果

- CPU cache/ownership：65/65 通过，包括 10,000 次 checkpoint churn 和 100,000 次 request handle churn。
- ASan+UBSan+LeakSanitizer：65/65 通过。系统 GCC sanitizer 符号链接损坏且账号无 sudo；从当前 MTOS 官方仓库下载匹配的 `libasan-10.3.1-20`、`libubsan-10.3.1-20` RPM，仅解压到 `/tmp/Paged-Batch-Engine-v4-m0-sanitizer-runtime` 使用。
- CUDA 定向数据流：75 通过、2 skip；skip 均为构建明确关闭 NCCL。
- CUDA 广域回归：170 通过、2 skip。`test_load.*` 因缺失仓库外 `../stories15M.bin` fixture 明确排除。
- 真实 Qwen2-0.5B BF16：源码重新构建通过；双请求 serving 冒烟通过；双 GPU P2P 与单模型 greedy token 一致；两次 checkpoint 后 sampled tokens 与不中断路径一致。

## 本轮修复

1. `Qwen2Model::init_mem` 的显式 KV block 分支曾在 `total_blocks` 声明前赋值，并跳过 GPU 权重 materialize。显式容量选择已移至 KV sizing，权重始终先物化。这个错误只在 `QWEN2_SUPPORT=ON` 的真实模型构建中暴露。
2. 旧 CUDA build 配置为 `compute_86/sm_86`，H20 上需要 driver JIT CUDA 12.8 PTX，但当前 driver 只支持 CUDA 12.4，导致后续 kernel 返回 222 并伪装成 SplitKV 数值失败。M0 固定 `CMAKE_CUDA_ARCHITECTURES=90` 后，SplitKV 与 baseline 最大误差约 `6.7e-8`，20 轮重复通过。
3. Paged/SplitKV 测试改由 `CudaConfig` 唯一销毁 stream，去掉重复 destroy；测试在临时 K/V buffer 归还 caching allocator 前等待 scatter 完成，并显式检查 stream 创建、kernel launch 与同步错误。

## 口径与后续入口

旧 hex-JSON/ZMQ KV handoff 继续作为 compatibility/oracle，不作为 V4 主线传输性能代表。当前 M0 验收完成，V4 总进度为 1/10；M1 从真实 `Qwen/Qwen2.5-VL-3B-Instruct` checkpoint、processor 与数值 ABI 探针开始。
