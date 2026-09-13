# 数据流重构：当前执行摘要

## 最新任务：多角色与多模态架构 V4

- 当前定位为“PBE：支持多角色与多模态数据共享的分页推理引擎”。先读 [V4 架构](PBE_MULTI_ROLE_MULTIMODAL_ARCHITECTURE_V4_20260912.md) 和 [V4 执行计划](PBE_MULTI_ROLE_MULTIMODAL_EXECUTION_PLAN_V4_20260912.md)；后续执行以 M0–M9 为主，旧 V3 保留详细安全合同。
- 本轮完成源码研究和文档，未修改 runtime；V4 已验收 **0/10（0%）**。用户本轮要求架构与执行计划，未要求本轮直接实现整个 V4。
- 源码已存在 PD/ZMQ/NCCL 交接，不能说 PBE 没有任何跨角色基础；缺的是通用多消费者数据生命周期及模型外部池所有权。
- 当前 checkpoint 已出现 byte budget、stale commit 拒绝与取消清理，旧摘要状态滞后。M0 先核验当前代码和证据，既不重复修复也不直接算完成。
- 首个模型候选 Qwen2.5-VL-3B：Python 视觉塔 + PBE C++ 语言核心，需 M1 数值验证；没有已验证的兼容性或性能结论。
- 本轮 encoder cache/aligned sequence 测试 **34 passed in 7.29s**，限 CPU/helper/fake client。源码版本/hash 与命令见 [审计清单](data_flow_evidence/v4_architecture_audit_20260912.json)。
- SGLang 当前官方文档已有 EPD 和全局多模态 embedding cache；不把这两个功能称为 omni-flow/PBE 独有。

以下为前轮 V3 摘要与历史证据，状态以以上 V4 入口为准。

更新：2026-09-12。当前入口为 [V3 执行计划](DATA_FLOW_REFACTOR_PLAN_V3_20260912.md)，研究依据为 [omniFlow 二次审计](OMNIFLOW_DATA_FLOW_SECOND_AUDIT_20260912.md)。

## 当前任务与状态

- 用户本轮要求重新研究 omniFlow、比较 SGLang 的设计差异，并准备后续可交给 AI 的执行计划。本轮已完成源码复核、48 项 omniFlow CPU/集成合同检查、增长准入等待边界复现及计划编写；没有修改 PBE runtime。
- 新计划 DF0–DF6 的实现为 **0/7，未开始**。首项是 checkpoint 的旧 revision 覆盖、新 preparing ticket 在取消后提交，以及 records/bytes/metadata 生命周期；先做失败回归再修复。
- 其后依次补语义 representation 编译、联合容量、方向 lane/deadline/restore waves、真实压力 checkpoint/输出队列、成本/水位策略及等价实验。X1 provider、X2 私有 layer-ready 单独选择。
- 上轮 P0a–P5 的 **8/8（100%）仅是历史原型验收口径**；二次审计确认它不能代表全部运行时集成和实验主张已经完成。原始证据保留，不抹掉已通过测试，也不沿用过强结论。
- 用户允许缺失依赖时安装；本轮所需模型、CUDA/GTest 与用户级 sanitizer 均已可用，无新增缺失依赖。
- 工作区改动尚未提交；保留任务开始前已有的 `demo/main_qwen2Instruct.cpp` 修改。

## 上轮已交付的原型能力

1. 64 位代际 request handle、请求级 RNG seed/counter、逐 token gap 指标与可复现环境清单。
2. `PageSchema → TransferPlan → BoundTransfer` 语义搬运链，支持 K/V/scales、物理乱序与执行前全量预检。
3. generation-aware block ownership、compute/I/O leases、logical radix page 与 GPU/host 多 residency。
4. 有界 pinned HostStore、完整页 D2H/H2D 事务、CUDA fence、取消后 drain、unknown-completion quarantine。
5. 统一 `TransferScheduler`：H2D/D2H/P2P、per-page single-flight、独立 waiter cancel、priority donation、依赖 wave、容量和 aging。
6. 进程内 `RequestCheckpointStore`：prepare/commit revision、不可变 KV snapshot、pending token、RNG/output 游标、整步 suspend 与换址 restore。
7. 固定预算 P5 消融脚本和机器可读/原始结果，分别记录收益与负收益。

## 上轮验证记录（本轮没有重新运行 PBE GPU 性能测试）

- CPU ownership：53/53。
- ASan+UBSan/LeakSanitizer：53/53。
- CUDA 回归（排除三个环境中不存在的 `test_load.*` 外部 fixture）：157 passed、2 个 NCCL-disabled skip。
- 真实 H2D/P2P、FP8 snapshot 的 compute-sanitizer：0 errors。
- Qwen2-0.5B-Instruct BF16 固定采样在两次 checkpoint 后与不中断输出完全一致。
- P5：GPU warm 吞吐 +39.27%/TTFT −64.84%；host restore 吞吐 +20.44%/TTFT −40.05%；checkpoint 短负载执行 +9.01%（负收益）；host token-gap P95 +1.82%（负收益）。

汇总证据：

- `data_flow_evidence/p3_runtime_acceptance.txt`
- `data_flow_evidence/p4_runtime_acceptance.txt`
- `data_flow_evidence/p5_runtime_acceptance.txt`
- `data_flow_evidence/p5_ablation/results.json`

旧 single-flight 微基准使用不同 logical keys 对照，raw snapshot 与 save/free/restore 工作量不同；这些数值不能独立归因为去重/事务机制收益。新实验要求相同轨迹、明确开关和完整保存/恢复成本。真实压力抢占与 emitter cursor 的集成仍待 V3 完成。

## 可复用路径

- 模型：`/tmp/Paged-Batch-Engine-models/Qwen2-0.5B-Instruct.bf16.bin`
- tokenizer：`/tmp/Paged-Batch-Engine-models/Qwen2-0.5B-Instruct/tokenizer.json`
- Qwen/CUDA build：`/tmp/Paged-Batch-Engine-build-qwen05`
- CUDA regression build：`/tmp/Paged-Batch-Engine-build-p1`
- CPU build：`/tmp/Paged-Batch-Engine-ownership-cpu`
- sanitizer build：`/tmp/Paged-Batch-Engine-ownership-asan`

P4 的保证限于进程存活期间；不包含崩溃持久化和网络重连。P5 数字限于当前机器上的 Qwen2-0.5B 短负载，不能外推通用恢复/重算 crossover。
