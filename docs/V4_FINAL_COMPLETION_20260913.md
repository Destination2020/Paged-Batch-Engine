# PBE V4 最终完成报告

日期：2026-09-13

结论：执行计划中的原阶段 **M0–M9 已验收 10/10**，扩展阶段 **E1–E3 已验收 3/3**。本报告是 `V4_FINAL_ACCEPTANCE_REVIEW_20260913.md` 的整改闭环；其后针对实际回收/选路/实验路径的复核缺口也已完成，详见 `V4_REVIEW_GAPS_CLOSURE_20260913.md` 与 `V4_PERSISTENT_PD_CLOSURE_20260913.md`。原复核、失败记录与负收益结果均保留。

## 最终复核缺口闭环

| 复核项 | 完成结果 | 可执行证据 |
| --- | --- | --- |
| M5 20-case 数值门禁 | 20/20 通过；15 个全序列相同，5 个 BF16 分叉逐例验证首次分叉处同历史、logits max/mean、双侧 top-2 margin 与选择 token；不要求逐 token 全等 | `data_flow_evidence/v4/M5_numerics_final/validation_final.json` |
| M5 在线协调 | JSONL v2 有界异步入口；推理中外部 cancel 在 2 token 后生效；绝对 deadline 从 Encode 前传递并在 10,503.42 ms 触发 | `data_flow_evidence/v4/M5_online_lifecycle_final/validation.json` |
| M7 设备预算 | 使用实际模型 allocation、真实 workspace profile、同卡 Vision/service 既有占用、external KV owner 和 CUDA 峰值；10,028,777,472-byte 峰值低于准入线 | `data_flow_evidence/v4/M5_online_lifecycle_final/result.json` |
| M7/M8 压力恢复 | Vision feature 262,144 bytes GPU→Host→GPU；Language 5 个完整前缀页与 14-token 私有尾页恢复；checkpoint 保存 feature 依赖、mRoPE、RNG、outbox，0 failure | `data_flow_evidence/v4/E3/feature_gpu_host_recovery/result.json`、`data_flow_evidence/v4/E3/real_recovery/final_run/validation.json` |
| M9 实验范围 | feature cache、KV share、常驻统一 P/D 与独立 Prefill→KV handoff→Decode、Host/IPC/跨卡、singleflight、生产 TransferScheduler lane、pressure、placement 均有对照；旧冷启动部署和绕过生产调度器的 lane 微基准明确不计最终验收；单/双完整语言副本实验仅计副本成本 | `data_flow_evidence/v4/M9/persistent_pd_deployment_final_gpu0_v4/validation.json`、`data_flow_evidence/v4/M9/final_results.json`、`data_flow_evidence/v4/review_gap_closure_checks.json` |

## 扩展验收

### E1：语义前缀与缺页传输

真实 serving 在 Prefill 前根据模型/表示、完整 token、媒体 ContentId、processor、span 和 mRoPE 查询最长连续前缀。最终正例各命中 80 token，processor 与图片内容负例均为 0；不同图片 fixture 已纳入仓库。跨 GPU 每个 Decode 仅复制所需 3,538,944 bytes，而不是 37,748,736-byte 全池，并进入真实 attention/COW。详见 `data_flow_evidence/v4/E1/checks.json`。

### E2：数据位置感知 placement

两个真实 Language worker 各持 32-slot grant。固定、round-robin、data-aware 在相同端点、64-slot 总预算和请求轨迹下各完成 5 次、55 请求。data-aware 每次从 worker 的实际 KV allocator、bundle/staging 预算和语义 radix/PageDirectory probe 读取状态；15 轮共验证 90 份候选容量样本。吞吐按首次 dispatch 到最后 future completion 的统一窗口计算。简单成本模型的 1,037.47 ms 平均预测误差保留，不宣称全局最优。详见 `data_flow_evidence/v4/E2/checks.json`。

### E3：恢复依赖分层缓存

实现版本化有界依赖 DAG、可重算/必须保留、精确/容差恢复、独立 active/cache/recipe 引用及最后必要副本保护。依赖图已接到真实 radix/PageDirectory D2H 淘汰和 H2D 恢复路径，并在实际页 clear/evict 时回收对象。真实 VLM worker 各记录 35 次提交下沉和 25 次恢复；19 个实际路径聚焦测试与 10,000 次生命周期均通过。Vision feature、私有尾页 checkpoint 与五次负收益策略消融继续保留。详见 `data_flow_evidence/v4/E3/checks.json`。

## 最终回归

- 两套构建成功：默认构建与 Qwen2.5-VL BF16 构建。
- 可用功能门禁：最终全量数字见 `data_flow_evidence/v4/final_regression/checks.json`；双 GPU P2P、Qwen2 checkpoint、Qwen2.5-VL 数值/分块、跨语言 bundle 与共享 external prefix 顺序恢复均纳入回归。
- Python：5/5 通过。
- compute-sanitizer：Host plain/FP8 scatter 与 Host-only radix restore 2/2 通过，`ERROR SUMMARY: 0 errors`。
- 仓库旧 `test_load.*` 三项依赖未提供的 `./tmp/test.bin`，不属于可执行功能门禁，最终命令显式排除；初次完整运行的失败日志保留在 `final_regression/test_default_full.log`，没有伪装为通过。

最终日志位于 `data_flow_evidence/v4/final_regression/`。所有服务进程均正常退出，最终检查无遗留 PBE worker/data-service 进程。

## 结果边界

保留 Agent-owned GPU 主路径、跨卡副本、四分支 COW、常驻 Vision 及所有负收益 A/B。BF16 验收是严格的同历史数值合同，不是全 token 相等；multimodal TTFT 是 Coordinator 可观测的 Encode+首 token 时间，不声称流式网络首包；本项目不声称优于 SGLang。NCCL、跨机 RDMA、权重共享、视频/音频、DiT 与 Draft/Verify 仍是计划明确排除的非分母范围。
