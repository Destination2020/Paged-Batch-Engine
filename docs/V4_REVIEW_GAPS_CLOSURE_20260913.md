# V4 最终复核缺口收口

日期：2026-09-13

结论：`V4_FINAL_ACCEPTANCE_REVIEW_20260913.md` 后续指出的 E2、E3 与 M9 四项实现/实验缺口已按真实运行路径收口。其后复核发现本文件旧版把“双完整 language 副本轮询”误记为 P/D 分离；该声明已纠正，真正的常驻 P/D 收口见 `V4_PERSISTENT_PD_CLOSURE_20260913.md`。既有通过项没有重做或改写；失败尝试、冷启动实验和负收益结果继续保留。

| 缺口 | 收口结果 | 最终证据 |
| --- | --- | --- |
| E2 固定容量、推断驻留、吞吐口径 | Language worker 暴露非变异的真实语义 radix/PageDirectory probe；Coordinator 每次决策读取 KV、bundle、staging 当前容量和 GPU/Host 可服务页；完成时刻来自 future callback；吞吐按首次 dispatch 到最后 completion 的统一窗口重算。15 轮、165 请求通过可执行公式复核，采集 90 份候选容量样本。 | `data_flow_evidence/v4/E2_final/runtime_state_validation.json` |
| E3 依赖图未接实际淘汰 | 实际 radix 页的 D2H 下沉、Host 提交、GPU 撤销、H2D 恢复与活跃引用全部经过 `RecoveryDependencyGraph`。清空/淘汰时同步回收页依赖对象，防止 4096 上限被长期页 churn 耗尽。19 个聚焦测试通过；两个真实 VLM worker 各记录 35 次提交下沉和 25 次恢复。 | `data_flow_evidence/v4/E3/dependency_eviction_integration_final2.log`、`data_flow_evidence/v4/M9/model_transfer_scheduler_lane_final_gpu0_v2/results.json` |
| M9 lane 绕过真实调度器 | 第一层 CUDA 基准直接构造生产 `TransferScheduler`，5×2 轮均有 3 次物理提交且全 payload byte 校验通过；lane on/off 唯一开关为 `direction_lanes_enabled`。第二层真实 VLM waiter 5×2 轮均在 2 个后台 D2H 页期间恢复 80-token 前缀；实际 D2H 并发为 on=1/off=2。CUDA demand queue 中位数从 4.926629 ms 降至 0.000489 ms，但模型客户端中位数 on=389.32 ms、off=380.55 ms，负收益如实保留。 | `data_flow_evidence/v4/M9/transfer_scheduler_lane_final/validation.json`、`data_flow_evidence/v4/M9/model_transfer_scheduler_lane_final_gpu0_v2/results.json` |
| M9 单/双完整副本成本实验 | 一个完整 language worker 与两个完整 language worker 在同一 GPU 常驻并预热，按固定 seed 交错 5 次。每个请求仍在单一 worker 内完成完整 Prefill+Decode，因此此结果只保留为副本数成本实验，不计 P/D handoff 验收。统一/双副本 TTFT 中位数为 50.99/57.70 ms；双副本模型 allocation 和 workspace 均为两倍。 | `data_flow_evidence/v4/M9/persistent_deployment_final_gpu0_v2/results.json` |

统一执行门禁现已改为读取真正的常驻 P/D 结果，见 `data_flow_evidence/v4/review_gap_closure_checks.json`，结果 `ok=true`。旧 `M9/lane_ab` 仅作为历史微基准、旧 `M9/deployment_ab` 仅作为含冷启动实验；本表的单/双完整副本结果以 `replica_count_cost` 名称保留，明确不计 P/D handoff 验收。

最终回归：带 Qwen2/Qwen2.5-VL fixture 的 236 个 C++/CUDA 用例中 232 通过，4 个 NCCL-only 用例因构建关闭 NCCL 而跳过；Python 5/5 通过。所有临时服务退出后 IPC slot 128/128 空闲、active grant=0、lease=0。
