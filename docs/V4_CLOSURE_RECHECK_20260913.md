# V4 四项收口再复核

本次独立执行 `validate_review_gap_closure.py`，输出到 `/tmp/pbe_review_gap_independent_20260913.json`，门禁返回 ok=true。380 个源码与 9 个验收文件逐一重算 SHA-256，全部匹配。代码确认真实 worker probe、future 完成时刻、实际 KV 恢复依赖图和生产 TransferScheduler 已接入。未重新运行全部模型实验或 232 项全量回归。

## 尚不能确认全计划完成的原因

M9 新部署实验已经排除冷启动，但测量对象仍不是 P/D 分离。

`tools/bench/data_flow/run_persistent_deployment_ab.py` 第 68–70 行创建一个 64-slot language worker 作为 unified，两个 32-slot 的同类 language worker 作为 split。第 87–92 行通过 `repetition % len(workers)` 选择其中一个 worker，将完整请求作为 `op=infer` 发给它，该 worker 完成全部 Prefill 与 Decode。此路径没有同一请求的 Prefill→KV 交接→另一 worker Decode。

因此当前数据正确描述的是“一个完整语言模型副本 vs 两个完整语言模型副本轮流执行请求”的预热对照。两倍权重/workspace 是副本数造成的实测成本，不能作为 P/D 分离实现已验收的证明。

统一门禁只检查冷启动排除、同时驻留、KV 总 slot 和模型进程数，没有检查每个请求的生产/消费角色和 KV 交接，所以 ok=true 与这个范围缺口并不矛盾。

## 最小补齐要求

保留现有实验，改名标明单/双副本。新增常驻 P 与 D 路径：一个请求在 P 生成 KV 后，由 D 导入受正确 lease 保护的页表继续 Decode；记录相同 request/generation 的 P PID、D PID、handoff、实际 prefix tokens 与 output。对照统一 worker 使用相同媒体/模型/到达轨迹、相同声明总预算，预热后每模式至少五次，并报告重复权重与 workspace。

给门禁增加 `prefill_pid != decode_pid`、有效 handoff、D 未重算全部 prompt、输出数值合同和最终资源回收检查，不能只验证 weight_processes=2。历史一次性 PD 演示可支持交接正确性，但不能代替这个常驻对照。

当前可确认上轮 E2、E3、lane 的针对性修复已进入代码；本次发现的剩余阻塞属于 M9 部署实验。不要再次重做已关闭问题，也不要将全部项目成果降回零。完成这条对照或明确正式缩减对应验收范围后，再宣称整个原计划全部完成。
