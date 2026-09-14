# V4 N1–N6 独立复核（2026-09-14）

结论：存在真实多角色集成和可引用的受控实验数据，但不能确认第 21 节全部完成。现有 `validate_n1_n6.py` 重跑通过，输出保存在 `/tmp/pbe-n1-n6-independent-review.json`；门禁实现未覆盖下列原计划要求。此复核没有重跑 GPU 实验或完整测试，不覆盖历史结果，也没有修改实现及其 manifest。

## 阻塞 1：N6 把同窗口请求数当作独立试验数

`run_n6_topology_matrix.py` 的 `args.repeats` 控制单 cell 中提交的请求数，`five_trials_per_cell` 实际检查 completed==repeats。90 cells/450 requests 是每 cell 一个窗口、5 个请求，不是每 cell 5 个独立 trial。`validate_n1_n6.py` 的 `N6_all_90_cells_and_450_trials` 只检查记录数；计划 20.2/20.6/21.7 要求至少 5 次随机顺序独立 trial。

runner 未显示正式 cell 前的独立预热阶段；`near_saturation` 的输入间隔为 0，实际是 5 请求突发，不能仅据名称确认持续近饱和吞吐。unified 的该突发确实使用 5-request batching，这项修复保留，但不能替代独立重复、缓存初态和持续负载验证。

整改：分离 trials 与 requests-per-trial；每 trial 有独立窗口和可核验的预热/缓存初态，随机化比较顺序。门禁检查 trial ID、窗口、请求数和协议，而非把请求叫 trial。现有最佳拓扑 +8.62%～+18.67% 等仅作单窗口探索结果，不作稳定提升结论。

## 阻塞 2：N2 未实现要求的请求关键路径归因

`analyze_n2_attribution.py` 对 B2 使用 `median(TTFT) + 7*median(ITL)` 构造模型窗口，再从两臂 median(round) 的差值推导 434.24 ms 残差。分项中位数之和/差不等于逐请求时段之和/差的中位数；该结果不能精确证明这些毫秒属于模型外某条路径。

E4 去掉在线 oracle 后仍观察到 779.41→841.49 ms，可以排除 oracle 是唯一原因，但没有完成所要求的 GPU/CPU/同步等分段归因。独立 Nsight 文件存在且排除于正式统计是有效证据，但不替代 B2/E4/E3 各自关联请求的 span 分解。

整改：按 request/trial 关联实际开始结束和逐 token 记录，先算每请求实际模型窗口与残差再汇总；补齐主要增长区间的低扰动埋点和单变量对照。保留未解释残差，禁止用聚合中位数做精确时间守恒结论。

## 阻塞 3：N3 的 reserve 仍是状态查询

`python/pbe_roles/pd_runtime.py` 的 `ProductionPDCoordinator.submit` 在 reserve 状态调用 `registry.refresh(decode)` 和 `registry.refresh(prefill)`，然后直接启动 pd_prefill。状态观察不是 D 增长/P 计算容量的实际预留，无法兑现计划 21.4 的顺序预留、失败回滚与有界重选合同。

整改：引入或复用真实请求 reservation/grant，D/P 所需容量按固定顺序获取并绑定 generation/deadline，失败安全回滚。加入两个请求争用剩余容量的主路径测试，检查 reserve 成功与后续计算使用的是同一份授权。

## 阻塞 4：多 P/D Coordinator 的 deadline/外部取消未贯通

`submit` 向 pd_prefill 传入 deadline，但调用 pd_decode 时只传 request/generation、KV 身份、provider incarnation 和 max_new_tokens，RPC timeout 固定 180 秒；Decode 返回后直接 finished，没有在该调用链中传入或检查原绝对 deadline。

Coordinator 当前仅检查预置 `cancel_before_decode`，没有外部 cancel 方法及可定位活跃 stage 的请求状态表。原单 LanguageProcess 的外部取消能力不能自动证明新多 P/D submit 路径也支持。finally 中 pd_release 未隔离异常，也应覆盖释放失败不能覆盖原请求终态的情形。

整改：原 deadline 传递至 D 及其计算步骤，RPC 等待受剩余时间约束；活跃请求以 generation 关联取消入口与 stage，取消后禁止新 stage，已有 GPU 工作按完成证明收敛。真实测试 Decode 中超时/取消、迟到完成与 release 失败，不用提交时设置布尔字段代替运行中取消。

## 已有性能数据及允许结论

- N1 无在线 oracle 的 1P1D 各 5 次：固定 KV 下物理显存 16872→9608 MiB（-43.05%），周转耗时 779.41→841.49 ms，周转吞吐 -7.69%。这是明确的显存收益及性能代价。
- N6 shared 模式的 5 请求突发窗口：长输入短输出 unified 0.41337→1P2D 0.44899 req/s；混合 unified 0.39169→2P2D 0.44618；短输入长输出 unified 0.37003→1P1D 0.43913。只限观测窗口，最佳值存在选择偏差，需独立重复确认。
- 18,000 MiB 外部分配夹点仍不是最大在线并发；最大实际访问 48 页的边界保留。

下一步应修正门禁对试验单位、真实 reservation、Decode deadline/取消及请求 span 的检查，再补测受影响范围。代码存在、hash 相符、旧门禁通过与完整实现原验收合同是不同层面的证据。
