# V4 与 E1–E3 完成声明复核

结论：本轮确实补充了大量真实执行证据，20-case 数值门禁与版本完整性复核通过；但仍不能确认原 M0–M9 和扩展 E1–E3 全部验收。主要阻塞在 E2 真实位置/容量信息接入、E3 依赖策略接入及 M9 实验归因，不再重复将已解决的数值门禁列为失败。

## 本次实际验证

- 运行 `python3 tools/bench/data_flow/validate_persistent_vlm_fixtures.py docs/data_flow_evidence/v4/M5_numerics_final/result.json`，退出码 0，20-case 通过当前可执行数值合同。15 个序列全等、5 个首次分叉有同历史 logits/top-2 证据，141/160 token 相同。20/20 是数值合同通过数，不是全序列相同数。
- 对 `final_completion_manifest.json` 的 374 个源码文件和 9 个验收文件逐一重算 SHA-256，全部匹配。
- 阅读最终结果、E1/E2/E3 checks、真实启动脚本、placement、feature 恢复与 language checkpoint 调用。没有重新运行完整 228 项 GPU/C++ 回归，也未重新执行全部模型性能实验；这些结果仍属于所提供日志。
- 本次只新增复核文档，未改运行时、旧验收记录或 manifest。

## 1. E3：依赖图组件与实际淘汰路径仍分离

`RecoveryDependencyGraph` 及 `choose_pressure_action` 在 `infMain/demo/python` 的检索结果只有自身声明/实现；测试调用它，但没有发现 serving、NodeAgent 或 Vision 回收路径使用它。实际 Vision 的 `demote_features` 清除 GPU 字典，`restore_features` 按本地 recipe 从服务获取 Host feature；这是实际副本恢复，但不等于恢复依赖图驱动了淘汰决定。

因此“依赖图单测通过”与“真实 feature/KV 恢复通过”两组证据尚未证明：真实资源压力下，最后必要副本由该图保护、版本失效和引用关系参与实际回收。若存在另一套等价实现，需给出完整调用链与对应故障结果，不能只列类型名。

代码入口：`infMain/source/cache/recovery_dependency.cpp`、`python/pbe_roles/vision/persistent_worker.py::demote_features/restore_features`。

验收补齐：在真实 eviction/withdraw 前查询恢复能力，并在 copy/fence/commit 后更新图中 residency；对最后 Host 副本、依赖版本变化、Host 满和活动 consumer 做实际数据路径测试。现有手动 checkpoint continuation 证据可以保留，但不能代替由真实容量压力触发的策略选择。

## 2. E2：候选信息仍有模拟值与未验证的驻留推断

`python/pbe_roles/multi_worker_coordinator.py::snapshots` 把可准入容量默认写为 `64 << 20`；测试通过 `unavailable_worker` 设为 0、`stale_worker` 人为倒退观测时间。`resident` 集合在请求完成后加入 request bundle content，未通过 E1 页目录确认当前仍可服务的前缀及实际缺页。

这会导致请求完成后页面已淘汰仍被当作命中；同图片不同问题的 request bundle 身份也不等于可共享前缀身份。worker 确实是真实进程，但候选容量/位置证据不能仅靠进程真实来成立。运行参数中的默认带宽/prefill/decode 常数未由该 launcher 读取 calibration 文件。

验收补齐：接入实际 worker 队列、预算与前缀 residency/epoch；根据真实 reservation failure 有界改选。人为 stale/unavailable 值可留作故障注入，正常负载必须使用实际状态和冻结校准值。与 E1 的 semantic prefix/missing pages 对接，不能仅按曾处理过的 request key 推断可服务副本。

另外当前 `throughput_requests_per_s = len(latencies) / (sum(latencies)/1000)` 是平均请求延迟的倒数。多 worker 并发下它不是系统吞吐，应使用统一观测窗口内完成请求数/墙钟时间。client_ms 在串行收集 future 时计算，晚收集的已完成任务可能被记入额外等待；应记录真实 completion timestamp。

## 3. M9 lane 数据是独立 CUDA 排队实验

`demo/pbe_lane_cuda_benchmark.cpp` 直接调用 cudaMemcpyAsync；没有调用 PBE TransferScheduler。开关改变三笔 copy 在两个 stream 上的分布，说明方向隔离可能减少队头等待，但不能证明 PBE lane 的实现和 serving 路径获得同样收益。

`max_total_active=2` 是写入结果的声明；本实验没有通过真实调度器执行 admission。源内容使用常量，仅核验 D2H 首末两个字节，不能充当完整传输 correctness 验收。

验收补齐：相同任务轨迹/bytes/总 active/stream 下，切换实际 TransferScheduler lane；让真实模型等待其中的 demand，记录 token 级 ITL、queue time、copy/fence timeline 和完整 payload 校验。保留当前实验，但标为 CUDA stream 微基准。

## 4. M9 deployment 仍是包含冷启动的单请求部署比较

`tools/bench/data_flow/run_deployment_ab.py` 每次启动整个 launcher，测 subprocess wall time。`separated_vlm_one_request.sh` 依次运行一次性 Vision、Prefill、Decode，Prefill 完成退出后才启动 Decode；merged 路径运行常驻能力的角色但也仅提交一个请求后关闭。

这能比较“启动+运行+退出”的总耗时，不能作为同预算常驻合并/拆分 serving 的性能对照。separated 的峰值内存仍为 None；same_gpu_count/same_kv_blocks 不能替代权重/workspace/传输的实际预算验证。

验收补齐：双方权重预热、常驻，提交相同到达轨迹，在同资源上限下记录端到端与逐 token 指标；冷启动另列。若决定只交付冷启动部署实验，需明确修改原计划范围，不能将其视为原常驻性能合同完成。

## 收口建议

先完成 E3 依赖策略与真实回收的连接，及 E2 候选数据与真实 worker 状态的连接；修正 E2 吞吐/完成时间口径；最后通过实际调度器与常驻部署补 M9。无需重做已通过的 20-case 或否认已有真实 attention、稀疏 P2P、COW 与恢复成果。

当前建议状态：保留各子项成果；E2/E3 与 M9 标记“待上述主路径/实验验收”。本次没有逐条重新验收所有其他阶段，因此不另给一个未经完整核算的总百分比。
