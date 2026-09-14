# V4 第 21 节 N1–N6 完成报告

日期：2026-09-14

结论：N1–N6 已完成 **6/6**，并据此将第 20 节 B5 与 B1–B6 更新为 **6/6**。验收范围仅为同机、同一 NVIDIA H20-3e、TP=1、Qwen2.5-VL-3B BF16，以及 unified/1P1D/1P2D/2P1D/2P2D 的冻结实验矩阵。

## 分阶段结果

| 阶段 | 实证结果 | 主要证据 |
| --- | --- | --- |
| N1 | 正式 1P1D private/shared 各 5 次，在线 oracle=0；共享显存 16872→9608 MiB（-43.05%），请求周转吞吐 -7.69% | `performance_attribution_multi_pd/N1/oracle_free_e4/results.json` |
| N2 | B2/E4/E3 均先逐请求核对关键路径守恒再聚合，保留未解释残差；独立真实多 P/D Nsight trace 不进入正式延迟 | `performance_attribution_multi_pd/N2/ATTRIBUTION_REPORT.md` |
| N3 | 生产 Coordinator 静态 registry、1P1D 独立 PID、worker-owned D→P 顺序预留/回滚、绝对 deadline 与在线取消通过 | `performance_attribution_multi_pd/N3/result.json` |
| N4 | 请求级 provider-generation attach、真实 1P2D、两个 D、fan-out/COW、P 正常退出和新 D 代际通过 | `performance_attribution_multi_pd/N4/result.json` |
| N5 | 1P1D/1P2D/2P1D/2P2D 共 9 条必需 P→D 路径、原子容量竞争、推理中外部取消、Decode 绝对 deadline 与回收矩阵通过；真实 1P1D memcheck 为 0 errors/0 leak | `performance_attribution_multi_pd/N5/multi_pd_reservation_cancel_v2/`、`N5/compute_sanitizer_reservation_v2/` |
| N6 | private/shared × 5 拓扑 × 3 workload × 3 load，每 cell 5 个独立窗口；低/中每窗口 5 请求、近饱和每窗口 11 请求，共 90 cell/450 窗口/3150 请求；另有 50 个固定物理预算容量探针 | `performance_attribution_multi_pd/N6/` |

## 当前回归

- C++ 受影响门禁：73 项，71 passed，2 skipped；skip 仅为本构建未启用 NCCL。
- Python coordinator/placement/vision identity：8/8 passed。
- 当前二进制 20-case 数值：20/20 通过冻结同历史 BF16 合同；19 个全序列相同，1 个首次分叉发生在同历史 step 4，双方 token 均在共享 top-2，margin 为 0/0.125，logits max/mean abs 为 0.25/0.0518；156/160 token 相同。
- 真实 P/D compute-sanitizer：三个受检 language 进程均为 `ERROR SUMMARY: 0 errors`、0 leak。
- 最终进程检查：无遗留 PBE Data/Vision/Language 或 GPU compute 进程。

## 实验边界

- 非流式接口的客户端 TTFT/ITL 与 SLO goodput 仍为 N/A；模型侧 TTFT 不改名。
- N6 每个 cell 包含 5 个独立 observation window；低/中每窗口 5 请求，近饱和每窗口 11 请求。不支持可靠 p99 或显著性声明。
- 18,000 MiB 容量阶梯的 private 夹点为 1024/2048 blocks，shared 为 12288/16384；请求最多访问 48 blocks，因此只能表述为外部分配夹点。
- 近饱和采用非零 200 ms 间隔的 11 请求到达列车（2,000 ms）后 cohort drain；这是有限持续过载，不是渐近饱和。unified 另有不计时的 5-request 生产批处理能力探针；全多模态主路径受上游 Vision 串行交付而未观察到 language 合批，按实记录。
- 五窗口中位数下，最佳 split 相对 unified 的完成窗口吞吐变化为 private **+0.18%～+0.82%**、shared **-0.13%～+1.09%**，各 trial 范围重叠；这不能证明稳定吞吐提升。旧 +8.62%～+18.67% 数字仅属于单个 5-request 零间隔窗口的探索性结果，不作为最终简历指标。
- N6 固定 GPU/模型/请求轨迹、总 128 KV slots 和设备准入预算；新增角色的 CUDA context、模型创建 stream 与 workspace 增量保留在实测成本中。CPU/stream 使用共享池而非等额硬分区，因此不把结果表述为严格同总 context/stream 配额实验。
- 历史 E4 -6.65% 包含在线 oracle Decode，已保留并标记范围；正式 oracle-free 值为 -7.69%。

完整复算入口为 `performance_attribution_multi_pd/REPLAY.md`；最终接受值以 `performance_attribution_multi_pd/checks.json` 和新 completion manifest 为准，历史 manifest 未覆盖。
