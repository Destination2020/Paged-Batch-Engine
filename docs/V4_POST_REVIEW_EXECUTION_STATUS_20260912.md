# V4 执行后复核整改状态

日期：2026-09-13

此文件覆盖各 M2–M9 历史 `accepted` 摘要的当前状态；原始日志保留，不改写为新实验结果。当前阶段验收为 10/10（M0–M9）。

## 已落地的 P0 整改

- R1：Data service wire protocol 升级到 v2。release 使用 service incarnation、consumer incarnation、lease id 组成的完整 `LeaseToken`；跨服务代际 release 返回 `OWNER_RESTARTED`。Acquire 携带稳定 operation id，同操作的 reply-lost 重放返回原 lease，不增加引用。
- R2：NodeAgent 使用覆盖 header、body 和 response 的单个绝对请求 deadline；跟踪已接受连接；stop 会中断活跃 fd、关闭排队 fd。回归覆盖 8 个 worker 被半包 header 占用、半包 body，以及每 20 ms 发送一个 header 字节仍不能延长 deadline。
- R6：Vision 缓存身份包含实际模型/processor 元数据与权重分片文件清单；singleflight 锁包含 endpoint 和 service incarnation，并使用进程退出时由内核释放的 `flock`。Vision 在 forward 前估算并 reserve 输出 bundle，失败释放 producer reservation。

## 新增回归

- `ContentRegistryTest.AcquireReplayIsIdempotent`
- `ContentRegistryTest.StaleServiceTokenCannotReleaseReusedLeaseNumber`
- `NodeAgentTest.SlowPartialHeadersCannotOccupyWorkersIndefinitely`
- `NodeAgentTest.PartialBodyUsesTheSameAbsoluteRequestDeadline`
- `NodeAgentTest.HeaderDripCannotRenewTheAbsoluteRequestDeadline`
- `NodeAgentTest.StaleWireTokenCannotReleaseLeaseAfterServiceRestart`
- `NodeAgentTest.ReplyLostAcquireCanReplayWithoutAddingALease`
- `VisionIdentityTest.test_model_or_processor_change_invalidates_cache`
- `VisionIdentityTest.test_service_incarnation_separates_locks_and_crash_releases_flock`

标准 CMake `test_llm` 目标增量构建通过；定向数据层二进制运行 24/24 通过，其中 partial body、header drip 与 8-worker shutdown 回归分别约 104、106、105 ms。Python 身份测试 2/2 通过；Python client 与 C++ v2 service 的 reserve/seal/acquire/release/shutdown 互操作通过。

## 动态 VLM 闭环整改

Vision bundle 新增请求实际生成的 `input_ids`、三轴 `position_ids` 和 `rope_delta`。PBE Prefill 从 bundle 读取这些字段及动态 feature width；KV payload 携带 prompt 与 position state，Decode 不再读取固定 fixture，也不再写死 `delta=-56` 或 hidden size 2048。

CPU 先与冻结 oracle 比较，95 个 token、`[3,95]` positions 和 `rope_delta=-56` 精确一致。随后在 H20 双 GPU 执行无 fixture 闭环：Vision PID、Prefill PID、Decode-A PID、Decode-B PID 独立；两路 Decode 与 Prefill oracle 的 16 个 token 完全一致，服务退出前 `leases=0`。原始日志位于 `docs/data_flow_evidence/v4/post_review_dynamic_epd_no_fixture/`。

另外运行了两个不同分辨率的真实 Vision 混合 batch。两个输出在 forward 前分别完成 reserve，视觉 token 数为 81/121，bundle 为 332056/495896 bytes，原始日志为 `docs/data_flow_evidence/v4/post_review_batch.log`。

## M2/M3 Agent-owned GPU KV 主路径（2026-09-13）

Data service 现在创建并持有自描述的 CUDA IPC KV 池，集中分配 slot；Qwen2 的每层 `BlockAllocator` 可绑定该外部池，真实 Prefill/Decode paged attention 通过 `KVPoolView` 直接读写 Agent-owned allocation。外部请求状态只发布页表、有效/已提交 token、mRoPE delta、prompt 和完整池 lease token，不再经 Host 发布 KV payload。正常错误路径用作用域 guard 归还尚未发布的 grant；发布后的 prefix grant 由协调方持有，Decode 在 CUDA stream 同步完成后归还私有 grant。

H20 实机主路径由一个 Data service、一个 Vision、一个 Prefill 和两个并发 Decode 独立进程组成。Prefill 退出后，两路 Decode 仍映射同一 pool owner；运行中观测到 prefix 加两路 private grant 共 3 个活跃授权、40/64 空闲页。两路输出均与 Prefill oracle 的 16 token 精确一致，每路真实未满尾页 COW 为 589,824 bytes。Decode 退出后仅保留 prefix grant；协调方释放后 64/64 slot 全部空闲，`active_grants=0`，metadata service `leases=0`。原始证据在 `docs/data_flow_evidence/v4/post_review_agent_owned_kv/`，复现入口为 `tools/launch/multi_role_vlm_ipc.sh`。

该次证据只验收 M2 的 external pool/view 子项、M3 的同 GPU Agent-owned attention 与正常生命周期子项，以及 M6 的跨角色双分支未满尾页 COW 子项；后续 serving 与跨卡结果见下一节。

## M2/M3 serving 主路径与跨 GPU copy（2026-09-13）

`serving_qwen` 新增 `--data-service-endpoint` 正式配置。模型初始化前经 typed DataClient 获取 Agent-owned CUDA pool、幂等 reserve 全部 slots 并导入 IPC view；原 Scheduler → MixedBatchBuilder → Qwen Prefill/Decode attention 调用链直接使用外部池。H20 GPU0 的两请求真实运行包含 71 个 Prefill token、32 个 Decode token，2/2 完成。运行中服务端为 `free_slots=0 active_grants=1`；CUDA 同步、模型/page table、import view 依次销毁后才归还完整 token，最终为 `free_slots=64 active_grants=0`、metadata `leases=0`。

跨 GPU 主路径把 Data service/Prefill 放在 GPU0、两个并发 Decode 放在 GPU1。每个 Decode 对 37,748,736-byte owner pool 执行一次可计费 P2P replica（432.271/407.309 ms），随后真实 attention 输出的 16 token 均与 Prefill oracle 精确一致；每路未满尾页 COW 为 589,824 bytes。退出后 64/64 slots 与全部 lease 回稳。

原始证据分别位于 `docs/data_flow_evidence/v4/post_review_serving_external_kv/` 与 `post_review_cross_gpu_kv/`。重链接后的标准测试二进制定向回归 23/23 通过。至此 M2 和 M3 按原计划验收；总阶段数更新为 4/10。常驻多模态、四分支取消矩阵、线上统一准入与公平 serving 实验仍不得提前验收。

## M4 常驻 Vision、动态 batching 与取消（2026-09-13）

新增常驻 typed Vision role：Qwen2.5-VL processor/vision tower 在单一 PID 只加载一次；请求携带 request id/generation、有序 image/text parts、绝对 deadline，入口为有界队列，支持 cancel/shutdown。两个不同分辨率请求在同一真实 forward 中形成 81/121-row 变长 batch，334,024/498,568-byte 输出均在 forward 前 reserve。

数据身份拆成 image/processor feature object 与 text-specific request bundle。后续同图不同问题命中同一 feature，新增 forward 为 0，但 input ids/mRoPE 各自发布到不同 request bundle，避免旧路径用图像 key 误复用文本 position state。另一个排队请求按 generation 在 forward 前取消，未发布 request bundle。进程退出后数据服务 `leases=0`。

原始证据在 `docs/data_flow_evidence/v4/post_review_persistent_vision/`，复现入口为 `tools/launch/persistent_vision_role.sh`。结合此前跨语言、双消费者、身份/接管和真实数值证据，M4 按计划验收；总阶段数为 5/10。M5 仍需把 Prefill/Decode 改为常驻 typed 多请求协调，不能由 Vision 常驻单独推断。

## M6 四分支、半数取消与退出顺序（2026-09-13）

一个真实 95-token 多模态 Prefill 的 Agent-owned prefix 同时被四个独立 Decode PID 获取；服务端实测 `active_grants=5`。四支首次 append 均对未满尾页执行 589,824-byte COW。两支生成 16 token 并与 Prefill oracle 精确一致；另两支真实生成 4 步后经 `Scheduler::cancel_request` 进入 terminal cancelled 状态，只释放自身引用，未改变幸存分支结果。

四支退出后仅 prefix grant 保留（48/64 free）；最后释放 producer prefix 后为 64/64 free、0 grant、0 data lease。两次失败尝试及原因均保留，第三次验收日志在 `docs/data_flow_evidence/v4/post_review_four_branch_cow_retry2/`，入口为 `tools/launch/four_branch_vlm_ipc.sh`。结合既有页对齐/未对齐单元语义回归，M6 验收；总阶段数为 6/10。

## M5 常驻在线多模态主路径（2026-09-13）

新增常驻 C++ PBE language role 与 typed Coordinator。语言模型和 Agent-owned 64-slot CUDA IPC 池均只绑定一次；请求通过 Encode → Join → Scheduler mixed Prefill/Decode → terminal output，generation、绝对 deadline、Join 取消和真实 Decode 中取消贯穿主路径。20 个固定图文 case 在同 PID 分别进行四请求混合和单请求重放，出现 5 个 Prefill/Decode 同 step。18/20 序列完全一致，153/160 token 一致；另两例各只有一个首次 greedy 分叉点，按 M1 已冻结的 BF16 tie 口径记录，未写成全等。首次运行暴露并保留了临时 schema component 悬空指针，修复为持久化已校验 offset/length 值后重跑。

原始证据位于 `post_review_persistent_vlm_online/`、`post_review_persistent_vlm_20/`（失败）和 `post_review_persistent_vlm_20_retry/`（验收）。进程退出后 64/64 页、0 grant、0 lease。M5 验收，总阶段数为 7/10。

## M7 线上统一准入与压力恢复（2026-09-13）

常驻 language 主路径在提交 Scheduler 前，对实际获取的 bundle bytes、视觉 staging bytes 与按 prompt+decode 保守估算的外部 KV blocks 执行单次跨池全量预留；固定 weights/workspace 在模型存活期占用，容量初始化后冻结。资源 guard 覆盖成功、拒绝、取消、deadline 和异常。

1,000,000-byte staging 上限下，同批 6 个 308px 请求明确准入 2、拒绝 4，拒绝原因为 `unified_budget_exhausted`，峰值 991,232 bytes；释放后同一 PID 下一批重新准入 1/1，staging 回到 0。结合既有 lane、demand reserve、aging、checkpoint/recompute/auto、quarantine 与 10,000-cycle 合同回归，补齐线上实际资源连接。原始证据为 `post_review_online_pressure_retry/`；M7 验收，总阶段数为 8/10。

## M8 故障矩阵复验（2026-09-13）

生命周期定向回归 57/57 通过。额外在常驻 language 主路径发布 malformed bundle，首笔 Scheduler/GPU compute 前因 shape/coverage 拒绝，预算仍为 0 且同 PID 可继续服务。随后 SIGKILL Data service 并在同 endpoint 重启，incarnation 改变，旧对象明确失败而不透明复活；language 可安全关闭，重启池为 64/64、0 grant、0 object、0 lease。原始证据为 `post_review_fault_matrix_regression.log` 与 `post_review_online_fault_matrix_retry/`；过长 Unix socket 的首次脚本失败保留。M8 验收，总阶段数为 9/10。

## M9 公平 serving 与交付（2026-09-13）

使用冻结 Qwen2.5-VL-3B language checkpoint、GPU0、相同两请求/71 Prefill token/32 Decode token、64 pages/layer 与相同 batch/chunk 配置，随机顺序各运行 5 次 process-owned KV 与 Agent-owned IPC KV。全部 10 次为 2/2 完成、0 失败，外部池每次退出均为 64/64 free、0 grant。

中位数 local/external 分别为：TTFT 84.140/90.812 ms，ITL 6.212/6.188 ms，request latency 177.491/183.453 ms，throughput 180.248/174.367 token/s。低复用小负载下 external 的 TTFT +7.93%、latency +3.36%、throughput -3.26%，这是诚实的 overhead control，不表述为加速。原始日志和 `results.json` 在 `docs/data_flow_evidence/v4/post_review_external_serving_ab/`。

新增常驻多模态 cache off/on 公平实验：同一 Vision/language PID、同图同文同生成配置，seed 20260913 随机顺序，每模式 5 次且十次输出 token 相同。off/on 中位数为：多模态 TTFT 2491.70/2483.70 ms、ITL 46.17/46.23 ms、端到端 3193.22/3178.80 ms；cache-off Vision forward 37.27 ms，hit 为 0。processor/RPC 占主要时间，因此不把 saved forward 宣称为等量端到端收益。

最终矩阵引用四分支 COW、同/跨 GPU transport、线上压力拒绝/恢复与故障证据，产出 `post_review_final_matrix/results.json`、原始 JSONL trace 和 standalone SVG。等预算 merged full-VLM 在 V4 中没有可比实现，字段保持 `null`，不拿不同路径推断。部署指南已切换到常驻单命令演示并修正 protocol v2、COW bytes 与五分钟流程。M9 验收，总阶段数为 10/10。
