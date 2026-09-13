# omniFlow data flow 二次研究：PBE 的下一轮设计重点

日期：2026-09-12。范围：真实源码、有限 CPU 合同测试和 SGLang 官方资料复核；本轮不修改 PBE 推理实现，不产生新的 GPU 性能结论。

后续实施入口：[数据流执行计划 V3](DATA_FLOW_REFACTOR_PLAN_V3_20260912.md)。本报告补充此前 [omniFlow/SGLang 对照](OMNIFLOW_SGLANG_DISTINCTIVE_DATA_FLOW_20260912.md)，不将此前原型验收等同于完整生产能力。

## 1. 建议的项目主线

把项目讲成一个问题：**固定 KV 容量下，怎样让计算拿到语义正确、及时可用的数据，并在拥塞、抢占和取消时维持进展？**

最值得深入的三条线：

1. **KV 传输的语义编译。** 同一份 KV 的内容身份、表示格式、物理地址分别建模；从语义覆盖生成 DIRECT/发送端 pack/接收端 unpack 计划，再按当前资源绑定。这比“支持 CPU offload”更能展示对 GQA、布局、DMA 粒度和内存带宽的理解。
2. **计算准入与迁移资源共同调度。** 一个请求到底增加多少独占页，多少共享页已付费，恢复前要腾出多少空间，后台复制会不会挡住 decode？用资源账本、方向 lane、依赖提升和真实等待指标回答。
3. **真实抢占中的精确恢复及成本选择。** 把 checkpoint 接到实际内存压力、RNG 和输出队列，测量保存、恢复、重算与受影响请求的成本；解释为什么有时应该重算。

omniFlow 贡献主要是组织这些合同的方式。C++ owner/RAII、逐次分配 generation、输出队列提交、成本模型和更严格的恢复协议，是 PBE 可以独立完成和解释的加强。不能把参考实现包装为原创，也不能把整个领域已有能力说成行业独有。

## 2. 比较版本与证据等级

| 对象 | 本次边界 |
|---|---|
| omniFlow | 本地 `omni_flow_sglang`，HEAD `e0827035fd4ceeb1df5a68a87a69fc575b627976`；审计的是当前工作树，`slots_manager.py`、相关测试及 `memory_manager.py` 有未提交修改 |
| SGLang 本地对照 | `omni_flow_sglang/sglang_0516`，HEAD `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`；代码结论限定该快照 |
| SGLang 当前能力交叉检查 | 2026-09-12 访问官方 HiCache、PD、Unified Cache 和 session 文档/源码；没有声称审计整个最新 main |
| PBE | HEAD `f3597a763b6cd5cf60b826baa2dc94395c3d02fd` 加现有未提交实现；以代码调用链和测试覆盖为准 |
| 硬件 | 本次读取到 2 × H20-3e，143771 MiB/卡，GPU 间 NV18；支持后续同机实验，不代表独占或跨机 RDMA 已验证 |

关键源码哈希记录于 [研究快照](data_flow_evidence/second_audit_source_manifest_20260912.txt)。文中“缺口”来自源码，“已通过”只用于实际运行过的测试；PBE checkpoint 的新反例目前是静态推导，实施时必须先做失败回归。

本次运行 omniFlow capacity/page-allocation/transfer-scheduler/three-lane 的 44 项 CPU 合同，以及 4 项 owner/join/wave/final-barrier 集成合同，全部通过；命令与结果记录在研究快照。它们不替代真实 GPU/远端性能测试。

## 3. 哪些差异值得借鉴

### 3.1 语义表示驱动的搬运，与具体设备布局分开

omniFlow 的 `LogicalAtomV1` 按 layer、KV、head 等标签识别数据。`build_remote_reshard_plan` 校验完整覆盖，生成 gather/scatter；`choose_permutation_placement` 决定 DIRECT/SENDER/RECEIVER。地址在执行时才绑定。

- [remote_reshard.py](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/remote_reshard.py)：重点看 `LogicalAtomV1`、`build_remote_reshard_plan`、`choose_permutation_placement`，约 344、784、998 行。
- [remote_reshard_runtime.py](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/remote_reshard_runtime.py)：115–255 行附近为有界 LRU、冷 miss 合并、clear generation 和固定尺寸 schema fingerprint；306 行附近明确规划不含 tensor/page/slot。

SGLang 已有异构 TP 的 gather→staging→transfer→scatter。可比较的是“用语义覆盖生成通用计划并选择转换位置”，不能说 SGLang 不支持异构 TP。[官方 PD 说明](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)

PBE 当前已有简单模板缓存，但只接受相同 schema fingerprint；生产迁移引擎构造时创建一次局部 planner，复用一张 `plan_`。下一步是增加受限的 head 分片表示、跨布局计划与真正长期使用的有界缓存，避免重新造一个同功能 map。

**面试追问：** GQA 的 KV heads 与 query heads 有何区别？什么情况下是 replica，什么情况下是 shard？为什么相同 token 不能跨模型/位置编码复用？为什么 48 次小 copy 有时不如 pack 一次、但小页时又可能相反？

**代价与边界：** 编译、预检、pack、staging 都有成本。omniFlow placement 目前主要是启发式，不能说已有在线最优成本模型；纯张量分片转换通过不代表 PBE 已支持异构 TP 模型 serving。

### 3.2 按共享内容计费的容量准入，而非只看当前空闲块

omniFlow 用 `PageAllocationIntent` 分开 immutable reads、COW source/private target、新私有页，`PageAdmissionDemand` 对 shared content keys 去重计费，`PageAdmissionLease` 管理持有、增长、释放及 poison。它把“业务需要什么”放在“分配哪些 slot”之前。

源码：[page_allocation.py](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/page_allocation.py)、[capacity_admission.py](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/capacity_admission.py)。

SGLang 同样有 token budget、共享前缀和内存保护；这里值得学习的是把共享/COW/私有需求写成显式可验证的容量合同，不能将“有准入”作为独有功能。

**本次发现的限制：** 原子首次准入不等于任意动态增长无死锁。预算 4 页，A/B 各持 2 页，再同时请求增长到 3 页，二者都可能持有原资源等待剩余容量。本次最小 CPU 实验观察到两者均 pending；取消并 release 后容量恢复。omniFlow 的常见 generate 路径预估 `max_new_tokens` 全程 footprint，有助于避开这一情形；PBE 若按 chunk horizon 增长，必须额外定义 headroom、非阻塞拒绝或抢占规则。

**面试追问：** 32 个请求共享 100 页时如何计费？最后一个 token 触发新页时为什么可能死锁？pending eviction 能否当 free？source pin 是否应该再次计成一份物理内存？

### 3.3 统一生命周期，同时保留独立方向资源与前台入口

omniFlow 的 OFFLOAD/UPLOAD/REMOTE_COPY_TO 有独立 lane；每 waiter 有绝对 monotonic deadline；PRELOAD 留出 DEMAND 准入额度，前台能先进入系统，再 join 和提升后台 flight。presence 区分 READY、KNOWN_MISS、FAILED_SAFE、UNKNOWN_QUARANTINED、STALE_SCHEMA。

源码：[transfer_scheduler.py](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/transfer_scheduler.py)、[upload_presence.py](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/upload_presence.py)、[DESIGN.md 第 4、6 节](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/DESIGN.md)。

SGLang HiCache 已有 load/write streams、优先级和预取策略；差异是 omniFlow 在上下迁移与 peer copy 上共用 task/waiter/终态合同。[HiCache 官方设计](https://docs.sglang.io/docs/advanced_features/hicache_design)

PBE 现有全局 `max_active_` 不能等同于方向 lane；按 poll tick 的 aging 也不等同于实际 deadline。前台若连 waiter 配额都拿不到，priority donation 没有入口。新增设计须同时解决方向准入、共享 byte budget 和 decode 依赖的 D2H reclaim 提升。

**代价与边界：** lane 不隔离 HBM/PCIe/NVLink 带宽；不能抢占已经开始的 DMA。等待上界只在资源可满足、后端持续完成的前提下成立。fixed wave 是调度时机，不自动把任意请求合成一笔 DMA。

### 3.4 共享传输与共享冷计算，应采用不同策略

omniFlow 明确移除 cold compute prefix single-flight：未发布 KV 的请求可以分别用私有页计算，完整发布时再收敛；同页已存在数据的搬运则合并 flight。理由是冷计算等待可能损害 TTFT 与 continuous batching，不能只优化重复 FLOPs。

依据：[DESIGN.md](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/DESIGN.md) 1363、1596 行附近的冷前缀决策，以及 `PageAllocationIntent` 的 private allocation 路径。对照本地 SGLang [schedule_policy.py](../../omni_flow_sglang/sglang_0516/python/sglang/srt/managers/schedule_policy.py) 278–308 行：批内共享长前缀的低命中请求可被暂时降优先级。后者是调度启发式，不应描述成所有 SGLang 请求都被强制 single-flight。

对 PBE 最有价值的是保留两种目标：迁移按目标页去重；计算等待作为可选策略，用相同到达轨迹比较“独立 prefill”和“有界等待首个生产者”，测 saved tokens、TTFT/P99、吞吐和公平性。

**canonical 合同：** 只有已确认完整、可服务的副本才能胜出；完成调用方页表/lease 的安全接管后，输家新资源还须等全部 compute/I/O 引用结束且 fence 完成才能回收，不能覆写胜者或提前释放生产者仍使用的私有页。两个 producer 因 batch 数值差异不一定逐字节相同；要求同格式物理迁移位级一致，与要求独立计算数值一致应分别测试。

### 3.5 “保存了字节”与“能对外服务”分开

omniFlow 的 `cprov` 表示可直接远端读取的完整 L1 provider。下沉到 L2/L3 只证明有本地副本，不能继续宣告同样的服务能力。strict 模式将 copy/fence、metadata withdraw ACK、L1 reuse 排序；超时不等于 Redis 操作被取消，迟到 absent 与补偿 present 走同一个有序 reconciler。

- [slots_manager.py](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/slots_manager.py)：4074–4101 行串行 reconciliation；4302–4349 行等待预算；5047 行附近 strict gate；5697–5741 行 ACK 后复验并复用。
- 同文件 1770 行附近 safe-only watermark：lower-backed 不能绕过 provider 可服务性检查。
- [remote_copy_admission.py](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/remote_copy_admission.py) 与 [memory_pool.py](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/memory_pool.py)：逻辑 prepare/admission 后固定源，phase 2 校验目标 descriptor；token 与精确 page/component 集绑定。

SGLang 官方 HiCache 说明 L1/L2 是实例私有层，跨实例复用经过配置好的 L3；PD 又有单独的 peer transport。因此差异应限定为“直接 provider 发现、撤销和驻留生命周期的组织”，不应说 SGLang 没有跨实例共享或远端传输。[官方 HiCache 层级范围](https://docs.sglang.io/docs/advanced_features/hicache_design)

这是很好的后续面试加深项，但需要真正的跨进程目标才值得引入 provider 协议。本轮主线先做好本地资源账本；不为了架构图增加 Redis 和独立 MemoryManager 服务。

### 3.6 内容页、生成历史和已发送输出，是三种不同状态

omniFlow history/KV 分离，明确最后采样 token 可能还没有 KV；history 读取使用乐观 plan→data→plan 验证，不是跨对象 MVCC。SGLang 本地 [SessionSlot](../../omni_flow_sglang/sglang_0516/python/sglang/srt/session/streaming_session.py) 68、101 行也有 save/restore 和 ownership 转移。

因此 PBE 的价值是把真实压力抢占、不可变快照、revision、RNG、stop 状态及输出队列提交连起来；“有 session”或“保存一个计数器”都不足以支撑精确恢复。当前 SGLang 官方的 session-aware radix reference 策略又是另一种能力，不能与 streaming session 或完整 checkpoint 混称。[官方 session-aware cache](https://docs.sglang.io/docs/advanced_features/session_radix_cache)

**面试追问：** sampled、KV committed、queued-to-client、client-acknowledged 为什么不能共用一个 token count？保存完成但尚未释放 GPU 时取消应怎么处理？有输出队列去重是否已经是网络 exactly-once？

## 4. PBE 现状：需要先修复的事实

此前“8/8、100%”保留为历史原型验收记录。以下发现说明它不能作为全部设计主张、压力运行和端到端发送语义均已完成的证明。

| 项目 | 代码证据 | 本次结论 |
|---|---|---|
| revision 回退 | `infMain/source/serving/request_checkpoint.cpp:55` | commit 未拒绝旧 revision；prepare r1/r2→commit r2→commit r1 可令 current 回退并删除新记录；静态反例，待执行复现 |
| cancel 后 preparing 复活 | 同文件 143 行 | cancel 只记录 current committed revision，prepare→cancel→commit 缺少覆盖 preparing ticket 的取消代际检查 |
| checkpoint 容量 | 同文件 27、125、157 行及 header 84–88 行 | 限记录数但不限 bytes；restore 后仍 READY，完成请求未清理；revision maps 也需生命周期边界 |
| 真正的压力抢占 | `infMain/source/serving/scheduler.cpp:757,891` | 仍 free/re-register/recompute；手动 suspend/restore API 未成为压力策略 |
| 输出提交 | `infMain/source/serving/serving_online_engine.cpp:972,1000` | emitter 的文本缓冲、push_token 与 checkpoint emitted_cursor 未统一 |
| lane 与失败类型 | `infMain/source/cache/transfer_scheduler.cpp:42,114`；`page_migration_flight_adapter.cpp:19` | 单 active 配额；多种失败折叠成 bool/FailedSafe；拥塞重试与永久 miss 难以区分 |
| 模板缓存范围 | `infMain/source/cache/transfer_plan.cpp:26`；`page_migration.cpp:43` | 已有缓存但仅相同 schema；生产持有一张预生成 plan，未验证多布局缓存收益 |
| 性能证据 | `test/test_p5_ablation.cpp:68,99,107,125` | 常量 payload；snapshot vs roundtrip 工作量不同；不同 key vs 同 key 并非同流量的开关消融 |

已有字节往返、在途取消和模型恢复测试仍有价值。需要补的是未覆盖的反例、真实调用链和等价实验，不能通过重命名 API 或增加单测计数来代替。

## 5. 哪些能力不适合当差异化卖点

- 多级缓存、radix logical/physical 分离、异步 memcpy、prefix reuse、session 保存、异构 TP staging：SGLang 都已有相关路径。
- 把 `cudaMemcpyAsync` 写进代码不证明 overlap；要有 GPU timeline 和受影响请求的 token-gap 证据。
- 逐层就绪：SGLang HiCache 已实现。omniFlow 当前明确采用 complete-page 可见性，因此它适合作为“正确性边界与流水粒度”的比较题，而非 omniFlow 独有功能。[官方逐层 overlap 说明](https://docs.sglang.io/docs/advanced_features/hicache_design)
- Unified Cache 已把逻辑树与组件资源/I/O 分开；不能再说 SGLang 的树必须直接管理所有物理资源。[官方 Unified Cache 设计](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/unified_cache/components/README.md)
- “吞吐提高 X%”只能引用同预算、同轨迹、同功能的实测。此前 GPU warm vs recompute 的单点结果不等于新架构比 SGLang 更快。

## 6. 建议的面试演示

完成 V3 实施后，用三张证据图展开：

1. **语义与地址图**：同一逻辑页在不同 layer/head layout 和 block 地址间迁移；缺一个 scale/head 时首笔 copy 前失败。展示实际 copy 数、pack 成本和 plan cache 命中。
2. **资源与时间线图**：后台 D2H 拥塞、前台 decode 需要 H2D，展示联合准入、依赖提升、独立 waiter 取消、资源守恒与后台最终进展。
3. **恢复/重算盈亏图**：横轴 prompt/已计算长度或 KV 压力，纵轴 TTFT、recompute tokens、P99 gap 和 goodput；同时标保存成本与恢复不划算的区间。

可用表述应在实现和实验后再写入简历：

> 我参考 omniFlow 的语义数据流，在 C++ 推理引擎中把 KV 内容、表示和地址拆开，按资源预算生成并执行搬运计划；将共享加载、计算增长和抢占恢复纳入同一套所有权与进展约束。我用故障注入证明取消不会提前复用地址，并用真实压力负载测出恢复与重算各自适用的条件。

更有说服力的不是功能数量，而是能解释一个看起来更快的方案为什么有时会更慢，以及代码如何守住这个边界。
