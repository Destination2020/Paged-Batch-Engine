# PBE 数据流改进执行计划 V3：从原型走向可解释的推理运行时

> 执行入口已由 [V4 多角色多模态计划](PBE_MULTI_ROLE_MULTIMODAL_EXECUTION_PLAN_V4_20260912.md) 替代，架构见 [V4 设计](PBE_MULTI_ROLE_MULTIMODAL_ARCHITECTURE_V4_20260912.md)。跨进程共享现为主线。以下保留 V3 发布时的状态；2026-09-12 后续源码复核已发现部分 DF0 修复代码，不能照抄下文“未开始”作为当前事实。V4 M0 负责复验，本文安全合同继续适用。

日期：2026-09-12。状态：**执行中：DF0 核心实现完成，DF1 核心实现完成，DF2–DF4 已部分集成，DF5–DF6 尚未完成**。本次执行证据见 [V3 执行记录](data_flow_evidence/v3/execution_report.md)；环境缺失或未运行项目不会标为通过。原 P0–P5 的成果作为代码基础；旧“100%”只保留为历史原型验收，不覆盖本计划的修复、真实压力集成与严格消融。

研究依据：[omniFlow data flow 二次研究](OMNIFLOW_DATA_FLOW_SECOND_AUDIT_20260912.md)。历史计划：[V2 正文，沿用 V1 文件名](DATA_FLOW_REFACTOR_PLAN_V1_20260911.md)。证据入口：[源码与合同验证快照](data_flow_evidence/second_audit_source_manifest_20260912.txt)。

本轮用户要求研究与后续实施计划，因此本文件不是已经执行的功能报告。后续 AI 从 DF0 开始执行，不根据旧完成比例跳过缺陷修复。

## 1. 交付目标与取舍

目标：固定 GPU KV 容量下，建立可验证的 **语义布局计划 → 联合资源准入 → 有界迁移调度 → 计算/输出状态恢复 → 成本驱动选择**，用真实模型和相同工作负载证明每个设计的作用与代价。

核心成果分三组：

- **语义编译**：将逻辑 KV、物理 representation、当前地址分开，支持受限 head 分片及 DIRECT/SENDER/RECEIVER 转换，具备有界计划缓存与真实执行后端。
- **进展与所有权**：共享页/COW/私有增长联合计账；方向 lane 与 demand 保留；请求级多页 restore wave；取消、deadline、未知完成分别处理。
- **真实压力恢复**：checkpoint 接入内存不足抢占和输出队列；根据测量的 save/restore/recompute 成本决策；避免后台下沉/恢复反复抖动。

保留现有 C++ owner 线程、BlockAllocator、CUDA attention/scatter、continuous batching 和 plain/FP8 路径。新增对象应接入真实调用链；不以接口数量或测试数量衡量完成。暂不要求通用 tensor RPC、独立 MM 服务、Redis、任意 dtype/page-size 转换、崩溃恢复或异构 TP 模型执行。

## 2. 当前基线与必须纠正的边界

| 当前事实 | 后续工作 |
|---|---|
| 已有 schema、plan、directory、leases、完整页迁移、single-flight 和 checkpoint 原型 | 增量修改，不另造一套平行实现 |
| planner 支持相同 logical schema；缓存只在局部构造中使用 | 扩展 representation/coverage，长期持有有界 immutable plan cache |
| transfer scheduler 只有一个全局 active 配额 | 增加方向/设备 lane、共享 bytes、demand 入口及依赖提升 |
| checkpoint prepare/commit 存在乱序提交和取消窗口 | DF0 先写可失败回归，再修复 |
| checkpoint 没有 payload byte limit，完成请求未统一回收 | byte admission、记录/元数据生命周期、长期 churn |
| 实际 KV 压力仍触发 free/re-register/recompute | 集成 `recompute/checkpoint/auto`，保留可回退策略 |
| emitted_cursor 未接 emitter，stop 文本状态独立 | 进程内输出序号与入队提交，覆盖 stop-string 延迟发送 |
| 旧 P5 是单点、小负载，部分对照工作量不同 | 保留原始数据，收窄旧结论，新增等价消融与压力轨迹 |

上轮 CPU/GPU 通过记录是历史证据。DF0 对当前 checkout 重跑相关检查并记录当前结果；没有测试资源的项明确 skipped/blocked，不照抄旧 passed。

## 3. 全阶段不变量

### 3.1 身份和覆盖

- `ContentIdentity` 表示模型计算语义：至少包括已冻结的模型 namespace/权重标识、影响 KV 的模型配置和 position/RoPE 语义，以及完整前缀链身份。相同当前页 tokens、不同历史前缀不能碰撞。
- `RepresentationIdentity` 表示 dtype/storage/scales、logical shape、layout/shard coverage 与版本。representation 转换不改变内容身份；量化误差转换暂不进入 exact-copy 路径。
- `PhysicalHandle` 表示 pool/device、index、allocation generation；它不是内容身份。schema epoch、request generation、allocation generation、checkpoint revision 各有独立含义。
- 编译只看 schema/coverage/topology，不持 raw pointer、page slot 或 lease；绑定在准入后全量复验。缺/重/重叠 coverage、非法 dtype/shape/bounds/epoch 在首笔 copy 前失败。

### 3.2 资源与进展

- owner 线程不能阻塞等待一个必须由自己 service 的 completion/admission。运行循环在没有 runnable 请求时仍能处理物理完成和回收。
- 每个物理池独立满足 `free + reserved + resident + quarantined = capacity`，分类必须互斥。source pin 是 resident 上的引用属性，不再重复加一份物理页。
- 同时报告逻辑 credit 与真实分配，不能把二者加总成使用量。共享内容只计唯一 resident/已预留 target 一次，COW 额外目标和 private growth 按请求计费；所有分层池与 staging 分别守恒。
- `pending_free` 只用于避免重复维护，不能给当前 compute allocation 使用。只有 fence/commit 与最后 owner 释放后才进入 free。
- 初始准入全量可回滚；动态增长不得让所有请求持有部分资源永久等其余资源。第一版使用保守全请求 footprint 准入，或非阻塞 chunk growth 加明确 headroom/抢占，必须记录选择。
- 物理 completion unknown 必须保留或隔离两端/staging，不把它降为普通 miss 后在同地址重试。

### 3.3 可见性与输出

- 主线只向普通 prefix index 发布完整 page；同页多个 producer 的 commit 返回实际 canonical winner，输家只回收自己取得且已结束全部 compute/I/O 引用的资源。fence 完成不代表生产者已放弃其页表。
- 逻辑 waiter timeout/cancel 不证明 DMA 停止；最后 waiter 退出后仍由物理 flight drain/fence。绝对 monotonic deadline 不因换 tier 或 retry 重置。
- checkpoint 在整步计算 fence 后冻结，payload 在 READY revision 内不可变；新运行尾页取得独立写所有权。
- `kv_committed_tokens`、`sampled_output_count`、`output_enqueued_cursor` 分开。后者是已提交 outbox item 的连续序号，item 另存关联 sampled-token 区间；stop/EOS/文本缓冲使两种序号不一定一一对应。只承诺进程内输出队列，不等于客户端接收 ACK 或跨重连 exactly-once。
- 性能优化关闭后必须可回到可工作的 oracle。失败回退不能绕过资源所有权或“首笔 copy 前全量预检”。

## 4. 阶段与依赖

| 阶段 | 交付 | 前置 | 当前状态 |
|---|---|---|---|
| DF0 | 修复 checkpoint 反例；重新建立证据与预算/指标口径 | 当前 checkout | 核心完成；sanitizer 与真实模型验收受环境阻塞 |
| DF1 | 语义 representation 编译、转换执行与有界缓存 | DF0 | 核心完成；forced-placement 性能实验待补 |
| DF2 | 共享/COW/private/restore 联合容量准入 | DF0 | 部分完成：分配前准入、共享 restore target 预留已集成；完整 COW/账本待补 |
| DF3 | 方向 lane、deadline、restore waves 与活性 | DF2；使用 DF1 的可用执行路径 | 部分完成：lane、deadline、typed outcome、demand reserve 已实现；完整依赖图与字节账本待补 |
| DF4 | 真正内存压力 checkpoint 与输出提交 | DF0、DF2、DF3 | 部分完成：forced checkpoint、自动恢复、进程内 outbox 状态已接入；异步 save 与流式压力验收待补 |
| DF5 | 水位维护、恢复/重算与转换位置成本策略 | DF1、DF3、DF4 | 未完成 |
| DF6 | 等价消融、压力评估、演示与最终验收 | DF1–DF5 | 未完成；当前无模型 fixture |
| X1 | 两独立进程 provider 内容复用与撤销协议 | 主线完成后的可选首选 | 未选定 |
| X2 | request-private layer-ready 流水 | 主线完成且 timeline 证明整页屏障是瓶颈 | 未选定 |

DF1 与 DF2 可在接口约定后分工实现。DF4 不等待 X1；X2 不改变普通 prefix READY 的完整页合同。

## 5. DF0：先修复可推导反例，建立真实基线

### 修改入口

- `infMain/include/serving/request_checkpoint.h`
- `infMain/source/serving/request_checkpoint.cpp`
- `infMain/source/serving/scheduler.cpp`
- `test/test_request_checkpoint.cpp`、`test/test_scheduler_radix_cache.cpp`
- `tools/bench/data_flow/`、`docs/data_flow_evidence/v3/`

### 实施

1. 保存当前 commit、dirty diff/hash、构建参数、模型 revision/hash、工具链和硬件拓扑；不要把不含未提交改动的 HEAD 当完整实验版本。保留用户已有修改，尤其 demo 文件。
2. 先复现 `prepare(r1)→prepare(r2)→commit(r2)→commit(r1)` 的 revision 回退；新 commit guard 拒绝旧票据并保留当前 READY，不借“正常调用不会乱序”回避公开协议。
3. 先复现 `prepare→cancel_client→commit`；为 request/session 生命周期增加取消代际或等价机制，覆盖尚未 committed 的 ticket。不能仅将 canceled-through 设置为当前 READY revision。
4. 明确 request 完成/cancel、checkpoint 被消费/保留、revision 替换和 store shutdown 的回收规则；旧 ticket 不能复活新请求。元数据 map 也必须有界，不能用永不清理的 tombstone 换取表面安全。
5. 增加 checkpoint byte/record 双上限及预分配预算。字节计算覆盖 K/V/scales、partial tail 的实际保存形式和必要 metadata，校验 size overflow；拒绝时无 payload 部分分配。此处可先同步实现，异步化在 DF4。
6. 修正旧 P5 解释：raw snapshot 与 save/free/restore 工作不同；不同 key 与同 key 不是开关消融；常量数据不能证明乱序正确。原始日志保留，旧数字只用于描述当时两个过程。
7. 增加显式 GPU `kv_cache_blocks_per_layer` 或等价 bytes 上限；实际输出 allocated blocks/bytes。单纯 `memory_utilization=0.02` 受设备状态和分配重试影响，不作为严格公平预算。

### 验收

- 上述两个反例在修复前失败、修复后通过；prepare/commit/cancel/restore 排列测试覆盖 stale handle、重复完成和错误 owner。
- 连续完成至少 `10 × max_checkpoint_records` 个不同请求，再重复至少 10,000 次生命周期；records、payload bytes、revision metadata 回到定义的稳态。
- 满容量时同请求换 revision 的策略明确，失败不破坏旧可恢复 revision；恢复失败仅回滚新目标。
- CPU 与 ASan/UBSan/LSan 相关集合通过；真实 CUDA snapshot/plain/FP8 的定向回归通过。外部 fixture 缺失与真正失败分别报告。
- 输出一份 `v3/df0_baseline.md`，分为“实现存在 / 已集成 / 已运行验收 / 尚待实施”。不从测试数直接推断完成度。

## 6. DF1：把布局转换做成受限的语义编译过程

### 修改入口

现有 `cache/page_schema.*`、`layout_codec.*`、`transfer_plan.*`、`page_migration.*`、`cuda_transfer_backend.cpp`，以及 `serving/pd_handoff.*` 和 P2P adapter。必要时新增 `semantic_transfer_compiler.*`、`transfer_plan_cache.*`；不要复制原 planner。

### 实施

1. 保持逻辑整页 schema 完整性，把物理分片放入独立 representation descriptor。第一版只支持现有 MHA/GQA 的 layer、K/V、KV-head contiguous ranges、token、head dimension，以及 FP8 scales 配套描述。
2. 编译时将 source/destination coverage 归一化。精确 replica 可选择其一，缺 coverage、歧义部分重叠或同一目标多写必须失败；不能按 rank、注册次序推断对应关系。
3. 生成 immutable 计划，操作限于 Copy、Gather/Pack、Scatter/Unpack。支持模式：ExactDirect、SenderPack、ReceiverUnpack；unsupported 返回明确结果。先无量化转换、无任意 page_size 转换。
4. 在真实 Host/P2P 执行器中绑定 component spans 和 staging leases。仅地址确实连续且布局允许才 coalesce；不能跨独立 allocation 假造连续地址。
5. 将 planner/cache 变成 runtime 长寿命成员；容量按 entries 和模板 bytes 限制。key 含 logical/representation fingerprints、codec/topology 版本，不含当前地址；cache hit 返回共享 immutable plan，不复制展开计划。
6. schema/topology 换代后旧模板不能复用；若实现异步 compile，clear-generation 防止旧 compile 回填。当前单 owner 不引入无用途的编译线程；cold compile single-flight 只在存在真实并发调用时实现。
7. P2P adapter 接入同一 compiled plan/绑定合同，不能仅在外层共享 scheduler 而继续绕过语义校验。

### 验收与实验

- 用按 layer/K/V/head/token/element 编码的非均匀 pattern，测试独立 layer block IDs、注册乱序、head split/merge/重排、plain 与 FP8 scales。每个目标完整页对照 CPU reference。
- 验证纯传输 TP1→2、TP2→1 的 coverage mapping，明确这是 shard 转换测试，不宣称 TP 模型 serving。
- 缺 head/scale、重复 destination、bounds/epoch/codec 错误必须 copy_count=0；staging 满时 typed retry/reject，零越界。
- 同 schema 1,000 个换地址页只需一次编译；模板数量/bytes 有界；换 schema 必须 miss。
- 相同语义转换分别强制逐字段、SenderPack、ReceiverUnpack，测 compile/bind/pack/copy/unpack/commit、实际 copy 数/bytes、staging 高水位。小页负收益必须保留。

## 7. DF2：联合准入及共享页/COW 的精确账本

### 修改入口

`base/kv_cache_manager.*`、`sequence_kv_manager.*`、`cache/page_directory.*`、`serving/scheduler.*`；新增 `cache/resource_admission.*`，复用现有 allocation intent。

### 实施

1. 定义可解释的 `RequestPhysicalDemand`：唯一 immutable shared pages、需要物化的 target representation、private growth、可选 COW target、checkpoint bytes、staging bytes。去重范围是同目标池内的同一内容/表示；跨设备副本分别占容量。
2. 引入 owner 管理的 move-only reservation；区分“软活跃预算”与“硬物理容量”。oversized 请求最多独占软预算，不能越过硬容量或整数范围。
3. 选择首版增长策略并写 ADR：推荐非阻塞 growth + 明确保留 decode headroom + 必要抢占；同时实现保守全请求 footprint oracle 作为对照。不能在 owner 中同步等待 delta。
4. COW 最小范围是 P−1/P/P+1 页尾：源保持只读，私有目标 copy/fence 完成后才可写。旧快照/其他请求看到的源不变；先前仅整页共享路径保持可用。
5. 多生产者竞争同 content 页时在 commit 再查 canonical：若已有兼容 READY winner，返回其真实 handle，完成调用方页表/lease 的安全接管；loser 在自身所有 compute/I/O 引用结束且 fence 完成后才回收。若格式/namespace 不兼容则拒绝。主线无需合并不同 producer 的半页内容。
6. restore 对最终同时存活的 GPU 页集合和 private tail，先非阻塞取得整请求容量承诺，再按 wave 物化；如选部分准入，必须定义可回滚/抢占的进展规则。已物化 target 持续计入 resident，不能随 wave 结束归还 credit。staging 是每 wave 的瞬时需求，fence 后可复用，不对所有 wave 累加预留。已 resident 页可被 pin，因此日志必须区分 existing-hit pin 和未准入 miss 的新 pin。

### 验收

- 32 请求共享 N 页时，唯一共享页只计一次，private/COW 按请求累加；Host hit 仍占 GPU restore target，不能漏账。
- 两请求分别持 2 页、总预算 4、同时增长到 3 的反例必须在指定策略下前进或明确拒绝/抢占，不允许双 pending 永久持有。
- 多层后段 allocation failure、grant 后取消、request generation 更新均全量回滚新增资源。
- 两 producer 先后交换/重复完成时只剩一个 canonical 映射；loser 仍有在途 compute 或后续 decode 引用时 slot 不可复用；partial/failed 页不可进入 attention；迁移字节与独立计算数值的断言分开。
- 检查每类池守恒、source pin 持有时间、reservation 峰值、被保守准入浪费的额度和实际 batch size。

## 8. DF3：方向 lane、绝对 deadline 与多页恢复进展

### 修改入口

`cache/transfer_scheduler.*`、`page_migration_flight_adapter.cpp`、`serving/kv_connector_flight_adapter.cpp`、`base/kv_cache_manager.cpp` 的 restore 流程、owner run loops。

### 实施

1. `LaneKey` 按方向与设备/peer 建模：D2H、H2D、P2P 独立 active 配额；同时服从 DF2 的总 bytes、staging、target credits。lane 是软件资源队列，不承诺物理带宽隔离。
2. 从后台 waiter/flight 配额中给 demand 留明确入口；后台占满普通额度时，decode 能 join 既有 flight 或提交关键任务。容量检查顺序不能先把所有 demand 拒绝再讨论 donation。
3. waiter 保存 original priority、absolute deadline 和生命周期；flight effective priority 从存活 waiter 重算，取消/到期后可撤销 donation。decode 的 reclaim/restore dependency 也继承关键优先级；依赖图拒绝环。
4. 保留 aging，在 monotonic 时间或明确可解释的调度时间上定义；前台/后台等待指标按实际毫秒输出。不得将 busy poll 次数当等待延迟。
5. submit/ensure_resident 返回 `Admitted/Joined/RetryCapacity/MissSource/UnsupportedSchema/Stale/Unsafe` 或等价 typed outcome。确定拥塞不立即变永久 cache miss；unknown 不在同目标地址 fallback。
6. 增加请求级 `RestorePlan`，区分 resident、own-flight、join-flight、private-tail。按 DF2 先保证持久目标容量，再调度自身 owner 页、等待 joined 页；每 wave 单独借用并归还瞬时 staging credit。最后重新检查完整连续 prefix 才安装页表。
7. logical target key 基于 content+destination representation+epoch。PBE 原 key 已没有 source，不重复做“去掉 source”；未来 Host/P2P 同目标来源选择在 intent 中完成，不能因后端路径不同产生互相覆写的 target。
8. request timeout 解除等待，物理 job 使用独立执行/回收状态；source/destination/staging 等到可证明安全时释放。shutdown 停准入、drain、提交/回滚，再销毁资源。

### 验收

- 32 waiter/取消 31 保留已有测试；新增短 deadline demand + 长 preload，queued/running/dependency 未完成三种时点都正确。
- 后台填满配额，decode→D2H 腾空间→H2D 恢复仍能获得资源；记录 P99 demand wait 与最大后台等待。所有活性测试声明后端持续完成且工作集可准入的前提。
- 强制 `[A,B]`/`[B,A]` owner/join 交错，小 staging 下多 wave 完成、每页物理迁移一次。一个 waiter 取消不能污染另一请求。
- 同 key/schema 换代不合流；source safe failure 后可换源，unknown 旧目标隔离后才允许新目标。
- real gated CUDA copy 在各取消点验证不早复用；CPU 状态机/故障注入、定向 compute-sanitizer、资源 churn 均通过。

## 9. DF4：真实抢占 checkpoint 与输出队列语义

### 修改入口

`serving/scheduler.*` 的 `preempt_sequence`、`request_checkpoint.*`、`base/kv_cache_manager.*` snapshot、`serving_online_engine.cpp`、streaming channel/stop 文本缓存、`SequenceState`、实际模型测试。

### 实施

1. 增加 `preemption_policy=recompute|checkpoint|auto`。先实现 forced checkpoint 与 forced recompute 的真实压力模式；auto 在 DF5 启用。不得仅从测试直接调用 suspend 来代替调度器触发。
2. 选择 victim 后进入 Saving；只冻结受害请求，复用 DF3 有界迁移后端异步保存。禁止每次 checkpoint 先 `drain_cache_transfers()` 全局等待所有无关传输。
3. payload prepare 完成且整请求 fence 成功，再发布 READY manifest、释放 victim GPU 所有权；只把实际释放的块计入可调度 free。取消/部分失败不能破坏仍可运行的源状态。
4. restore 在 DF2 准入后分配 fresh private target，通过所有页/模型/revision/request generation 验证，再原子安装 KV、next input、RNG、stop 状态及 runnable 队列位置。
5. 输出明确以有界进程内 outbox 为提交边界：sampled token 有稳定索引，outbox item 另有连续提交序号及关联 token 区间；入队和推进 item cursor 在 owner 中共同完成。恢复后不重复创建已提交 item。EOS/stop 吞掉的 token 不要求对外出现；不能用 socket write 返回就宣称客户端确认。
6. 保存 stop-string 跨 token 缓冲、UTF-8/解码残余及尚未入队文本的必要状态，或将其放入独立持续存活的 request output object 并由 checkpoint 明确引用。不能只保存 token_count 后丢掉隐藏文本。
7. 输出队列 backpressure 不得阻塞物理回收；完成/cancel 后清理 checkpoint、outbox、revision metadata。复用 client ID 时用 request generation 防旧 completion 混入。

### 验收

- 显式 KV block 数不足时自动触发 victim save→free→其他请求进展→restore，证据含调度决策与实际 preemption/recompute counters。
- 在 P−1/P/P+1、prefill chunk 中途、已采样未 forward、KV commit 后未入队、已入队、restore 中取消/新 revision 等边界运行。
- 同计算路径固定 seed/counter 下 token 序列精确相同；batch 不同则分别检验 RNG 输入/生成状态和精度定义的 logits，不能忽略差异也不能错误归因。
- 真实 streaming consumer 收到已提交 outbox item 的连续序号，无重复/缺口；token 区间与 stop/EOS 截断对应，stop-string 跨多个 token 仍正确。网络断连只按明确的 abort/进程内语义验收。
- 受害请求保存期间，另一个 running 请求持续产出 token；timeline 与 P99 gap 证明 owner 未因全局同步停住。
- 保存和恢复成本都进入报告；10,000 次压力/取消生命周期后内存、tickets、events、outbox 均无持续增长。

## 10. DF5：解释得清楚的成本决策与水位维护

### 10.1 成本模型

分别测量：

```text
T_restore = queue + bind/pack + physical_copy + unpack + fence/commit
T_recompute = recompute_prefill_for_committed_context
checkpoint decision additionally includes T_save and victim/other-request delay
```

上式是解释维度，不得直接把重叠区间相加冒充 wall time；最终对照真实 critical path。以模型、representation、页数/长度、并发与 lane 拥塞分桶，先静态测量表，再简单 EMA；样本不足时回到明确定义的保守策略。决策日志记录估算项、选项、实际结果和预测误差。

- restore/recompute 选择要求目标资源可准入；快但永远拿不到内存的候选不能选。
- 使用 hysteresis/min-residency time，避免轻微波动导致 offload→restore→offload。
- 对 checkpoint，不把 save 成本藏在预热里；对已有 host cache，分别报告 restore-only 与含先前创建副本成本的端到端场景。
- Sender/Receiver placement 先 forced-mode 测量和静态启发式，再决定是否启用成本选择；不能声称全局最优。

### 10.2 后台水位

新增单机 `WatermarkController`：

- free 低于 low 触发，补到 high 停止；只提交 target/deficit，不在排队时固定 victim。
- admission 时重新采样 actual free 与 pending-free，过期维护可 no-op。
- pending-free 用于防止重复下沉，不进入当前 compute free；取消/失败撤销预测。
- 限制每 wave bytes、background active 和 host 容量；victim 被重新 pin 后跳过；无可回收页时 backoff。
- 主线仅处理本地完整 cached pages；X1 才增加 alternate serviceable provider 检查。

### 10.3 冷前缀计算策略

保留 transfer single-flight；增加实验性 `cold_prefix_policy=independent|bounded_wait`。后者只等待可复用 prefill page 发布，设有界等待与超时独立计算回退，绝不能等待首请求整个 decode 完成。先做相同轨迹两模式，只有数据支持才加入 adaptive。

### 验收

- free 从 low 以下回到 low/high 之间仍继续维护，到 high 停；排队期间已满足 high 则 copy=0。
- copy failure/pin race/host 满不会虚报 free 或持续空转；测 urgent eviction 次数、background bytes 和短期 re-restore 比例。
- 构造 restore 快、recompute 快、lane 拥塞使选择反转三类场景。决策偏离离线 oracle 时报告 regret，不人为调参隐藏负收益。
- 冷/热/部分热共享 prefix 的突发与持续到达，比较 TTFT/P99、actual prefill tokens、batch size、private pages、goodput。不能只用 saved FLOPs 评价策略。

## 11. DF6：等价消融、可复现实验与演示

### 11.1 先修正实验方法

1. 开关对照必须使用同一组 token IDs、到达时间、共享关系、采样参数、GPU block 上限、host/checkpoint/staging bytes。single-flight off/on 不改变 logical page keys；off 使用独立私有目标完成重复 copy，维持相同总 target/staging 预算，最终 canonical 单赢家，输家按完整引用合同回收，禁止并发写同一目标。lane 消融固定总 active/streams/backend capacity 和共享 byte budget，只改变额度分配/优先级；增加总并发的实验另列。
2. 布局对照执行相同语义转换，使用非均匀 pattern 和输出 checksum；每轮通过 benchmark barrier/ClobberMemory 或等价黑盒消费保留实际搬运，并单列/控制屏障开销。仅最终 checksum 不足以阻止重复 memcpy 被合并。将 correctness assertions 放在计时区间外，记录 Release/CPU affinity。
3. 事务消融两组都做同样 payload save+restore；只在隔离 benchmark 下比较额外 journal/validation 成本。生产路径不提供关闭所有权/安全校验的危险开关。
4. checkpoint 基线从相同压力触发点开始，计入 victim 保存、其他请求延迟、恢复或重新 prefill；不中断请求不是“重算抢占”对照。
5. 模型/kernel 预热与 cache 温度分开；随机或交替模式顺序。每个正式案例至少 3 次独立进程，主结论建议 5 次，输出所有 run 与波动。
6. 小样本不能声称可靠 P99。延迟尾部结论需要足够请求/时间窗口，注明数量、超时/失败和排除规则；failed/timeout 请求不能从统计中消失。

### 11.2 最小实验矩阵

| 实验 | 变量 | 必需证据 |
|---|---|---|
| E1 语义编译 | exact/reorder/split/merge；plain/FP8；冷/热 plan | payload、compile/bind P95、cache hit、copy 数/bytes、staging |
| E2 共享加载 | 相同 32 waiter 流量，flight off/on；取消 0/31/32 | 实际物理 bytes、queue P95/P99、资源释放、剩余 waiter |
| E3 lane/依赖 | 单全局配额 vs 方向 lane；D2H 后台 + H2D demand + P2P | wait、deadline misses、后台进展、decode gap、GPU timeline |
| E4 准入/COW | 并发 1/8/32；共享 0/50/90%；工作集占额度 50/100/150% | 守恒、OOM/拒绝、batch size、增长前进、COW bytes |
| E5 压力恢复 | 512/2K/8K 可支持长度；recompute/checkpoint/auto | T_save/T_restore、真实重算 tokens、goodput、TTFT/gap、盈亏曲线 |
| E6 水位 | off/on；低复用/高复用；突发/持续到达 | urgent evictions、churn、wasted bytes、前后台等待 |
| E7 冷计算等待 | independent/bounded_wait；冷/部分热；并发 1/4/16 | 重复计算、首 token、batch 效率、公平性、负收益 |

长度先检查模型和当前 kernel/workspace 上限，超出时标 unsupported 并选仍能触发压力的替代值，不能偷偷修改模型语义。Qwen2-0.5B 用于完整路径；7B 级模型不是前置，只有需要改变计算/传输比且可用时追加。无需把全部维度做全笛卡尔积，但每条主张必须有直接对照与至少一个不占优场景。

### 11.3 指标与原始数据

- 业务：完成/失败/超时、output tokens/s、goodput（预先固定 TTFT/gap SLO 后满足者）、TTFT、每 token gap 分布、端到端延迟。
- 计算：实际 prefill/recompute tokens、scheduled/committed tokens、batch size、victim/retry/preemption 次数。
- 传输：logical demand、physical flights、component copy calls、bytes、join/cancel/late completion、safe/unsafe/stale 各 outcome。
- 时间：admission/queue/bind/pack/copy/unpack/fence/commit；CPU wall 与 CUDA event 口径分开。
- 容量：各 pool free/reserved/resident/quarantined，source pin 峰值及持续时间，staging/checkpoint/outbox/模板 bytes。
- 稳定性：重复完成、allocation failure、迟到 fence、无 runnable drain、最终守恒；fault quarantine 保持不可分配不算泄漏失败。

每条 trace 关联 request generation、page/representation、flight/waiter、checkpoint revision。开启详细 trace 与正式吞吐分别跑，避免追踪开销污染主结论。至少保留一条真实 Nsight Systems 或等价 GPU timeline，证明或否定 overlap。

### 11.4 最终交付

`docs/data_flow_evidence/v3/` 保留 manifest、commands、raw logs、results.json/csv、原始 trace 和自动生成图表；`tools/bench/data_flow/v3_*.py` 可一键重复有限矩阵。

最终报告逐项列“来自原项目 / 借鉴 omniFlow / PBE 加强”，写清机制收益与容量增加收益。面试材料只引用本轮等价实验；旧 +39.27%/+20.44% 等单点数字可留历史，不当作新设计或相对 SGLang 的提升。

## 12. 可选扩展：先选一个，单独验收

### X1：同机两进程跨请求 provider reuse

这是最能延伸 omniFlow 与 HiCache 组织差异的加分项，前提是主线已有稳定证据。

- 两个独立模型进程、固定 endpoint registry，B 起始无本地 prefix。第一版使用可用的同机 IPC/受控 transport；真实数据必须过进程边界，不能用同进程 P2P 代替。
- `PrepareToken` 绑定 content/representation、provider generation、过期时间和精确组件集合。provider logical queue 不捕获 slot/address；admitted 后固定源；requester 已准入的目标/staging 仍占容量。
- attach descriptor 完整预检；replay、缺/额外 component、pointer/bytes/codec 变化在 DMA 前拒绝。
- registry 发布“可服务能力”，区分有 lower bytes 与能提供指定 representation。withdraw ACK 前持必要 lease；ACK unknown 和晚到回复按有序状态机补偿，不以 TTL 替代物理 fence。
- 先本地 registry/fault adapter 证明协议；只有部署需要再引入 Redis。需要跨机 RDMA 时另查 NIC、GID、连通性和所有权协议。
- 验收 B 实际少做 prefill、TTFT/bytes、token/payload 对照；provider 退出、withdraw reply 丢失、迟到 completion、独占最后 provider 和关闭过程均安全。

### X2：私有 layer-ready 与 canonical READY 分离

SGLang 已有逐层 H2D/compute overlap；omniFlow 当前完整页发布。PBE 可研究在保持普通 prefix 完整页合同下，让特定恢复请求持有私有目标并按 layer fence 消费。

- 只有请求私有 `MaterializedGpuView` 能看到 layer-ready；普通目录仍在全部 layer/component 完成后发布 READY。
- later-layer 失败时取消该请求后续计算，drain 所有已用该页的 compute/copy；不能将半页交给新请求或假装回滚已执行输出。
- 必须先证明该模型按层依赖和输出提交能支持此边界，明确不与 graph capture 的静态地址假设冲突。
- 只有 timeline 显示整页等待确为关键瓶颈、且阶段 DF4 无关请求已不被全局同步阻塞时启动。

X1/X2 不纳入 DF0–DF6 的分母；只有明确选择后才建立独立执行清单，不能静默勾选完成。

## 13. 交给后续 AI 的执行指令

可直接使用：

> 执行 Paged-Batch-Engine/docs/DATA_FLOW_REFACTOR_PLAN_V3_20260912.md。先读二次研究和当前源码，从 DF0 的 checkpoint 反例、生命周期/byte 容量和实验口径开始。保留已有修改，只安装缺失依赖，沿用可用模型/构建目录。按 DF0–DF6 逐阶段实现、运行验收、保留原始证据并更新状态；有通过的真实集成和实验才算完成。复用现有代码，不以另起一套接口、fake backend 或手动 API 测试代替真实压力/输出/GPU 证据。不要复用历史 100% 宣称；明确报告失败、跳过和负收益。X1/X2 单独选择和计数。

实现过程每个阶段至少交付：当前触发问题、代码变化、设计不变量、定向失败测试、相关集成、实验命令与原始数据、剩余限制。可用逻辑变更分组或提交组织审查，不能将不属于本任务的既有改动混入。遇到依赖/硬件限制时先继续不依赖它的工作，并准确标记验收缺口。

## 14. 完成清单

- [ ] DF0 反例修复、生命周期与实验基线
- [ ] DF1 语义转换与有界模板缓存
- [ ] DF2 联合容量、COW 与 canonical 收敛
- [ ] DF3 lane、deadline、restore waves 与活性
- [ ] DF4 真实压力 checkpoint 与输出队列提交
- [ ] DF5 成本选择、水位与冷计算策略
- [ ] DF6 等价实验、可复现证据与面试演示
- [ ] 对 X1/X2 给出有证据的选择记录；未选择不算实现
