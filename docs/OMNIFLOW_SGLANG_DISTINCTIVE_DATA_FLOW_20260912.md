# omniFlow / SGLang 数据流对照与 Paged-Batch-Engine 选题

日期：2026-09-12。性质：源码审计与设计建议；本报告中的 C++ 新接口、实验和性能收益尚未实现或验证。

本文件是上一轮实施前的历史研究。当前原型、复核发现和后续选题以 [二次审计](OMNIFLOW_DATA_FLOW_SECOND_AUDIT_20260912.md) 与 [V3 执行计划](DATA_FLOW_REFACTOR_PLAN_V3_20260912.md) 为准；不再将本文件中的“尚未实现”或上一轮完成百分比直接当当前状态。

## 1. 结论与比较范围

建议将项目定位收敛为：**具有语义化搬运计划、可验证迁移事务和请求恢复能力的 C++/CUDA KV 数据运行时**。

比较的价值主要在接口边界、资源所有权和失败语义。无法从检查若干版本证明某项技术为 omniFlow 或行业独有，也没有依据宣称整体性能优于 SGLang。

参考版本：

- omniFlow：工作区 `omni_flow_sglang`，HEAD `e0827035fd4ceeb1df5a68a87a69fc575b627976`，2026-08-25；工作树包含修改，特别是 `slots_manager.py` 和 `memory_manager.py`。本报告引用当前工作树，不能视为该 commit 的完整原样快照。
- SGLang 主比较：`omni_flow_sglang/sglang_0516`，clean HEAD `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`，2026-07-24，release/v0.5.16。
- `omni_flow_sglang/sglang` 为本地兼容版本 `f7ed34d835`，2026-08-25，存在修改，不将其等同于最新 upstream。
- 另交叉核对 upstream 官方当前文档的 HiCache、Unified Cache Components、异构 TP staging；没有对最新 main 做全仓审计。
- Paged-Batch-Engine：`f3597a763b6cd5cf60b826baa2dc94395c3d02fd`，保留上次 demo 修复和环境文档。

omniFlow 的 data_flow/README.md 存在旧描述；应结合实际代码与 [DESIGN.md](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/DESIGN.md) 判断。目前基线明确是 complete-page、chunk-wise 发布，不是逐层发布。

## 2. 不能作为独有卖点的能力

| 候选卖点 | SGLang 已有证据 | 合理的比较表述 |
|---|---|---|
| GPU/CPU/远端三级缓存 | `hiradix_cache.py`、`cache_controller.py`、Mooncake storage | 比较管理边界、调度和失效语义 |
| 逻辑前缀与物理驻留分离 | `radix_cache.py:217` 的 key/value/host_value；GPU evict 后保留 host 数据 | 不宣称 SGLang 驱逐 GPU 就丢失全部逻辑索引 |
| 树与 I/O 分离 | upstream UnifiedTreeCore / NodeId / deferred CacheAction | 将 C++ PageDirectory 当工程选择，不当独有发明 |
| 异步传输、计算搬运重叠 | `cache_controller.py:70,797` 的逐层完成 event | omniFlow 完整页提交有更粗的可见性边界，也会损失部分流水机会 |
| 异构 TP 和 staging | `disaggregation/mooncake/conn.py:715` M-to-N TP；官方 GPU staging 文档 | 比较语义标签规划、coverage 校验和转换执行位置 |
| 多类型状态缓存 | `hicache_storage.py:58,93,156` 多 PoolTransfer 与各池命中规则 | 不宣称 SGLang 只能存标准 KV |
| session 保存恢复、迟到完成保护 | `streaming_session.py:69,101`；`session_radix_cache.py:17` | 比较迁移与生成进度的共同提交合同 |

上述 SGLang 源码路径位于 `../../omni_flow_sglang/sglang_0516/python/sglang/srt/`。

官方交叉证据：[HiCache design](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/hicache_design.mdx)、[Unified Cache Components](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/unified_cache/components/README.md)、[异构 TP staging](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/pd_disaggregation.md)。

## 3. 优先项 A：迁移事务、延迟绑定地址和完整页可见性

### omniFlow 实际实现

- [transfer_plan.py:178](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/transfer_plan.py#L178)：任何 copy 前检查全部源/目标 component、shape、dtype、slot 范围和有效副本位置。
- [memory_pool.py:3424](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/memory_pool.py#L3424)：`prepare_batch_copy` 的 provider 排队项只含 name/page/schema/semantic keys；`execute_remote_copy_to` 在 worker admission 后才 `search_for_transfer` 并 pin 源页。
- phase 2 才附加 requester 地址，token 是一次性且绑定 page/component/schema 的；目标、staging 是 requester 已准入的 UPLOAD 资源，不能误写为“全系统排队不占任何内存”。
- [slots_manager.py:530](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/slots_manager.py#L530)：计算请求与 loading/offloading/uploading 分开持有引用集合。
- [slots_manager.py:3057](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/slots_manager.py#L3057)：`commit_uploaded_slots` 在锁内检查 schema epoch、独占 target 和 canonical page；并发生产者已发布同页时采用已有 canonical slot，回收安全的重复 target。
- completion 未知时保留 I/O 所有权、隔离地址；不是把错误转换成普通 cache miss 后立即复用内存。

可概括为：

```text
逻辑意图排队 → 准入 → 解析并固定物理资源 → 全量预检
           → 搬运 → fence → 复验 → 提交驻留 → 向计算暴露
```

“事务”指内存驻留提交边界，并非跨 Redis、history、全部请求状态的数据库事务。部分低层级搬运已成功时可以保留成功副本，而不伪装全路径回滚。

### 相比 SGLang 的差异

SGLang 也有锁、完成 event、取消预取和延后释放。omniFlow 的特点是将本地上下迁移与远端搬运纳入同一套显式阶段和页面可见性合同。SGLang HiCache 可逐层 H2D 就绪并与计算重叠；omniFlow 当前只允许完整页所有 component 完成后发布。这是正确性复杂度与流水细粒度之间的取舍。完整页发布仍可与其他请求的计算重叠，限制的是该页内逐层就绪即消费。

### C++ 落地

将现有 `KVTransferConnector` 的 `Pending/Completed/Failed/Cancelled` 拆成两个维度：waiter 是否继续等待、物理 flight 是否已确认完成。增加 move-only `PageLease`、`TransferTicket`、`MaterializedGpuView`，在准入后绑定 blocks；计算与 DMA 都通过 owner 管理的引用保活。

建议额外加入 `BlockHandle{index,generation}`。这是对 omniFlow 的加强：其 SlotRef 本身没有逐次分配 generation 字段；schema epoch 与 staging chunk generation 不应混同为普通 slot generation。代际校验不能代替 I/O pin，旧 DMA 不能在检测出迟到之前已经写坏新页。

实验：控制 copy 延迟，在 QUEUED、已提交 copy、已 fence 三个位置取消；检查旧写不会污染新请求、未确定完成的地址不会入 free list。大量排队传输时测 source pinned pages 和 pin 持续时间，展示延迟绑定的容量收益。

代价：检查、journaling 与严格提交增加 CPU 开销；quarantine 在故障时损失可用容量；整页屏障可能增加 TTFT。

## 4. 优先项 B：统一传输调度与 per-page single-flight

[transfer_scheduler.py:395](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/transfer_scheduler.py#L395) 定义含 name/page/tier/lane/schema_epoch 的 key，[TransferTask:418](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/transfer_scheduler.py#L418) 管理 reason、priority、deadline 和 dependency。

三类流量分别使用 resource lane：OFFLOAD、UPLOAD、REMOTE_COPY_TO。REQUEST/DEMAND 的优先级高于 PRELOAD/WATERMARK。lane 表示调度资源，不保证 DMA 引擎、PCIe 或 NIC 物理带宽互不竞争。

同一页的预取和真实请求可 join 一个 flight，真实请求可提升排队预取的优先级。每个 waiter 有独立 deadline；一个请求超时不取消其他请求需要的搬运。`TransferFuture.result` 与 `drain` 明确区分调用者等待和物理完成。对应代码：[transfer_scheduler.py:639](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/transfer_scheduler.py#L639)、[737](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/transfer_scheduler.py#L737)、[894](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/transfer_scheduler.py#L894)。

`UploadPresenceRegistry` 的 ready/known miss/safe failure/unknown quarantined/stale schema 是不同结果；唤醒不直接证明 residency，仍做最终全页验证。多页交叉请求先提交自己负责的页再等待共享 flight，避免 `[A,B]` 与 `[B,A]` 互相等待。

SGLang 已有 prefetch 策略、load/write stream、资源限制；差异是此处用相同任务和 waiter 合同覆盖多种传输路径。不能将它表述为已实现全局带宽最优调度：未发现 bytes/s 整形器；fixed wave 也不等于自动合并任意请求的 DMA。

C++ 可实现 `TransferScheduler` + `unordered_map<PageFlightKey, Flight>`，completion 回送元数据 owner。先配置 H2D/D2H 两个 lane，P2P 用已有 connector 接入；设置 staging/in-flight 上限，后台预取不能占满 demand 入口。

实验：人为塞入后台预取，再提交命中其中一页的前台请求。验证同页只有一个共享物理迁移 flight、priority donation 降低排队等待；取消一个 waiter 后其他 waiter 正常完成。一个 flight 仍可包含多次 component copy，应报告实际 copy 次数与总字节，以及 demand p95/p99 等待、lane 利用率和背景任务饥饿情况。

## 5. 优先项 C：语义化传输计划与布局转换位置

这是更有辨识度、也更适合展示 C++ 设计能力的方向。

[transfer_plan.py:34](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/transfer_plan.py#L34) 的 LogicalMemoryKey/ComponentKey 去掉物理 slot 和注册顺序；[remote_reshard.py:344](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/remote_reshard.py#L344) 用 `LogicalAtomV1(sub_labels,slice_labels)` 表示 layer/KV/head 等语义。

例如同一段 KV 在 source 的 GPU block 7、target 的 block 93，layer 或 head 的注册顺序还不同。计划器按语义字段找对应关系，不将 list 下标或 rank 编号当作数据身份。

[build_remote_reshard_plan:784](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/remote_reshard.py#L784) 验证源目标语义覆盖完全一致；完全等价覆盖可当 replica，缺字段或含糊的部分重叠会被拒绝。它先产出纯逻辑 gather/scatter 计划，执行器再绑定地址。**本地 TransferPlan 已包含 TensorEndpoint 和 views；不要将本地已绑定执行计划也称为无地址的纯计划。**

[choose_permutation_placement:998](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/remote_reshard.py#L998) 选择 DIRECT/SENDER/RECEIVER：exact layout 直接搬；源更集中且都位于同一已知 peer 时可在 sender 转成目标布局；多 peer gather 或目标更集中时放 receiver。当前是 coverage/peer 启发式，不是在线带宽代价最优化。

[remote_reshard_runtime.py:218](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/remote_reshard_runtime.py#L218) 用 schema fingerprint 缓存规划结果；缓存 key 不含 page/slot/raw pointer，物理资源仍在执行时重新绑定。

SGLang 已支持异构 TP 和 staging。可讲的差异是语义描述符驱动的通用规划和校验，以及统一的转换位置选择，不是“支持 TP1→TPN”这个结果。

C++ 第一版无需实现真正多卡 TP：为现有各层 K/V/scales 写 `PageSchema` 和 `ComponentDescriptor`，实现 GPU layer-first ↔ CPU page-first 的 plan/pack/unpack；乱序 layer 注册、不同 block IDs 和 scales 完整性都用同一规则处理。再用双 GPU P2P 接入同一计划。

实验：随机重排 component、改变 src/dst block ID，验证往返 payload 位级相同；缺 V/scale 或 layout 不兼容在第一笔 copy 前拒绝。对照多次逐字段 copy 与 pack+bulk copy，分别报告 planning、pack、transfer、unpack 和 staging bytes，短页可能因打包成本更慢。

## 6. 增强项 D：history 与 KV 内容分离，再发展成请求 checkpoint

omniFlow history 是按 session 寻址、带 schema 的历史行，KV 是可共享的 immutable content chain。二者不可用同一种 ID。接口分别见 [history_schema.py:41](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/history_schema.py#L41)、[history_snapshot.py:241](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/history_snapshot.py#L241)。

[sglang_backend.py:508](../../omni_flow_sglang/omni_flow/compute_flow/llm/sglang_backend.py#L508) 显式区分已物化 KV 的输出与 `pending_history_*`：最后采样的 token 尚未送入下一次 forward，有 history，但没有其 KV。这为可抢占恢复提供很好的状态建模参考。

边界必须保留：当前 history 是 plan→data→plan 的乐观校验；KV 与 history 不是跨对象原子提交；失败时可能 poison request lease，见 [sglang_backend.py:4111](../../omni_flow_sglang/omni_flow/compute_flow/llm/sglang_backend.py#L4111)。不能称作完整 MVCC、进程崩溃恢复或已经持久化 RNG 的通用 checkpoint。

SGLang 的 SessionSlot 已保存/恢复请求进度和 KV 所有权。C++ 项目的新增价值应是将同一份 checkpoint 与分页下沉、完整恢复和输出提交联系起来。

建议增加自己的 `RequestCheckpoint{revision, kv_committed_tokens, output_tokens, pending_next_token, emitted_cursor, rng_seed, rng_counter, stop_state, page_manifest}`。第一版在整步 forward 完成后冻结，先把 page payload 准备好，再由 owner 发布一个完整 checkpoint manifest；恢复先验证所有页、模型/schema、revision，再安装页表和生成状态。私有尾页保存 valid_tokens，不能作为完整跨请求共享页。

实验：prefill 后、decode 中及已采样未消费的边界抢占，换一组物理 blocks 恢复；同计算路径/确定性配置下对照连续生成，检查 token 不漏不重、RNG counter 不回退、emitted_cursor 不重复发送。batch 改变可能带来浮点数值差异，应同时验证状态与随机输入，不把所有 token 差异归咎于 checkpoint。

这部分是借鉴后的 C++ 新设计，不能当成 omniFlow 已有的完整功能，更不能在实现前写成面试业绩。

## 7. 共同基础：逻辑准入与物理分配分开

[page_allocation.py:83](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/page_allocation.py#L83) 的 PageAllocationIntent 明确区分 immutable reads、尾页 COW、private pages；short match 不能偷偷变成 writable scratch。`sglang_backend.py:1842` 实际调用并等待完整 readiness。

[capacity_admission.py:109](../../omni_flow_sglang/omni_flow/data_flow/global_params_memory_pool/capacity_admission.py#L109) 按共享 content key 去重计费，私有/COW 各自计费，原子准入防止持有半套资源等待其余资源。预算大于限制时的独占路径只是准入策略，不意味着能突破实际 GPU 物理容量。

PBE 应在已固定 GPU KV blocks 限额内，同时计入共享唯一页、私有增长页、in-flight reservation 和 staging；不可把 pending eviction 计为已经空闲的块。

## 8. 建议实施顺序与验收

| 阶段 | 实施内容 | 必须展示的证据 |
|---|---|---|
| 1 | PageSchema/ComponentDescriptor、CPU pack/unpack oracle、执行前完整预检 | layout 乱序不改值；缺组件零写入；新旧 GPU 路径输出回归 |
| 2 | PageDirectory、leases、完整页迁移事务、waiter/flight 分离 | 各阶段取消/故障与迟到 copy；无提前复用、无永久正常路径泄漏 |
| 3 | GPU↔CPU + 原有 P2P 接统一调度，single-flight、priority donation | 一页多请求一次搬运；前台等待分位数；有界 staging/队列 |
| 4 | checkpoint manifest 与请求进度整体恢复 | 多次抢占、换址恢复、无重发/漏发；恢复与重算的盈亏区间 |
| 后续选一 | 语义化异构分片 / 更好的转换位置成本模型 / 逐层计算流水 | 每项单独控制变量，不能由 transport 子测试推断端到端收益 |

现有 PBE `KVPoolDescriptor`、`KVBlockManifest`、`LayerKVTransferConnector` 可作为 adapter；保留 kernel 的 GPU block table。首先修改管理与传输合同，不要求更换 attention kernel。

Qwen2-0.5B-Instruct 可验证完整模型路径。通过显式限制 KV blocks 制造压力，而非填满 H20 真实显存。小模型重算便宜，出现搬运不划算是有效结论；报告各段成本与 crossover，不预设 restore 必胜。

面试表述建议（完成对应实现与实验后）：

> 我参考 omniFlow 的数据流，把 C++ 推理引擎里的 KV 迁移建模成独立任务：排队时保留逻辑页意图，准入后绑定带所有权的物理地址，以完整页完成屏障提交驻留。GPU/CPU/P2P 共用语义化搬运计划和调度合同，多个请求可以共享一次搬运、独立取消。我另外实现了生成进度与 KV manifest 一致的抢占恢复，并在固定 KV 容量下测量恢复与重算的收益边界。

本次只做静态源码与测试定义审计、官方文档交叉验证；没有执行新的 omniFlow/SGLang 性能实验，也未修改推理实现。
