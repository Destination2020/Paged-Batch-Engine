# Paged-Batch-Engine 数据流改造执行计划 V2

> 最新执行入口为 [V4 多角色多模态计划](PBE_MULTI_ROLE_MULTIMODAL_EXECUTION_PLAN_V4_20260912.md)，架构见 [V4 设计](PBE_MULTI_ROLE_MULTIMODAL_ARCHITECTURE_V4_20260912.md)。下文是历史原型计划和验收，当前实现状态需由 V4 M0 复核。

初版：2026-09-11；本次修订：2026-09-12。代码基线：`f3597a763b6cd5cf60b826baa2dc94395c3d02fd`。沿用原文件路径，内容版本升级为 V2，作为后续实施入口。

后续执行入口已更新为 [V3 计划](DATA_FLOW_REFACTOR_PLAN_V3_20260912.md)，依据 [二次源码研究](OMNIFLOW_DATA_FLOW_SECOND_AUDIT_20260912.md)。本次复核发现 checkpoint 乱序提交/取消窗口、容量生命周期及真实压力/输出集成缺口；旧实验也未覆盖全部等价消融。因此下述 8/8 仅保留为历史原型验收记录，不能表示全部设计主张已兑现。V3 新增实现尚未开始。

历史验收记录（2026-09-12）：上轮将 P0a、P0b、P1、P2a、P2b、P3、P4、P5 按本地测试记为 8/8（100%，非工作量占比）。当时 P6/P7 按第 12.19 节未启动；其后续选择现在按 V3 的 X1/X2 门槛执行。当前摘要见 `DATA_FLOW_EXECUTION_CONTEXT_20260912.md`，原始过程与数据保留在第 12 节。

设计依据：[omniFlow/SGLang 数据流对照](OMNIFLOW_SGLANG_DISTINCTIVE_DATA_FLOW_20260912.md)。环境操作与既有结果：[本地环境说明](QWEN2_05B_LOCAL_SETUP_20260911.md)。版本、源码证据与 SGLang 比较边界以对照文档为准；执行依赖、阶段编号与验收以本文为准。

## 1. 目标与范围

项目定位：**具有语义化搬运计划、可验证迁移事务和请求恢复能力的 C++/CUDA 推理引擎**。

核心问题：同一段 KV 在不同布局和物理地址之间如何保持语义；如何在有限容量下准入并安全执行迁移；多个请求如何共享加载并独立取消；如何让恢复后的 KV、下一次 forward 输入和已发送输出保持一致。固定 GPU KV 容量下，用实验确定恢复相对重算的收益与负收益边界。

实施主线按依赖组织为：

1. `PageSchema/ComponentDescriptor → TransferPlan → BoundTransfer`：按组件语义规划，准入后绑定物理地址。
2. `PageDirectory/PageLease → copy/fence/validate/commit`：建立完整页可见性与取消、失败、资源回收合同。
3. `TransferScheduler/PageFlight`：GPU↔CPU 与现有 P2P 接同一框架，支持 per-page single-flight、独立 waiter、priority donation 和有界资源。
4. `RequestCheckpoint`：分开管理会话历史与不可变 KV 内容，通过 revision/manifest 实现整步暂停和换址恢复。

CPU 冷前缀缓存是上述主线的运行载体；P1–P3 是第一交付里程碑，P4 的精确恢复是第二里程碑，不再只列为可有可无的增强。两实例跨请求内容复用、真正异构 TP serving、跨机 RDMA/SSD 为后续独立扩展；单机 P2P 执行器接入不等待这些扩展完成。

保留现有 Qwen2 serving、CUDA attention/scatter kernel、GPU KV layout、逐层 block allocator。新增进程内 `CacheRuntime`；第一版不引入独立 MemoryManager 进程、Redis、任意张量 RPC、完整 history 数据库。源项目允许大幅修改，但每阶段应有可工作的回归边界。

### 1.1 借鉴范围与贡献归属

| 内容 | 本计划中的定位 |
|---|---|
| 多级缓存、远端前缀复用、逻辑索引与物理驻留分离、异步加载 | SGLang 已有；作为基础能力，不作为 omniFlow 独有卖点 |
| 语义组件规划、远端 provider 延迟绑定、完整页事务、统一 flight/waiter 合同 | 借鉴 omniFlow 的具体组织方式，用 C++ 强类型和 owner 线程实现 |
| 逐次分配的 block generation | PBE 加强设计；不能把 omniFlow 的 schema epoch 或 staging generation 混称为普通 slot generation |
| 带 revision 的完整请求 checkpoint、RNG 与输出进度一起恢复 | PBE 加强设计；omniFlow 当前 history/KV 非跨对象原子提交，SGLang 也已有 session 保存恢复 |
| DIRECT/SENDER/RECEIVER 转换位置选择 | 后续深化；SGLang 已有异构 TP/staging，不能仅以支持不同 TP 作为区别 |

比较对象是已审计版本及官方文档。独立贡献通过接口合同、失败测试和性能消融体现，不宣称单项技术行业独有或未经实测的整体性能优势。

## 2. 本机实测与构建情况

2026-09-11 22:56（Asia/Shanghai）附近的硬件快照，非本次实时空闲承诺：

| 项目 | 观察 | 计划含义 |
|---|---|---|
| 主机 | `set-hldy-llm-multimodal-worker27.mt` | 当前仅检查此机器 |
| GPU | 2 × H20-3e，各 143771 MiB；各用 1 MiB、util 0%，无可见计算进程 | 可做单 GPU 分层缓存和双 GPU 实验；不是长期独占承诺 |
| 拓扑 | GPU0↔GPU1 为 NV18，CPU affinity 0–41 / NUMA 0 | P2P 是优先验证后端；仍需运行时 peer-access 测试 |
| CPU/内存 | 42 个在线逻辑 CPU；392 GiB total、359 GiB available | 可分配有界 host pool；实际额度另查 cgroup 和锁页限制 |
| 存储 | `/tmp` 为本地 NVMe，约 5.4T 可用；工作区为 beegfs-fuse | 本地 build/SSD 实验放 /tmp，不能把网络文件系统当成本地 SSD |
| 网卡 | mlx5_0…mlx5_7 可见 | 未验证端口、GID、远端连通性和 RDMA 吞吐 |
| 工具链 | GCC 10.3.1、CMake 3.31.8、nvcc 12.8.61 | CMake 已识别编译器 |
| 驱动 | 550.127.08，nvidia-smi 显示 CUDA 12.4 | 不混淆驱动展示版本与 toolkit；后续 Qwen2 BF16 GPU 冒烟已通过，其他路径仍逐项验收 |

### 2.1 初次探测记录

```bash
cmake -S Paged-Batch-Engine -B /tmp/pbe-plan-20260911-build \
  -DUSE_CPM=OFF -DQWEN2_SUPPORT=ON \
  -DKUIPER_BUILD_TESTS=ON -DKUIPER_BUILD_DEMOS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=90
```

最初 `USE_CPM=OFF` 配置停在 `find_package(glog REQUIRED)`；当时 `pkg-config libzmq` 也未命中。此记录只说明初次探测结果，已不构成当前 demo 构建阻塞。

### 2.2 已完成的环境工作（P0a）

- [x] 项目 `.venv`：Python 3.12.3，使用 `--system-site-packages`；复用 torch 2.9.1+cu128、Transformers 5.3.0，补装 Accelerate 1.12.0。这不是完全独立锁定的 Python 环境。
- [x] HF 权重与 tokenizer：`/tmp/Paged-Batch-Engine-models/Qwen2-0.5B-Instruct`；已在 Transformers 上运行 GPU 推理。
- [x] C++ BF16 权重：`/tmp/Paged-Batch-Engine-models/Qwen2-0.5B-Instruct.bf16.bin`。
- [x] `USE_CPM=ON` 构建：`/tmp/Paged-Batch-Engine-build-qwen05`，CUDA arch 90，`qwen_instruct_infer` 已编译并运行。
- [x] 修复 `demo/main_qwen2Instruct.cpp`：逐 token 建立 prompt embedding，避免整段 prompt 超过 serving workspace 的预分配容量。

既有 C++ 单次结果：`steps=76, duration=0.218239s, steps/s=348.242`。steps 包含 prompt 处理，且该 demo 为逐 token prefill；不能当作 output tokens/s、连续批处理 serving 基线或数据流改造收益。

当次构建关闭 tests/NCCL/ZMQ，未证明这些路径可用。模型与构建产物在 `/tmp`，后续实施先检查是否仍存在；仅在失效时重建。详细命令沿用环境说明，不重新创建或覆盖现有 venv。

### 2.3 已完成的基线工作（P0b）

- [x] 记录 Python 系统依赖来源、模型 revision/hash、C++ 依赖版本、编译参数，补齐可重现 manifest。
- [x] 开启并运行已有 cache/PD/scheduler 测试，按目标路径确认 GTest、P2P 和相关构建依赖。
- [x] 跑实际 serving 的 greedy/采样、吞吐、TTFT 和逐 token gap 基线。
- [x] 复现并修正第 3.3 节的 request handle、请求 RNG、统计口径问题；取消合同改造与 P2 联动。

P0b 的实现与验收记录见 12.3–12.11；checkpoint 随机采样验收在请求 RNG 隔离完成后执行。

## 3. 当前调用链与代码发现

```text
OnlineServingEngine::run_loop
  → Scheduler::schedule_step
  → KVCacheManager / SequenceKVManager 分配 slot
  → MixedBatchBuilder 构造逐层 block tables / slot mapping
  → InProcGpuWorker → PDWorkerPair → forward / sample
  → Scheduler::process_outputs
  → publish_radix_cache / free_request / preempt_sequence

PD：DecodeKVReservationManager::reserve → KVBlockManifest
  → KVTransferConnector::submit/poll → decode-ready
```

### 3.1 可复用基础

- [BlockAllocator](../infMain/include/base/block_allocator.h)：逐层物理池、引用计数、K/V/scales payload。
- [SequenceKVManager](../infMain/source/base/sequence_kv_manager.cpp)：多层批量分配与失败回滚、页表、共享前缀采用。
- [CompressedRadixCacheTree](../infMain/include/base/compressed_radix_cache_tree.h)：独立匹配、split、pin、叶子淘汰。
- [PD handoff](../infMain/include/serving/pd_handoff.h)：池描述、逐层源/目标映射、connector 接口。
- [PagedKVRuntime](../infMain/include/model/paged_kv_runtime.h)：kernel 消耗 GPU allocator 与物理页表，可隔离于 host/remote 管理。

PD 已包含整块/逐层 connector 与 P2P/NCCL 实现；新增工作是让这些执行器接入统一的语义计划和生命周期合同。`KVPoolDescriptor::compatible_with` 当前要求 block_size、维度、dtype、storage mode 相容；扩大支持范围须增加明确的 codec/转换计划，不能只放宽条件判断。

### 3.2 必须改变的语义边界

| 代码依据 | 当前行为 | 改造含义 |
|---|---|---|
| radix `Node::segment_block_ids_per_layer` | 索引直接持物理 blocks | 引入 logical page records，才能表示 host-only 前缀 |
| `KVCacheManager::evict_radix_cache_blocks` | erase leaf 后 free blocks | 分离 GPU 下沉与逻辑索引删除 |
| `SequenceKVManager::append_tokens_internal` | 分配时增加 `num_tokens_` | 已分配不等于已完成计算 |
| `Scheduler::process_decode_output` | 记录 sampled token，没有统一更新全程 KV committed counter | 不能直接用现有 `computed_tokens` 保存 decode checkpoint |
| `maybe_attach_radix_cache` | 最多复用 `floor((prompt_len-1)/P)*P` tokens | 保留用于产生首个 logits 的 seed tail |
| `adopt_shared_prefix` | 空 sequence、完整块边界 | 恢复接口需严格验证，部分共享尾页 COW 后置 |
| `maybe_publish_radix_cache` | 发布 prompt_tokens | 生成历史 checkpoint 与跨请求 prefix publish 要分开 |
| FP8 allocator/format validation | CPU 执行池不允许 FP8；K/V/scales 独立存储 | CPU 缓存应保存原始 payload，不创建 FP8 CPU 执行 allocator |

### 3.3 应先修复或复现的基线问题

1. **请求 ID 生命周期上限。** [KVCacheManager](../infMain/include/base/kv_cache_manager.h) 的 RequestId 为 int32，20 位槽＋11 位 generation；[register_request](../infMain/source/base/kv_cache_manager.cpp) 在同一槽 generation 达 2047 后再次复用触发 CHECK。顺序请求即可不断复用该槽。改为 64 位代际 handle，处理极限退休；同步修改 wire/日志/测试，GPU batch row index 无需一起变 64 位。
2. **随机采样依赖 batch。** [qwen2.cpp](../infMain/source/model/qwen2.cpp) 1646/1692 附近以 `0xC0FFEE + sample_count` 每次创建 RNG。改为每请求 seed＋生成步 counter；否则改变 batch 组成会改变请求随机输入，无法隔离恢复正确性。
3. **取消与物理完成混淆。** [pd_handoff.cpp](../infMain/source/serving/pd_handoff.cpp) 中 InProcKVBlockCopy/CudaP2P 的 cancel 销毁 event 并设 Cancelled；[PDCoordinator](../infMain/source/serving/pd_coordinator.cpp) 将取消视作终态。event destroy 不等待已排队工作完成，不能作为可释放内存的证明。调用方可能另有同步，须复现后才宣称端到端缺陷；新异步路径必须修订合同。
4. **现有取消测试仅覆盖状态。** [test_pd_handoff.cpp](../test/test_serving/test_pd_handoff.cpp) 的 CoordinatorCancelBeforeCompletion 使用 PendingConnector mock，不包含迟到真实 GPU copy；需新增受控延迟与地址复用测试。
5. **ITL 统计命名不精确。** [SequenceState](../infMain/include/serving/sequence_state.h) 的 `itl_ms()` 为每请求首末 token 区间均值，部分工具再对这些均值取分位数，不是逐 token gap 的 p95。保留旧值并准确命名，补充逐 token 时间戳。

以上为静态发现，尚未执行复现。[CUDA 12.4 event 文档](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-runtime-api/group__CUDART__EVENT.html) 明确允许未完成 event 被异步销毁，不能将 destroy 理解为停止 copy。

## 4. 目标架构与线程模型

```text
Scheduler / OnlineServingEngine
        │ match / allocate_intent / ensure_resident / checkpoint
        ▼
CacheRuntime（owner 线程修改元数据）
  ├─ PrefixIndex：token prefix → LogicalPageId[]
  ├─ PageDirectory：身份、有效范围、格式、各 tier 驻留
  ├─ PageSchema / ComponentDescriptor：K/V/scales 的语义覆盖
  ├─ TransferPlanner：无地址的逻辑计划与 layout codec
  ├─ AllocationIntent / Admission：共享只读、COW、私有页需求
  ├─ GpuPageStore：包装逐层 BlockAllocator
  ├─ HostPageStore：有界 pinned memory，完整 K/V/scales
  ├─ CheckpointStore：revision、history、KV manifest、输出/RNG 游标
  ├─ ResidencyPolicy：admission / eviction / restore-vs-recompute
  └─ TransferScheduler
       ├─ H2D / D2H / P2P lanes、优先级、依赖、完成队列
       ├─ PageFlight + 独立 waiters；single-flight / priority donation
       └─ admission 后 BoundTransfer 持有 leases 与有界 staging
            → copy → fence → validate → commit/rollback/quarantine

MaterializedGpuView → MixedBatchBuilder → 原有 CUDA kernel
```

元数据、radix split、allocator refcount、完成提交均由 owner 线程串行修改。I/O 线程提交 immutable completion，不直接调用现有非线程安全 allocator。跨结构更新在 owner 内统一提交；仅在各模块分别增加细粒度锁不足以保证这一点。

无可运行请求时也要处理 completion；改造现有 `run_loop` 的等待条件，确保 host restore 可以唤醒 owner，最后一个请求取消后仍能完成后台回收。

Planner 不触碰 allocator、CUDA stream 或地址；I/O executor 只消费已绑定且经过预检的计划，不决定 radix、canonical page 或 checkpoint 的发布。单 owner 串行化是 PBE 的实现选择，不照搬 Python 锁与 RPC 组织方式。

## 5. 核心数据合同

拟议类型，非已实现接口：

```cpp
struct BlockHandle { uint32_t index; uint64_t generation; };
struct GpuPageRef { int device; std::vector<BlockHandle> layer_blocks; };
struct HostPageRef { uint64_t offset; uint64_t bytes; uint64_t generation; };
struct PrefixMatch { int logical_tokens; int gpu_ready_tokens; /* page ids */ };
struct ComponentDescriptor;  // layer、K/V/scale、head range、shape、dtype
struct PageSchema;           // 完整逻辑组件集合，物理排列由 layout 描述
struct TransferPlan;         // 逻辑组件路由/codec，不含 slot、指针或 lease
class BoundTransfer;        // admission 后持有物理 spans、generation、leases
struct AllocationIntent;    // read_page_ids、可选尾页 COW、private_pages
class PageLease;             // move-only，release 回送 owner
class TransferTicket;        // 一个 waiter；共享 PageFlight 的结果
class MaterializedGpuView;   // 计算期间固定物理映射
struct RequestCheckpoint;   // revision、history、进度、RNG、KV manifest
```

### 5.1 页身份与格式

- 一个 logical page 是 P 个连续位置的全部层 KV；物理上仍为 L 个独立 blocks，不要求各层 block ID 相同。
- 内容 key 使用版本化序列化：模型/权重、输入与位置语义 namespace、前缀链 hash、块 token、P。格式 key 独立包含 logical/storage/scale dtype、storage mode、head/layer 维度与 schema version。
- 同一块 token 在不同前文下可能得到不同 KV，因此不能只 hash 当前块。输入包含 embeddings 等时必须纳入身份或拒绝跨请求共享。
- 第一交付仅接受同模型、同逻辑 dtype、同 page_size 和明确支持的物理 layout 转换；不同精度、TP 或 page_size 不得直接 memcpy。稳定导出 key 不使用进程相关 std::hash；本地可用递增 LogicalPageId。
- 共享 prefix 仅发布完整已提交页。私有 checkpoint 尾页可 `valid_tokens<P`，不作为完整共享页；未用字节清零或只传有效区。

### 5.2 驻留与资源所有权

一页可同时有 GPU/CPU 副本，所以不能用唯一 GPU/CPU/Moving 枚举。维护多个 residency records，各有 ready、generation、format；in-flight transfer 独立记录。

不变量：

1. compute 只接受完整 GPU-ready 且持 lease 的页。
2. D2H 完成并提交 host 副本前，不释放源 GPU pin。
3. H2D 全层全组件完成、generation 校验通过后，才安装页表。
4. RequestId generation 与 block generation 分开；检查代际不能替代 I/O pin。
5. 索引存在不等于可用命中；只能采用连续且可用/可恢复的前缀。
6. compute/I/O 引用未清零，不进入 free queue；缓存保留引用与执行引用分别统计。
7. 无法确认物理完成时隔离资源；必要时停止后端接流并重建，不假装取消成功后空间可用。
8. 元数据与 host cache 同样有容量上限；GPU demote 不 erase prefix，但最终 host eviction 必须允许清理不可达索引。

### 5.3 长度与发布

新增 `allocated_tokens`、`kv_committed_tokens`、`published_full_tokens`，另保留 prefill/recompute cursor。现有 computed_tokens 不承担全部含义。forward 完成 fence 后才能推进 KV commit；allocation 成功和 sampled token 已返回都不能独立证明全部待发布 KV 可见。

第一交付中的新缓存驻留遵守完整页可见性：全部层 K/V/必要 scales 完成后才可读。部分有效尾页仅由私有 checkpoint 的显式 valid_tokens 合同处理。该边界允许其他请求的计算与迁移重叠，但不提供该页内逐层 ready 即消费。原有 layer connector 的局部 event 不能直接提升成完整页 READY；逐层消费优化单独放到 P7 验收。

### 5.4 语义计划与执行期绑定

`PageSchema` 表示一页“需要哪些组件”，`LayoutDescriptor` 表示其 shape/stride/offset/packing，`ComponentDescriptor` 标明 layer、K/V/scale、head 范围与 dtype；注册顺序、block ID 不参与语义身份。

Planner 生成完整的 component coverage 与 copy/pack/unpack 操作，不读取裸地址。相同格式支持 direct；GPU layer-first ↔ host page-first 使用显式 codec。缺组件、重复或含糊 coverage、shape/dtype 不匹配在执行前拒绝，不退回猜测性 raw copy。

执行前由 owner 完成 admission、选取当前可用来源、申请目标、pin 两端，再构造 `BoundTransfer`。绑定阶段再次检查 schema/pool epoch、block generation、bounds、residency 和全部组件，任何错误在第一笔 copy 前零 payload 写入。排队期间源可能下沉或消失，应重新选择合法来源或返回 typed miss，不能沿用旧地址。

本计划的 C++ `TransferPlan` 特指纯逻辑计划；omniFlow 本地同名 `TransferPlan` 已包含 tensor views，而远端 reshard plan 才是纯逻辑，移植时明确分层。可缓存的 schema/codec 模板 key 不包含 page ID、slot 或指针；具体 transfer intent 仍必须标识要搬的逻辑页。计划缓存命中不能跳过执行期资源校验。

初期 host pack/unpack 用可检查的 CPU oracle 验证映射；GPU 逐组件 copy 为执行基线，pack kernel/bulk copy 在有性能证据后启用。真正异构 TP、sender/receiver placement 的扩大实现放在 P7。

P1 用 CPU/fake endpoint 验证绑定预检的纯函数、组件乱序与虚拟 block ID 映射，不依赖尚未实现的 PageDirectory/leases。真实 GPU allocator、generation 和 owner 资源绑定在 P2 接入并重跑这些合同，避免阶段依赖倒置。

### 5.5 分配意图与有界资源准入

`AllocationIntent` 显式分开必须完整命中的 immutable reads、需要复制的尾页和新建私有页；返回 `read_slots/cow_slot/private_slots`。缺只读页返回 miss/wait，不能用新分配 scratch 顶替而改变位置语义。第一交付可以先不支持 COW，但必须显式返回 unsupported；checkpoint 私有尾页恢复不依赖跨请求部分页共享。

准入按本地唯一共享页计费，私有/COW 每 owner 单独计费；全局目录存在某页不代表本地已占有该页。原子预留当前 wave 的完整资源集合，失败整体释放本次新增 reservation。分别限制 GPU 每层 blocks、host bytes、staging bytes、in-flight tasks 与公开队列，后台预取保留 demand 入口额度。

不在持有部分 staging 的 planning loop 中等待同一批后续空间；未进入执行的任务不保留源裸地址或长期 source pin。远端 provider 排队是逻辑意图，但 requester 已准入的 UPLOAD 可以持有有界 target/staging，两者不能混为“排队零资源占用”。

## 6. 模块级改造

| 模块 | 改动 | 保留边界 |
|---|---|---|
| 新 `cache/page_schema.*`、`cache/layout_codec.*` | 语义 component、完整 coverage、CPU oracle、layout 转换合同 | 先同模型/dtype/page_size；scales 必须纳入 |
| 新 `cache/transfer_plan.*` | 无地址计划、schema 模板缓存、BoundTransfer 预检 | copy 前全量检查；cached plan 不代替执行期复验 |
| `base/block_allocator.*` | generation handle、GPU store adapter、payload enumeration | GPU pool 地址、kernel stride、owner 分配 |
| `base/compressed_radix_cache_tree.h` | payload 改 logical handles，split/erase 管索引引用 | 前缀算法和 pin 语义；不把 layer 数量语义丢失到猜测中 |
| `base/kv_cache_manager.*` | facade 委托 CacheRuntime，match 与 attach-ready 分离 | 旧 GPU-only 路径可回归，host IDs 不进入 batch metadata |
| `base/sequence_kv_manager.*` | 原子安装 restored prefix/checkpoint、GPU leases | 跨层失败回滚，append 不覆写共享页 |
| 新 `cache/page_directory.*` | 身份、驻留、schema/epoch | identity 不包含地址 |
| 新 `cache/host_page_store.*` | 有界 pinned arena、页 payload、LRU | FP8 scales 完整；CPU 存储不是 CPU FP8 计算 |
| 新 `cache/page_lease.*`、`cache/allocation_transaction.*` | compute/I/O 所有权、rollback journal、quarantine | 回滚只撤销本次新增资源；不 free_all_by_req_id |
| 新 `cache/capacity_admission.*` | immutable/COW/private intent、共享页去重计费、有界预留 | allocation budget 不能突破物理容量 |
| 新 `cache/transfer_executor.*` | 同步 oracle、有界异步 executor、现有 P2P adapter | 只执行 BoundTransfer，fence 后回送 completion |
| 新 `cache/transfer_scheduler.*`、`cache/page_presence.*` | lanes、priority、dependencies、独立 waiters、typed outcomes | cancel_waiter 不提前 free；single-flight 不等于一次 DMA |
| 新 `cache/cache_runtime.*` | owner、admission、materialize facade | 管理与 kernel 解耦 |
| 新 `cache/checkpoint_store.*` | request revision、history/RNG/输出游标与 page manifest | owner 提交可见性；第一版不承诺进程崩溃恢复 |
| `serving/scheduler.*` | wait-restore/offloading/suspended、完成事件接入 | 等待 I/O 不误报 no-progress；decode 优先可回退 |
| `serving/mixed_batch_builder.cpp` | 消费已固定 GPU view | kernel 仍看到物理 block tables |
| PD connector/reservation | 接 BoundTransfer executor、waiter 与 physical-done 两轴 | 保留已有 layer-wise 传输，完整页 READY 须聚合必要 fences |
| online loop/metrics | completion wakeup、drain、统计快照 | 指标读取不竞争 owner 的可变状态 |
| model sampling/config | request-local seed/counter | batch 重排不改变请求随机输入序列 |

## 7. 下沉、恢复与取消的执行流程

### 7.1 冷前缀下沉

排队 logical deficit/page intent → admission 时重新选择无活动 compute 引用的 immutable pages → 预留完整 host/staging 需求 → 固定 GPU 源 → BoundTransfer 全量预检 → copy stream 等待 producer fence → D2H → 完成确认 → 复验 → 提交 host-ready → 释放 GPU cache 引用。

第一版 host layout 为 page-first，其内按 layer/K/V/scales 排列。分散 GPU 地址先用多次 cudaMemcpyAsync 作基线；profile 后再做 pack/bulk copy，不声称天然一次 DMA 搬走整页全部层。水位任务 admission 时重新采样 deficit，避免根据过期快照过量下沉。

事务 journal 只记录本次新取得的 refs/reservations；同一请求在调用前持有的 cache hit 不随本次失败被释放。若将来增加 L2→L3 级联，按依赖先完成低层腾挪，再提交高层下沉；已提交的合法低层副本可以保留，不假装全路径同时回滚。

### 7.2 前缀恢复

match logical pages → 选择连续可恢复前缀 → claim/join page presence → 调度 admission → 选择并 pin 当前来源 → 预留 GPU blocks/staging → bind/preflight → H2D → fence/复验 → commit 驻留 → 全前缀 readiness barrier → 安装 MaterializedGpuView → 转 runnable。

`ensure_free_blocks_available()` 不可同步阻塞等待后台下沉；调度转 waiting 并运行其他 ready 请求。pending eviction 仅用于预测，不计为已经 free 的容量。

admission 同时考虑每层容量、恢复中的 reservation 和未来 decode 增长。共享页按唯一物理页计容量。大传输分有界 wave；reserve 顺序固定且失败整体回滚，避免持部分资源等待导致循环阻塞。

若完整工作集超过 GPU 可行容量，截短恢复前缀并选择可行重算/拒绝，不能无限等待无法放入的请求。第一版不实现 token 级 demand paging attention。

### 7.3 取消

```text
waiter: WAITING → READY / TIMED_OUT / CANCELLED / FAILED

flight: QUEUED → ADMITTED → BOUND → COPYING → FENCED → VALIDATED → COMMITTED
          └无剩余 waiter 可撤销     └失败且物理安全：ROLLBACK
                                  └无法确认完成：QUARANTINED
```

waiter outcome 与物理 flight 终态分开；COMMITTED 表示驻留已发布，不表示缓存页立即回收。只在全部计算/I/O/保留引用满足释放条件后回收对应资源。排队任务的最后一个 waiter 离开可撤销；copy 发出后，waiter 可 detach，但两端仍保留 pin 并继续 drain/fence。

一个 waiter 超时不能取消其他 waiter 共用的迁移。deadline 使用端到端绝对单调时间，不在每 tier 重置；超时控制 waiter 等待与是否启动新工作，不能强停已发出的 DMA。waiter 可立即 detach 并返回 TIMED_OUT/CANCELLED，不必等待 drain；物理 flight 继续持有两端资源。合法 copy 完成后可提交驻留，超时请求只撤销自己新增的 ownership。

共享 flight 的物理终局结果在 commit、safe rollback 或 quarantine 完成后才发布，区分 READY、KNOWN_MISS、FAILED_SAFE、UNKNOWN_QUARANTINED、STALE_SCHEMA。不能把提前返回的 waiter 取消结果当成 flight 已结束；普通 Event 唤醒也不是可读证明。

RAII 析构向 owner 交回所有权，不在任意线程直接 free 或全局同步。shutdown：停止 admission → detach 等待者 → drain 物理任务 → 释放 leases → 销毁 pools/streams。

### 7.4 共享 flight、优先级与 resource lanes

`PageFlightKey = (model_namespace, logical_page_id, destination_device/tier, layout_id, pool_epoch)`；来源不进入页面 readiness 的 single-flight 身份，H2D 与 remote 是解析该需求时的来源选择。具体 bound operation 必须记录所选来源与其资源版本。

相同目标页的 PRELOAD 与 DEMAND join 同一 flight，每个 waiter 保有自己的 deadline/取消状态。DEMAND 可以提升尚未运行的 PRELOAD；不承诺抢占已经提交的 DMA。实际需要的其他目标 location 必须另行补齐并 fence，不把“一处 ready”扩展为“任意副本 ready”。

H2D、D2H、P2P 按实际设备和资源设置独立 lane，有界 worker/stream 配置与 completion queue。它们共用任务状态和准入合同，物理 PCIe、显存、NIC 带宽仍可能竞争；第一版不实现 bytes/s 整形器。必须为 demand 保留入口，确保前台能进入 presence 并完成 priority donation。

容量回收作为 demand 的依赖任务时传播所需优先级；不能把关键路径 D2H 一律视为最低优先级后台工作。使用 aging/有界后台进展策略，记录饥饿时间。依赖任务不能在同一 lane worker 内同步等待尚未获得该 worker 的另一任务。

多页交叉请求先提交自己 owner 的可执行页，再等待 joined 页；大批按 staging 容量分 wave，wave 间释放已完成的工作空间，最后统一做所需前缀的 readiness barrier。覆盖 `[A,B]` 与 `[B,A]` 并发，避免互相持有资源等待。

single-flight 是“同页一个共享迁移任务”，内部可以包含多个 layer/K/V/scales copy；分别统计 flight 数、实际 copy 调用数和字节。fixed wave 仅是有界准入/批次策略，不声称任意跨请求 DMA 自动合并。

### 7.5 P2P 接入与远端二阶段边界

P3 先把现有同机 P2P connector 包装为 BoundTransfer executor，验证与 H2D/D2H 共享 ownership、取消和提交合同。无需同时建设分布式目录，也不把原有 P/D handoff 当作新的跨请求缓存命中。

P6 扩展至两进程跨请求内容复用：provider phase 1 接收逻辑 page/schema/component 意图，lane admission 后解析并 pin 当前 source，返回绑定请求者/页集合的一次性 token；phase 2 附加 requester 目标描述符，校验 exact page/component 集和 epoch 后执行。requester 的 target/staging 已由本地 UPLOAD admission 有界预留，不能据 provider 排队无裸地址推导全流程无资源占用。phase-2 replay、缺项/额外项、过期 token 在写入前拒绝。

跨进程描述符必须使用所选后端可导入或已注册的 transport handle/内存区域信息，不能把另一进程的裸 CUDA 指针直接解引用。token 注册表有容量与过期清理上限；phase 2 从未到达且未发出 copy 时安全释放 source pin，已发出的 copy 仍按物理完成 drain/quarantine，过期不授权提前回收。

## 8. 请求抢占保存与恢复

P4 在完整页迁移事务、P3 共享调度和 request-local RNG 成熟后开展。借鉴 omniFlow 的 history/KV 分离与 pending tail，增加 PBE 自己的 revision/manifest 一致性合同。第一版是进程存活期间的整步暂停/恢复，不承诺断电、进程崩溃或跨 Redis 数据库事务恢复。

不可变 KV 以 content ID 共享；可变请求状态以 `(request_handle, revision)` 寻址。保存 prompt/output tokens、KV committed 长度、pending next token、采样 seed/counter、stop 状态、emitted cursor、schema/model namespace、page manifest 与既有时间戳。已 emit token 不一定已经进入 KV，恢复后不能重复 emit 或重复采样；状态中要能明确指出下一次 forward 的输入。

提交步骤：在整步 forward fence 后冻结 owner 状态 → 获取新 revision → 保存进度与 immutable/shared pages、私有尾页 payload → 确认 payload 全部完成并持有相应引用 → owner 一次性发布 READY manifest。准备过程失败时未完成 manifest 不可见；保留仍合法的旧 checkpoint 或让请求明确失败，不发布半份进度。

恢复步骤：读 READY manifest/revision → 验证模型、schema、有效长度及所有页面 → 按迁移合同恢复到新的 physical blocks → owner 复验 request generation/revision → 在调度器下一次读取前，一次安装多层页表与生成/RNG/输出状态。已取消或被新 revision 替代的恢复不得使请求重新 runnable；安全完成的缓存副本可保留。

共享完整页继续用内容引用，私有页与部分尾页保存 valid_tokens；它们只为该 checkpoint 服务，不冒充可跨请求共享的完整页。checkpoint 必须持有防淘汰引用或可验证的 backing，不能发布只含已经可能失效的裸页 ID 的 manifest。

READY checkpoint 的 payload 对该 revision 不可变，pin 只保证地址存活并不能阻止内容被覆写。恢复尾页必须分配新的私有可写目标；完整页可以继续只读共享，写入前按 COW/独立副本合同取得私有目标。旧 revision 的替换、过期和 manifest 删除按引用释放，不能让新一轮计算原地修改仍可恢复的旧快照。

新增 `Suspend → Offloading → Suspended → Restoring → Runnable`。当前 free/re-register/recompute 作为回退保留，但不得与同一空间的在途读写并行释放。

保存 checkpoint 与发布生成历史的跨请求内容页分开上线；前者先证明精确续算，后者再扩展共享范围。部分共享尾页 COW 是后续功能，不是冷前缀缓存的先决条件。

这比 omniFlow 当前 plan→data→plan 的乐观 history 校验更强，但只在上述 PBE owner/manifest 范围内成立；不称其为已实现的完整 MVCC。验收同时覆盖 P−1/P/P+1、已采样未消费的 tail、恢复期间取消/更新 revision、连续多次换址；不同 batch 造成的浮点数值变化与 RNG/状态回退分别判断。

## 9. 分阶段提交、依赖与验收

本节替换初版阶段含义：旧 P1 的 PageDirectory/leases 移至 P2a；新 P1 前置语义 schema/plan；旧 P3 的底层异步安全提前到 P2b，P3 聚焦共享 flight 调度与 P2P adapter。后续执行按本表，不把旧编号直接当完成记录。

| 阶段 | 状态与依赖 | 交付 | 验收门槛 |
|---|---|---|---|
| P0a | 已完成既有 smoke | 环境、权重下载/导出、HF/C++ demo、prompt embedding 修复 | 仅代表第 2.2 节事实；不替代 serving 回归 |
| P0b | 已完成本地基线验收；2026-09-12 | 依赖/model manifest、已有回归、64 位 request handle、请求 RNG、逐 token 指标、serving baseline | 同槽至少 100000 次周期；请求随机输入不随 batch 重排；基线原始日志可重现 |
| P1 | 已完成；2026-09-12 | PageSchema/ComponentDescriptor、逻辑 plan、CPU/fake endpoint 绑定预检与 pack/unpack oracle | layer/component 乱序、虚拟 block ID 映射往返正确；缺 K/V/scale、dtype/shape 不符在首笔 copy 前零写入 |
| P2a | 已完成本地同步路径验收；2026-09-12 | GPU-only PageDirectory、generation/leases、radix logical payload、allocation intent | split/adopt/evict 回归；stale handle 拒绝；short match 不变 scratch；功能关闭时输出回归 |
| P2b | 已完成本地验收；2026-09-12，见 12.12–12.16 | 有界 host store、同步 oracle 与最小有界异步 executor、完整页事务/journal、waiter/flight 基础两轴 | K/V/scales 往返位级一致；host 冷前缀命中；各阶段取消不早释放；unknown completion 隔离；无 runnable 仍完成 drain |
| P3 | 已完成本地验收；2026-09-12，见 12.17 | 统一 TransferScheduler/PagePresence，H2D/D2H/P2P adapter、single-flight、priority donation、依赖/wave | 同页 32 waiter 一个迁移 flight；取消 31 个不影响剩余者；交叉请求无死锁；容量有界；前后台等待/饥饿可观测 |
| P4 | 已完成本地验收；2026-09-12，见 12.18 | revision/manifest checkpoint、pending token、输出/RNG 进度与页表整体恢复 | 多次换址、页边界、恢复中取消/新 revision；无漏/重 token、无重发、无 RNG counter 回退 |
| P5 | 已完成固定预算消融；2026-09-12，见 12.19 | 布局/事务/调度/checkpoint 消融，水位/aging、恢复与重算选择 | 固定预算下分别报告收益与负收益；原始日志与重现脚本保留 |
| P6 | 本轮不启动；可选独立扩展 | 两进程同布局的跨请求内容复用、provider 二阶段 token | 需要独立模型实例/endpoint 目录与安全协议验收，不能由单机 P2P 结果代替 |
| P7 | 本轮不启动；可选研究项 | sender/receiver placement、异构 TP、逐层 ready、SSD 或跨机 RDMA | 每项需要单独硬件、正确性和收益门槛；当前明确 unsupported |

每阶段拆成可审查提交，保留已通过的默认运行路径。P2b 的最小异步 executor 只需有界执行、fence 与完成队列，先不实现 P3 的复杂优先级/共享逻辑；同步后端持续作为 oracle。P3 的每个 backend 上线前必须重跑 P2b 的完整生命周期合同。

P4 建议拆为 checkpoint 数据结构与冻结边界、manifest prepare/commit、换址恢复、模型边界回归四次提交。P4 只在单机进程内确认整体安装语义，网络断线重连/进程崩溃恢复不隐含在“无重复输出”的保证中。

P3 同机双 GPU P2P 是现有 executor 的统一接入；P6 再做两个独立模型实例、固定 endpoint 目录和跨请求内容复用。P/D 同请求 handoff 与不同请求的缓存复用分别统计。租约超时不能替代物理完成证明；没有第二台机器就不报告跨机 RDMA。

P7 的 DIRECT/SENDER/RECEIVER 先基于显式 layout 与 topology 做启发式，再与实测成本模型对照。tensor 重分片测试通过不代表模型已支持异构 TP serving；第一版也不做任意 dtype 或 page_size 转换。

## 10. 缓存策略与实验

### 10.1 容量公平性

H20 显存大，新增显式 KV blocks/bytes 上限，不靠占满真实 GPU 触发压力。模型、精度、token budget、并发、GPU KV 额度一致，host 额度单列。承认 CPU 缓存是在增加 host memory 后换取重算减少。

### 10.2 成本策略

`T_restore ≈ admission_wait + planning + bind/validation + pack + transfer + unpack + fence/commit`，其中 `transfer ≈ bytes / measured_bandwidth + launch_cost`；各段可能重叠，最终以 wall time/timeline 验证，不能重复相加共享区间。与同模型、长度区间的实测 `T_recompute` 比较；checkpoint 策略还应单列 save/offload 成本，不能只计恢复时间。

先静态阈值与简单 EMA，再加 hysteresis。低复用页可不进入 host；紧急压力时不要让 D2H 吃掉全部恢复和 decode 资源。P3 的关键请求优先级配合 FIFO/aging，复杂淘汰算法须有消融支持。omniFlow 的 coverage-based placement 是启发式，PBE 若实现动态成本选择，应作为自己的新增策略单独验收。

### 10.3 实验矩阵

| 维度 | 取值建议 |
|---|---|
| 模型 | 已下载 Qwen2-0.5B-Instruct 验证完整路径及初始成本曲线；必要时再引入可加载的 7B 级模型扩大计算/传输比，非首阶段依赖 |
| 边界 | P−1/P/P+1、2P−1/2P/2P+1，完整命中仍保留 seed-tail |
| 长度 | 约 512/2K/8K，受模型实际上限约束 |
| 复用 | 无复用、部分共享、长公共前缀、多轮；相同块 token 不同前缀反例 |
| 并发 | 1/8/32 起步；闭环和固定到达速率分别报告 |
| KV 预算 | 工作集约 25%/50%/100%，按模型实际 bytes 换算 |
| 温度 | GPU/host 都冷、GPU 冷 host 热、GPU 热；模型/kernel 预热与缓存温度分离 |
| 消融 | GPU-only、同步 host oracle、逐字段 copy/pack、单 FIFO/多 lane＋priority、single-flight 关/开、重算抢占/checkpoint 恢复；FP8 独立组 |

采集 TTFT、逐 token gap p50/p95/p99、每请求平均 TPOT、E2E、成功率、output tokens/s、真正执行的 prefill/recompute tokens、L1/L2/remote 命中 tokens、planning/bind/pack/transfer/unpack/fence/commit 时间、source pinned 页数及持续时间、staging 峰值、CPU/GPU 峰值、reserved/in-flight/quarantined、flight 数、实际 component copy 次数/字节、合并 waiter 数、demand 排队 p95/p99、后台最老任务等待及完成率。

### 10.4 主张与对应实验

| 设计主张 | 对照与触发 | 证据与限制 |
|---|---|---|
| 语义计划不依赖物理排列 | 注册顺序/各层 block ID 随机化；缺组件、格式错误 | payload 位级相同；预检失败零写入；记录 CPU planning 和缓存命中开销 |
| 延迟绑定降低排队 source pin 压力 | 固定 worker 数、1000 个逻辑排队任务，对照入队即 pin | source pin 高水位/持有时长随已准入工作变化；单列 target/staging 预留，不能宣称零总资源 |
| 取消不会污染新请求 | 受控 CUDA 延迟，在 QUEUED/BOUND/COPYING/FENCED 取消并施加 block 重用压力 | event 前不 free，旧 DMA 不写入新对象；unknown 组隔离而非伪造容量恢复 |
| single-flight 与 priority donation 有价值 | 同页 32 waiter、31 取消；后台 PRELOAD 后加入 DEMAND | 一个物理迁移 flight，记录真实 copy 次数/字节；剩余 waiter 成功；不抢占在途 DMA |
| 分 lane/有界 wave 能维持进展 | `[A,B]`/`[B,A]`、小 staging、前台依赖 D2H 腾空间、无 runnable | 无互等；deadline 有界；后台不永久饥饿；记录共享硬件带宽导致的 decode gap |
| checkpoint 恢复生成状态正确 | P−1/P/P+1、pending token、多个 revision、换址、多次暂停 | 页表/进度共同安装；无重复发送；固定随机输入与状态先独立验证 |
| 恢复优于重算的条件可解释 | 同预算、同负载，扫描长度/命中率/并发与 cache 温度 | 同时计 save 和 restore；报告 crossover 与负收益，不预设小模型必须加速 |

Qwen2-0.5B 重算较便宜，恢复或 pack 在部分配置下更慢是有效结果。CPU 层增加了可用存储，应单列 host 容量，不能将其收益归因于更优的纯 GPU 算法。

保存 token IDs/seed、权重与格式标识、commit、启动命令、拓扑、原始请求日志、GPU timeline。至少 3 次独立运行并报告波动。先同引擎消融，再做其他引擎对照。新功能关闭时吞吐/TTFT 回归预设调查阈值 5%，不是已达成数字。page_size 只扫描 kernel 已确认支持的集合。

通过 timeline 证明真正 overlap，不能凭使用 cudaMemcpyAsync 就认定重叠。[CUDA 异步执行说明](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html) 指出异步 host copy 对 pinned memory 有要求。

## 11. 测试与稳定性

- CPU core：PageSchema/coverage、逻辑 plan/模板缓存、pack/unpack oracle、identity、radix、generation、move-only lease、admission、独立 waiter、priority donation、依赖状态机；拆出无 CUDA 工具链依赖的测试构建入口，当前根 CMake 强制启用 CUDA。
- GPU 无模型：pattern K/V/scales roundtrip、不同层乱序 blocks、bind 后 bounds/epoch 复验、producer fence、取消迟到 copy、完整性失败不发布；fake backend 可确定性注入先后顺序，但不能代替真实 CUDA 在途写测试。
- 模型：cold/warm、host eviction/reload、seed-tail、多次 suspend/resume、pending token、换址/revision、同/不同前缀并发；底层迁移同格式字节必须一致，logits 按精度定义 tolerance。
- 随机输入与 logits/采样分开验证；batch kernel 的数值差异可能改变接近 tie 的输出，不能将所有差异都归因于迁移。
- 故障：host/staging 满、某层分配失败、bind/submit 前后错误、一个/全部 waiter 取消、重复完成、stale block/schema/pool generation、checkpoint 半准备/旧 revision 完成、shutdown、provider 消失/phase-2 replay。
- 压测：100000 次 handle 周期、缓存 churn；free/allocated/reserved/quarantined 守恒，无持续增长 tickets/events；故障 quarantine 组检验“不再分配”，不要求立即恢复所有 free。
- CPU ASan/UBSan，受控并发用 TSan，GPU 针对性 compute-sanitizer；工具未安装不代表检查通过。

## 12. 设计取舍与交付标准

借鉴 omniFlow：语义 component/plan、compute/I/O ownership、先 admission 后源地址绑定、copy/fence/validate/commit、per-page presence、多 waiter 生命周期与有界 staging。用 C++ 强类型、RAII、单写者完成队列实现，RNG/revision/checkpoint 与普通 slot generation 明确标为 PBE 加强。

暂缓通用 tmp tensor、history 数据库、独立 MM 与真正 TP semantic reshard。P4 的进程内 RequestCheckpoint 属于主线，不因暂缓 history 数据库而删除。完整页可见性简化失败合同；若 H2D 成为瓶颈，可用项目已有逐层传输基础在 P7 单独验证 layer-ready。

“事务”限定为完整页驻留或 owner 内 checkpoint manifest 的提交边界，不等价于跨节点数据库事务。分 lane 不等价于硬件带宽隔离；single-flight 不等价于单次 DMA；layout 重排子测试不等价于异构 TP 模型 serving。这些限制必须进入最终演示与实验说明。

每阶段交付可审查 commit、设计不变量、故障复现、性能消融、原始结果与重现脚本。注明原项目基础、借鉴来源、本人实现；移植代码前核对使用条件和归属。简历中的收益百分比只来自实测。

面试主线：组件语义为何独立于 block/rank/注册顺序；排队为何只保留意图；取消为何不等于物理完成；真实请求如何共享并提升预取；已采样 token 为何未必已有 KV；如何用故障注入证明安全，并解释恢复不划算的配置。

完成对应实现与实验后可使用的表述：

> 我参考 omniFlow 的数据流，为 C++ 推理引擎实现了语义化 KV 搬运计划和完整页迁移事务。任务在准入后绑定物理资源，GPU/CPU/P2P 共用执行合同，多个请求共享加载但独立取消。我进一步实现了生成进度与 KV manifest 一致的抢占恢复，通过在途 copy 故障注入和固定 KV 预算实验验证正确性与性能边界。

### 12.1 当前执行清单

- [x] P0a 环境、模型、导出与 HF/C++ demo 冒烟；记录在第 2.2 节。
- [x] V2 设计和 SGLang 对照更新；本清单不是运行时功能完成记录。
- [x] P0b：manifest、handle/RNG、指标、实际 greedy/采样冷热基线已有本地验收证据，见 12.3–12.11。
- [x] P1：新增语义 schema/plan 与 CPU oracle，完成乱序/缺字段/零写入预检测试；独立纯 C++ 测试入口见 `test/cache_core/`。
- [x] P2a：页面所有权、logical radix、allocation intent 与同步 producer fence，本地验收见 12.11。
- [x] P2b：完成有界 host store、迁移事务、CUDA executor、host-only radix 恢复与真实模型冷命中，见 12.12–12.16。
- [x] P3：共享加载、优先级调度和 P2P executor 已统一接入并通过本地验收；达到第一里程碑。
- [x] P4：进程内 checkpoint、revision/manifest、pending token、输出/RNG 游标和换址恢复通过验收；见 12.18。
- [x] P5：固定预算完成布局、事务、调度、checkpoint 与恢复/重算消融，正负结果均保留；见 12.19。
- [x] P6/P7 取舍：作为独立可选扩展留待出现跨进程复用或异构硬件需求时立项，本轮主线不启动。

### 12.2 P1 实施记录（2026-09-12）

- 新增 `cache/page_schema.*`：版本化 schema、K/V/scales 完整覆盖、shape/dtype/storage 一致性与稳定 fingerprint。
- 新增 `cache/layout_codec.*`：组件语义到 offset/bytes/虚拟 block ID 的布局描述，拒绝缺项、重复、越界与重叠。
- 新增 `cache/transfer_plan.*`：不含地址的逻辑 plan 与模板缓存；`BoundTransfer` 在执行期一次性预检两个 endpoint，CPU oracle 只消费已绑定操作。
- 新增 `test/cache_core/` 的纯 C++ 测试入口；5 个合同测试覆盖 plain/FP8、乱序布局、虚拟 block 映射、pack/unpack 位级往返，以及缺组件和 dtype/shape 错误时零目标写入。
- 主库 Qwen2 配置编译通过；同时修正 serving 对 `nlohmann_json` 的无条件依赖声明，以及关闭 Qwen2 时仍编译 Qwen2 专用测试造成的链接失败。
- P0b 已有 cache/radix/scheduler/PD 定向回归完成 53 项：51 项通过，2 项因本构建显式关闭 NCCL 而跳过；NCCL 测试现在按构建能力正确跳过。P0b 的 handle/RNG/指标与 serving baseline 仍未完成。
- 补齐 GCC 10.3.1 对应的用户级 `libasan`/`libubsan` 运行库；P1 的 5 项 CPU 合同测试在 ASan、LeakSanitizer 与 UBSan 下全部通过。主机强制预加载 DolphinFS 客户端，因此测试时须将 `libasan.so.6` 放在 `LD_PRELOAD` 首位。

### 12.3 P0b request handle 实施记录（2026-09-12）

- 新增同槽 100000 次注册/分配/释放回归，在旧实现上实测 generation 2047 后 CHECK 中止；原始日志 `/tmp/p0b-handle-before.log`。
- `RequestId` 改为 int64，保留 20 位 slot、扩为 43 位 generation，达到极限的槽释放后永久退出可复用列表；新增极限代际注入测试，确认旧 handle 失效、新请求分配新槽。
- 修正 engine core 两处及 online engine pool 一处 JSON request_id 默认值的 int32 推导，改为显式 int64。PD layer request 的 JSON 读取使用 RequestId 字段默认值，随类型同步拓宽。
- 集成构建与 KVCacheManager/Scheduler/PD 定向回归 36 项：34 通过、2 项 NCCL 关闭跳过；包括 100000 次同槽复用和极限退休。日志 `/tmp/p0b-handle-build-final.log`、`/tmp/p0b-handle-final.log`。
- P0b 尚未整体验收：请求 RNG、逐 token 指标、manifest、实际 serving baseline 仍待完成；跨进程大 ID 运行验证尚待补充。
- 64 位 handle 改造后 Qwen2 配置主库编译通过，日志 `/tmp/p0b-handle-qwen-build.log`。

### 12.4 P0b RNG 与指标实施记录（2026-09-12）

- SamplingConfig 新增 uint64 seed；HTTP 请求与内部 generation_config JSON 传递该值。默认 seed 为 0xC0FFEE。
- Qwen2 CPU/CUDA configured sampling 使用同一个 SplitMix64(seed, counter) 映射出的 [0,1) float。SequenceState 独立保存 sampling_counter，只在 token 提交时递增；统计计时重置不修改 counter，P/D 首 token 接管初始化为 1。
- SequenceState 保存 token 时间戳并计算实际相邻 gap；原 itl_ms 保留为兼容的请求内均值。FINAL_SUMMARY 新增 token_gap_count、token_gap_p50/p95/p99_ms、request_mean_token_gap_p95_ms。
- 新增 3 项测试覆盖乱序 batch 的随机输入、提交前重试/换 handle、进度复制与指标重置、非均匀 token gaps。相关测试与旧回归共 21 项通过；日志 `/tmp/p0b-rng-tests.log`。这不是模型输出或 checkpoint 整体验收。
- Qwen2 主库及 serving_qwen 构建通过。两请求、每请求 24 token 的真实 greedy serving 冒烟日志保存在 `data_flow_evidence/p0b_greedy_smoke_20260912.log`。采用一次 warmup，radix 开启，因此该结果含热前缀命中，不能用作冷启动或改造收益。
- 重现命令：`/tmp/Paged-Batch-Engine-build-qwen05/demo/serving_qwen /tmp/Paged-Batch-Engine-models/Qwen2-0.5B-Instruct.bf16.bin /tmp/Paged-Batch-Engine-models/Qwen2-0.5B-Instruct/tokenizer.json 'Explain KV cache in one sentence.' 'What is continuous batching?' --max-new-tokens=24 --max-batched-tokens=128 --kv-cache-memory-utilization=0.02 --warmup-rounds=1 --quiet=1 --final-summary=1`。
- 待完成：实际 seeded sampling 的复现与 batch 对照、HTTP/RPC seed 和大 ID 验证、完整环境/model manifest、冷/热 serving 基线。P0b 保持进行中。

下一步完成剩余 P0b 验收，再进入 P2a 的 PageDirectory、generation 与 lease 接入。无需重新下载有效模型或重建已可用环境；不得把旧 demo 数字写成改造后的 serving 收益。

### 12.5 P0b 实际 HTTP 采样与 provenance（2026-09-12）

- 新增 `tools/bench/data_flow/seeded_http_probe.py`，每轮启动独立 serving 进程，执行两次固定 seed 串行请求及三请求并发对照，共三轮；原始 command、payload、response、wall_ms 与服务日志在 `data_flow_evidence/seeded_http/`。
- 三轮 fixed-seed 串行重复均相同；并发组至少一个同 seed 请求与串行输出不一致。不能据此宣称 batch 无关的模型输出；随机输入单测通过，但实际差异来源尚未定位，需对照 CPU/GPU sampling、token IDs 与 logits，P0b 仍进行中。
- `tools/bench/data_flow/capture_manifest.py` 记录本地模型/导出权重 SHA256、Python 包版本和安装位置、GCC/CUDA、CMake 参数、Git HEAD/dirty 状态、GPU UUID/驱动/拓扑；产物 `data_flow_evidence/environment_manifest.json`。上游 revision 未知，使用本地文件哈希锁定实际内容；manifest 不表示工作区已提交。
- 新增 RequestWireTest：完整宽度 uint64 seed 和大于 32 位的两个 PD handle 经实际 JSON codec 往返无损；与采样/指标共 4 项通过，日志 `data_flow_evidence/p0b_wire_test.log`。这是 codec 验证，不等价于跨进程 ZMQ 验收。
- 下一步优先定位并发 sampling 差异，补冷/热与 CPU/GPU 对照，完善 manifest 的 C++ 依赖版本及源码差异哈希，再继续 P2a。

### 12.6 实际采样输入与 logits 分离验收（2026-09-12）

- 新增 opt-in `KUIPER_TRACE_REQUEST_SAMPLING=1`，记录每个实际样本的 handle/seed/counter/batch/uniform24/token；CUDA 模型在 counter=19 额外记录 top-16 logits 与同一行 CPU sampler 复算。仅诊断时开启，会增加同步与日志开销，不用于性能结论。
- `tools/bench/data_flow/analyze_sampling_trace.py` 检查每轮 5 请求、每请求 24 连续 counter，比较同 seed 请求的实际随机输入。CPU sampling 与 CUDA sampling 各三轮均通过；证据目录 `seeded_trace_cpu/`、`seeded_trace_cuda/`。
- 找到 CPU/CUDA 对相同 logits 的平局候选排序差异：CPU 原 comparator 未规定 token 次序，CUDA 合并依赖线程次序。统一为 logit 降序、相同值 token ID 升序。新增跨 CUDA 线程的 top-k tie 用例，与其他 GPU sampler/RNG 测试共 7 项通过，日志 `data_flow_evidence/p0b_tie_tests.log`。
- 修复后再跑三个独立进程（`data_flow_evidence/seeded_tie_fixed/`）：串行 fixed-seed 复现、每请求 counter 连续、随机输入与 batch 无关均通过。检查 counter=19 的相同行 logits，CUDA token 与 CPU oracle 一致。
- 并发与串行 token 仍可能不同：在同一历史及同一随机输入下，记录的 BF16 logits 已不同。例如旧诊断 token 438 的 logit 由 19.500 变为 19.625；这可改变概率及候选次序。已证明差异不由随机输入改变引起，不能承诺 batch 数值逐位一致；完整前向数值容差/迁移对照留在模型验收中。
- P0b 剩余：补无诊断的三次冷/热性能基线与完整 C++ 依赖/source diff provenance。P2a 运行时接入时继续回归默认模型路径。

### 12.7 P0b 冷/热基线及 P2a block lease 起步（2026-09-12）

- `tools/bench/data_flow/serving_baseline.py` 完成 6 次独立进程测量，诊断关闭；两请求各 24 token、固定 workspace 128、KV utilization 0.02，cold/warmup=1 各三次，原命令、FINAL_SUMMARY 与均值/标准差保存于 `data_flow_evidence/serving_baseline/`。
- cold 吞吐均值 319.121 token/s（标准差 2.536），TTFT 89.920 ms；warm 吞吐均值 745.480（标准差 9.380），TTFT 3.142 ms。token gap p95 均值分别 2.600/2.640 ms。warm 同时含模型预热和 radix 命中，不是纯缓存消融，不是本次重构收益。
- manifest 已增加 C++ CPM 来源 commit/版本线索、工作区源码 SHA256 清单及实际 libllama/serving_qwen 哈希；证据是 P2a 编辑前的 P0b 快照。上游模型 revision 仍未知，以本地权重 SHA256 固定实物。
- P0b 尚需补 configured sampling 的吞吐/TTFT/gap 基线，不能仅用 HTTP 请求总时长替代全部指标；已通过的 handle/RNG/缓存相关回归允许并行开展 P2a（单 agent 顺序执行）。
- P2a 已扩展真实 BlockAllocator：pool identity、每次 allocate 的 uint64 block generation、耗尽退休；新增 move-only BlockLease，compute/I/O 各自计数并持有物理引用，最后一个 lease 释放前不进入 free queue。元数据按 owner 线程操作，allocator 必须晚于 lease 析构。
- 新增 BlockLeaseTest，覆盖请求引用提前释放、compute+I/O 交叠、move/覆盖释放、stale generation 和跨 pool 拒绝。集成 BlockLease/KV/radix/scheduler 回归 35 项通过，日志 `data_flow_evidence/p2a_block_lease_tests.log`。
- P2a 未完成：PageDirectory、radix logical payload、allocation intent、compute fence 接入与默认路径模型回归仍需实现。本轮没有把 lease 原语测试等同于真实异步 copy 生命周期验收。

### 12.8 configured sampling 基线与 PageDirectory（2026-09-12）

- BenchConfig/CLI 增加 temperature/top-k/top-p/seed/ignore-eos，并传入实际 scheduler request（含 P/D offline 入口）。`serving_baseline.py --sampling` 跑固定 seed、top-k=16、top-p=0.9、temperature=0.8 的冷/热各三次独立进程，诊断关闭，证据 `data_flow_evidence/sampling_baseline/`。
- sampling cold 平均吞吐 281.226 token/s、TTFT 92.498 ms、gap p95 3.440 ms；warm 分别 571.736、4.034、3.575。均为 2 请求 × 24 token 小规模基线，波动和完整命令见 results.json，不外推到生产吞吐。
- P0b 的本地功能和基线条目已有对应证据：manifest、既有回归、100000 handle、实际 RNG 输入对照、逐 token gap、greedy/configured sampling 三次冷/热运行。最终验收仍须核对所有证据与实际构建，不将 codec 测试扩大表述为跨进程 ZMQ 通过。
- P2a 新增 `cache/page_directory.*`：有界 logical page 索引、格式隔离、全层 handles 预检、目录缓存引用、整页 move-only leases；erase 后在途 lease 继续保留源物理块。调用者负责在 producer fence 完成后发布，当前尚未把此约束接到模型执行链。
- PageDirectory/BlockLease/KV/radix 共 31 项通过。覆盖目录回收与 I/O 存活、跨模型内容 key、capacity、重复发布、最后一层 stale 预检失败时无引用泄漏。当前目录接受调用者内容 key，稳定内容序列化和 radix logical payload 还需实现。
- 后续优先把 PageDirectory 接入 radix 发布/匹配/回收，再接入 compute fence、allocation intent 与模型回归，不能把 standalone directory 测试等同于 P2a 完成。

### 12.9 radix logical payload 运行时接入（2026-09-12）

- CompressedRadixCacheTree 泛型化 payload，保留原 int32 独立测试别名；KVCacheManager 使用 uint64 LogicalRadixCacheTree，只保存一列逻辑页 ID，PageDirectory 持有所有层的缓存物理引用。
- 发布完整页时生成显式 LE32 全前缀身份（目录限定在单一模型 allocator 集内），新页进入目录后再插入树；重复前缀不新增物理缓存引用。当前身份是 pool-local，尚不是跨进程/跨模型导出身份。
- 命中时先通过目录取得全页 compute lease 并复验 pool/generation，再解析各层 block IDs 给现有 request page table；请求 adopt 后持有自己的物理引用。缓存 eviction/clear 删除目录引用，外部在途 leases 不受影响。删除旧 radix 直接 retain/release block 的私有路径。
- 新增双层不同 block ID 的真实 KVCacheManager 用例：发布→短前缀 split→完整前缀命中→清理，层映射正确且最终物理块全部回收。PageDirectory/BlockLease/KV/radix/scheduler/PD 56 项定向回归：54 通过、2 项因关闭 NCCL 跳过。
- Qwen2 serving 构建通过，`data_flow_evidence/p2a_radix_baseline/` 完成冷/热各三次真实生成。对 P0b 同工作负载基线：cold throughput -1.15%、TTFT +1.97%；warm throughput +0.91%、TTFT -0.84%，未触发 5% 调查阈值。数据量仅两请求各 24 token，不据此声称性能收益。
- P2a 未验收部分：producer fence 后显式 KV commit/publish、allocation intent、执行期 compute lease、不可用 residency 时最长可用前缀解析、功能关闭时输出对照；P2b 以后尚未实施。

### 12.10 KV commit、producer completion 与 allocation intent（2026-09-12）

- RequestSlot 分开记录物理 allocated（现有 get_context_len）、kv_committed_tokens、published_full_tokens；新请求全部从零开始，prefix adopt 继承目录完整页的已提交长度。commit_kv 拒绝 stale handle、回退和超过已分配长度；radix 仅发布 committed 完整页。
- Qwen2 mixed/decode forward 在提交 kernel 前取得请求的 compute BlockLeases，成功同步 compute queue 后才 commit_kv，随后释放 leases。当前采用同步 completion 基线；错误沿既有 CHECK 终止路径，不把未知完成当成功。该路径未使用 graph capture；未来异步 executor 接入需保留物理完成合同。
- 新增 KVAllocationIntent，仅含 request generation handle、expected allocated length、append token count。plan 不分配内存，admit 复验请求与长度；decode 单 token 和 mixed prefill 分配已使用该入口。
- 新测试覆盖分配而未提交不命中、部分已提交仅发布完整页、stale/回退/越界 commit 拒绝、取消请求后 compute pin 保留物理块、过期 allocation intent 零状态变化。合计 59 项定向测试：57 通过、2 NCCL 关闭跳过；日志 `data_flow_evidence/p2a_commit_tests.log`。
- Qwen2 serving 构建和冷/热各三次真实运行通过，结果在 `data_flow_evidence/p2a_commit_baseline/`。这是同步屏障基线，不能宣称已经具备异步迁移 overlap。
- P2a 仍待统一验收：关闭功能的输出对照、可用前缀边界、CPU core 独立入口覆盖新目录/leases；P2b 的 host store、迁移事务和真实在途取消尚未实现。


### 12.11 P2a 前缀边界、CPU-only 与验收汇总（2026-09-12）

- radix 增加只读 probe，不分裂节点/触碰 LRU；匹配先取得连续可用页 leases，再 materialize 可采用的前缀边界。不可用页之后不继续采用；首个页不可用时零页表改变、零 tree split。
- 新增高位 logical ID 和短 probe 用例，以及真实 KVManager 缺中间 residency/缺首 residency 故障注入。集成 prefix/directory/block/KV/radix 37 项通过。
- 纯 C++ CMake 入口新增 KUIPER_TEST_CPU_OWNERSHIP，直接编译生产 CPU allocator/Tensor/BlockAllocator/PageDirectory/KVManager；不启用 CUDA、不链接 CUDA。CPU-only 分支明确拒绝 CUDA/pinned 操作，普通 GPU 构建不定义该宏。配置步骤见 test/cache_core/README.md。
- 完整 CPU 26 项全部通过；ASan/UBSan/LeakSanitizer 同样 26/26；日志 `data_flow_evidence/p2a_cpu_ownership.log`、`data_flow_evidence/p2a_cpu_ownership_asan.log`。包括真实生产块引用和目录生命周期代码，不是替代 allocator 测试。
- 最新 Qwen2 serving 构建通过；radix-cache=on/off 各三轮固定 seed 串行输出一致，证据 p2a_radix_on/ 与 p2a_radix_off/。并发输出数值差异延续 12.6 已记录边界，不声明跨 batch token 一致。
- P0b/P2a 本地阶段门槛通过。保留限制：目录身份是单模型 pool-local；当前 producer completion 为同步屏障；没有异步 host migration，也未承诺跨进程/异构 TP。P2b 必须补真实 CUDA 在途取消、资源隔离、完整页事务和 host 冷前缀命中。
- 最近同步屏障基线 cold token gap p95 相对 P0b 上升约 5.7%，throughput/TTFT 变化小于 5%；原始数据保留，后续 P5 重复实验继续检查该指标，不隐藏负向结果。

### 12.12 P2b HostStore 与真实在途取消（2026-09-12）

- 新增 `cache/host_store.*`，同时限制 byte/page capacity。reservation 在分配前检查预算，分配失败不占容量；单调 ticket 防止旧 completion 误关联后续保留。
- 状态为 RESERVED/COPYING/READY/QUARANTINED；只有完整 copy fence 和 validation 后 complete 才转 READY。copy 后取消只标记，不释放目标容量；物理完成后 safe rollback。未知完成进入 quarantine，继续占预算且不可读/不可重用，显式 backend quiescence 后才可回收。
- HostReadLease 为 move-only，读者存活时禁止 eviction。HostStore 析构要求 DMA 已 drain、quarantine 已处理且读者已释放，否则 fail closed；后续 executor shutdown 必须满足该约束。
- 默认 allocator 是 CPU oracle，可注入实际 pinned allocator。5 项 HostStore 单测覆盖容量/失败/整页 TransferPlan 往返/read pin/late-write cancel/quarantine；完整 CPU 和 ASan+UBSan+LSan 分别 31/31。
- 新增真实 CUDA gated-stream 测试：使用 cudaLaunchHostFunc gate 延迟 D2H，确认 event 尚未完成后取消目标并释放请求源引用；source I/O pin 阻止重新分配，pinned host reservation 不可读且不归还预算；放行后验证迟到 K/V 字节正确，成功 drain 后释放两端。集成 HostStore/Directory/BlockLease 10/10，compute-sanitizer memcheck 该真实 CUDA 用例 ERROR SUMMARY: 0 errors。
- 原始证据 `data_flow_evidence/p2b_host_cpu.log`、`p2b_host_asan.log`、`p2b_host_cuda.log`、`p2b_host_compute_sanitizer.log`。
- P2b 尚未完成：HostStore 尚未接入目录多 residency、scatter BoundTransfer executor、迁移 journal、scheduler 恢复等待和真实模型 host 冷前缀命中。上述取消用例不是生产迁移 executor 的整体验收。

### 12.13 P2b 分散组件绑定与真实 GPU 往返（2026-09-12）

- 新增 ComponentSpan、Const/MutableComponentEndpoint 和 BoundTransfer::BindComponents；每个组件显式绑定独立 allocation，layout offset 仅用于连续端点适配。连续 Bind 复用相同预检，不跨 allocation 推算地址。
- 绑定前检查 schema/plan、全组件覆盖、重复、dtype/shape、长度、地址溢出和端点内实际区间重叠；失败不改输出 binding，不写目标。保留两端 epoch 与 virtual block ID。
- 新增 CPU plain/FP8 分散多层 → 乱序 packed host → 分散目标往返及错误预检；CPU 32/32，ASan+UBSan+LSan 32/32。日志 `data_flow_evidence/p2b_scatter_{cpu,asan}.log`。
- 新增真实 CUDA plain/FP8 K/V/scales 从 BlockAllocator 源块，经 pinned host，恢复到不同 block 地址的同步 oracle；包含 producer fence，逐字节核验全部组件。与真实在途取消、TransferPlan 一起运行 compute-sanitizer，6/6 通过、0 errors，见 `data_flow_evidence/p2b_scatter_cuda_memcheck.log`。
- 这仍不是生产异步迁移验收。下一步接入目录 GPU/host residency、迁移 executor/journal 和 scheduler 恢复；P2b 尚未完成，主线阶段比例仍为 4/8。

### 12.14 P2b 目录多副本生命周期（2026-09-12）

- PageDirectory 保存 schema 与 host cache pin，支持 attach_host/acquire_host/demote_gpu/install_gpu/drop_host。attach 检查完整 schema fingerprint、READY 可读性和本目录内 ticket 独占；host store 必须晚于目录和读者析构。copy 内容验证仍是迁移 executor 在发布前的责任。
- demote_gpu 要求 host 可读且无 compute/I/O pins，仅释放目录 GPU 引用；host-only 保留 logical ID 和内容索引。install_gpu 在成功 H2D fence 后由调用者提交，验证 source store 身份、ticket、logical ID 和全部目标 block generation/格式后统一保留引用。
- HostStore 新增 retire：立即阻止新 reader，已有 reader 保持物理内存/预算，最后释放时归还容量。drop_host 删除最后副本时也删目录索引，仍有 GPU 副本时保留索引。目录 erase/clear 对 host-only 安全。
- 新增目录生命周期与 schema version/ticket 错配测试：CPU 34/34，ASan+UBSan+LSan 34/34，CUDA 构建下 HostStore/Directory/KVManager 27/27。测试包含 host-only 恢复、stale source/目标拒绝、compute/I/O pin、在途 reader 延迟回收。
- serving 旧构建暴露 HostStore 新文件未被配置时源码扫描纳入的链接错误；cache 源码改用 CONFIGURE_DEPENDS 自动重配置。
- 证据 `data_flow_evidence/p2b_residency_{cpu,asan,cuda}.log`。这些是目录和生命周期合同，尚非真实请求 host 冷命中；下一步继续有界迁移 executor/journal 与 scheduler 接入。P2b 保持未验收。

补充验证：修复源码发现规则后 Qwen2 serving_qwen 构建成功；日志 `data_flow_evidence/p2b_residency_qwen_build.log`。

### 12.15 P2b 有界迁移事务与 CUDA 后端（2026-09-12）

- 新增 `cache/page_migration.*`：owner-thread PageMigrationEngine，jobs/bytes 双上限，submit/cancel/poll/drain/completion/consume；完成记录未消费前也占 job 预算，quarantine 不可消费。准入后才取得源 leases、目标 reservation 和分散组件地址。
- D2H 持整页 GPU I/O leases 与 host reservation；H2D 持 HostReadLease 和新分配目标的 I/O leases。事务只回滚新取得的资源，不撤销旧副本。完整 fence 后分别 attach_host/install_gpu，目录/source store/ticket/block generation 不匹配则回滚。GPU demote 是完成后的独立策略操作。
- CPU 同步 oracle 与 CUDA backend 共用事务合同。CUDA 后端预建固定数量 stream/event 槽位，submit 前全组件检查 GPU device 与 pinned host 类型；提交失败/未知完成保留全部资源，成功 stream drain 后安全回滚，不能把 event destroy 当完成。
- CUDA 可借用一个 dependency event，所有提交显式 wait；event 必须晚于 backend 析构。正常已提交目录页可省略。shutdown 取消未终结 waiter、drain physical copy 后回滚；无法证明 quiescence 时 terminate，禁止释放未知在途内存。
- 新 CPU 故障测试涵盖容量拒绝、未知页、后续层分配失败、safe submit failure、取消后迟到 copy、未知完成隔离、删除目录后迟到 H2D、重复 poll、shutdown。完整 CPU 41/41，ASan+UBSan+LSan 41/41。
- 真 CUDA 事务测试以 gated dependency event 阻塞 copy，plain/FP8 各覆盖正常 D2H→demote→H2D 及 D2H 后取消/删除目录；验证在途 source I/O pin 与 host 预算、所有 K/V/scales 字节。compute-sanitizer 8/8，0 errors。
- 证据 `data_flow_evidence/p2b_migration_{cpu,asan,cuda_memcheck}.log`。这是实际异步事务后端验收，尚未接入 KVManager/scheduler；真实请求 host 冷前缀命中、无 runnable 的 owner drain 和压力策略仍待实现，因此 P2b 不标完成。

补充：Qwen2 serving_qwen 构建通过，证据 `data_flow_evidence/p2b_migration_qwen_build.log`；本轮所有构建/测试进程已退出。

### 12.16 P2b KVManager、scheduler 与真实模型验收（2026-09-12）

- KVCacheManager 可选配置有界 pinned host cache 与迁移并发数。radix 命中逐页区分 GPU-ready、host-ready 和缺失；含 host-only 页时请求保持 restore pending，逐页 H2D 完成后才一次性 adopt 连续完整前缀。取消请求把物理 flight 转为 orphan，由 owner completion service 继续 drain，request slot 可安全复用。
- scheduler 每步先处理 completion。等待 host restore 的请求不进入 batch，也不会被 no-progress/rejection 误判；无 runnable 时仍轮询 completion。离线和在线循环在仅等待 DMA 时做 50µs 有界等待，真实用例 no-progress polling 从 133 次降至 3 次。
- GPU 压力下 host-enabled 路径先迁移可下沉的 radix 页，物理完成并提交 host 后才释放目录 GPU 引用；不会沿旧路径直接删除 logical prefix。host 禁用时保留原 radix eviction 行为。
- 新增 CPU 生产对象测试：host-only 多层恢复、取消后 request slot 复用、压力下沉且 logical prefix 保留、scheduler 恢复后只调度 seed tail。CPU 44/44，ASan+UBSan+LSan 44/44；CUDA/集成定向 30/30。
- CUDA KVManager 测试对 plain 与 FP8 完成 publish→D2H→host-only→新物理 block H2D→radix adopt，并逐字节核验 K/V/scales；最终 compute-sanitizer 定向 3/3，0 errors。
- 真实 Qwen2-0.5B-Instruct：64-token prompt，warmup 后 3 个完整页下沉 host，新请求恢复 48 tokens，仅执行 16-token seed tail，生成文本与 GPU warm-cache 对照均为 `A paged key value cache is a`；完成 1、失败 0、host_restore_requests=1、host_restored_blocks=3、host_restore_failures=0。
- P2b 的验收门槛均已有本地证据，阶段标记完成。P3 将在此合同上增加 per-page single-flight、独立 waiter、priority donation、D2H/H2D/P2P lanes 与容量/饥饿观测；当前实现不提前宣称这些 P3 能力。

### 12.17 P3 共享 flight 调度与同机 P2P 验收（2026-09-12）

- 新增统一 `TransferScheduler`，以 `(logical page, target, destination)` 合并物理 flight；waiter 独立完成/取消，最后 waiter 取消后进入 drain，直到后端 fence 给出安全终态才释放资源。H2D/D2H 经 `PageMigrationEngine` adapter 接入，现有同机 P2P connector 经相同 `FlightExecutor` 合同接入。
- 调度器提供 decode/prefill/background 三档优先级、同 flight priority donation、显式依赖 wave、有界 flight/waiter/active 准入以及 aging。两档优先级差为四个 aging quantum，测试证明持续 decode 到达时后台任务在有限 tick 内获得执行；指标记录 merge、physical submission、donation、最大容量和最老排队 tick。
- 实际 CUDA H2D 中 32 个请求命中同一 host 页，只产生 1 次物理 restore；实际双 GPU P2P 中 32 个 waiter 只产生 1 次物理传输，取消 31 个不影响剩余 waiter，目标 KV 与源逐字节一致。direct P2P 与既有 host migration 生命周期回归也重新通过。
- CPU normal 与 ASAN/UBSAN/LSAN 各 49/49；CUDA 集成回归 35 passed、2 个 NCCL-disabled skip；compute-sanitizer 对共享 H2D 与共享双 GPU P2P 报告 0 errors。汇总证据见 `data_flow_evidence/p3_runtime_acceptance.txt`。
- P3 验收门槛均已有本地证据，阶段标记完成；主线进度为 6/8（75%）。下一步进入 P4 的 revision/manifest checkpoint、pending token、RNG/output 进度与页表整体换址恢复。

### 12.18 P4 请求 checkpoint 与精确恢复验收（2026-09-12）

- 新增进程内 `RequestCheckpointStore` 和不可变 `KVRequestSnapshot`。prepare 阶段不可见，commit 才发布 READY revision；manifest 保存模型 namespace、KV valid/committed 长度、pending token、请求 RNG seed/counter 与 emitted cursor。
- scheduler 增加 Suspend、Offloading、Suspended、Restoring 状态和整步边界检查。恢复分配新的 request handle 与全层物理页表；模型/schema 不匹配、取消、旧 revision 和新 revision 竞态均在安装前拒绝并回收新资源。
- CPU 用例覆盖 P−1/P/P+1、plain K/V、三次换址、旧 restore 与新 revision 竞争、恢复中取消和 prepare 可见性；CUDA FP8 用例逐字节验证 K/V/scales。scheduler 用例验证部分 prompt、pending token 恰好消费一次、输出/RNG 游标不回退以及 step 内暂停被拒绝。
- 实际 Qwen2-0.5B-Instruct BF16 固定 seed、temperature/top-p/top-k 生成在第 1、3 step 暂停恢复，最终 token 序列与不中断执行完全一致。P5 又完成一次预热和三次独立计时重复，四次均保持一致。
- 最新 CPU 与 ASan/UBSan/LeakSanitizer 各 53/53；CUDA 全量排除三个缺失外部 fixture 的 `test_load.*` 后 157 passed、2 个 NCCL-disabled skip；FP8 snapshot compute-sanitizer 0 errors。汇总见 `data_flow_evidence/p4_runtime_acceptance.txt`。
- P4 保证限于当前进程存活期间的 owner/manifest 安装原子性；不包含崩溃恢复、持久化数据库和网络重连。阶段完成后主线为 7/8（87.5%）。

### 12.19 P5 固定预算性能消融与扩展取舍（2026-09-12）

- 新增 `test/test_p5_ablation.cpp` 与 `tools/bench/data_flow/p5_ablation.py`。固定 2 请求、每请求 8 个新 token、128 batched-token、64 prefill chunk、0.02 GPU KV 利用率；host 预算 64 MiB/128 页/2 in-flight，所有 serving 模式各跑 3 个独立进程并保留原始日志。
- GPU 热前缀相对重算：吞吐 +39.27%、TTFT −64.84%、wall −28.19%，每轮复用 256 tokens。host 恢复相对重算：吞吐 +20.44%、TTFT −40.05%、wall −16.98%，每轮恢复 16 blocks、失败为零。
- single-flight 把 32,000 个独立物理提交压到 1,000（−96.875%），固定 CPU 调度循环耗时降低 47.45%。布局语义的代价也单列：reversed-layout semantic transfer 比连续 memcpy 慢 36.59%。
- checkpoint 的负收益边界明确：200 次事务 save/free/restore 比 raw snapshot 慢 59.25%；真实短 Qwen 生成的恢复执行比不中断执行慢 9.01%，checkpoint copy 平均 2.172 ms。host restore 的 token-gap P95 增加 1.82%，因此收益来自减少 prefill 和改善 TTFT，并未改善 decode gap。
- 结果、波动、命令和日志见 `data_flow_evidence/p5_runtime_acceptance.txt` 与 `data_flow_evidence/p5_ablation/results.json`。本实验只刻画本机 Qwen2-0.5B 短负载，不外推通用 crossover。
- P6 需要两个独立模型实例、endpoint 目录和 provider token 安全协议；P7 需要异构 TP、逐层 ready、SSD 或跨机 RDMA 的独立硬件目标。本轮没有这些产品/硬件触发条件，且它们在阶段表中明确为可选，因此不把其加入 8 阶段主线，也不以现有同机 P2P 冒充完成。
- P0a/P0b/P1/P2a/P2b/P3/P4/P5 全部完成，主线最终进度为 8/8（100%）。
