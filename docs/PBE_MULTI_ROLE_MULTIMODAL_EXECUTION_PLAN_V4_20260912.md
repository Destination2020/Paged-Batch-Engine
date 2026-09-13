# PBE 多角色与多模态执行计划 V4

> 新增交付任务：第 20 节 B1–B6 为“性能证据整理与补测”，当前 **0/6，待执行**，本次仅补充计划。保留 M0–M9 10/10、E1–E4 4/4 的历史功能验收口径；新增实验尚未完成，不据此预先宣称性能提升或任意 N:P/D 拓扑已验收。

> 2026-09-13 E4 执行完成：同卡 P/D 跨进程只读权重共享已通过真实主路径、故障、数值、回归、sanitizer 与两组五次实验，见 `V4_E4_SHARED_WEIGHT_COMPLETION_20260913.md`；当前 **E4 为 1/1**，历史 M0–M9 为 10/10，扩展 E1–E4 为 **4/4**。逐层传输/计算重叠、流式角色图/背压、异构 TP 布局转换仍不在此次范围。

> 2026-09-13 最终整改：复核指出的数值、在线生命周期、实际设备预算、压力恢复与实验缺口均已补齐并通过可执行门禁；E1/E2/E3/E4 均已完成，当前验收为 **原阶段 10/10，扩展 4/4**。实际依赖回收、实时选路状态、生产调度器 lane 的收口见 `V4_REVIEW_GAPS_CLOSURE_20260913.md`；常驻 Prefill→KV handoff→Decode 对照见 `V4_PERSISTENT_PD_CLOSURE_20260913.md`；同卡只读权重共享见 `V4_E4_SHARED_WEIGHT_COMPLETION_20260913.md`。原始复核仍保留在 [V4 最终验收复核](V4_FINAL_ACCEPTANCE_REVIEW_20260913.md)。

## 1. 交付范围与进度口径

架构依据：[PBE 多角色多模态架构](PBE_MULTI_ROLE_MULTIMODAL_ARCHITECTURE_V4_20260912.md)。本计划替代旧 V3 的执行顺序；V3 的不变量、故障测试和公平实验要求作为详细规范继续生效。发生范围冲突时以本计划为准。

交付目标：同机双 GPU 上，真实 Vision worker、PBE Prefill worker、两个独立 Decode 消费者通过框架数据层共享视觉特征与 KV；完成数据独立保留、取消、COW、压力恢复和可复现实验。语言推理必须运行于 PBE C++/CUDA；不能以外部完整 VLM API 代替。

截至 2026-09-13，逐阶段整改与扩展执行均已完成：**M0–M9 为 10/10，E1–E4 为 4/4**。历史研究原型、失败尝试和负收益实验继续保留；最终结论以各阶段完成报告及 `docs/data_flow_evidence/v4/` 原始证据为准。

首版：图像+文本、单一 Qwen2.5-VL-3B 候选模型、TP=1、同机、可信进程、BF16 模型正确性主路径。原纯文本 plain/FP8 路径继续回归。同卡 P/D 只读权重共享作为独立 E4 已验收。跨机 RDMA、任意分片多 provider 聚合、视频/音频/DiT、Draft/Verify 继续后置。

## 2. 阶段与依赖

| 阶段 | 核心交付 | 前置 | 状态 |
| --- | --- | --- | --- |
| M0 | 当前代码核验、旧问题回归、构建与证据冻结 | 当前 checkout | 已验收 |
| M1 | 真 VLM 数值探针与输入 ABI 冻结 | M0 | 已验收 |
| M2 | 数据对象、唯一所有权、本地 runtime、typed 协议 | M0；多模态字段取 M1 | 已验收：原 serving Prefill/Decode 主路径使用 Agent-owned external pool |
| M3 | Data service、跨进程池、真实多消费者文本 KV | M2 | 已验收：同 GPU IPC 与跨 GPU 计费 copy、多消费者及生命周期通过 |
| M4 | Tensor bundle 池、Python client、真实 Vision role | M1–M3 | 已验收：常驻 typed role、变长 batching、feature/request 身份与取消通过 |
| M5 | PBE 多模态 Prefill/Decode 与完整 EPD 闭环 | M1、M3、M4 | 已验收：常驻 typed role、20-case 混合/单请求、阶段取消 |
| M6 | 跨请求特征复用、Prefix Builder、分支 COW | M3、M5 | 已验收：跨角色四分支、半数中途取消、尾页 COW 与退出顺序通过 |
| M7 | 联合预算、lane、背压与压力恢复 | M5、M6 | 已验收：实际 bundle/staging/KV 联合准入、typed 拒绝与同 PID 恢复 |
| M8 | 故障矩阵、服务退出与长期稳定性 | M3–M7 | 已验收：57/57 回归及线上 malformed/restart 代际矩阵 |
| M9 | 等价实验、部署文档、演示与最终验收 | M0–M8 | 已验收：统一 P/D 与常驻独立 P→KV handoff→D 各五重复；旧单/双完整副本实验仅计副本成本 |

M1 优先暴露模型适配风险，不等分布式框架建完才发现语言模型无法承接视觉输入。每个阶段以验收证据完成；代码存在、fake backend 通过、模型能输出一句话都不能独立证明完成。

## 3. M0：核验当前基线

修改/检查入口：`infMain/source/serving/request_checkpoint.cpp`、`infMain/source/serving/scheduler.cpp`、`infMain/include/model/qwen2.h`、`test/test_request_checkpoint.cpp`、`test/cache_core/CMakeLists.txt`、`tools/bench/data_flow/`。

实施：

1. 记录 HEAD、全部 tracked/untracked 源码 hash、dirty diff、工具链、CUDA、GPU 拓扑、模型 revision；保护原有 `demo/main_qwen2Instruct.cpp` 和任何新出现的未提交改动。仅安装缺失依赖，Python 模型依赖使用独立 venv 和 lock，避免改坏现有 omni-flow 环境。
2. 当前已见 checkpoint byte accounting、旧 revision 拒绝、cancel 清理和固定 KV block 配置。先读现有测试与证据，再执行乱序 commit、prepare/cancel/commit、完成请求回收、10,000 次 churn；失败才修复。不要按旧 V3 状态重复实现。
3. 重跑适用 CPU、sanitizer、CUDA 和纯文本模型小回归，记录所有 skip 的具体缺失条件。历史 53/53、157 passed 等不作为当前验收。
4. 保存同模型、同到达轨迹、同显存上限的统一执行与旧 PD oracle。把旧 hex JSON KV 通道标为 compatibility/oracle，不作为新主线性能代表。

验收产物：`docs/data_flow_evidence/v4/M0/manifest.json`、`checks.json`、原始命令日志、失败修复前后证据。旧缺陷已修复且通过时记录“复验通过”，不要求故意回退当前代码制造失败。

## 4. M1：模型兼容性探针

新增建议：`tools/models/export_qwen25_vl.py`、`tools/models/qwen25_vl_reference.py`、`test/test_model/test_qwen25_vl_input.cpp`、`docs/data_flow_evidence/v4/M1/compatibility.md`。改动入口：`tools/export_qwen2.py`、模型配置/权重 loader、`model/qwen2_batch.cpp`、RoPE kernel dispatch。

实施：

1. 检查本地是否有目标权重，缺失时下载官方候选和 processor，冻结 commit、文件 hash、依赖和许可证记录；不要把旧 Qwen2-0.5B 模型作为视觉语言模型替身。
2. 参考实现拆出 processor → visual+merger → inputs_embeds/positions → language model。只做数值探针，先不引入跨进程协议复杂度。
3. 核对词表、embedding/lm_head tied 权重、QKV bias、RMSNorm eps、RoPE 配置、attention mask、权重 key 映射和 BF16。导出格式显式版本化，保留现有文本格式 reader。
4. 冻结 `MultimodalSequencePlan` 和 bundle schema：展开 token、media spans、feature refs、grid、三轴位置、decode position state。token offset 与 mRoPE 值分离。
5. 为 C++ 最小单请求 forward 增加受控 embedding 替换和 mRoPE；暂不调整 batching 调度。比对视觉特征、位置、选定层 KV/hidden、logits 与 32 步 greedy decode。
6. 至少覆盖纯文本、一图、两图、不同分辨率、batch 内不同视觉长度。先在参考路径重复运行确认误差范围，冻结阈值与 fixture，再评估 PBE。记录 top-k、max/relative error 及 token 分歧，不能只目测答案。

验收：真实 checkpoint 在 PBE 语言核心完成图文生成、关键中间状态符合冻结阈值。若模型确实不兼容，输出精确差异与最小修复范围，保持 M1 未完成；替换候选需留下 ADR 并保持真实 PBE 多模态目标，不能降级为调用外部完整模型。

## 5. M2：统一数据合同与本地实现

新增模块位置建议：`infMain/include/data/`、`infMain/source/data/`，包含 `data_ref`、`tensor_bundle`、`data_runtime`、`budget`、`protocol`；新增 `test/data_core/`。现有 `cache/` 继续承担 KV 特化，不复制同名目录/迁移器。

实施：

1. 定义 ContentId/RepresentationId/AllocationHandle、DataKind、OperationId、OwnerIncarnation 和 typed error。sequence 内容包含媒体身份、插入位置及位置配置，禁止全局目录继续仅靠 token ids。
2. 以 `LocalDataRuntime` 包装现有 PageDirectory、HostStore、TransferScheduler；实现 reserve/seal/acquire/ensure_local/release。整 bundle/page preflight 在首笔写入前完成。
3. 拆分 `BlockAllocator` 的 physical pool ownership 与 `KVPoolView`。现有本地模式继续使用旧 allocator owner；外部模式只借用 pool/slot，不另建独立 free queue。attention/scatter 改接 pool view，保持布局与现有内核行为。
4. 权威账本在 runtime owner，KVCacheManager 维护请求页表和 prefix 引用。批量 lease/reservation 由 owner loop 非阻塞提交，完成队列在空闲时仍推进。
5. 定义 Write/Read/Compute/IO 引用、producer fence、consumer fence、seal 的 canonical 返回值、unknown quarantine 和有界元数据清理。
6. 定义 wire schema/version，控制面只传描述符；跨语言协议使用明确整数宽度、枚举、shape/bytes/offset 校验和错误码，不传 C++ ABI、pickle 或裸指针。

验收：原纯文本路径回归；资源守恒、stale generation、重复 seal/release、错误 coverage、两 producer canonical 竞争通过。至少一个原 serving 调用链使用 LocalDataRuntime，不能只是孤立接口测试。

## 6. M3：跨进程共享成为主线

新增：`demo/pbe_data_service.cpp`、`data/data_client.*`、`data/node_agent.*`、`data/content_registry.*`、`data/ipc_pool.*`、`tools/launch/multi_role_text.*`。改动：`serving_zmq_rpc.*`、`pd_handoff.*`、`pd_engine.*`、`kv_cache_manager.*`。

实施：

1. 单节点 Registry+Agent 服务，固定 endpoints、incarnation、协议版本与有界 metadata；进程内 local 模式使用同一合同。
2. 两步建立 transport：先 bounded binary Host copy oracle，再验证 Agent-owned CUDA IPC 池的导出/导入、pool view、生产/消费事件。跨设备采用可测 P2P/copy，不能假设同 GPU IPC 自动消除跨 GPU 传输。
3. 控制面支持幂等 reserve/seal/acquire/release/withdraw，服务端执行阻塞 GPU/RPC 工作的线程与 metadata owner 分开。逐请求/批次交互，禁止每生成一个 token 做目录 RPC。
4. Prefill 向 Agent 请求 slots，产生完整页后 seal；Decode 通过 ContentId 获取并安装本地页表。旧 handoff manifest 可作为 adapter 输入，但不再决定对象寿命。
5. 同时启动独立 Prefill、Decode A、Decode B。A/B 消费同一已发布前缀，生产请求结束后 B 再获取；缓存保留 lease 使对象独立存活。第一版可保留完整页边界，partial tail 在 M6 完整化。
6. 读写权限由协议和 worker 执行合同约束，不把共享 CUDA 映射当作硬件只读保护。已有 lease 终结及 fence 完成前不得释放 slot；RPC 超时不代表取消 GPU 工作。

验收：真实 Qwen 文本模型输出与 oracle 对照；独立 PID、pool/export/import 日志、传输字节和 prefill token 数证明实际复用。生产 worker 正常退出后 service 保留数据，后续消费者仍可获取。记录同 GPU/跨 GPU 两种路径；不支持 IPC 时阶段保持其子项未通过，不把 Host 路径标成零拷贝。

## 7. M4：视觉特征作为框架数据

新增：`python/pbe_roles/vision/`、`python/pbe_data_client/`、`data/tensor_bundle_pool.*`、`data/media_object.*`、跨语言 golden message 测试。借鉴 omni-flow encoder hook 与 bundle 模式，保留来源/license，避免整体复制框架依赖。

实施：

1. 公共 Vision role 管有界输入队列、batching、device/workspace 和完成发布；模型 hook 管 prepare/collate/forward/split。只加载目标模型视觉塔与 merger。
2. 媒体物化后生成内容摘要；cache key 覆盖 processor/encoder revision 与有效预处理选项。路径内容变化、相同图片不同 resize 必须可区分。
3. Agent 创建对齐 bundle allocation；Python client 初版通过 binary copy 写入，之后使用经过 M3 验证的导入视图。IPC tensor 的 owner capsule/lease 活到最后 stream 完成，Python GC 不决定服务端物理回收。
4. 输出包括特征与 grid/长度等 schema 所需组件，原子 seal。LRU 保留引用与消费者活跃引用分开；cold miss 合流按编码任务处理，有取消和超时边界。
5. 限制图片解码大小、视觉 token、bundle bytes、batch workspace；输出容量不足时在 forward 前背压。

验收：真实 encoder 数值符合 M1；两个独立消费进程获取同 bundle；一个取消另一个仍正确。重复图片跳过真实 forward，换 processor 参数不误命中。所有组件有 byte 校验与 checksum；同 key 并发、变长混合 batch、失配 manifest 拒绝。

## 8. M5：完整视觉语言执行路径

改动入口：`serving/serving_online_engine.*`、`serving/mixed_batch.h`、`mixed_batch_builder.*`、`sequence_state.h`、`model/qwen2_batch.cpp`、模型 adapter；新增 `serving/role_runtime.*`、`serving/request_coordinator.*`、`model/multimodal_input.*`。

实施：

1. 在线 API typed text/image parts 转为媒体句柄与 sequence plan，保留旧 prompt 请求。明确首版支持的格式、顺序、尺寸、错误返回，不声称完整多模态 OpenAI API 兼容。
2. Coordinator 驱动 Encode → Join → Prefill → Decode，不把图像解码或 data wait 放入语言模型 owner 的阻塞路径；每个 stage 有 generation、deadline、失败与取消传播。
3. 将 M1 的最小输入适配接入 mixed batch、chunked prefill 与 paged KV。校验视觉 span 与特征长度；chunk 从 span 中间开始、prefix 命中跨过图像时仍正确。
4. 保存每请求 mRoPE position state，PD 传递和 checkpoint 可重建它；Decode 只处理新 token，不重跑视觉塔。
5. 纯文本和图文请求混合批处理；pool view 到 kernel 地址只在本进程绑定，外来 wire descriptor 不直接驱动任意写入。

验收：至少 20 个固定图文 fixture（纯文本、一/两图、多分辨率、长文本穿越视觉 span），分别单请求、混合 batch、分块、PD 下比较参考中间状态和输出。真实 Encoder PID 与 PBE PID 分离，两个进程实际经数据服务获取特征；全过程 trace 可追踪。

## 9. M6：复用与分支

实施入口：Coordinator、Data Registry、KVCacheManager、prefix index、branch snapshot、采样与 output outbox。

1. 同图不同问题复用视觉特征；只在完整多模态前缀相同时复用 KV。对同 token placeholder/不同图片建立必不命中回归。
2. Prefix Builder 是显式 Prefill task，发布受预算管理的前缀对象；无活动原请求仍可被后来的请求发现。构建任务和 online cold compute 的去重策略分开。
3. 支持 BranchSnapshot：完整页共享、有效尾页不可变保存、每 branch append 前 COW。记录 computed/sampled/pending token 边界，明确分支首 token 是共同继承还是独立采样；独立采样需可用 logits/等价重算步骤。
4. 每个 branch RNG、请求 generation、输出 outbox 独立；分支取消只释放自身引用。同 GPU 能共享物理页，跨 GPU 副本分别计费。

验收：一个 Prefill 两/四分支；包含页对齐和未满尾页，比较分别独立运行的语义结果。中途取消一半分支，其他分支 KV 不变；producer 与分支按不同顺序退出后资源回稳。实测 prefill tokens、physical pages、COW bytes，而不是只数 cache hit。

## 10. M7：预算与真实压力

继承 V3 DF2–DF5 的详细合同，实施时按新 owner 边界接入。已有功能复验后复用，不另起平行 scheduler。

1. 建立设备级 weights/workspaces/KV/bundle/staging/quarantine 总预算，各池独立守恒。所有角色加载完成后固定池容量；已有进程不可因新角色启动被动 OOM。
2. 每请求共享/COW/private/restore 计账，首版保守 footprint 或明确非阻塞增长。复现预算 4、两请求各持 2 再各增至 3：必须明确拒绝/抢占/前进，不永久互等。
3. Encoder 输出、Prefill 输入保留与 KV 分配之间有有界队列和释放规则。不同池联合预留失败应全量回滚本次新增资源；不许 hold-and-wait。
4. 方向 lane、demand 入口、绝对 deadline、可撤回 priority donation、dependency donation、aging、restore waves。staging 按 wave 复用，持久 target 不随 wave 归还。
5. 实际容量不足触发 `recompute|checkpoint|auto`；异步 save 不 drain 无关全局传输。checkpoint 保存多模态依赖、position state、RNG 与输出状态，数据可重算性和精确恢复要求明确。
6. 带滞回的水位维护；restore/recompute 成本用实测校准。后台 eviction 不回收最后必要且不可恢复的对象；auto 必须报告选择原因。

验收：Encoder 积压、Decode 持续增长、bundle 与 KV 同时压力、后台传输满载四类负载。记录 demand wait P99、ITL P99、池守恒和进展；不可准入工作集返回明确拒绝。至少 10,000 次 acquire/cancel/restore 生命周期无持续增长。

## 11. M8：故障与生命周期矩阵

| 注入点 | 必须成立 |
| --- | --- |
| reserve 后生产者取消/退出 | 未 seal 对象不可见；未证明安全的写入不能提前复用 |
| seal/release 重复、旧 incarnation 回包 | 幂等或拒绝，不影响新 allocation |
| 两消费者之一超时 | 另一个仍能完成；物理 copy 由独立 flight 回收 |
| copy 完成前 worker 异常退出 | 隔离或经可靠 quiescence 回收，不用 TTL 假装 fence |
| provider 撤销与 acquire 并发 | 要么拿到有效 lease，要么 retry/stale；无悬空地址 |
| Data service 重启 | 旧句柄全部失效；请求明确失败或重新构建，不透明复活 |
| partial bundle/错误 shape/bytes/coverage | 首笔 copy/compute 前拒绝 |
| 最后一份不可恢复数据受淘汰压力 | 保留/拒绝新工作/明确错误，不能静默重算 |
| decode 输出后 checkpoint/重试 | 进程内 outbox 连续 item 序号，无重复入队；不宣称网络 exactly-once |

运行 CPU 状态机、跨进程故障注入、真实 gated CUDA copy、针对性 compute-sanitizer。正常 shutdown 顺序为停新准入 → 停生产 → drain/隔离 → 释放 importer → 释放 exporter/pool。硬故障允许 fail-stop；没有可靠停止证明时，不要求冒险在线回收来满足“零泄漏”。

验收证据区分正常 churn 回稳与故障 quarantine；后者必须有隔离上限和服务重启回收流程，不能以 quarantine 无限增长通过稳定性测试。

## 12. M9：实验与交付

实验使用冻结模型、processor、媒体内容、到达轨迹、随机 seed、显存与 stream/active 总预算。统一执行和分离执行计入全部权重、workspace、CPU/RPC 成本；顺序随机化，至少 5 次重复，报告中位数及波动，数据量不足不外推。

| 实验 | 控制变量与结果 |
| --- | --- |
| 特征缓存 off/on | 相同图片轨迹，分别报告命中比例、encoder GPU 时间、TTFT、缓存 bytes |
| KV 共享 off/on | 相同多模态前缀/分支，记录 prefill tokens、COW bytes、实际页数 |
| 合并部署/角色分离 | 相同总设备预算与工作量；重复权重成本单列 |
| Host copy/IPC/跨卡 copy | 相同 payload、checksum、同步语义；真实跨进程，不用 memcpy 微基准代替 |
| singleflight off/on | 相同 key 轨迹，off 使用私有目标再 canonical 收敛，禁止并发写同一目标 |
| lane off/on | 相同总 active/stream/bytes；额外资源不是调度收益 |
| pressure policies | 保存+恢复+重算的总成本、ITL 尾延迟、拒绝率、不可恢复内容处理 |

负载矩阵至少含图片重复率 0/50/90%、分支 1/2/4、短/长前缀、两档分辨率、低/中/高压力。先选代表组合和单因素消融，不盲跑所有笛卡尔积；至少保留低复用的负收益场景。

输出 `results.json`、原始 trace、绘图脚本、资源时间线和 standalone 图。记录 logical hits 与 actual saved compute 的差别、TTFT/ITL P50/P95/P99、吞吐、传输 bytes、每池峰值、OOM/retry/拒绝和恢复次数。

最终交付：一个启动命令、纯文本回退配置、模型准备说明、协议/故障文档、架构图、五分钟演示脚本和面试问答。所有性能表注明硬件/模型/revision，未测项留空；不宣称超越 SGLang，也不拿不同模型的吞吐直接比较。

## 13. 与旧计划的映射与明确后置项

| 旧 V3 | 新去向 |
| --- | --- |
| DF0 checkpoint 修复与基线 | M0 复核已有改动；M7/M8 扩展多模态状态 |
| DF1 identity/plan/preflight | M2/M3 必需部分；通用 head 重排与 placement 优化后置 |
| DF2 共享/COW/容量 | M2/M6/M7 |
| DF3 lane/deadline/waves | M3 物理生命周期 + M7 完整调度 |
| DF4 真实压力恢复/输出 | M7/M8 |
| DF5 成本/水位 | M7；仅测量支持的策略 |
| DF6 公平实验 | M9 |
| X1 跨进程 provider | M3–M6 核心，必须完成 |
| X2 layer-ready | 后置，先完整 page/bundle READY |

后续候选分别建立新验收清单：跨机多 Agent/RDMA、跨 provider slice 聚合、异构 TP representation 转换、视频/音频、DiT、Draft/Verify。同卡只读权重共享已按第 19 节 E4 完成验收；不得由此推断上述候选范围已实现。

## 14. 后续执行入口与阶段摘要格式

后续执行者先读架构与本计划，核对工作区最新改动；当前接手 E4 时直接按第 19 节执行，先冻结现有基线并复验相关门禁，不重新实施已验收的 M0–M9/E1–E3。仅安装缺失依赖。保留现有 C++/CUDA 与 PD 能力，通过 adapter 逐步迁移，不能先删除旧实现。M1 先证明真实模型接口，M3 先证明真实进程边界，M5 必须使用 PBE 语言核心。每阶段有证据才更新状态，遇到失败继续诊断可独立推进的部分，禁止用 fake 测试替换必需 GPU/模型验收。

每阶段摘要包含：当前 checkout/hash、改动文件、已通过与失败/skip、证据目录、仍存在的风险、下一阶段入口。进度报告为“已验收阶段数/10”和阶段内部 checklist；不要按代码行数、接口数或主观工作量给完成百分比。

- [x] M0 当前基线复验
- [x] M1 真 VLM 数值探针
- [x] M2 数据合同与本地 runtime
- [x] M3 跨进程 KV 多消费者
- [x] M4 真实 Vision 与 bundle 共享
- [x] M5 PBE 真实 EPD 闭环（常驻 typed 多请求 mixed Prefill/Decode 与取消）
- [x] M6 前缀/特征复用与分支 COW
- [x] M7 预算、调度和压力恢复（实际资源联合准入与压力恢复）
- [x] M8 故障与长期稳定性（线上故障矩阵与 57/57 定向回归）
- [x] M9 公平实验与演示交付

## 15. 扩展阶段与执行顺序

新增扩展不替代 M5/M7/M8/M9 的整改，也不把原未完成内容包装成新增成果。保留旧实现和证据，执行时先按最新源码确认已有能力；已存在的 prefix key、目录、HostStore、lease、scheduler 必须复用或演进，不再建一套平行账本。

| 阶段 | 必做交付 | 依赖 | 验收状态 |
| --- | --- | --- | --- |
| E1 | 计算前语义前缀查询、最长前缀命中和缺页获取 | M2/M3 所有权；M5 输入数值合同；M7 真实 target/staging 准入 | 已验收 |
| E2 | 按队列、数据位置和计算成本选择角色 | E1 目录与缺页估算；M5 异步在线入口/deadline；M7 实际设备预算 | 已验收 |
| E3 | 按恢复依赖实施分层保留、淘汰与恢复 | E1 页面/内容身份；M7 多模态 checkpoint；M8 物理完成与故障合同 | 已验收 |
| E4 | 同 GPU 常驻 P/D 共享不可变物理权重，统一计费与安全生命周期 | M3 IPC/代际；M7 设备预算；M8 故障合同；真实常驻 PD 对照 | 已验收：真实共享 attention、单次物理上传、统一计费与安全生命周期通过 |

建议顺序：原验收整改 → E1 → E3 的必要依赖保留/回收 → E2 → E3 的成本策略与联合实验。E2 的统计、打分实现可在 E3 完整策略之前进行，但不能越过容量及生命周期门禁。每个阶段均包含真实集成、故障测试和公平实验，不单独用接口或单测计完成。

## 16. E1：语义前缀查询与页级按需获取

目标：新请求不依赖上次请求输出的 payload key，在计算前即可找到最长可复用前缀；只传输当前消费者缺少的页面，未命中部分继续正常 prefill。

### 实施入口与合同

优先检查 `data/multimodal_prefix_key.*`、`data/content_registry.*`、`data/data_client.*`、`base/kv_cache_manager.*`、prefix/radix index、`serving/prefix_builder.*`、`python/pbe_roles/coordinator.py` 和实际 language serving 主路径。新增建议接口：`query_prefix(SemanticSequence, destination) -> PrefixMatch`、`ensure_pages(PrefixMatch, deadline) -> RestoreTicket`。

1. 冻结 `SemanticSequence` 编码：模型权重/配置与 adapter revision、完整有序文本 tokens、媒体内容与 processor/encoder revision、媒体插入区间、position/RoPE 语义及版本。输出采样参数若不影响已算 KV，不作为该段 KV 身份；生成前缀以实际生成 tokens 纳入链。不能只比较 placeholder tokens 或当前页 tokens。
2. 页面身份由前缀链和有效语义区间确定，不能依赖 GPU 地址、request id 或算完的 KV 字节才能得到。相同语义的不同 dtype/layout 使用独立 RepresentationId；不兼容模型/量化表示拒绝复用。独立计算的数值差异按既有 canonical 合同处理，不能暗改成任意近似复用。
3. `PrefixMatch` 返回最长连续可复用 token 数、页列表、representation、可服务副本与版本；目录查询只是候选，acquire/attach 时必须复核 generation、coverage 和 provider 可服务性。出现中间缺页时不能把后面的孤立页算作完整连续前缀；重新查询或从安全连续边界补算。
4. 将命中集合拆为本地 resident、需迁移、已加入 flight、未命中和私有尾页。先确保持久 target 容量，再按 wave 借 staging；同 GPU 可直接 acquire 共享页，跨 GPU 只复制请求所需缺页，禁止为了少量命中页复制整个 owner pool。
5. 对同一目标内容/表示合并传输，source 不进入 singleflight 身份；一个 waiter 取消不终止其他 waiter。源撤销、旧 generation、copy 未知完成分别处理，只有安全目标可以安装到 attention 页表。
6. 完整页进入普通 prefix index；未满尾页按专用 snapshot 与 COW 合同处理。命中全部 prompt 时仍需正确处理最后 logits/首 token：保留合法的生成启动状态或明确重算必要尾步，并计入 saved-prefill 统计，不能直接复制旧请求采样结果。
7. serving 在 prefill 前实际调用查询，按 matched_tokens 设置计算边界；后续请求新增的 token 和视觉区间使用当前 sequence plan，不能用缓存里的旧 request bundle 替代。

### 验收与证据

- 真实模型同图同前缀、同图不同问题、同 placeholder 不同图片、不同 processor/position、不同历史但同当前页分别验证应命中/不命中；对照无缓存 oracle，并采用已通过的数值合同。
- 多请求首次生产后退出，再从全新请求计算输入身份并查找；日志证明未读取生产者日志中的 key。覆盖页对齐、未对齐、部分本地命中、provider 撤销和多消费者取消。
- 跨卡请求所需 N 个缺页时实际 transfer bytes 对应 N 页及明确协议开销；不能按 pool 容量计一次全池 copy 后宣称按需拉取。传输前后做 byte/coverage 校验，复用结果进入真实 attention。
- off/on 使用同请求轨迹、模型、页预算与相同采样；测 matched tokens、actual computed tokens、saved prefill、目标页数、copy bytes、TTFT/ITL、查询开销。低复用负收益保留，独立重复至少 5 次。
- 产物：`docs/data_flow_evidence/v4/E1/` 中的 key golden cases、匹配/搬运 trace、故障日志、`checks.json`、`results.json` 和执行报告。

## 17. E2：依据数据位置选择计算角色

目标：Coordinator 根据排队、可用副本、预计搬运和计算成本选择 Prefill/Decode worker，证明数据位置对端到端延迟的影响；不是单纯选择命中率最高的节点。

### 实施入口与合同

复用 Coordinator、role endpoint/能力注册、ContentRegistry、DataClient 与 NodeMemoryBudget。新增建议 `RoleCostSnapshot`、`PlacementDecision`，决定只保存 worker/内容/版本等逻辑信息，不预先持有裸地址。

1. worker 周期报告 incarnation、模型/表示能力、队列中 token 工作量、实际可准入容量和观测时间。元数据有时效；未知统计使用保守默认，不把未知队列当作零等待。
2. 首版使用可解释估计：`queue_wait_ms + missing_bytes / measured_bandwidth + compute_ms`，加实际测量支持的 RPC/布局转换成本。带宽、prefill 和 decode 成本用不同长度/批量实测校准，保存估计与实际值；不宣称全局最优。
3. Prefill 选择计入 feature/KV 命中与后续 P→D 搬运；Decode 选择计入所需前缀、私有增长和输出长度预算。第一版选择有界候选角色对或先选 D 再选 P，必须记录策略，避免局部最优导致两次不必要搬运。
4. 打分只针对模型/布局兼容且可能准入的候选；最终 reserve 失败则有限次数重新选择，沿用原绝对 deadline。使用幂等 request/generation 防止重复启动计算，资源回滚受同一生命周期合同约束。
5. 增加热点约束、有限 in-flight placement 和滞回，避免所有请求同时追逐同一热门副本。默认只放置尚未运行的 stage；运行中请求迁移仍走 checkpoint，不能以改 endpoint 代替状态迁移。
6. 每次决定输出候选成本分解、统计年龄、选择/拒绝原因、实际等待/搬运/计算时间。不把一次 query 命中视为已 pin 的服务保证。

### 验收与证据

- 至少两个真实兼容 worker，构造“缓存命中但队列长”“无缓存但空闲”“统计过期”“预算竞争”“worker 重启”场景；证明能在有依据时选择远离热点的角色。
- 比较固定分配、round-robin 和数据位置策略，保持相同 endpoint 集合、模型副本数、资源总量、请求轨迹和预算；不能通过多启动 GPU/模型副本取得收益。
- 至少 5 次随机顺序重复，报告客户端 TTFT、token 级 ITL、吞吐、拒绝率、copy bytes、队列等待、决策开销与预测误差；保留错误预测和负收益。
- 产物：`docs/data_flow_evidence/v4/E2/` 的 calibration、placement trace、故障记录和公平对照结果。模拟器只能帮助诊断，不能代替实际多 worker 验收。

## 18. E3：按恢复依赖管理分层缓存

目标：视觉特征、共享前缀、私有尾页采用不同保留策略；压力下根据恢复能力决定下沉、重算或拒绝，避免淘汰最后一份必要数据。本阶段扩展 E1 的数据对象，不替代 M7 原有多模态 checkpoint 门禁。

### 实施入口与合同

复用 `cache/host_store.*`、`cache/page_directory.*`、`cache/page_migration.*`、`data/data_runtime.*`、`data/content_registry.*`、NodeMemoryBudget、RequestCheckpointStore 和 Vision feature cache。主线先实现 GPU/Host 两层，SSD 与跨机持久化继续后置。

1. 为对象定义恢复描述：`Recomputable(recipe, dependencies, versions)` 或 `PreserveUntilReleased`，以及恢复要求 `ExactContinuation`/已定义的数值容差。依赖包括可用的媒体字节、processor/encoder/模型版本、序列位置、采样状态；URL 或“模型还在”不构成充分恢复保证。
2. 依赖图必须无环，metadata/依赖引用有界；缓存保留引用、活动消费者和恢复配方引用分开。发布配方时验证依赖，淘汰/撤销时再次验证；缺依赖、旧版本或不可达 provider 返回明确不可恢复状态。
3. 视觉特征按复用与重算成本保留；共享完整前缀按扇出/重算成本保留；私有尾页仅在计算 fence 后快照。正在写入或 in-flight 的对象不能直接 LRU 回收；副本同样区分“有字节”和“可服务”。
4. GPU→Host 下沉先取得 Host/staging 容量，copy/fence、校验、提交 Host 副本，再撤销 GPU 可服务状态，最后按引用回收。`pending_free` 不是当前可分配 free；Host 满时明确拒绝下沉或选择可重算对象。
5. exact continuation 所需特征必须保留精确副本或经过证明等价的恢复路径；重新编码可能产生数值差异，不自动视为无损重算。最后一份必要副本不可淘汰；容量不足时背压/拒绝新任务，不能悄悄丢数据。
6. restore 以 M7 checkpoint 为入口，先恢复/验证依赖，再安装 KV、position state、RNG、outbox；取消只能释放本请求持有的引用。恢复失败时不得部分运行，未知 DMA 仍隔离到确认安全。
7. 增加 trigger/target 水位滞回、最低驻留时间与后台配额，避免 feature 与 KV 反复驱逐。成本统计覆盖保存、加载、必要重新编码/重算和依赖保留内存，不能只统计恢复拷贝。

### 验收与证据

- 真实同图多问/多分支压力负载：分别淘汰 GPU feature、完整 KV 前缀和私有尾页，再恢复并继续生成；比较冻结 oracle、输出序号和所有池占用。
- 覆盖原图不可用、模型/processor revision 变化、最后 Host 副本、Host 满、provider 撤销、下沉中取消、恢复中服务重启；确认不可恢复对象被保留或明确失败，无悬空依赖。
- 至少 10,000 次对象/依赖 acquire-release 生命周期以及真实 GPU 压力序列；报告元数据与数据池回稳、quarantine 上限和最终回收，不允许以永久 tombstone 换安全。
- 与普通 LRU/固定保留策略在同预算下对照，至少 5 次重复；报告真实保存/恢复/重算成本、TTFT、token 级 ITL、重算 token/encoder forward、拒绝率与副本 bytes。完整压力场景不能由低容量单元测试代替。
- 产物：`docs/data_flow_evidence/v4/E3/` 中的依赖图快照、恢复配方版本、压力/故障 trace、资源时间线和策略消融结果。

### 扩展交付清单

- [x] E1 计算前语义查询、最长连续前缀、缺页迁移进入真实 serving，并完成数值/故障/性能验收。
- [x] E2 实际多 worker 数据位置选路、准入回退及公平对照完成。
- [x] E3 恢复依赖、GPU/Host 分层保留、多模态精确恢复与压力实验完成。

报告格式：历史结果为原 M0–M9 **10/10**；扩展 E1–E4 为 **4/4**，其中 E4 独立验收 **1/1**。这些是阶段验收比例，不代表工时比例，也不包含明确后置范围。

## 19. E4：同卡跨角色只读权重共享（已验收）

### 19.1 目标、边界与接手入口

目标：两个独立常驻 PBE language 进程执行同一请求的 Prefill→KV handoff→Decode，同时映射框架持有的同一份 GPU 权重；降低角色拆分的重复显存成本，并验证节省的容量能否用于 KV/请求准入。语言计算继续使用 PBE C++/CUDA 的真实 attention 主路径。

范围限定同机、同一物理 GPU、可信进程、TP=1、相同冻结 Qwen2.5-VL-3B BF16 language 权重和布局。不共享 Vision 权重、workspace、激活、RNG、可变量化状态或请求私有 KV；embedding/lm_head 若原模型 tied，保留别名关系，不能凭 tensor 名不同重复分配。跨 GPU 映射、任意 TP 转换、热更新权重、训练/原地修改不在本阶段。CUDA IPC 不提供硬件只读隔离，本阶段保证的是可信 worker 的只读使用合同。

优先阅读：

- `V4_PERSISTENT_PD_CLOSURE_20260913.md` 及 `docs/data_flow_evidence/v4/M9/persistent_pd_deployment_final_gpu0_v4/`：必须延续真实 P/D 分进程交接，不能退回双完整副本各跑整条 infer。
- `infMain/source/model/qwen2.cpp`、`model/raw_model_data.cpp`、`op/layer.cpp`、`tensor/tensor.cpp`、`base/buffer.cpp` 及对应头文件：核对 CPU 权重解析、dtype 转换、GPU 分配、tensor 绑定和析构所有权。
- `data/ipc_pool.*`、`data/protocol.*`、`data/data_client.*`、`data/node_agent.*`、`serving/node_memory_budget.*`、`demo/pbe_data_service.cpp`、`demo/pbe_vlm_language_role.cpp`：复用现有 IPC、完整 LeaseToken、operation-id 和设备计费合同。
- `tools/bench/data_flow/run_persistent_pd_deployment_ab.py`、`validate_persistent_pd_deployment.py`、`validate_review_gap_closure.py`：在既有真实实验与门禁之上增加权重共享维度。
- 本地参考 `../omni_flow_sglang/omni_flow/compute_flow/utils/shared_weight_v2.py` 与 `../omni_flow_sglang/omni_flow/data_flow/memory_manager.py`。记录参考 checkout/hash，区分上游设计与本地改动；借鉴 owner/follower、加载发布、身份和峰值控制，不照搬 Python monkey patch 或将“内容发生变化”作为唯一权重判定依据。直接移植代码时保留适用许可证与来源。

新模块可采用 `data/shared_weight_registry.*`、`model/shared_weight_binding.*`，名称由实现者按现有结构调整；不要另建与 Data service 无关的权威账本。

### 19.2 实施顺序与不可变条件

**E4-A：基线与显式权重清单。** 冻结 dirty workspace/source hash、模型及配置 hash、构建选项、GPU UUID、CUDA 版本和当前 PD 门禁。导出显式 manifest，逐 tensor 记录稳定名称、shape、dtype、stride、对齐、offset/bytes、别名、源 checkpoint 区间及推理布局版本。必须覆盖 embedding、attention/MLP、norm、bias、输出头；只共享已确认的不可变参数。校验溢出、越界、非法重叠，合法 tied alias 单独标记。默认保留 process-owned 路径作为 oracle 和回退模式。

**E4-B：框架拥有物理权重，计算进程借用。** 首选扩展现有 Data service/Agent 作为 exporter，以模型内容身份+布局版本+dtype+TP 拓扑+GPU UUID 索引共享权重集合；物理句柄还绑定 service incarnation/allocation generation。权重使用专门 allocation，不与可回收 KV 混放。由框架分配兼容 CUDA IPC 的内存，loader 按有界 staging 分块上传，最后校验并 seal；消费者只获取 manifest、描述符和 lease，不接收裸指针或每 tensor JSON payload。相同模型文件路径不等于相同内容，GPU ordinal 也不等于稳定设备身份。

状态至少区分 loading、ready、failed、draining；完成全部写入、布局转换、checksum 和 producer fence 后才能 ready。并发首次加载合并为一次实际上传，支持超时和失败重试；失败半成品不可见，迟到 seal 不得发布新代对象。不能阻塞 metadata owner 等待 follower，否则多角色启动可能死锁。GPU allocation 可按 slab/分段实现，但必须列明分配次数、真实 bytes 与 IPC 生命周期。

**E4-C：loader/算子真实绑定。** shared 模式在初始化时直接把算子权重绑定为借用的 GPU tensor view，capsule 持有 import/lease 生命周期；析构关闭 import 而不 cudaFree 外部 allocation。避免“先给每个 worker 加载完整私有 GPU 权重，再替换指针”的稳态省显存假实现。允许 CPU 文件映射和有界暂存，但其峰值需要实测。审查 `to_cuda`、dtype cast、contiguous/repack 等路径，防止暗中生成完整私有副本；必须转换的布局由 owner 在发布前统一生成并计费。

模型结构、dtype、布局、设备或版本不匹配必须在首个 compute 前报 typed error。显式 `shared` 模式失败不得静默退化为私有加载；若提供 auto 模式，必须报告 fallback 原因和实际模式，不能进入 shared 验收样本。保持现有非共享 BF16/plain/FP8 接口行为；不强求把 FP8 也加入共享支持范围。

**E4-D：生命周期与故障。** 复用完整 token 与幂等 operation-id；每个独立 importer 有可追踪 lease。正常停机先停准入、完成/停止 compute、同步必要 fence、销毁模型 view、关闭 import、释放 lease，最后 owner 才能释放物理权重。一个计算角色正常退出不应销毁其他角色仍使用的权重，新角色可重新获取 ready 对象。

consumer 崩溃不以 RPC 超时/TTL 当作 GPU quiescence；无法证明停止时隔离到可靠清理或有序重启。Data service/exporter 崩溃允许 fail-stop：旧映射不得继续调度新 compute，关联 worker 明确失败并销毁旧 CUDA 上下文/进程，再启动新代；不承诺 CUDA IPC 内存在 exporter 消失后仍可安全使用。测试必须保存异常返回/退出及隔离记录，不能把“未崩溃”视为正确。正常退出要求回稳，硬故障采用有界隔离与明确回收流程。

**E4-E：统一物理计费与在线准入。** 同 GPU 共享权重按物理 allocation 在节点账本中只收费一次；每个 worker 可报告相同 logical/imported bytes，但不得把它们累加为物理显存。分别报告 physical weight bytes、imported logical bytes、私有 workspace/激活、KV、Vision、staging、CUDA context/其他开销，以及加载峰值、实测总设备占用和账本差额。禁止仅以两个 worker 各自 `cudaMemGetInfo` 的差值认定共享容量。

所有常驻角色权重加载/导入及实际 workspace 初始化完成后，再确定可用 KV 容量；避免先启动角色吞掉后启动角色的显存。共享不是权重复制带宽优化，跨进程 IPC 映射也不代表进程 GPU 调度无开销。只有物理 free 和账本一致时，节省容量才能通过已有联合准入成为新 KV slots，不能只修改报告数字。

### 19.3 必须通过的正确性与故障门禁

| 场景 | 必须提供的证据 |
| --- | --- |
| 同模型两个独立进程 | 不同 PID、同 GPU UUID、相同 owner generation/allocation 身份、各自 import/lease；owner 仅一次物理分配/上传，真实 attention 读取这些 view |
| 权重/布局身份 | 不同文件内容但同路径、dtype/shape/layout 不兼容、旧代、错误 offset/bytes 均在 compute 前拒绝；tied alias 不重复计费 |
| 并发启动/加载 | 多 follower 同时启动，仅一次上传；loading 不可计算；上传中失败、reply-lost 重试及迟到 ready 不泄漏或误发布 |
| 真实 P/D | 同一 request/generation 的不同 P/D PID，KV handoff 有效、Decode 不重算 prompt、输出符合冻结数值规则；不能用两个完整 infer 代替 |
| 多消费者/生命周期 | P/D 并发计算与至少两个独立 importer；一个正常退出另一个继续正确计算，新 importer 可加入；重复/迟到 release 不损坏新代 |
| 故障恢复 | consumer 强杀、owner 重启、加载中取消、映射失败、容量不足；按合同 fail-stop/隔离，无旧句柄继续准入；重建后恢复服务 |
| 数值回归 | 当前冻结 20-case 真实 VLM shared off/on；同历史 logits/top-2 规则不放宽，并保留 token 分歧；适用纯文本回归通过 |
| 资源回收 | 正常 churn 后仅保留明确的常驻权重/模型基线，无活跃请求 lease 泄漏；卸载所有角色与 owner 后物理 allocation 回收；硬故障隔离单列 |

增加有意义的 CPU manifest/状态机测试、真实 CUDA IPC 跨进程与故障测试、定向 compute-sanitizer；缺 CUDA/模型时写明 blocked/skip，不能用 fake backend 计通过。检查权重共享期间没有写入，包括初始化后的原地修改；校验可以用于诊断，但不能代替访问路径审查。

### 19.4 公平实验与收益口径

实验必须保留以下两个独立问题，不能混成一个加速结论：

1. **固定 KV/工作量的共享开销。** 同一组常驻独立 P/D worker、相同 GPU/模型/请求轨迹、相同 KV slots、workspace 配置与总并发/stream 上限，对比 private weights / shared weights。两组分别充分预热，随机顺序各至少 5 次；每个 trial 启动/清理完整拓扑，不能让另一组权重残留影响设备预算。报告加载时间与启动峰值、稳态设备总占用、physical weight bytes、TTFT/ITL p50/p95、成功吞吐与错误率。稳态显存和启动峰值必须分开；不同进程的显存统计不能简单相加推断物理共享。
2. **固定总显存预算的容量收益。** 在相同设备物理预算下让共享节省容量进入真实 KV 准入，比较可用页、并发准入/拒绝、队列等待与有效吞吐；相同输入/输出长度和到达轨迹，各至少 5 次。新增页必须真实分配并被请求访问，释放后同 PID 恢复准入。报告权重理论可节省 bytes 与实测差额，不预设“显存减半”或“吞吐必然提升”。

重用真实 persistent PD runner，新增明确的权重模式及机器可读指标。旧 unified/P-D、单/双副本结果保留原名和原数据；共享 on/off 对比必须保持角色拓扑相同。若收益为负或受算力/带宽限制，照实记录；正确性、物理共享和真实准入成立即可讨论适用边界，不为达成加速而改变负载口径。

### 19.5 交付物、证据冻结与进度

新增证据统一放在 `docs/data_flow_evidence/v4/E4/`：`baseline_manifest.json`、`weight_manifest.json`、`lifecycle_trace.jsonl`、故障/原始执行日志、`memory_timeline.jsonl`、`results.json`、`checks.json` 与执行报告。新增单命令 private/shared 常驻 P/D 演示和可执行 validator，检查 allocation/lease 身份、真实 P/D 调用、数值、峰值、计费、重复次数及资源回稳，不接受手写 `ok=true`。

本次计划修改可能使旧 final manifest 中的计划文件 hash 失配，这是新增范围后的预期版本变化；**不要只为通过旧 hash 检查而覆盖历史 manifest**。执行者先保存历史基线及差异，E4 全部门禁通过后生成新的 E4 completion manifest，记录最新源码、构建二进制、模型/配置、验收文件 hash 和可重放命令。旧 M0–M9/E1–E3 报告保持其历史验收语义；对受修改影响的路径做回归并保存当前证据。

- [x] E4-A 冻结基线与显式权重/别名清单。
- [x] E4-B owner 分配、单次上传、ready 发布、完整代际与幂等协议。
- [x] E4-C 真实 PBE loader/attention 使用共享权重，无完整私有 GPU 副本启动峰值。
- [x] E4-D 多消费者、退出/崩溃/重启/迟到消息生命周期门禁。
- [x] E4-E 物理权重单次计费、私有状态计费与真实 KV 准入闭环。
- [x] E4-F 20-case 数值、相关回归、sanitizer 与两组五次公平实验通过；完成演示、validator 和证据 manifest。

以上全部已有真实主路径证据，E4 更新为 **1/1**、扩展合计 **4/4**。执行摘要、失败/skip、证据路径和重放入口见 `V4_E4_SHARED_WEIGHT_COMPLETION_20260913.md`。

## 20. 性能证据整理与补测：面向简历和技术面试（B1–B6，待执行）

### 20.1 目标、执行范围与先后顺序

目标：将“实现了什么”整理成“在什么模型、硬件、负载和资源约束下，减少了多少成本，以及付出了什么代价”。先审计已有 M9/E1–E4 原始记录，再补测不可支撑结论的缺项。不得预设正收益、挑选最优 seed、放宽数值门禁或把组件节省量当作端到端加速。

本次仅新增数据整理/测试计划。执行者从 B1 开始；允许补充测量埋点、参数化 launcher、validator 和必要的测试入口，保护现有用户改动。发现主路径缺陷时先记录并修复再重新冻结受影响实验；不能借实验名义实现逐层传输重叠、流式角色图或 TP 转换。B5 所需通用多 P/D 注册、页表授权或协调若尚不支持，记录精确缺口并保持该项未验收，不用轮流执行完整 infer 的替身冒充通过。

执行顺序：B1 证据审计与协议冻结 → B2/B3/B4 主实验 → B5 单卡拓扑对照 → B6 汇总与复算。最终独立报告 B1–B6 的完成数/6，不重置历史功能阶段，也不把“没有正收益”视为实验未完成；不支持的必做场景须明确 blocked，不能标作完成。

### 20.2 B1：已有证据审计与统一测量协议

1. 优先读取 `data_flow_evidence/v4/M9/final_results.json`、E1/E2/E3/E4 的 results/checks、`V4_E4_SHARED_WEIGHT_COMPLETION_20260913.md` 和常驻 PD 实验。建立 `evidence_inventory.json`，逐条记录原始 trace 路径、源码/二进制/模型 hash、硬件、样本数、配置、计算公式、可复算指标、缺失字段和结论边界。分类为可直接引用、仅机制证据、需补测、已失效；历史版本有效数据可保留历史身份，不能混入当前版本配对比较。
2. 实际少算 token、减少 encoder forward、两请求准入差异、两进程共享映射分别只证明对应机制；不能直接写成最大并发、最大容量、吞吐翻倍或端到端加速。审查旧 fixed-budget 实验是否给 private/shared 同一硬预算且同等机会分配 KV；人为限定 private 为很少页的实验不能证明容量上限差异。
3. 冻结模型/processor/媒体 hash、dtype、位置语义、采样、输入/输出长度、GPU UUID、驱动/CUDA、构建选项、CPU 线程、MPS/MIG 状态、CUDA Graph、总 stream/active 上限、KV/Host/staging/工作区预算。保存 PID、启动命令、独立 GPU 空闲基线与完整 dirty diff；只安装缺失依赖，隔离 Python 环境。
4. 所有 serving 实验包含预热；启动耗时/启动峰值独立测量。区分预填缓存后的 steady-state 与从空缓存开始的 cold-cache 轨迹，两臂使用相同种子和请求序列。预填成本单列，不能通过免费预填获得未披露收益。测量任务不与其他 GPU 实验并跑；发现外部负载则记录并按预先规则重测整对 trial。
5. 预先固定每个 cell 的负载、重复次数和停止规则，随机化臂顺序，各至少 5 次独立 trial。先导试验只用于确定可运行长度、到达率和时长，配置冻结后不能再根据收益调参。正式 cell 每 trial 建议至少 100 个请求且持续至少 60 秒；若成本不可接受，须先在协议中记录缩小样本及结论限制，不能用少数请求估算可靠 p99。
6. 开环负载按预定到达时刻发起，不等待前一请求完成；同时记录 scheduled arrival 和实际发送时刻，识别客户端瓶颈。闭环实验必须标明固定并发。队列等待纳入客户端 TTFT；server 内部计时与客户端时间分开，同机跨进程单调时钟的可比性先核验。

统一指标与公式：

| 指标 | 定义与约束 |
| --- | --- |
| 客户端 TTFT | 首个实际可见 token 到达时间减请求发送时间，包含排队与数据准备；记录是否真实 streaming。仅最终响应含完整 tokens 时不能伪造客户端 TTFT |
| ITL/TPOT | 实际相邻 token 可见时间差分布；TPOT 为首末 token 间时长/(输出 token 数−1)，两者不混称；不足 2 token 标 N/A |
| 请求/输出吞吐 | 成功完成请求数或其实际输出 token 总数/明确墙钟窗口；固定 cohort 使用首次发送至最后终态，含 drain，不用平均延迟倒数 |
| 错误/拒绝/超时 | 分别给出占 offered requests 的比例，失败不从分母消失；报告 goodput 防止大量拒绝造成延迟虚低 |
| SLO goodput | 预先冻结 TTFT/TPOT 或 ITL 门限及允许失败率；同时正确、完成且满足 SLO 的请求数/墙钟窗口。门限在看结果前确定并记录理由 |
| 复用与传输 | 请求/媒体/实际 token 命中率分别记录；actual computed/saved tokens、encoder forwards、missing pages、实际 copy bytes 与协议开销分列 |
| 显存/容量 | GPU 总物理使用量、owner allocation、imported logical bytes、每进程私有占用、启动峰值、KV 实际页与活跃页分别报告，注明 MiB/GiB 或十进制 bytes |

每个 trial 保存请求级原始记录和资源时间线，声明采样周期及采样峰值可能遗漏瞬时分配，结合 allocator 高水位核对。展示各 trial 值、中位数与范围；如给置信区间，按独立 trial 重采样，不把同 trial 的请求当作独立重复。延迟下降率=(baseline−candidate)/baseline；吞吐增幅=(candidate−baseline)/baseline；基线为 0 时标 N/A。

### 20.3 B2：多模态缓存的计算节省与端到端收益

固定同一 serving 拓扑，比较 feature cache off/on × semantic KV cache off/on 四种配置；不能只关闭目录查询但仍隐式复用 radix，或通过关闭缓存改变模型执行精度。对照保留同一设备/池预算，隔离缓存机制影响。

- 负载包含同图不同问题、同图同前缀、不同图片但相同 placeholder，以及不同历史/位置的正确性反例。分别定义图片复用率和连续 KV token 复用率，不用一个“90%”覆盖两个概念。
- 图片复用比例采用 0/50/90%；选择两档可运行上下文长度和两档图像分辨率，保存实际展开 token 数。先在代表长度/分辨率跑完整四臂复用矩阵，再对固定的长前缀/高分辨率场景做 stress 对照，无需盲目做所有变量的笛卡尔积。
- 记录视觉 forward 数/设备耗时、实际 Prefill tokens、查询耗时、缺页 bytes、排队与客户端延迟。GPU 各段耗时有重叠时不得简单相加当作总延迟。
- 缓存预热按每条请求的真实前缀构造，不能直接塞 oracle 输出；按冻结的数值规则比较 off/on。0% 场景必须保留，以展示管理开销；如观察到由负到正，仅在测得的相邻负载间给出收益转折区间，不外推通用阈值。

交付：复用率—TTFT/吞吐图、计算节省/搬运量表、正确性反例，以及至少一条有完整证据链的结论；全为负收益时明确报告。

### 20.4 B3：权重共享的物理显存、容量与服务收益

在同 GPU、相同常驻 1P1D 拓扑下比较 private/shared 两臂，沿用 E4 的真实 handoff、独立 PID、零前缀重算和完整输出门禁。

1. **固定 KV 的开销实验：** 相同页数、总并发、请求轨迹和 workspace；测加载峰值/时间、稳态物理显存、每进程私有成本、TTFT/ITL、吞吐。owner 权重仅计一次；导入映射的逻辑 bytes 不重复累加，不用“文件大小×副本数”代替物理测量。
2. **固定显存的容量实验：** 预先冻结两臂相同的总硬预算（可小于物理整卡）和安全余量；计入 Vision、context、workspace、staging 与 CUDA 峰值。两臂使用相同容量探测算法和实际分配/准入约束，逐步增加 KV/并发，在每档运行真实请求并访问新增页；保存最后成功档与首个失败档/拒绝原因。可用页上限、可保持请求数和满足 SLO 的并发上限分开表述。
3. **容量到 goodput：** 在相同到达轨迹和显存预算下比较准入/拒绝、完成请求、排队、SLO goodput。页数更多但算力已饱和时可能仅增加排队，必须报告。不得把“shared 两请求成功、private 一请求失败”写成最大容量提升 100%。

交付：物理显存分项图、容量探测完整记录、显存→KV→准入→goodput 的对应表，以及固定 KV 下的性能代价。

### 20.5 B4：位置选路与压力恢复的代价边界

先复算 E2/E3，原始记录满足当前协议时只补汇总；缺失的压力/在线时序才重测。

- 选路对比 fixed、round-robin、data-aware，保持相同 worker 集合/模型副本/总容量。覆盖命中但排队、未命中但空闲、容量竞争和过期状态；测排队、实际传输、决策耗时、预测误差及客户端结果。单卡 IPC 与跨卡 copy 单列，不能把同卡映射称为 RDMA 收益。
- 压力恢复对比现有可执行 LRU/固定保留基线与依赖策略，同预算、相同依赖可用条件。覆盖正常负载、接近容量上限及短时过载；测下沉/恢复/必要重算的全部成本、拒绝率、尾延迟和恢复后数值。
- 基线不支持保持不可恢复内容时，只能记为正确性失败，不能拿丢失请求后的低延迟作为有效性能基线；输出明确“策略确保正确服务”与“策略加速”两种不同结论。

交付：策略有利/不利场景表、原始可重算 metrics、故障与正常性能数据的独立标签。

### 20.6 B5：单卡多 P/D 拓扑与共享成本实验

目的：确认能否在同 GPU 部署多个 P/D 角色并安全共享资源，以及进程增加的代价；不预设进程越多越快。首先核查 launcher、Coordinator、KV grants/initial slots 是否支持多个 Prefill provider 和多个 Decode 消费者；不得把单 P 的初始化 slot 清单直接复用于多 P。

| 拓扑 | 必查内容 |
| --- | --- |
| 统一单进程 P+D | 使用现有 mixed/continuous batching 的有效基线；不能为了衬托多进程故意串行化或缩小基线 batch |
| 1P1D | 同卡常驻真实交接与共享权重，作为拓扑扩展起点 |
| 1P2D | 两个 D 都实际完成请求；共享不可变前缀、私有增长页隔离 |
| 2P1D | 两个 P 都实际生产；D 能按 provider/generation 获取正确页表，无 slot 冲突 |
| 2P2D | 覆盖每条可用 P→D 路径，四进程均参与；跨请求错配、取消与退出后回收验证 |

每个可支持的拆分拓扑分别跑 private/shared；各比较臂固定总 KV/active/stream/CPU 配额做开销对照，再在固定总设备预算下测容量。workspace 随进程增加的不可避免成本必须计入；不可仅固定每 worker 配额导致总资源随进程数增长。统一进程基线保持同等总预算，并使用其既有批处理能力。

负载冻结为 Prefill 较重（长输入短输出）、Decode 较重（短输入长输出）、混合三类；长度在先导阶段按实际可运行范围确定。各正式 cell 至少 5 次独立 trial，记录每角色吞吐/队列与公平性、设备利用率、显存、TTFT/ITL、goodput。MPS 状态各臂保持一致并披露；不把多 PID/多 stream 当作 kernel 真实并发或硬件隔离证明，需要时以独立 profiler trace 验证，profiler 时序不用于正式延迟统计。

证明共享时同时记录 GPU UUID、owner allocation/generation、各 importer 的 lease、权重上传次数、KV 内容/物理页身份、请求 P/D PID 和 actual computed tokens。多个进程共享一份权重不意味着任意语义 KV 都可以共用；KV 仍必须满足 E1 的身份与 E4 的生命周期条件。

交付：拓扑支持矩阵（已验证/不支持/未测）、不同负载下的规模曲线、每增加一个角色的私有显存成本和最佳观测配置。结论限于测试范围，不宣称任意 N/M、弹性伸缩或严格资源隔离。

### 20.7 B6：可重放产物、简历指标与最终门禁

证据目录：`docs/data_flow_evidence/v4/performance_characterization/`，保留历史文件，禁止覆盖旧 manifest 来消除本次计划变更造成的 hash 差异。

必交付：

- `evidence_inventory.json`、`protocol.json`、当前源码/二进制/环境 `manifest.json`、精确命令与随机顺序。
- `raw/` 请求级 JSONL（request/trial/arm、到达/发送/首 token/逐 token/终态时刻、tokens、状态、命中/搬运、P/D PID）和独立资源时间线；失败和超时保留。
- `summary.csv`、`results.json`、可独立从 raw 复算的分析脚本和 validator；自动检查配对条件、重复次数、数值、计时边界、计费、拓扑参与、资源回收及百分比公式。
- 标准绘图工具导出的 SVG/PNG：复用率—性能、显存分解、固定预算容量/goodput、单卡拓扑规模曲线。图中注明模型、硬件、负载、样本量和误差范围，不使用截断坐标夸大效果。
- `PERFORMANCE_REPORT.md`：收益、负收益、缺失/不支持场景和瓶颈解释。`RESUME_METRICS.md`：每条候选表述附 baseline/candidate、绝对值/相对值、实验条件、trial 数、原始证据链接和适用边界。

候选简历句式（只有实测支持才填写）：

> 在 [模型/硬件]、[媒体/token 复用率和输入输出长度]、[相同资源预算] 下，相比 [明确基线]，TTFT 从 [A] 降至 [B]；代价为 [查询/传输/低复用开销]。
>
> 在单卡 [P/D 数量] 部署下，将物理显存从 [A] 降至 [B]；相同硬预算内实际可用 KV 从 [C] 增至 [D]，满足 [SLO] 的 goodput 从 [E] 变为 [F]。

禁止“全面加速”“领先其他框架”等未经同条件外部对照的表述。单卡多进程本身不是独占能力；例如 Dynamo 的 SGLang 后端有[同卡 P/D 示例](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/sglang/launch/disagg_same_gpu.sh)。该链接只证明存在同卡部署能力，不能据此推断其他框架全部支持或不支持共享物理权重。若未来做跨框架性能对照，应另行冻结版本、模型、kernel、缓存与资源条件，本轮不强制新增外部框架部署。

- [ ] B1 完成证据审计、缺口清单及冻结测量协议。
- [ ] B2 完成缓存复用率/开销/端到端效果矩阵。
- [ ] B3 完成公平显存、容量与服务收益实验。
- [ ] B4 完成选路/恢复证据整理及必要补测。
- [ ] B5 完成单卡 1P1D/1P2D/2P1D/2P2D 与统一执行对照；不支持项明确未验收。
- [ ] B6 完成复算门禁、图表、报告、可引用简历指标和资源清理。

完成标准是证据完整、可重放且结论与数据一致，允许所有性能结果为零或负收益；不要求为获得正收益继续增加新功能。当前本节仅为执行计划，**B1–B6 为 0/6**。
