# V4 执行后审计与后续改进建议

## 结论

V4 已有真实模型、跨进程 Host 数据服务、视觉特征缓存、分页模型输入适配和局部生命周期测试，不能再描述为仅有计划。但原计划要求的“数据服务拥有 GPU 池、计算角色借用 slots、在线多模态协调与统一资源准入”尚未形成完整的真实调用链。阶段记录全部 accepted，与原始验收范围存在差距，尤其 M3、M5、M6、M7、M8、M9。

本次按源码和原始记录逐项复核，并使用现有构建运行了 16 项定向测试及两个独立数据服务故障复现。未重新构建全部目标，未重跑 VLM GPU 性能实验；历史模型结果与本次运行结果分开。未修改 runtime、旧 accepted 标记或原始实验记录。当前工作区仍包含大量未提交改动，不将 HEAD 当作完整实验版本。

## 已有证据支持的成果

- M1/M5 原始记录支持真实 Qwen2.5-VL-3B 视觉特征、PBE 语言执行、mRoPE 和部分 chunked-prefill 数值验证。
- M3/M5 支持生产进程退出后，两个后续 Decode 进程获取同一 Host KV 快照、恢复并输出相同 token。
- M4 支持真实视觉编码产物发布及命中时跳过 forward。
- M6 有同进程页共享、尾页 COW、取消后页数回稳、临时 PrefixBuilder 请求退出后复用的单元测试。
- M2/M7/M8 有对象身份、引用、预算账本、重启 allocation handle 拒绝和局部状态机测试。

这些成果构成可用的研究原型；“通过局部测试”与“接入跨角色服务主路径”应分别计数。

## 高优先级发现

### R1：重启后的迟到 release 会释放新消费者的 lease（已复现）

`ContentRegistry::release` 只接受数字 lease_id；每个新 registry 的 next_lease_id 从 1 开始。allocation 有 incarnation 校验，但 release 协议没有携带它。

复现顺序：服务 A acquire 得到 lease 1 → kill A → 同 endpoint 启动服务 B → 新消费者 acquire 得到 lease 1 → 旧客户端 release(1)。结果新服务 active_leases 从 1 变为 0。旧客户端错误改变了新消费者引用；在后续 withdraw/reclaim 中可能破坏本应保留的生命周期。本次未声称已复现 GPU use-after-free，当前服务 payload 是 Host 拷贝。

源码：[registry](../infMain/source/data/content_registry.cpp) 第 35–56 行、[wire dispatch](../infMain/source/data/node_agent.cpp) 第 69–72 行、[Python client](../python/pbe_data_client/client.py) 第 59 行。

修复方向：release 使用完整 LeaseToken（service incarnation、consumer incarnation、lease id，必要时 allocation generation），服务端核验归属与版本；重复 release 幂等，跨代 release 拒绝或安全 no-op。Acquire/reply-lost 也需要 operation-id 重试合同，不能每次重试新增不可达 lease。

### R2：半包客户端使正常 shutdown 等待不退出（已复现）

客户端连接后仅发送一个请求头字节并保持连接；另一客户端执行 shutdown 返回 0，但服务 1 秒后仍未退出。关闭第一条连接后服务退出 0。`ReadAll` 使用阻塞 recv，无连接读取期限；stop 关闭监听路径而不打断活跃 fd，随后 join worker。

源码：[node_agent.cpp](../infMain/source/data/node_agent.cpp) 的 ReadAll、stop、serve_connection、run worker join。

修复方向：连接级绝对 deadline、活跃 fd 管理、stop 时 shutdown 活跃连接、队列清退。测试部分 header/body、慢 reader、8 worker 全被占用、reply 丢失、正常关闭上界。设置每次 recv 超时还不等价于防止持续滴流，应使用绝对请求时间预算。

### R3：统一 GPU 所有权尚未接入真实 EPD

当前 `LocalDataRuntime::Object::storage` 是 `vector<uint8_t>`；ContentRegistry acquire 复制整个 payload。真实 VLM demo 由各模型自行初始化 KV allocator，通过 snapshot/Host 传输后 restore 到各自池。IPC owner/import 只在独立 probe 演示常量数据，没有接到模型 attention 使用的共享 pool。

源码：[data_runtime.h](../infMain/include/data/data_runtime.h)、[content_registry.cpp](../infMain/source/data/content_registry.cpp)、[VLM demo](../demo/pbe_multi_role_vlm.cpp)、[IPC probe](../demo/pbe_ipc_pool_probe.cpp)。

这证明“跨进程快照复用”，尚不等于“Agent 拥有 GPU 页，计算 worker 借用 slot”。优先把 Agent-owned GPU pool、导入后的 KVPoolView、WriteLease/ComputeLease 和真实 kernel fence 接到模型调用链。同 GPU 同前缀双消费者验证唯一物理页；跨 GPU 副本单独计账，Host 路径保留为 oracle。

### R4：VLM 闭环仍是固定 fixture 演示

`pbe_multi_role_vlm.cpp` 读取 `one_image_224.input_ids.i32.bin` 和 `position_ids.i32.bin`，写死 delta=-56、hidden size 2048、image placeholder，并含生成时固定 fixture 路径。启动脚本顺序运行 Vision、Prefill、Decode A、Decode B，能够证明跨 PID 交接，但不能证明在线并发角色调度。

`RequestCoordinator` 在当前检索范围中仅有实现和单测，未见 serving/demo 消费调用；`OnlineGenerateRequest` 仍为 prompt 等文本字段。M5 的在线 typed parts、动态 join、按请求 position state、阶段取消需要继续接入。

验收方向：同一个常驻服务连续提交不同图片/文本/分辨率，多请求混合且两 Decode 同时活跃；tokens/grid/positions 由请求和 processor 实际生成；运行时无需读取参考 fixture。fixture 保留为数值测试 oracle。

### R5：联合预算与跨角色分支主要是局部组件证据

检索 `NodeMemoryBudget`、`PrefixBuilder`、`RequestCoordinator` 在 `infMain/source` 和 `demo` 中的调用，当前只找到自身实现；单元测试显式创建这些对象。M7 账本测试不能证明 Encoder 输出、语言 KV、staging 与权重共享同一个线上准入约束。

Vision worker 在完成模型加载和 forward、bundle 序列化后才 reserve；这与 M4/M7“forward 前保障输出容量并背压”不一致。M6 的 COW 验证发生在单个 KVCacheManager，同一 Host 快照恢复到两个模型实例并不共享同一套 GPU page 引用。

改进方向：统一设备预算和真实分配绑定，消除独立账本；常驻 Encoder 先估算输出/workspace 再准入；并发 Decode 分支使用 M3 的真实共享页；checkpoint/mRoPE/feature 依赖在真实压力路径联合恢复。

### R6：视觉缓存身份与 singleflight 有遗漏（静态发现）

worker 的 key 使用媒体字节、size 和硬编码 REVISION，实际 `--model` 加载内容和 processor revision 未核验。更换本地模型/processor 后可能错误命中旧缓存。

singleflight 使用 `/tmp/pbe-vision-{content}.lock`，未包含服务 endpoint/incarnation；两个独立服务可能错误串行等待。锁创建后 processor/模型加载、forward 等异常不在清理 try 范围内；leader 崩溃或这些步骤失败会残留文件，后续请求等到 120 秒超时且不能自动接管。

源码：[vision worker](../python/pbe_roles/vision/worker.py) 第 32、43–79 行。本次没有触发大模型失败复现。

改进方向：从冻结模型 manifest/processor 配置推导身份；编码 singleflight 由服务管理，使用 attempt generation、独立 waiter、绝对 deadline、失败状态与接管合同。接管只解决编码任务，不表示旧 GPU 写入可直接回收；写入目标要遵守独立物理生命周期。

### R7：M9 的公平性能验收尚未完成

`results.json` 明确保留未测字段：同预算合并/拆分、GPU lane timeline、五重复真实模型压力策略、multimodal TTFT/ITL。其余 branch/budget/fault 数据是测试进程 wall time，不是 serving workload 结果；matrix 列出 0/50/90% 重复率等参数不代表这些轨迹已运行。

Vision off 约 15.6 秒与 on 约 0.43 秒是一次性进程路径：off 包含模型/processor 加载，on 在加载前返回。不能当常驻服务缓存带来的约 36 倍推理加速。现有报告对此有说明，应保留这一诚实边界；但不能因此把原 M9 全部验收标作完成。

源码与证据：[实验 runner](../tools/bench/data_flow/run_v4_experiments.py)、[M9 results](data_flow_evidence/v4/M9/results.json)、[M9 report](data_flow_evidence/v4/M9/execution_report.md)。

## 后续优先级：先完成 V4 合同，再做新增设计

| 优先级 | 工作 | 价值与验收 |
| --- | --- | --- |
| P0 | R1/R2/R6 生命周期和身份修复 | 重启迟到 release 不影响新消费者；半包与 leader 崩溃不永久占资源 |
| P1 | R3 Agent-owned GPU 页接入 | 真正实现框架所有权；用模型读写和物理页计数证明 |
| P1 | R4 在线 sequence plan 与常驻角色 | 去掉 fixture 运行依赖；动态请求、混合 batch、真实取消 |
| P1 | R5 联合准入/分支/压力接入 | Encoder、KV、staging 同预算，跨角色并发共享和 COW |
| P1 | R7 公平实验 | 相同常驻模型、轨迹与预算，测 TTFT/ITL/bytes/拒绝率 |
| P2 | 语义前缀查询和页级按需获取 | 从整请求 snapshot 复用升级为不同请求的最长前缀复用 |
| P2 | 基于实际副本的计算放置 | 在 Encoder/Prefill/Decode placement 中量化省下的搬运与等待 |
| P2 | 分层缓存与恢复依赖 | 一图多问、内存压力下保留精确视觉特征和必要 KV；不同数据采用不同保留策略 |
| P3 | 分片覆盖、多 provider 聚合、异构布局计划 | 只有真实多来源/TP 需求后加入，先 schema/coverage 预检与完整提交 |

特别值得加入的 P2 是“语义前缀查询”。当前 demo 对计算完成后的 payload 求 ContentId，消费者从生产者日志获得 key。它适合验证数据传递，但新请求无法在计算前仅凭输入推导出该快照 key。应以模型/processor revision、完整有序文本与媒体内容及位置语义建立前缀身份，页链返回最长可服务前缀；消费者只拉缺页，私有尾页另算。这会直接体现缓存命中、少算 prefill 和传输成本的联系。

计算放置可先做简单可解释策略：选择 estimated queue wait + transfer bytes/measured bandwidth + compute time 最小的角色，限制热点和迁移抖动；与 round-robin、固定 PD 对照。资源不足时仍服从联合准入，不以命中率最高替代端到端延迟最小。

不建议此时扩展 DiT、音视频全覆盖或权重共享。它们会增加模型与生命周期复杂度，当前 P1 缺口更直接影响 V4 的核心主张和面试可信度。

## 本次验证记录

使用现有 `/tmp/Paged-Batch-Engine-build-qwen05/test/test_llm`：

```text
--gtest_filter=NodeMemoryBudgetTest.*:RequestCoordinatorTest.*:LocalDataRuntimeTest.*:ContentRegistryTest.*:KVCacheManagerTest.PartialTailForkSharesThenCopiesOnFirstAppend:KVCacheManagerTest.PrefixBuilderPublishesAfterTemporaryRequestExits
16 tests from 5 suites, 16 passed
```

初次旧 CPU build 筛选只匹配一个 HostStore 测试，不能用它覆盖 V4；随后使用上述 V4 构建。未声称此次对源码做了全量 rebuild。

两个故障测试使用独立临时 Unix endpoint 和现有 `pbe_data_service` 可执行文件，所有临时进程退出并清理，未影响已运行的用户服务：

```json
{"old_incarnation":49719689048638862,"new_incarnation":49719659110316095,"old_lease":1,"new_lease":1,"leases_before_stale_release":1,"leases_after_stale_release":0}
{"shutdown_rpc_exit":0,"service_still_alive_after_1s":true,"service_exit_after_partial_header_client_close":0}
```

这两项是实际观察，其余未列动态复现的缺口为源码/调用图与验收证据审查结论。后续实现首先把它们加入可重复回归测试，再修复，不用增加 passed 数量代替对应场景。
