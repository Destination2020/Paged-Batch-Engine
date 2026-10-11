# PagedBatchEngine 在 RTX 4060 WSL 上的多模型并发与 CUDA MPS 升级方案

日期：2026 年 10 月 11 日（Asia/Shanghai）。代码基线：`f3597a763b6cd5cf60b826baa2dc94395c3d02fd` 加当前工作区内容。本文为设计与研究结论，配置、接口和新增模块均为拟实现项。

## 1 建议先做什么

针对你明确的条件——**本机 RTX 4060 Laptop、8 GiB 显存、WSL，任意两个很小的 LLM 或 VLM**——建议采用以下路线：

1. **先交付两个小 LLM 常驻同一 GPU**：使用 Qwen2.5-0.5B-Instruct 和 Qwen2.5-Coder-0.5B-Instruct 两套不同权重，分别处理对话摘要和简单代码任务。复用现有 Qwen serving 路径，增加模型路由、绝对显存预算及独立运行状态。
2. **本机并行主线采用单进程多 ModelRunner 与 CUDA 非阻塞 stream**：先做模型间异步调度，再做同模型 prefill/decode 的独立执行状态。在支持条件和微基准通过后评估 Green Context。普通多进程可作为部署对照，但不能把进程同时存在当成 kernel 同时执行。
3. **完整设计 MPS 多进程后端，但设置环境门槛**：当前 WSL 环境未满足已验证的 MPS 部署条件，不应承诺加一个开关即可使用。若目标环境验证失败，严格的 MPS 进程并行目标需要另一个经过验证的原生 Linux 环境；这是一项部署选择，不是本次默认要求你更换系统。
4. **VLM 作为第二种可替换演示组合**：Qwen2.5-0.5B 与 SmolVLM-256M。首轮可用外部 Transformers worker 承载视觉模型；这能验证同卡多模型服务，但不能声称自研 CUDA 引擎已原生支持 VLM。
5. **暂不把跨进程共享权重、动态全局 KV Arena、分布式服务栈一起塞进首版**。在 8 GiB 卡上，先量化显存和同步开销，比首先追求复杂部署拓扑更有价值。

推荐的首个可验收成果是：一个 HTTP 入口、两个不同权重的小模型、每模型独立 tokenizer 和请求队列、统一显存接纳、正确的流式输出与取消；在有收益的工作负载中，用 GPU 时间线证实执行重叠。MPS 和单进程 stream 是这个服务层之下的不同执行后端。

**核心判断：两个小模型共卡是适合当前机器的目标；“必须在当前 WSL 上依赖 MPS”尚不能作为可兑现的前提。** 同卡 P/D 分离也不会天然比现有 chunked prefill 更快。最终选择依据是每模型 SLO 内吞吐和尾延迟，而不是进程数量。

## 2 当前环境与能力边界

### 2.1 本机只读检查结果

| 项目 | 检查结果 | 对升级的影响 |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU | 按消费级单卡方案设计 |
| 总显存 | 8188 MiB | 不能把完整 8 GiB 都分给模型 |
| 即时显存快照 | used 1200 MiB，free 6758 MiB | 是易变快照；驱动保留等使字段未必简单相加，预算按实际 free 和安全余量 |
| 驱动 | 566.26 | 不能直接使用要求新驱动的 CUDA 13 系列新能力 |
| CUDA 编译工具 | 12.6，nvcc 12.6.85 | 可以研究该版本已有 Driver API；不能照搬 13.1 Runtime API |
| 系统 | WSL2，内核 6.6.87.2-microsoft-standard-WSL2 | 必须区分 WSL 与原生 Linux 的 CUDA 支持范围 |
| MPS 二进制 | PATH、/usr/bin 及 /usr/local/cuda*/bin 常见位置未发现 | 说明当前未具备启动前提，不构成对所有 WSL 版本的普遍不支持证明 |
| 已有模型配置 | Qwen2-0.5B，24 层、2 KV heads、head_dim 64 | 可用于精确 KV 估算与第一轮正确性测试 |

本次没有更换驱动、安装 MPS 或运行并发性能实验；上表是环境观察，不是性能测量。

### 2.2 WSL 与 MPS 必须分别验证

NVIDIA 的 WSL 指南明确说明部分管理能力受限，同时列出 legacy CUDA IPC 自 R510 起支持。**CUDA 推理可运行、IPC 可运行、MPS 可运行是三个不同的结论。** 官方 MPS 支持说明列出 Linux/QNX，不能仅因 WSL 使用 Linux 用户态，就推导出完整 MPS 可用。[CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)、[MPS 平台限制](https://docs.nvidia.com/deploy/mps/595/when-to-use-mps.html)

实施时设置能力探测结果，而不是仅保存 `mps: true`：

- `cuda_basic`：单进程推理成功。
- `multi_stream_overlap`：两个独立非阻塞 stream 的探针出现实际 kernel 重叠。
- `cuda_ipc_memory_event`：两个独立进程能映射内存并正确等待跨进程事件。
- `mps_clients_registered`：控制端确认两个 worker 注册到同一 MPS server。
- `mps_kernel_overlap`：实际时间线证实跨进程 GPU 执行重叠。
- `green_context`：在具体驱动、CUDA、GPU、WSL 组合上创建并运行成功。

如果用户指定 `execution_backend=mps_process`，探测失败必须明确返回原因，不能静默改为普通多进程后仍报告 MPS 成功。本机默认配置采用 `execution_backend=inproc_streams`，这条路线也要经过微基准验证。

MPS 下 `EXCLUSIVE_PROCESS` 是官方推荐设置，不能错误地写成所有情形的强制必要条件，更不能仅凭 WSL 无法设置该模式就完成“不支持”的证明。[MPS 系统要求](https://docs.nvidia.com/deploy/mps/595/when-to-use-mps.html)

### 2.3 版本分层

| 机制 | 能解决什么 | 当前机器的定位 |
| --- | --- | --- |
| CUDA 非阻塞 stream | 单进程中提交互不依赖的工作 | 首选实验路线；重叠和收益仍取决于 kernel 与同步 |
| CUDA 12.4 起的 Driver API Green Context | 给执行上下文分配 SM 资源 | 本机 CUDA 12.6 可研究对应 API；WSL 支持需探测 |
| 传统 MPS 动态 provisioning | 多进程并发及执行资源上限 | 保留后端设计；当前 WSL 未验证 |
| CUDA 13.1 静态 MPS 分区 | 分配不重叠的 SM 分区 | 软件栈升级后再评估，不作为当前配置 |
| CUDA 13.4 MPS v3 | 新 CLI、配置及 cgroup 显存管理 | 要求匹配的新驱动和系统条件，当前不具备 |
| MIG | 更强的计算和显存资源隔离 | RTX 4060 不在支持清单，本机排除 |

Green Context 在 CUDA 12.4 已存在；13.1 增加 Runtime execution context 接口及静态 MPS 相关能力。13.4 新功能要求 R615 或更新驱动，不能依据一般的 CUDA minor compatibility 就认定当前 566.26 可用。[CUDA 12.4 发布说明](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/)、[CUDA 13.1.1 Green Context](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/04-special-topics/green-contexts.html)、[CUDA 当前发布说明](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)、[MIG 支持 GPU](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html)

## 3 从源码确认的起点

仓库已经具备 paged KV、continuous batching、chunked prefill、radix prefix cache，以及远程 P/D 通信骨架。升级重点是把单模型服务提升成有明确资源与生命周期边界的多模型运行时。

| 代码位置 | 当前事实 | 对设计的影响 |
| --- | --- | --- |
| `demo/serving_qwen.cpp:74`、`:100` | 按部署角色初始化 Qwen2Model，并传入对应设备 | 可复用 worker 初始化；不能照旧文档说固定 device 0 |
| `infMain/source/model/qwen2.cpp:418` | 已有带 device_id 的 init | 本机 worker 内用可见设备 ordinal，协议记录物理 GPU UUID |
| `infMain/include/model/qwen2.h:214` | 明确同一实例的 forward/sample 不线程安全，复用可变 workspace | 单进程并发需拆运行状态，不能只给同一模型对象加两个线程 |
| `infMain/include/serving/serving_online_engine.h` | 请求缺少统一 model/task 身份 | 增加模型、任务、请求及取消路由标识 |
| `infMain/include/serving/serving_online_engine_pool.h` | SingleOnlineEnginePool 和单 endpoint 的 ZmqOnlineEnginePool | 新增多模型池和 registry |
| `infMain/source/serving/serving_online_server.cpp` | 模板与响应模型信息存在单模型假设 | 模板随模型选择，响应返回实际 model |
| `infMain/source/serving/serving_online_engine.cpp:483`、`:518`、`:593` | 远程循环含同步 submit/poll，prefill 完成后申请 decode reservation | RPC 与容量接纳需从 decode 热路径解耦 |
| `infMain/source/serving/serving_online_engine.cpp:605` | 现有远程 NCCL 路径按请求创建通信相关状态 | 同卡后端不宜直接沿用这套成本 |
| `infMain/include/serving/pd_handoff.h:29` | 已有 pool descriptor、manifest、block mapping、connector | 扩展身份、代次和所有权，复用逻辑协议 |
| `infMain/source/model/serving_memory_planner.cpp:39` | 动态 KV 基于设备 free_bytes × 利用率计算 | 多模型必须改绝对字节预算，不能各自使用 0.8 |
| `infMain/source/base/cuda_backend_runtime.cpp:48` | compute_queue 与 transfer_queue 指向同一 stream | 新增独立 stream 和设备端依赖接口 |
| `infMain/source/base/cuda_backend_runtime.cpp:190` | wait_event 使用 cudaEventSynchronize | 增加 stream_wait_event/query_event，避免主线程等待一方而无法提交另一方 |
| `infMain/source/op/kernels/cuda/matmul_kernel_batch.cu:154` | 进程静态 heuristic_cache，key 包含形状和 workspace，未见锁 | 多 CPU 线程写入有竞态；配置变化后还应重新验证算法适用性 |
| `infMain/source/base/alloc_cu.cpp` | allocator 缓存释放 allocation | 逻辑释放不等于归还 GPU 物理显存 |
| `infMain/include/base/cuda_graph_manager.h` 与对应 cpp | 当前文件为空，当前主干未检出 CUDA Graph capture/replay 实现 | Graph 属于待接入或重新核查项，不能视为已有能力 |
| 当前源码全文 | 未发现 cudaIpcGetMemHandle/OpenMemHandle 等实现 | 同卡跨进程 IPC 数据面需要新增 |

代码定位以这次读取的工作区为准。已有设计稿 `docs/cuda-mps-multi-model-upgrade-plan.md` 与《多模型共享 KV Pool 的 Chunk 架构设计 V1》可复用其思路，但设计描述不能作为已实现证据。本方案单独保存，不覆盖它们。

**三个需要先修正的架构假设：** 设备空闲显存不是每个模型可以独占的预算；模型权重可只读不意味着整个 Model 对象可并发；P/D 类名分离不意味着 GPU 时间线已并行。

## 4 公开方案比较与选择

资料检索截至 2026 年 10 月 11 日。以下区分论文方案、开源组件和当前官方部署功能；不把论文对旧版本基线的加速倍数当作本项目收益。

### 4.1 与当前单卡目标直接相关

| 方案及时间 | 核心价值 | 本框架应如何取用 |
| --- | --- | --- |
| [Sarathi-Serve，OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/agrawal) | chunked prefill 与 decode 混合调度 | 当前已有类似能力，必须保留为强基线 |
| [MuxServe，2024](https://arxiv.org/abs/2404.02015) | 按模型热度及 P/D 特性联合做多模型复用 | 借鉴资源管理与跨模型调度，不照搬整个依赖栈 |
| [Nexus，2025，v5 为 8 月 7 日](https://arxiv.org/html/2507.06608v5) | 用 Green Context 和代价预测做单引擎内 P/D 并行 | 适合中期单进程路线；可借鉴预建资源布局与滞回切换 |
| [DuetServe，2025 起](https://arxiv.org/html/2511.04791v2) | 预测 decode 延迟风险，仅在需要时开启空间分区 | 借鉴按 SLO 选择混合或分离执行，避免永久切半 GPU |
| [POD-Attention，2024 起](https://arxiv.org/abs/2410.18038) | 在 attention kernel 内重叠 P/D | 当 attention 是瓶颈时评估，比直接再加进程更细粒度 |
| [NanoFlow，OSDI 2025](https://www.usenix.org/conference/osdi25/presentation/zhu-kan) | nano-batch 与算子级资源重叠 | 作为长期引擎优化参考，首版改造范围过大 |
| [Prism，2026 年 6 月版本](https://arxiv.org/abs/2505.04021) | 通过 GPU memory ballooning 提高多模型容量弹性 | 与仓库 Chunk 设计高度相关，适合固定预算版之后 |
| [kvcached 开源组件](https://github.com/ovg-project/kvcached) | 解耦 KV 虚拟地址与物理显存分配 | 研究稳定地址和物理页复用；接入需要适配自研 layout |

MuxServe 官方仓库仍列有 API serving 的 TODO，并依赖修改版 vLLM，属于需要审慎集成的研究系统。[MuxServe 仓库](https://github.com/hao-ai-lab/MuxServe)

DuetServe 文中使用 libsmctrl 对 SM mask 进行控制。它与 NVIDIA 官方 Green Context 接口不是同一层面的兼容承诺；这里优先吸收调度策略，执行层先选择官方 API，避免把底层非标准控制变成必需依赖。[DuetServe 实现说明](https://arxiv.org/html/2511.04791v2)

FlashInfer 当前 POD 文档已提供相应 wrapper，包括批量 POD 接口。引入时需核对 GPU 架构、head_dim、GQA、page layout、dtype 和编译方式；它不是直接替换任意自研 attention 的通用动态库接口。[FlashInfer POD API](https://docs.flashinfer.ai/api/pod.html)

### 4.2 更成熟服务组件与跨卡路线

| 方案 | 适用问题 | 本机优先级 |
| --- | --- | --- |
| [vLLM disaggregated prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill/) | 分开优化 TTFT/ITL，提供 KV connector 机制 | 外部基线和接口参考；官方仍提示实验性，不承诺吞吐必升 |
| [SGLang](https://github.com/sgl-project/sglang) | radix cache、调度及多后端服务生态 | 用作比较对象；不要为两个微型模型立即重写全部框架 |
| [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) 与 [NIXL](https://github.com/ai-dynamo/nixl) | 分布式路由、P/D 协作与异构传输 | 未来多 GPU 时接入 connector，单卡首版成本过高 |
| [Mooncake](https://github.com/kvcache-ai/Mooncake) 与 [LMCache](https://github.com/LMCache/LMCache) | KV 分层缓存、传输、复用 | 长上下文/多机时有价值，不能消除不同权重间的 KV 语义差异 |
| [Ray Serve fractional GPU](https://docs.ray.io/en/latest/serve/llm/user-guides/fractional-gpu.html) | 多服务共享 GPU 放置 | GPU fraction 是调度记账，不能当作硬件隔离或显存保护 |
| [NVIDIA Triton](https://github.com/triton-inference-server/server) | 多种模型后端及生命周期管理 | 若主要需求变成 LLM+VLM+其他网络，作为服务层替代候选 |
| [DistServe，OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) 与 [Splitwise](https://arxiv.org/abs/2311.18677) | 跨 GPU P/D 分离与资源配置 | 借鉴接纳和成本核算；跨卡结论不能直接用于同卡 8 GiB |

### 4.3 更新研究值得关注但暂不接入主线

[TRE，2026 年 9 月 24 日](https://arxiv.org/abs/2609.29160) 强调不同模型之间应比较经过校准的服务缺口，而非只看原始队列长度。这里可借鉴为跨模型公平性信号，无需引入其集群自动扩缩容系统。

[Talaria，2026 年 7 月 19 日](https://arxiv.org/abs/2607.17181) 强调会话连续性、权重驻留和 KV 局部性。对本机方案的启发是给同一会话优先路由回已有缓存的模型实例；百亿级以上多卡实验不构成微型模型单卡收益证据。

若两个任务最终能共享同一个底座，仅 adapter 不同，可以另选 [S-LoRA](https://arxiv.org/abs/2311.03285) 或 [Punica](https://arxiv.org/abs/2310.18547) 一类多 LoRA 路线，减少底座重复驻留。但用户允许任意两个小模型，首版没有必要引入训练、adapter kernel 与额外兼容约束。

**选择结论：** 当前最有价值的组合是已有 chunked prefill 基线、MuxServe 式跨模型资源管理、Nexus/DuetServe 式按需单卡并行，以及后续 kvcached 式稳定地址显存弹性。Dynamo、Mooncake 和跨卡 P/D 放到扩展层，不作为本机首版依赖。

## 5 推荐模型与 8 GiB 显存规划

### 5.1 第一轮用两个不同的 0.5B 模型

| 组合 | 任务 | 选择理由 |
| --- | --- | --- |
| 首选 A：Qwen2.5-0.5B-Instruct | 简单对话、摘要 | Qwen2ForCausalLM，最接近现有后端 |
| 首选 B：Qwen2.5-Coder-0.5B-Instruct | 简单代码生成、解释 | 不同完整权重，可验证真正的模型隔离 |
| 可复用的本地替代 | 已有 Qwen2-0.5B-Instruct 替换 A 或 B | 先减少下载和导出变量，仍须与另一套不同权重配对 |
| VLM 扩展 | Qwen2.5-0.5B + SmolVLM-256M | 文本和图像任务并存，需要新增后端能力 |

两套 Qwen2.5 0.5B 的官方配置均为 24 层、hidden 896、14 Q heads、2 KV heads，head_dim 64，使用 tied embeddings。它们结构相近但参数不同；不能共享计算出的 KV。[普通模型配置](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json)、[Coder 配置](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/blob/main/config.json)

按每个约 0.49B 参数估算，两个模型参数存储合计：FP32 约 3.65 GiB，BF16/FP16 约 1.83 GiB。**这些仅是参数字节下界**，不包括导出格式可能造成的重复 tensor、加载转换峰值、KV、workspace、cuBLAS、CUDA context 或 allocator 缓存。实际预算必须来自本框架加载后与最大形状 warmup 的测量。[Qwen2.5 模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)、[Coder 模型卡](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct)

先复用现有 exporter 验证新 checkpoint，检查 config、tied embeddings、QKV bias、tokenizer、EOS 和 ChatML 模板。先保证单模型与参考输出一致，再纳入双模型测试。不要以“同属 Qwen2”代替导出一致性验证。

### 5.2 全局预算公式

每个执行实例 r 的峰值预算为：

```text
M_r = W_r + KV_r + A_r + Workspace_r + Runtime_r + Margin_r

服务新增分配峰值
 = 所有私有 M_r 的和 + 服务共享物理分配及全局运行开销
 <= 启动前 free_bytes - 尚未承诺的安全余量
```

其中 W 为实际 GPU 权重，A 为最大在途 activation。free_bytes 已反映已有占用，不再重复扣除这部分外部显存；只额外预留其未来增长和测量误差的安全余量。测量基点必须在服务尚未分配 GPU 内存时，若服务已有常驻分配则先单独记账。若实现共享权重或共享 Arena，同一物理 allocation 全局只计一次，但各进程的 context、page table、workspace 等仍单独计费。初始化与 warmup 可能比稳定运行占更多显存，也必须纳入峰值。

本机第一轮建议给服务**总额先设为 5120 MiB**，两模型各 2560 MiB，所有额外服务端 GPU allocation 也必须纳入这个总额。这只是保守的实验起点：在本次 free 6758 MiB 快照下留出约 1638 MiB 未承诺空间；启动时重新测量，不保证其他程序不继续占用。如果任一模型 warmup 超过子预算，就降低并发/上下文或用 BF16，不能悄悄从另一模型池中借款。

两模型建议初始形状：每模型最多 2 个活跃请求、prompt 上限 1024 token、输出上限 256 token、prefill chunk 128。先验收并发 1+1，再升到 2+2 和 4+4。2560 MiB 是否足够取决于运行精度及实测工作区，配置加载必须允许失败并给出分项原因。

### 5.3 KV 按模型几何估算

对于该 Qwen GQA 配置：

```text
Plain KV bytes/token = 2 × L × Hkv × D × element_bytes

FP32: 2 × 24 × 2 × 64 × 4 = 24576 bytes = 24 KiB/token
BF16: 2 × 24 × 2 × 64 × 2 = 12288 bytes = 12 KiB/token
```

假设 block size=16，按最长保留 token 数向上取整：

| 场景 | FP32 KV | BF16 KV |
| --- | ---: | ---: |
| 一个 1280-token 请求 | 30 MiB | 15 MiB |
| 两个模型，各 2 个 1280-token 请求 | 120 MiB | 60 MiB |
| 两个模型，各 4 个 4096-token 请求 | 768 MiB | 384 MiB |

这些值不含元数据与额外对齐。首轮只有少量短请求时，权重和工作区可能比 KV 更值得优化；因此不要在测量之前就先重写全局 KV allocator。

当前 FP8 格式是每 token/head 的 K、V 各带一个 FP32 scale，按 `KVCacheBytesPerToken()`：

```text
FP8 KV bytes/token
 = 2 × 24 × 2 × 64 × 1 + 2 × 24 × 2 × 4
 = 6528 bytes = 6.375 KiB/token
```

相对 BF16 的节省约 46.9%，不是完整减半。FP8 KV 也不会减少模型权重；它需要额外数值验证和 K/V scale 交接，放到基础并发正确之后。

### 5.4 必须改掉的动态容量逻辑

现有 `estimate_dynamic_kv_blocks_per_layer()` 从 `cudaMemGetInfo` 的 free_bytes 取一定比例。在多个 worker 同时加载时，两者可能看到近似相同的空闲量；即使串行启动，后启动者也可能几乎没有容量。新的多模型模式采用：

1. 控制器分配绝对 `memory_budget_bytes`，按预算顺序加载与 warmup。
2. 测出权重与最大工作区后计算 `kv_pool_bytes`；不从全卡 free 再独立乘 0.8。
3. 若预算连最小 KV/workspace 都容纳不了，返回容量不足。不要沿用“强行返回 minimum blocks 然后再 OOM”的兜底。
4. 请求接纳同时预留 prompt 与允许的输出增长；保守首版按 `max_new_tokens` 预留。
5. 后续才改为滚动扩容预留，并设置可证明可完成的下限，避免所有请求都占着 prompt KV 却没有一个能扩出下一页。
6. 指标分别报告 allocated、in_use、reserved_for_growth、cache_evictable、allocator_cached；逻辑空页不是物理 free bytes。

## 6 统一服务层与两种执行后端

### 6.1 逻辑架构

```text
HTTP 请求
  → ModelRegistry / TaskRouter
  → 全局准入与显存预算
  → 每模型队列和调度器
  → ExecutionBackend
       ├─ 本机：单进程、多 ModelRunner、独立 stream
       └─ 可选：每模型进程或 P/D 进程、MPS、CUDA IPC
  → GPU
```

路由、任务、计费、取消、监控和显存协议应共用；后端只决定执行位置和交接方式。这样 WSL 验证出的功能不会因为后续迁到 MPS 而全部重做。

第一轮默认一个模型一个 mixed runner。不要默认把两个模型都变成四个 P/D 进程：两份模型各复制一次权重，再叠加 context 和 workspace，容易浪费 8 GiB 的容量。

### 6.2 请求与模型注册

拟增加：

```cpp
struct ModelIdentity {
  std::string model_id;
  std::string weights_revision;   // 文件摘要或明确版本
  std::string tokenizer_revision;
  std::string adapter_revision;   // 无 adapter 时为空
};

struct RoutedRequest {
  std::string global_request_id;
  ModelIdentity identity;
  std::string task;               // chat / summarize / code / image_text
  std::string session_id;
  GenerationConfig generation;
  // 文本或结构化 messages；多模态内容另行定义
};
```

`ModelRegistry` 描述权重、tokenizer、模板、可支持任务、上下文限制和执行后端。`MultiModelEnginePool` 将请求送往匹配模型。未知模型或不支持任务返回明确错误；不擅自改投另一个模型。

服务行为应包含：

- 支持请求中的 `model`；旧客户端省略时仅在配置了明确默认模型的情况下使用默认值。
- 增加 `/v1/models` 或等价模型列表。
- 响应和 SSE chunk 标记实际模型，请求 ID 全局唯一。
- 取消使用 `model_id + worker_epoch + global_request_id` 路由。
- tokenizer 和 chat template 随模型选择；不能在 HTTP 层先按固定模板拼完再决定模型。
- 各模型的 stop/EOS、随机数状态、采样配置、指标和 prefix cache 分开管理。
- 异构权重模型不进入当前同一次 forward；每模型自己 continuous batch，上层协调多个 batch。

### 6.3 本机多 stream 的改造重点

建议把不可变权重与可变执行状态逐步分离：

```text
ModelWeights
  └─ 只读参数与模型结构

ModelRunner A / B
  ├─ stream 与 cuBLAS handles
  ├─ activation、logits、sampler workspace
  ├─ batch metadata 与 token staging buffers
  ├─ KV manager 和 scheduler
  └─ in-flight StepTicket
```

第一阶段 A、B 是两个模型对象，各有一个 runner。第二阶段同一模型的 P、D 才共享一个只读权重对象，并分别拥有运行 workspace；KV 元数据仍由一个明确的控制线程修改。

现有 `pd_worker.cpp` 把 forward、sample、取回结果串在一个同步步骤。新接口应拆成：

```cpp
StepTicket enqueue_step(const ScheduledBatch& batch);
StepStatus poll_step(StepTicket ticket);  // 不阻塞
StepResult finish_step(StepTicket ticket);
```

一个 GPU 提交线程依次 enqueue A、enqueue B，然后轮询完成事件；HTTP 和 tokenization 可由 CPU 线程池处理。这样先避免 allocator 和 cuBLAS heuristic 全局容器的 CPU 数据竞争，再逐步消除全局状态。

必须落实的同步改造：

| 位置 | 当前问题 | 拟改造 |
| --- | --- | --- |
| `cuda_init_utils.cpp` | 普通 cudaStreamCreate | 使用 non-blocking stream，明确默认流交互 |
| `qwen2.cpp:1478` 附近 | greedy D2H 后立即等事件 | D2H 后返回 ticket，稍后 query |
| `qwen2.cpp:1684` 附近 | 部分采样 copy 使用 need_sync=true | 异步 copy，加局部完成事件 |
| `alloc.cpp:41` | 同步 copy 路径会调用 cudaDeviceSynchronize | 并发热路径禁用全设备等待，保留明确的同步维护接口 |
| `BackendRuntime` | 缺 query_event 与 stream_wait_event | 增加非阻塞查询和设备端等待 |
| 复用 metadata / logits / token buffers | CPU 已返回不代表 GPU 已读完 | 每在途 step 固定 buffer slot，按完成 fence 回收 |
| allocator cache | release 可立即标记可复用 | 增加 deferred free，GPU 使用完成后再放回 free list |

不能简单删掉 synchronize：删等待后必须补足依赖、buffer 生命周期和错误传播。每 runner 首版最多一个在途 step，可大幅简化正确性。每次 enqueue_step 都在该 runner stream 的最后一项操作之后记录统一 completion event，包括不产生采样输出的中间 prefill chunk；不能只靠 sampler 的 D2H event 判断 step 完成。

延后回收同时覆盖两层：CUDADeviceAllocator 的 allocation 复用，以及 `BlockAllocator::free()` 在 refcount 归零后的逻辑 KV 页入队。即使物理 pool 固定不释放，页在其他 stream 仍被读取时也不能被下一请求覆写。

### 6.4 同模型 P/D 分离的两条实现路线

**单进程路线**：独立 P 和 D runner，共享只读权重和一个受控 KV 管理域；以 cudaEventRecord / cudaStreamWaitEvent 表达先写后读。此时不需要 CUDA IPC，也不需要再复制一套权重。prefill 请求 A 与 decode 请求 B 可以重叠；同一请求必须满足前后依赖。

**MPS 路线**：P 和 D 为独立进程，默认各持一套权重与私有 KV pool，以 IPC 映射源池并复制到目标池。先证明正确，再考虑跨进程共享只读权重。一个模型拆 P/D、另一个保持 mixed 是首个合理三进程实验，而不是两个模型全部拆分。

两种路线都保留原 mixed 执行路径作为回退与比较对象。首次上线不做请求执行到一半时随意切换后端；按新请求选拓扑，旧请求排空。

## 7 同 GPU P/D 的 KV 交接协议

以下设计面向 MPS/普通多进程 IPC 后端；其中身份、页生命周期和取消原则也适用于单进程。

### 7.1 不要发送裸地址或直接搬用双卡 NCCL 模式

跨进程发送 CUDA IPC handle、allocation ID、字节 offset 和布局信息；导入者在自己的地址空间得到映射。不能发送 P 的裸 device pointer 后让 D 直接解引用。[CUDA IPC 指南](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html)

普通 NCCL 配置不能简单把已有双卡两个 rank 都指向 device 0。**NCCL 2.30 起新增实验性同 GPU 多 rank 开关**，但有资源耗尽挂起、CTA 数量与 NVLS 等约束；这修正了“任何版本都不支持”的旧结论，却不足以改变本项目首版优先 IPC 的选择。[NCCL 2.30.7 固定版本文档](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/env.html#nccl-multi-rank-gpu-enable)

### 7.2 长生命周期 pool 注册

新增 `CudaIpcKVConnector` 和 `IpcPoolRegistry`：

1. pool 在启动后导出一次，每层 K/V allocation 分别注册；FP8 增加 scale allocation。
2. 明确 allocator allocation 的真实边界，不对随意的 tensor 子视图单独导出。
3. D 只在握手时打开 handle，得到本地映射；不要每个请求反复 open/close。
4. 新 pool 有新 `pool_epoch`，旧进程重启后所有旧描述符失效。
5. exporter 的 allocation 必须活到所有 importer 关闭映射；导出资源不能被 allocator cache 提前复用。[CUDA Runtime IPC API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__DEVICE.html)

建议首版使用专门的可导出 pool allocation，避免与无关 tensor 混在同一子分配块。VMM 可在后续统一管理映射，但不必成为首版前置。

### 7.3 Manifest 需要的不只是 shape

现有 `KVPoolDescriptor::compatible_with` 主要验证几何与 storage mode。新描述符增加：

| 类别 | 字段 |
| --- | --- |
| 语义身份 | model_id、weights_revision、adapter_revision、tokenizer_revision |
| 执行几何 | layer_num、kv_heads、head_dim、block_size、TP 分片描述 |
| 表示格式 | logical_dtype、storage_dtype、scale_dtype、layout_version、RoPE/position 配置摘要 |
| 资源身份 | GPU UUID、worker_epoch、pool_epoch、allocation_id |
| 请求身份 | global_request_id、handoff_id、request_generation |
| 位置与状态 | prompt_tokens、computed_tokens、下次输入 token、next_position、采样状态 |
| 数据映射 | 各层 src/dst block IDs、有效 token 数、尾页范围 |
| 同步与租约 | ready_event_slot、event_generation、reservation_id、lease_generation |

不同权重即便 shape 完全一致也必须拒绝交接。prefix cache 的 key 同样需要模型/adapter 身份；多模态时还要纳入图像内容、预处理配置和视觉模型版本，不能仅对占位 token 做 hash。

### 7.4 首版采用复制式交接

```text
请求被接纳
  → D 预留 prompt 与输出增长容量
  → P 分配源页并执行 prefill
  → P 记录 IPC ready event，发送 manifest
  → D copy stream 等 ready event
  → 按 src/dst 映射复制 K/V 及 scales
  → D 记录 copy_done
  → D compute stream 等 copy_done
  → D 接续 decode；host 确认 copy_done 已完成后发送释放 ACK
  → P 放下 transfer pin；满足其他引用条件后释放源页
```

IPC event 使用 `cudaEventInterprocess | cudaEventDisableTiming`。发送消息时 GPU 工作可以尚未完成，但必须先在相应 stream 上提交 event record；D 以 GPU stream wait 表达数据依赖。消息收到本身不代表数据可读。[CUDA Event API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html)

首个生成 token 通常由 prefill 最后 logits 采样产生，协议明确由谁发给客户端、D 的第一步输入是什么、position 如何递增。覆盖 `max_new_tokens=1`、EOS 立即结束及 stop 匹配；否则容易重复或遗漏首 token。随机采样采用每请求计数器/RNG 状态，不能因切换进程改变随机序列。

所有权状态建议为：

```text
RESERVED → PREFILLING → READY → COPYING → DECODE_OWNED → FINISHED
                       └──────── cancel/failure → DRAINING → RECLAIMED
```

释放 ACK 只有在 `cudaEventQuery(copy_done)` 确认完成后才能发送；另一种合法协议是 ACK 携带完成事件并由 P 等待。仅仅 enqueue 了 D compute stream 的 wait，并不足以让 P 安全释放源页。

控制消息必须幂等。ACK 重复不能重复减引用；超时不能直接回收 GPU 仍可能访问的页；事件 slot 复用必须带代次并等待消费者释放，避免 ABA 问题。P 的 prefix cache 引用和 transfer pin 是两个独立引用，不能因为交接 ACK 就把仍被 prefix cache 使用的页覆盖。

首版等待全部层复制完再启动该请求的 decode，正确性最清晰。后续再利用现有 layer-wise connector 实现逐层流水，并证明每一层读取依赖都正确。

### 7.5 取消与进程退出

- P 未开始时取消：撤销 D reservation 和排队项。
- P 已提交 kernel 时取消：标记取消，等最后写入完成后释放源页。
- 复制中取消：D 目标页和 P 源页都保留到 copy fence 完成。
- D 接管后取消：停止新 decode step，等已提交 step 完成再回收。
- exporter 崩溃：导入内存失效风险必须触发相关 importer 停止访问和域重建；租约超时不是 GPU fence。
- worker 重启：分配新 epoch，拒绝迟到 ACK/manifest，原流式请求返回明确错误；已经发出的 token 不做透明整段重放。

MPS fatal GPU fault 可能影响同 GPU 的其他客户端，进程独立不等于完全故障隔离。正常停机先排空，异常终止按相应版本的 MPS 客户端终止机制处理。[MPS 内存保护与提前终止说明](https://docs.nvidia.com/deploy/mps/595/when-to-use-mps.html)

## 8 MPS 后端的正确配置与新版本路线

### 8.1 传统 MPS 的作用边界

MPS 为多个 CUDA 进程提供更有效的并发执行机制；框架仍要负责模型路由、batch 形成、显存接纳、页生命周期与公平性。

`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` 是执行资源上限，不是独占 SM 数量，也不是固定时间份额。两个 worker 配成 50/50 不意味着各自获得一半显存带宽；合计 100% 也不是必要的正确性条件。旧控制接口对 server 设置新百分比只影响后来连接的 client；已有 client/context 不变。[MPS 动态资源配置](https://docs.nvidia.com/deploy/mps/595/when-to-use-mps.html#dynamic-execution-resource-provisioning)、[MPS 管理接口](https://docs.nvidia.com/deploy/mps/latest/appendix-tools-and-interface-reference.html)

`CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` 名字中的 pinned 不应被误读成仅限制 host pinned memory；它针对 client 的设备内存分配限额。但它不能代替框架自己的容量规划，也不会在不同模型间自动安排 KV 优先级。[MPS 显存限制](https://docs.nvidia.com/deploy/mps/latest/appendix-tools-and-interface-reference.html#cuda-mps-pinned-device-mem-limit)

首轮在具备 MPS 的目标环境比较：

- 两个 mixed worker：100/100、50/50 等少量上限组合。
- 一个模型 P/D：100/100、30/70、50/50 等组合。
- 加第三个 worker 后重新测量，不沿用二进程最佳值。
- 传统 MPS 上限按排空、重建 worker 的慢路径调整；请求级控制使用 chunk 大小、放行速率、batch 与队列。

### 8.2 启动流程契约

以下是**未来受支持环境上的传统 MPS 启动示意**，不是当前 WSL 已可运行的承诺，也没有在本次执行：

```bash
export CUDA_VISIBLE_DEVICES=GPU-REPLACE-WITH-VERIFIED-UUID
export CUDA_MPS_PIPE_DIRECTORY=/tmp/pbe-mps-pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/pbe-mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
chmod 700 "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
echo get_server_list | nvidia-cuda-mps-control
```

Supervisor 在 CUDA 初始化前给每个 worker 设置设备、pipe directory、资源上限，再执行独立程序；不要在已初始化 CUDA 的父进程中 fork 后继续使用继承状态。使用匹配 UID 和明确 GPU UUID，处理可见设备重新编号，避免把 daemon 的设备索引与 worker 的 ordinal 混淆。

READY 顺序为：能力验证 → 全局预算分配 → 加载权重 → warmup → pool/事件握手 → MPS client 注册检查 → 模型就绪。仅 daemon 启动成功不等于 client 已接入，传统模式 server 还可能直到首个 client 连接才创建。

本机失败时输出 `MPS_UNAVAILABLE_ON_CURRENT_ENVIRONMENT` 及探测细节；`inproc_streams` 和 `process_default` 是单独可选配置，不伪装成 MPS。

### 8.3 新机制如何纳入设计

**CUDA 13.1 静态 MPS 分区**能够使用不重叠的 SM 分区，但受架构及分区粒度限制，动态线程百分比不再按原方式生效。使用中的分区不能任意删除；也不提供 MIG 那样的显存路径隔离。把它视为部署级资源配置，不作为每 token 快速切换工具。[静态 MPS 分区](https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html#static-sm-partitioning)

**CUDA 13.4 MPS v3**增加新的 CLI 和配置方式，显存软硬限额依赖 Linux、cgroup v2、non-MIG 等条件。软限额可能使客户端成为显存压力下的驱逐候选，硬限额可导致分配失败；对有状态解码服务，都必须接入 worker 退出和请求失败处理。不能把新旧命令混用，也不建议为了本机双 0.5B 演示先升级整套系统。[MPS v3 接口](https://docs.nvidia.com/deploy/mps/latest/mpsv3-interface.html)、[v3 显存分区](https://docs.nvidia.com/deploy/mps/mpsv3-memory-partitioning.html)、[CUDA 13.4 官方介绍](https://developer.nvidia.com/blog/cuda-toolkit-13-4-adds-windows-on-arm-support-and-greater-control-over-shared-gpus/)

实现上增加 `ExecutionCapabilities` 和版本化配置适配器，分别实现 `legacy_mps`、`static_mps`、`mps_v3`，避免在业务调度器中散布环境变量与命令字符串。

## 9 调度策略与收益判断

### 9.1 为什么并行可能更慢

prefill 通常有较高计算密度，低 batch decode 常对显存带宽敏感；它们有互补机会，但不是完全使用不同硬件资源。两者仍竞争缓存、显存带宽、寄存器、SM 以及调度资源。decode batch 增大后，线性层计算比例也会上升。VLM 的视觉编码阶段还可能与 prefill 一样对计算资源施压。

同卡拆分的收益条件可以写为：

```text
避免的阻塞时间 + 资源重叠节省
  > 拆分导致的 batch 效率损失
    + 数据交接及控制成本
    + 资源争用损失
    + 额外显存压缩带来的并发下降
```

只有 Nsight 时间线重叠而端到端 ITL 变差，不应判为升级成功。只报告总 tokens/s 也可能掩盖另一个模型被饿死。

### 9.2 两级调度

**全局层**按模型分配服务机会与显存 credit。先用有权重 deficit round robin；成本单位以实测 step 时间估计，加入输入长度、当前上下文及 batch 大小，而不只比较 token 个数。

**模型内层**复用现有 scheduler：decode 优先、chunked prefill、KV 接纳。每 runner 首版最多一个在途 step；允许两个 runner 各自一个 step 重叠，不在整个 GPU 周围加全局执行互斥锁。

防饥饿要求：

- decode 有延迟目标，但不能永远压制另一模型的新请求。
- prefill 配置最大等待时间与 aging。
- 有界队列、deadline、拒绝策略和每模型最低服务份额。
- 路由考虑 session/prefix 局部性，但有队列过载阈值。
- 扩容/回收物理显存时考虑既有输出增长预留，避免为了新模型饿死已在生成的请求。

### 9.3 先测量再做自适应

建立本机 profile 表：

```text
T_prefill(model, chunk_tokens, prefix_len, resource_config)
T_decode(model, batch, context_len, resource_config)
I(pair_of_stages, pair_of_models, batch_shapes)
M_peak(model, precision, maximum_shapes)
```

I 是实测干扰系数。4060 Laptop 的功耗限制和温度可能显著影响结果；记录运行功耗/频率/温度与是否插电，避免把降频当成调度效果。

首次控制策略可以很简单：decode p95 ITL 超出目标时减少其他模型的 prefill chunk 和在途 prefill 数；负载较轻时允许更大 chunk。以一段时间窗口内持续超标作为触发，设置最短保持时间，避免逐 token 反复改变模式。

中期再借鉴 Nexus/DuetServe 的预测决策，在 `mixed`、`streams`、`green_context` 的有限配置集合中选择。**不要把传统 MPS 环境变量当成这个快速控制回路的执行器。**

### 9.4 CUDA Graph 的加入顺序

当前源码没有可复用的 Graph 主干，建议在异步多模型正确后再实现。按 runner、模型版本和 shape bucket 维护 graph；capture 前预分配 metadata 和 workspace；buffer 地址变化或执行 context 变化后重建。

Green Context 下尤其不能把“原 stream 捕获的 graph”换个 launch stream，就宣称所有节点使用新的 SM 分区；节点 context 与捕获/创建时的 context 有关。要在正确 context 下捕获并核查节点执行归属。[Green Context 与 Graph 说明](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/04-special-topics/green-contexts.html)

## 10 VLM 接入需要额外完成什么

SmolVLM-256M 官方描述单图推理可低于 1 GB GPU RAM；这是特定条件下的模型声明，不是本机服务的峰值预算。它基于 Idefics3，使用视觉编码器和文本解码器，不能直接换一个 Qwen 权重路径就运行。[SmolVLM 模型卡](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)、[官方发布介绍](https://huggingface.co/blog/smolervlm)

### 10.1 两条可选接入路径

| 路径 | 需要完成的工作 | 验收能说明什么 |
| --- | --- | --- |
| 外部 Transformers worker | 输入 processor、模型加载、图片限制、生成/取消协议、统一路由和预算 | 同一服务入口管理 LLM 与 VLM，二者可在 GPU 常驻 |
| 原生 C++ VLM backend | 图像预处理、vision encoder、projector、视觉 embedding 注入、对应文本 decoder、权重导出及 kernels | 本框架原生执行完整 VLM |

第一轮选择前者时，外部进程不共享自研模型的 context；在当前 WSL 上不能承诺两进程的 GPU kernel 真正重叠。它是模型兼容性与共卡容量验证，不是完成 MPS 的证据。

原生路线要增加 `TaskBackend` 接口，至少覆盖 `prepare_input`、`estimate_memory`、`enqueue_encode`、`enqueue_prefill`、`enqueue_decode` 和 `cancel`。不要让 API 层把所有视觉输入强行压成文本。

### 10.2 多模态显存和调度

VLM 预算增加：

```text
M_VLM =
  vision_weights + language_weights + projector_weights
  + image_encoder_peak_activations
  + visual_embeddings
  + decoder_KV + decoder_workspace + runtime
```

视觉 token 进入 decoder 后会占 KV；计费和上下文上限不能只数用户文本 token。限制单请求图像数、图像大小、processor 生成的 crop/patch 数与视觉 token 上限；prefill 前先得到有效 token 估计。

首轮测试只用一张低分辨率图像和一条文本请求，并发 1+1。分别测图片预处理、视觉编码、文本 prefill 和 decode；若视觉编码影响另一模型 ITL，可先对 encoder 做全局有界准入，再研究视觉 encoder 与 decode 的资源划分。

视觉输入缓存的 key 至少包含图像内容摘要、processor 参数、vision weights revision、projector revision 和文本模型身份。未经验证不复用不同模型的图像 embedding 或 KV。

## 11 与现有共享 KV Chunk 设计衔接

仓库《多模型共享 KV Pool 的 Chunk 架构设计 V1》以单 GPU、单进程的全局 Arena 为起点，这与本机路线较一致。但当前实现仍是每层固定 pool，attention/batch metadata 存在以 layer 0 block table 为基础的假定。

因此分三级实现：

| 阶段 | 内存组织 | 必须保持的不变量 |
| --- | --- | --- |
| K0 首版 | 各模型固定私有 pool，统一预算 | 模型内现有层间 block ID 关系、独立 cache namespace |
| K1 单进程弹性 | 全局 ChunkAllocator，各模型租借空 chunk | 页还在 GPU 使用时不可回收，不能只看 CPU refcount |
| K2 VMM 或跨进程 owner | 稳定虚拟地址、动态物理映射/跨进程映射 | 统一所有者、mapping generation、重启失效和 fence 回收 |

K1 不是把所有模型的 `BlockAllocator` 指向同一 free_queue。不同模型可能具有不同层数、KV heads、dtype、scale 格式；共享的是物理容量，模型各自维持逻辑 page allocator。

若允许各层物理页独立移动，必须同时修改 `mixed_batch_builder.cpp` 与 `qwen2_batch.cpp` 的地址映射，显式提供每层 page table，或引入统一逻辑 block 到各层物理位置的映射。仅改 allocator 会破坏 kernel 看到的地址。

Prism/kvcached 提供了更先进的方向：保留虚拟地址，按需借还物理显存。现有可观测预算、异步 fence 和 allocator 所有权是其前置条件。其完整研究系统与独立缓存组件的支持范围不同，不能将任一方的限制外推到另一方。[Prism 论文](https://arxiv.org/abs/2505.04021)、[Prism 研究实现](https://github.com/Multi-LLM/prism-research)、[kvcached](https://github.com/ovg-project/kvcached)、[CUDA VMM](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html)

对于 8 GiB，优化优先级取决于观测：权重主导时先做 BF16/权重存储优化；KV 长期低利用时做弹性 pool；IPC copy 成本主导时做同模型共享 KV；不能用同一套高级架构解决所有瓶颈。

## 12 拟新增配置

以下 YAML 是设计契约，当前 `serving_qwen` 不能直接读取。首版使用两个 0.5B 模型、单进程 stream 后端；数值是待 warmup 验证的实验上限。

```yaml
deployment:
  gpu: 0
  execution_backend: inproc_streams
  total_service_budget_mib: 5120
  max_inflight_steps_per_runner: 1
  enable_cuda_graph: false
  fail_if_requested_backend_unavailable: true

models:
  - id: chat-small
    backend: qwen2
    weights: /models/qwen2.5-0.5b-instruct/model.bin
    tokenizer: /models/qwen2.5-0.5b-instruct/tokenizer.json
    tasks: [chat, summarize]
    execution_mode: mixed
    runtime_dtype: bf16
    memory_budget_mib: 2560
    kv_pool_mib: 256
    max_prompt_tokens: 1024
    max_new_tokens: 256
    max_num_seqs: 2
    prefill_chunk_tokens: 128
    sampler: greedy

  - id: code-small
    backend: qwen2
    weights: /models/qwen2.5-coder-0.5b-instruct/model.bin
    tokenizer: /models/qwen2.5-coder-0.5b-instruct/tokenizer.json
    tasks: [code]
    execution_mode: mixed
    runtime_dtype: bf16
    memory_budget_mib: 2560
    kv_pool_mib: 256
    max_prompt_tokens: 1024
    max_new_tokens: 256
    max_num_seqs: 2
    prefill_chunk_tokens: 128
    sampler: greedy
```

模型本身的 context 上限不等于服务必须开放的上限。只有最大形状 warmup 通过，才能进入 READY。FP32 也可作为第一轮正确性模式，但用实际加载峰值重新校验，不沿用 BF16 的预算假设。

MPS 配置应独立展开为 worker 列表，明确每个进程角色、显存预算、endpoint、MPS 版本、线程上限与 transport。只给上述两个模型增加 `mps: true` 不足以定义 P/D 生命周期。

## 13 分阶段实施与文件级工作包

建议按可独立验收的工作包推进；工时是熟悉仓库的一位开发者的粗估，不包含等待驱动/环境变更的时间。

| 阶段 | 工作及主要文件 | 验收出口 | 粗估 |
| --- | --- | --- | --- |
| P0 | 记录环境能力；新增最小 stream/IPC/MPS probe 与单模型 profile | 明确当前可用后端；两个模型独立结果与峰值 | 1–2 天 |
| P1 | `serving_memory_planner.*`、`serving_config.h`、`model_registry.*`、`multi_model_engine_pool.*`、HTTP 请求结构 | 两个不同模型同卡常驻，串行基线正确，超预算在 READY 前失败 | 3–5 天 |
| P2 | `backend_runtime.*`、`cuda_backend_runtime.*`、`cuda_init_utils.cpp`、`qwen2.cpp`、`pd_worker.*` | enqueue/poll，独立非阻塞流，无热路径全卡同步；KV 页和 allocation 均按 fence 回收 | 4–7 天 |
| P3 | `global_admission_controller.*`、scheduler、metrics、bench | 双模型公平服务，有可复现重叠或明确无收益结论，支持回退 | 3–5 天 |
| P4 可选 | 模型权重与 execution state 拆分，`qwen2_batch.cpp` 等；Green Context 实验 | 同模型 P/D runner 共享一份权重且不共享 scratch，满足依赖 | 5–10 天 |
| P5 条件性 | `cuda_ipc_kv_connector.*`、`ipc_pool_registry.*`、`worker_supervisor.*`、P/D 协议 | 在通过环境门槛的机器上完成真正 MPS P/D 进程并行 | 5–10 天 |
| P6 可选 | VLM 外部 backend 与多模态输入协议 | 一 LLM 加一 VLM 共卡，图像限制、路由与预算正确 | 2–4 天 |
| P7 研究 | CUDA Graph、Chunk/VMM、共享权重 IPC、POD backend | 按瓶颈逐项证明净收益，不能合成一个不可回退大改动 | 单独估算 |

P0–P3 为当前 WSL 主线，约 11–19 个开发日的初步范围。P4/P5/P6 各自独立；不需要做完全部高级阶段才能交付双模型服务。阶段用“完成条件”控制，实际工时由源码重构与问题暴露调整。

建议拆 PR 顺序：预算与身份 → 路由和串行多模型 → 事件查询与异步采样 → 多 stream 和延后回收 → benchmark/QoS。MPS IPC 后端另开系列，避免网络协议、内存布局、模型后端同时改变。

CMake 与测试注册同步更新；保留已有单模型启动方式作为兼容入口。新增配置解析与启动错误必须给出具体模型、所需字节、可用预算及后端能力。

## 14 正确性与性能验收

### 14.1 正确性先于并行收益

复用 `test/test_serving/test_pd_handoff.cpp`、`test_pd_engine.cpp`、`test_pd_worker.cpp` 与现有 KV/scheduler 测试，新增真正覆盖异步所有权的测试，而非仅检查新字段是否存在。

| 类别 | 必测情形 |
| --- | --- |
| 模型隔离 | 两个不同权重；相同 token 前缀不串 cache；相同局部 request ID 不串结果 |
| 单模型一致性 | 串行和并发下 greedy 输出对照；数值差异时比较 logits，不能直接归因于 MPS |
| Buffer 生命周期 | 一方 GPU 人为延迟；另一方持续执行；metadata/logits/host sampled buffer 不被提前覆盖 |
| KV 回收 | cancel、EOS、timeout、prefill 完成、复制中取消、旧 ACK；最后 GPU 访问完成前不能复用 |
| 采样 | greedy 首先通过；top-k/top-p/CPU fallback 各自恢复后再测；无全卡同步回归 |
| 首 token | 不重复、不遗漏；max_new_tokens=1、立即 EOS、stop 跨 token 边界 |
| 并发接纳 | 两模型同时突发，预算和输出增长预留不超额；饱和时按策略拒绝或排队 |
| 页布局 | src/dst block ID 不同、跨 page、尾页未满、各层映射、FP8 scales |
| 进程故障 | worker 在各阶段退出；旧 epoch 全部失效；exporter 退出时 importer 不再读旧映射 |
| VLM | 图像数/分辨率/patch 上限，错误图像，image token 计费，图片不同但文本相同 |
| 长稳 | 反复取消与多轮请求后，显存、page 引用、事件、handle 和线程数不持续增长 |

当前采样实现还有一项影响可重复性的细节：`qwen2.cpp` 中部分随机数种子由 sample_count 派生，批次组成变化可能改变输出。后续改为每请求 seed 与 token_position/counter 驱动，再验证随机采样的跨拓扑一致性。首轮使用 greedy，避免把调度导致的随机序列变化误判为模型计算错误。

### 14.2 对照矩阵

| 编号 | 配置 | 目的 |
| --- | --- | --- |
| B0 | 模型 A、B 分别独占 GPU | 各模型基准延迟、峰值与单独容量 |
| B1 | 两模型常驻，单进程按步交替 | 多模型功能与显存成本基线 |
| B2 | B1 相同模型和预算，异步多 stream | 隔离 stream 与异步改造带来的收益 |
| B3 | 两普通进程，无 MPS | 部署隔离对照；不预设 kernel 并发 |
| B4 条件性 | B3 开 MPS，其他条件相同 | 隔离 MPS 的实际作用 |
| B5 | 单模型 mixed 对比单进程 P/D runner | 分离收益是否超过 batch 损失 |
| B6 条件性 | 单模型 IPC P/D，无 MPS和有 MPS | 区分 IPC/进程拆分成本与 MPS 收益 |
| B7 可选 | LLM 加 VLM | 测视觉编码对文本 ITL 和显存的干扰 |
| B8 可选 | 当前版本 vLLM 或 SGLang | 外部工程基线，记录精确版本与配置 |

首版性能比较至少完成 B0/B1/B2；不能因当前环境不能运行 B4 而把 B2 的结果写成 MPS 测量。

扩展现有 `tools/bench/run_online_concurrency_bench.py` 与 `run_pagedbench.py`，加入模型字段、每模型统计和到达率控制。负载覆盖：

- 短输入长输出，如 128/256 tokens。
- 长输入短输出，如 1024/64，再逐步扩到可容纳范围。
- 长短交替，模型负载比例 50/50、90/10。
- 一模型突发长 prefill，另一模型持续 decode。
- 相同前缀高复用与全新前缀；冷/热 cache 分开。
- 闭环并发 1+1、2+2、4+4；另加开环到达率，观察排队和超载。

固定权重版本、精度、请求集、输出上限、总显存预算、warmup、是否启用 prefix cache 与调试同步。每配置至少重复 3 轮；记录驱动、CUDA、框架 commit、Windows/WSL、温度与功耗。若组间配置为了容量必须不同，明确写出差异，不能称为单因素比较。

### 14.3 指标与可接受标准

每模型分别报告 TTFT p50/p95/p99、ITL p50/p95/p99、首 token 至第二 token 间隔、E2E、成功率、输入/输出吞吐、排队时间和取消回收时间。GPU 侧记录峰值显存、分项 allocation、KV 已用/预留/可回收、kernel overlap、copy bytes 与 copy 时间。

goodput 定义为：单位时间内同时满足该模型 TTFT 与 ITL 目标、且正确完成的请求数。不同模型 token 成本不同，全局汇总时保留各模型分项。

可以在 P0 固定以下**建议工程门槛**，这些是验收决策阈值，不是对实际加速的预测：

- 正确性、隔离与生命周期测试全通过，无新增 OOM、漏释放和串流错误。
- 相同显存预算和请求集下，各模型 p95 ITL 不超过已确认的目标。
- 若没有绝对 SLO，先约定允许的基线退化幅度，例如 5%；不能事后为了好看的图再改阈值。
- 启用更复杂并发模式，至少应稳定改善一个核心指标，如 goodput 提升 10%，或目标模型 p95 ITL 改善 10%，同时其他模型满足其约定界限。
- 结果若落在测量波动以内，保留更简单的 mixed/轮转配置。
- MPS 路线额外要求控制端 client 证据和 GPU 时间线，不能只看 nvidia-smi utilization。

上述 5%/10% 是可以调整的产品决策示例；真正测试前固定它们。没有收益时记录负载边界和回退原因，同样是有效工程结论。

### 14.4 回退与演进决策

| 观测结果 | 下一步 |
| --- | --- |
| 双模型容量不足 | 降低并发/上下文，检查 workspace，使用已支持 BF16；不先增加 P/D 进程 |
| CPU 等待占主导 | 完成异步采样与 enqueue/poll，控制 polling/RPC 线程 |
| 多 stream 重叠但 ITL 变差 | 缩小 prefill chunk，降低同时活跃阶段；回退交替执行 |
| SM 配额导致明显算力空闲 | 比较共享与按需分区，禁止长期固定均分假设 |
| decode 已饱和显存带宽 | 限制竞争 worker；更多并发未必有效 |
| IPC 复制与双份 KV 成为瓶颈 | 研究同模型共享页/direct-write，补全 COW 与所有权 |
| 私有 KV 池大量空闲且另一模型缺页 | 推进 Chunk/VMM 弹性容量 |
| 两任务能共用底座且质量合格 | 评估多 LoRA，独立验证任务质量 |
| VLM 后端工作量压过主目标 | 先外部 worker，保持原生后端边界 |
| 要求真正 MPS 而本机探测失败 | 保留实现设计，在受支持环境独立验收，不替用户假定系统已更换 |

## 15 最终建议与交付边界

对这台机器，第一阶段应该是**两个小 Qwen 模型常驻、统一路由和绝对预算、采样异步化、单进程双 stream**。这能直接推进多模型目标，也能把框架最关键的同步和资源边界修清楚。

第二阶段再判断是否值得对一个模型做同进程 P/D runner 分离，或接入 SmolVLM 展示多模态任务。第三阶段在环境明确支持时实现 **MPS 多进程 + CUDA IPC**；并以相同模型与预算对照前两阶段。

“两个模型同时部署”“不同请求的 P/D GPU 工作重叠”“MPS 下不同进程并行”是三个独立验收目标。方案分别提供实现和证据要求，不用前两个替代第三个。共享权重、弹性 KV 和更先进的调度机制都在此基础上逐项引入。

## 附录 A 优先阅读的一手资料

下表用于实施时复核版本。滚动文档记录为 2026-10-11 访问，真正开发时锁定 release/tag/commit；论文日期采用正文或提交历史，不采用搜索引擎的相对日期。

| 资料 | 日期或版本 | 阅读重点 |
| --- | --- | --- |
| [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) | 当前文档，访问 2026-10-11 | WSL 功能范围、IPC 和管理限制 |
| [MPS r595 支持与资源配置](https://docs.nvidia.com/deploy/mps/595/when-to-use-mps.html) | 固定 r595 | 传统 MPS 行为、fault、配额上限 |
| [MPS v3](https://docs.nvidia.com/deploy/mps/latest/mpsv3-interface.html) | 13.4 路线 | 新管理接口，避免与 legacy 混用 |
| [MPS v3 Memory Partitioning](https://docs.nvidia.com/deploy/mps/mpsv3-memory-partitioning.html) | 13.4 路线 | soft/hard limits、系统条件 |
| [Green Context](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/04-special-topics/green-contexts.html) | CUDA 13.1.1 固定归档 | SM 配置、stream 与 Graph 归属 |
| [NCCL 多 rank 单 GPU](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/env.html#nccl-multi-rank-gpu-enable) | NCCL 2.30.7 | 实验条件及资源约束 |
| [MuxServe 论文](https://arxiv.org/abs/2404.02015) 与 [实现](https://github.com/hao-ai-lab/MuxServe) | 2024 | 多模型资源管理与研究原型边界 |
| [Nexus](https://arxiv.org/html/2507.06608v5) | 2025-08-07 v5 | 同引擎 P/D、Green Context、预测与滞回 |
| [DuetServe](https://arxiv.org/html/2511.04791v2) | 2025 起版本 | 按延迟风险触发空间复用 |
| [POD-Attention](https://arxiv.org/abs/2410.18038) 与 [FlashInfer POD](https://docs.flashinfer.ai/api/pod.html) | 论文 2024 起，API 当前文档 | attention 内并行、接入条件 |
| [NanoFlow](https://www.usenix.org/conference/osdi25/presentation/zhu-kan) | OSDI 2025 | 算子级异步与资源互补 |
| [Prism](https://arxiv.org/abs/2505.04021) 与 [kvcached](https://github.com/ovg-project/kvcached) | 论文 v3 为 2026-06-10 | KV 物理显存弹性与稳定地址 |
| [TRE](https://arxiv.org/abs/2609.29160) | 2026-09-24 | 跨模型服务缺口校准 |
| [Talaria](https://arxiv.org/abs/2607.17181) | 2026-07-19 | 会话与 cache 局部性 |
| [Dynamo 分离指南](https://docs.nvidia.com/dynamo/v1.2.0/user-guides/disaggregated-serving) | v1.2.0 | 何时分离、何时保留 aggregated |
| [NIXL 架构](https://docs.nvidia.com/nixl/getting-started/architecture/) | 当前文档 | 未来 C++ KvTransport 适配 |
| [vLLM WSL 安装](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) | 当前文档 | 外部基线环境选择 |
| [Ray 多小模型共卡](https://docs.ray.io/en/latest/serve/llm/user-guides/fractional-gpu.html) | 当前文档 | fractional GPU 的含义和边界 |
| [SmolVLM 官方发布](https://huggingface.co/blog/smolervlm) | 2025-01-23 | 极小 VLM、图像配置与后端要求 |

## 附录 B 本次方案未做的验证

已完成当前源码和本机环境检查，以及上述官方资料、论文、模型卡和开源实现的交叉阅读。本次交付是升级设计文档，**没有修改推理代码，没有验证 MPS 启动，没有测得任何并行加速比，也没有加载第二套模型做显存实测**。

因此本文的模型组合、内存总额、chunk 和并发上限是可供实施的起点，须按 P0 和 P1 验证。旧文档与历史 benchmark 中的能力说明，均以当前源码实际状态为准。
