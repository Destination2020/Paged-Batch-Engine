# PBE 项目面试讲义

日期：2026-09-14。对应简历：多角色数据共享、多模态语义缓存、多 P/D 协调与生命周期。依据当前代码和最新实验报告整理，不把报告阅读当作重新完成所有验收。第一人称讲稿中的“我实现/主导”应按个人实际负责范围使用。

## 1. 开场：先说明解决的问题

### 60–90 秒版本

> PBE 是一个 C++/CUDA 分页推理引擎，我围绕多模态和角色拆分扩展了它的数据管理能力。
>
> 我关注的核心问题是：Vision、Prefill、Decode 变成独立进程以后，哪些数据可以共享、由谁持有、什么时候才能回收。如果每个进程各自管理全部资源，会出现重复加载权重、重复计算视觉特征和前缀，以及跨进程退出后数据失效的问题。
>
> 我把物理内存所有权与计算角色分离，让框架数据服务持有共享权重和 KV 池，计算进程通过授权使用 GPU view。上层实现多模态语义前缀查询，下层用租约、代际、完成事件和 COW 约束共享与回收；生产 Coordinator 支持同卡 1P1D 到 2P2D 的执行链路。
>
> 数据上，固定 KV 配置的 1P1D 权重共享让整套部署物理显存降低 43.05%；高复用图文负载下，实际 Prefill 计算量减少 88.60%。我也测了端到端代价，当前没有证明多 P/D 带来稳定吞吐提升，因此会明确区分显存、计算和服务性能收益。

### 面试官只给 20 秒

> 我把分页推理引擎扩展为多角色图文推理系统，核心是让不同进程安全共享视觉特征、KV 和只读权重。重点实现了语义前缀复用、动态页授权和资源生命周期；固定配置下显存降低 43.05%，并通过真实模型实验量化了共享的开销和边界。

不要从类名、通信库或测试数量开始。先让对方知道问题，再用一个请求把实现串起来。

## 2. 白板怎么画：三层足够

```text
                 请求：图片 + 文本 + generation + deadline
                                   │
                    Coordinator / RoleRegistry
                     选择角色、预留、取消与回收
                                   │
           Vision ──特征与请求 bundle── Prefill ──handoff── Decode
          PyTorch                    PBE C++/CUDA       PBE C++/CUDA
                                       │                 │
                 Data service / Node Agent / 数据身份与授权
                    特征/bundle       KV 页池       只读权重
                                   │
                   GPU / Host、预算、依赖恢复、完成事件
```

说明：这是逻辑关系，不是所有 tensor 都经过 Coordinator。KV 的 handoff 主要传身份和页元数据；数据访问受 owner 的授权与生命周期约束。同卡 shared 权重配置使用同一 allocation；跨卡要真实搬运，不是“整个框架零拷贝”。

内部语言执行再画一行：

```text
Scheduler → MixedBatchBuilder → Qwen2Model → Paged Attention / CUDA → Sample
    ↑                         sequence state / next step                   │
    └─────────────────────────────────────────────────────────────────────┘
```

框架既有 continuous batching/chunked prefill，也有显式 pd_prefill/pd_decode 主路径。不能因为 Scheduler 支持 batching，就声称所有多模态实验中都发生了合批。

## 3. 追踪一条真实请求：你必须能讲通的调用链

以“图片 X + 请描述图片”为例：

1. 客户端提交 request id、generation、图片/文本和超时时间。Coordinator 建立绝对 deadline，登记活动请求，后续阶段不重新获得完整超时预算。
2. Vision worker 完成预处理与编码，或命中已有视觉特征。返回特征身份及请求 bundle：特征可复用，当前 tokens、位置和 decode 起始信息属于本请求。
3. RoleRegistry 提供可用角色，Coordinator 选择 P/D。当前多 P/D `_place` 支持轮询或指定角色；不要将它说成默认运行 E2 的成本最优选路。
4. 调用 Prefill probe 确认输入与预算需求，然后按全局 D→P 顺序调用 worker 的 `pd_reserve`。实际 grant 由 worker 持有；第二步失败则回滚第一步，不把 status 查询当作预留。
5. P 调用 `pd_prefill`，使用真实 PBE forward 产生 KV 与生成起始状态，发布 handoff。正式性能请求 oracle_steps=0，校验用 Decode 在测量外进行。
6. D 获取 handoff 元数据并请求 `acquire_ipc_attach`，校验 provider/target incarnation、源 grant、metadata allocation generation、有效页与 token 范围。
7. D 先 `attach_external_shared_pages`，再 `restore_external_shared_request`，通过 `RunPDDecode` 从恢复的 KV 开始生成。页表有效不代表允许随意写共享尾页；增长预算与 COW 仍独立管理。
8. 完成、超时或外部取消后，停止启动新的计算阶段，按实际 GPU 完成情况释放请求、attach、reservation 和 handoff；保留明确的常驻权重/模型基线。

主阅读入口：[pd_runtime.py](../python/pbe_roles/pd_runtime.py)、[language role](../demo/pbe_vlm_language_role.cpp)。不要把所有 prefix-cache、跨卡恢复与显式 PD 实验描述成每次请求都会完整经过的同一条代码分支；不同入口需要分别追踪。

## 4. 简历第一条：为什么把物理所有权移到框架

### 推荐回答

> 计算进程的生命周期不应直接决定可共享数据的生命周期。P 结束后，D 或后续请求可能仍要使用 KV；多个语言进程还可能使用相同模型权重。因此框架持有 allocation，worker 持有使用授权和逻辑 view。共享带来的难点是身份、发布时机和回收，而不只是拿到另一个进程的地址。

### 四类概念必须区分

| 概念 | 回答的问题 | PBE 中的例子 |
| --- | --- | --- |
| 内容身份 | 这是什么计算结果？ | ContentId、语义前缀与媒体身份 |
| 表示身份 | 它按什么格式存储？ | RepresentationId、dtype/layout |
| 物理身份 | 它在哪一代 allocation 里？ | owner incarnation、allocation id、generation |
| 使用授权 | 谁可以在当前生命周期内使用？ | lease、源 grant、目标 attach grant |

内容相同不保证表示兼容；页编号相同不保证是同一内存；目录里存在数据也不保证当前 acquire 一定成功。见 [data_ref.h](../infMain/include/data/data_ref.h) 和 [content_registry.h](../infMain/include/data/content_registry.h)。

### 权重共享具体怎么做

> Data service 分配一份 GPU 权重 slab，按有界 staging 上传并校验，完成后才发布 ready。P/D 在初始化时导入同一个 allocation，算子通过稳定 offset 绑定 GPU view。worker 不先加载一整份私有 GPU 权重再替换，所以要同时检查启动峰值和稳态占用。

实现中需要解释：

- 权重身份包含冻结内容、布局、dtype、设备等条件，不能只用文件路径或 tensor 名。
- `Layer`、`Matmul`、`Qwen2Model` 的权重绑定进入真实算子；capsule 维持 import 生命周期，borrowed view 析构不能 cudaFree owner 的 allocation。
- tied embedding/lm_head 保留别名，logical view bytes 可以大于 unique physical bytes，不能重复计费。
- workspace、激活、CUDA context、调度与采样状态不因共享权重而消失。
- CUDA IPC 不是硬件只读保护；当前范围是可信 worker。导出者崩溃需要 fail-stop，不能继续使用失效映射。

入口：[shared_weight.h](../infMain/include/data/shared_weight.h)、[shared_weight.cpp](../infMain/source/data/shared_weight.cpp)、[shared_weight_binding.h](../infMain/include/model/shared_weight_binding.h)、[layer.cpp](../infMain/source/op/layer.cpp)、[qwen2.cpp](../infMain/source/model/qwen2.cpp)。

### 高频追问

**为什么不直接放在同一个进程？**

统一进程是有效基线，通常更容易合批，也少一些上下文和通信成本。拆分让角色拥有独立执行状态、部署配置和可明确管理的数据交接边界；是否值得取决于负载。PBE 没有证明同卡多进程普遍更快，因此统一模式仍保留。

**43.05% 是权重压缩率吗？**

不是。这是同一 H20-3e、Qwen2.5-VL-3B BF16、1P1D、固定 128 KV blocks 下，整套部署观测物理显存从 16872 降到 9608 MiB，约少 7.09 GiB。模型 dtype 和数值合同没有靠压缩改变；不能解释成权重 tensor 缩小 43.05%。

**为什么引用计数为零还不能立即释放？**

Host 侧请求结束不证明 GPU kernel/DMA 已结束。逻辑引用和物理完成是两个条件；还在使用内存的计算或传输必须完成，或经过可靠隔离，才能复用内存。

**为什么需要 generation？**

allocation id/page slot 可以复用，旧请求迟到 release 可能误释放新对象。代际与完整 lease token 用来拒绝旧操作。operation-id 重放解决“服务已 acquire、回复丢失后重试”的重复持有问题，但这不是网络 exactly-once 承诺。

## 5. 简历第二条：多模态缓存到底复用了什么

### 推荐回答

> 我区分了两种复用：视觉特征取决于图片和编码条件；KV 还取决于完整前缀和位置。相同图片可以共享一次视觉编码，但不能直接复用不同问题或不同对话历史的全部 KV。

| 请求 | 视觉特征 | KV 前缀 |
| --- | --- | --- |
| 图片 X + 描述图片；图片 X + 数人数 | 编码条件相同时可复用 | 只复用实际相同的连续前缀 |
| 图片 X 与图片 Y，文本和 placeholder 相同 | 不应混用 | 对应多模态区间不能当成同一语义 |
| 同一图片放到不同对话历史中 | 视觉编码输入相同则可复用 | 前文/位置变化可能使 KV 不兼容 |
| 同一图片但 processor 或模型版本变化 | 重新校验身份 | 不得继续套用旧 KV |

### 本项目的落点

Vision 特征身份与当前 request bundle 分开；缓存查询键不能仅取视觉 placeholder token。`multimodal_prefix_key.h` 提供内容参与 radix key 的适配，其中搜索键与模型输入 tokens 分离，不能把 hash 编码当作真实 token 送入模型。真实调用还要结合模型命名空间、媒体位置与请求构造链检查，不能只背一个 hash 函数。

建议在面试前亲自读 `pbe_vlm_language_role.cpp` 的 semantic key 构造、probe、prefix lookup 和实际 computed tokens 记录，追到 `KVCacheManager`。这是“会讲抽象”与“真的能解释实现”的分界。

### 常见问题的回答

**为什么当前页 tokens 相同还不够？**

同一页的 hidden/KV 依赖前面的上下文。必须匹配前缀链及位置语义，不能把不同历史中的相同后缀页直接复用。

**有后面的缓存页，但中间缺一页怎么办？**

先确认能恢复所需缺页，或退回安全连续边界。目录中的孤立页不能凭存在就计为完整连续命中；实际 attention 需要正确的历史范围。

**最长前缀命中之后就可以直接返回答案吗？**

不能。命中只省掉对应前缀的计算，还要正确获取首个输出所需的 logits/生成起始状态，并继续当前请求的采样。显式 PD handoff 携带首 token/position 等状态；普通前缀缓存不能直接复制另一个请求的生成结果。

**共享尾页为什么要 COW？**

多个请求读同一前缀没问题，向未满尾页追加不同 token 会互相覆盖。写前复制必要尾页并更新当前请求页表；完整共享前缀仍复用。复制量、引用变化和幸存分支输出都要测试。

**Prefill 计算量减少 88.60% 为什么吞吐还下降？**

这组 B2 的模型侧首步变快，但完整请求还包含 Vision/预处理、bundle/RPC、Decode、缓存维护和回收等工作。实测完成窗口吞吐下降 10.50%。当前归因保留未解释区间，不能确定地甩给 Python、IPC 或某个 kernel；模型时间不是端到端时间。

## 6. 简历第三条：多 P/D、调度与恢复

### 两个层次的调度

| 层次 | 负责什么 | 项目中的准确表述 |
| --- | --- | --- |
| 角色选择 | 由哪一个 worker 执行 | E2 有数据位置成本选路及独立多 worker 实验；当前 `ProductionPDCoordinator._place` 默认轮询/指定角色 |
| 引擎内调度 | 本轮哪些请求、多少 token 进入计算 | Scheduler 的 token budget、continuous batching、chunked prefill、preemption |

面试不要说“当前 2P2D 默认按全局数据成本最优调度”。成本模块和多 P/D 协调均存在，但不同实验路径的策略要按实际代码解释。

E2 可用如下估计说明思路：排队时间 + 缺失数据字节/实测带宽 + 计算成本 + RPC/布局开销。只在模型兼容、状态足够新鲜且可能准入的候选中比较；最终 reservation 才提供资源保证，快照不能替代实际预留。入口：[role_placement.h](../infMain/include/serving/role_placement.h)、[role_placement.cpp](../infMain/source/serving/role_placement.cpp)。

### 2P1D 为什么比 1P1D 难

> D 不能只认识启动时某一个 P 的 slot 白名单。它需要按当前请求验证源 provider、物理代际、源 grant、metadata allocation 和有效页面，拿到目标消费者自己的 attach 授权后才安装页表。否则不同 P 的局部页编号相同，或者 P 重启复用了编号，就可能读取错误内容。

1P2D 主要增加消费者与尾页隔离；2P1D 主要检验多个来源的动态授权；2P2D 同时覆盖两类问题。证明方式是每条允许的 P→D 路径真实执行，不能只展示四个 PID。

### 资源预留与取消

> 当前 Coordinator 按 D→P 顺序拿到 worker-owned reservation；后续消费对应授权。部分失败回滚，避免先产生大量 KV 才发现 D 完全没有增长空间。超时始终沿用原绝对 deadline；外部取消按 request/generation 定位当前角色，阻止后续 stage，已提交的 GPU 工作按安全条件收敛。

cancel accepted 只表示控制面接受取消，不等于 GPU 立即停止或内存已经释放。当前模型执行点检查 deadline/cancel，不是任意 CUDA kernel 的硬抢占。

**为什么一个 D 取消不能直接取消物理传输？**

传输 flight 可能还有其他 waiter；需要分离物理操作和各请求等待关系。最后一个 waiter 退出也要按已提交传输的完成状态处理，不可立即复用目标。

### 压力恢复为什么要考虑依赖

LRU 回答“多久没用”，恢复依赖回答“丢了以后还能不能正确恢复”。视觉特征、共享完整前缀、私有尾页的恢复条件不同；保存配方需要实际可用的媒体、模型/processor 版本、位置和必要状态，不能仅说“有模型就能重算”。

GPU→Host 通常需先预留目标与 staging，完成 copy/校验并发布 Host 副本，再撤销 GPU 可服务状态；最后一份必要副本不能随意淘汰。Host 满、不可恢复或操作未完成时保留/背压/拒绝。入口：[recovery_dependency.h](../infMain/include/cache/recovery_dependency.h)、[page_migration.cpp](../infMain/source/cache/page_migration.cpp)。

数据：该压力场景重算从 92 降至 12 tokens，但总延迟从 751.75 增至 786.23 ms。这证明少重算和正确恢复，不证明恢复更快。

## 7. 性能速查：只背带条件的数据

| 项目 | 条件与样本 | 当前结果 | 禁止扩大成 |
| --- | --- | --- | --- |
| 权重共享 | H20-3e；Qwen2.5-VL-3B BF16；1P1D；固定 128 blocks；每臂 5 个 oracle-free trial | 16872→9608 MiB，-43.05%；周转吞吐 -7.69% | 权重压缩、普遍加速 |
| 多模态缓存 | 90% 图片复用、长上下文、280px；并发 1；每臂 5 trials/50 请求 | prompt 实算 15350→1750，-88.60%；Vision 50→5；模型 TTFT 135.95→64.66 ms | 客户端首字延迟降低 52.44% |
| 缓存端到端 | 同上 | 0.3159→0.2827 req/s，-10.50% | 吞吐提升 |
| 数据位置策略 | 同一两 worker/容量；每策略 5 trials/55 请求 | 0.3710→0.3955 req/s，+6.60% | 所有多 P/D 默认路径普遍提升 |
| 依赖恢复 | 固定压力场景，5 trials | 重算 -86.96%；延迟 +4.59% | 无成本恢复/恢复加速 |
| 多 P/D | private/shared × 5 拓扑 × 3 workload × 3 load × 5 窗口 | 450 窗口、3150/3150 请求；最佳 split 增幅很小且范围重叠 | 稳定吞吐提升、任意拓扑最优 |

N6 private 最佳 split 相对 unified 为 +0.18%～+0.82%，shared 为 -0.13%～+1.09%。保留低收益和负收益；旧单窗口 +8.62%～+18.67% 已被替代。

N6 近饱和命名对应有限到达列车：11 请求、200 ms 间隔、2 秒到达后 drain。不是长时间稳态饱和测试；没有可靠 p99 或显著性结论。实际正式多模态链路受上游 Vision 串行交付影响，未观察到 language 合批；单独的非计时探针证明引擎具备 5-request batching 能力。

固定显存容量探针只证明外部分配夹点：private 1024/2048，shared 12288/16384 blocks；最多实际访问 48 页。不能说最大并发增加 12 倍。

非流式多模态实验没有真实客户端首 token/逐 token 可见时刻，客户端 TTFT/ITL 和 token SLO goodput 为 N/A。既有文本 HTTP/SSE 支持不能自动补足这组实验的流式测量。

## 8. 正确性怎么证明

回答结构：机制证据、数值证据、故障证据、性能证据分别提供。

- 机制：独立 PID，同一物理权重 allocation/独立 importer，真实 KV handoff，动态授权与实际 attention 使用，Decode 不重算已完成前缀。
- 数值：当前 20-case 按冻结 BF16 同历史 logits/top-2 合同接受；19 个序列全等、1 个首次分叉有证据，156/160 token 相同。不能说 20/20 完全逐 token 一致，更不能说整个输出语义“差不多”就接受。
- 故障：运行中取消、绝对 deadline、原子容量竞争、旧代拒绝、P 正常退出后消费者继续使用、异常回收。owner 崩溃和普通 P 结束是不同事件。
- 工具：本轮受影响 C++ 71 passed/2 NCCL-disabled skip、Python 8 passed；三个受检 language 进程 sanitizer 0 error/0 leak。它们是具体测试覆盖，不证明系统无 bug；没有重跑完整历史套件就不要声称做过。

如果被问 BF16 分叉：先说明在完全相同历史输入上比较 logits、候选 token 与 margin。首次分叉后历史已不同，后续序列差异不能当作同输入误差。使用冻结的可执行规则，不能事后调阈值匹配结果。

如果被问 hash：它证明当前文件与记录的实验版本是否一致，不证明实验公平、样本充分或输出正确；这些由独立门禁和实验设计证明。

## 9. 三个可以深入讲的工程故事

### 故事 A：会传 KV 不等于会管理 KV

问题：最早的 P/D 演示容易让生产请求持有所有资源，生产者结束后数据能否继续使用不清楚。

设计：把物理 pool 与请求页表拆开，数据服务维护 owner/grant，D 拿独立 attach 授权。共享前缀与私有尾页分开。

验证：生产角色正常退出、另一个 D 继续计算、尾页 COW、旧代句柄拒绝与资源回稳。强调生产者不是物理 exporter；后者失败不保证旧映射继续可用。

### 故事 B：看起来公平的性能实验测错了对象

问题：E4 计时中请求了 oracle_steps=16，使 Prefill 额外 fork/Decode 校验分支后才交接。

整改：oracle 移到独立数值阶段，正式调用 oracle_steps=0，恢复数值对照但不让校验计算污染性能窗口。

结果：显存节省仍为 43.05%，oracle-free 周转吞吐仍下降 7.69%。因此发现测量问题不意味着已经找到负收益全部根因，后续仍需看关键路径。

### 故事 C：五个请求不是五次独立实验

问题：旧 N6 一窗口五请求被当作 five trials，最佳拓扑提升显得较大。

整改：把请求数和 trial/window 分开，做 450 个独立窗口，保存配置、预热、到达及 drain 边界。

结果：最佳 split 与 unified 基本持平，trial 范围重叠。工程贡献是获得可信边界，而不是维护一个好看的加速数字。

## 10. 面试中最危险的表述替换

| 不要说 | 改为 |
| --- | --- |
| 我做了单卡多进程，所以更快 | 我实现并验证了多角色数据共享，吞吐收益依赖负载，当前矩阵未证明稳定提升 |
| 全链路零拷贝 | 同卡特定路径通过 GPU view 共享；跨卡和 GPU/Host 需要真实传输 |
| 全局缓存完全独立于引擎 | 物理所有权移到框架；KV 页表、radix 和执行管理仍需要引擎侧协作 |
| 引用计数解决了所有回收 | 逻辑引用加物理完成条件共同约束回收，未知完成状态需要隔离 |
| 模型侧 TTFT 就是用户首字延迟 | 本组非流式实验只有模型侧 TTFT，客户端 TTFT 无有效观测 |
| 数据位置调度已经是多 PD 的默认最优策略 | 独立 E2 路径验证成本选路；当前多 PD 默认轮询/指定，不宣称全局最优 |
| 我已经查清所有负收益根因 | 已校正实验、关联请求和量化残差，部分底层原因仍未分解 |
| 通过门禁说明可以生产大规模使用 | 在单节点 TP=1 的指定模型/拓扑/负载范围完成验证 |

## 11. 面试前阅读与演示准备

按以下顺序阅读，每个入口至少能解释输入、输出、资源持有和失败分支：

| 顺序 | 代码/证据 | 准备目标 |
| --- | --- | --- |
| 1 | [pd_runtime.py](../python/pbe_roles/pd_runtime.py) | 画出 submit/reserve/cancel/finally 回收状态机 |
| 2 | [pbe_vlm_language_role.cpp](../demo/pbe_vlm_language_role.cpp) | 找到 pd_reserve、PDPrefill、PDDecode、RunPDDecode、语义查询 |
| 3 | [ipc_pool.h](../infMain/include/data/ipc_pool.h)、[node_agent.cpp](../infMain/source/data/node_agent.cpp) | 解释源 grant、目标 attach、generation 校验 |
| 4 | [shared_weight_binding.h](../infMain/include/model/shared_weight_binding.h) 与 [shared_weight.cpp](../infMain/source/data/shared_weight.cpp) | 从 loader 跟到真实算子 GPU view 和析构 |
| 5 | [scheduler.cpp](../infMain/source/serving/scheduler.cpp)、[kv_cache_manager.cpp](../infMain/source/base/kv_cache_manager.cpp) | 批处理、页表、共享/COW 与请求释放 |
| 6 | [N1–N6 完成报告](V4_N1_N6_COMPLETION_20260914.md) | 准确说出当前能力和数值边界 |
| 7 | [简历指标](data_flow_evidence/v4/performance_characterization/RESUME_METRICS.md)、[拓扑报告 v2](data_flow_evidence/v4/performance_attribution_multi_pd/TOPOLOGY_REPORT_V2.md) | 每个数字对应哪个基线、窗口和样本 |
| 8 | [归因报告](data_flow_evidence/v4/performance_attribution_multi_pd/N2/ATTRIBUTION_REPORT.md)、[重放命令](data_flow_evidence/v4/performance_attribution_multi_pd/REPLAY.md) | 解释已知原因、未解释残差和复现条件 |

现场演示提前准备一条小型真实 1P1D 或 1P2D 请求，展示 request ID、P/D PID、共享 allocation、attach grant、computed tokens 和结束后的资源状态。不要临场启动完整 450 窗口矩阵。用固定模型/图片及已验证配置，保留日志和失败返回。

个人贡献准备单独写下三列：亲自完成的设计/实现、工具或协作辅助的部分、亲自验证与能解释的部分。可以说明使用 AI 辅助编码与分析，但不要将未掌握的模块说成独立完成。面试的可信度来自能追踪代码、解释取舍并复现证据。

## 12. 结束时如何总结与回答下一步

> 这个项目让我从“把模型跑起来”深入到“计算与数据怎样协作”。最明确的成果是跨角色物理共享、多模态缓存正确性和请求生命周期。性能上，我证明了显存与重复计算收益，也通过更严格的实验否定了部分早期吞吐判断。下一步会先解决完整多模态链路中的阶段等待与合批机会，并补齐关键路径残差，随后再根据瓶颈考虑跨节点传输。

不要把未来工作说成当前已实现；也不需要在简历里塞进所有技术名词。围绕一份数据的产生、身份、授权、使用和回收讲透，足以支撑深入追问。
