# CUDA MPS 多进程与共享 KV Chunk：实现进展

实施日期：2026-10-11。依据 [升级研究档案](cuda-mps-multi-model-upgrade-research-2026-10-11.md) 和 [Chunk V1 设计](<../多模型共享 KV Pool 的 Chunk 架构设计 V1.md>)。验证模型目标按本次要求调整为 **Qwen2.5-0.5B-Instruct + Qwen2.5-1.5B-Instruct**。

## 已接入的执行路径

```text
HTTP/SSE API（无模型、无 CUDA context）
  ├─ model 路由 → Qwen worker A（独立进程、权重、workspace、stream）
  └─ model 路由 → Qwen worker B（独立进程、权重、workspace、stream）
                         ↓ CUDA IPC 映射、Chunk 控制 RPC
             长生命周期 KV owner（唯一 CUDA allocation、唯一 free list）
```

实现了一模型一个 mixed worker 的多进程后端。`mps_process` 模式在任何 CUDA worker 启动前设置可见 GPU UUID、MPS pipe、线程比例和客户端显存上限，worker READY 后要求所有模型进程注册到同一个 MPS server。控制程序不存在、连接失败、身份不匹配或注册失败，启动直接失败。`process` 是显式的普通多进程对照模式，不会由 MPS 模式静默回退而来。

新增模块：

- `GlobalChunkAllocator`：字节区间 best-fit、非二次幂对齐、合并、模型硬配额、epoch/租约代次校验。配额统计完整租入的 chunk，包括保留和退役页。
- `shared_kv_owner`：专用 `cudaMalloc` 一次分配 Arena，导出 CUDA IPC handle，串行处理注册、租赁、归还和指标；不同模型只复用归还后的物理容量。
- `SharedKVArenaClient`：每 worker 只映射一次，检查 GPU UUID、epoch 和租约范围；映射 lifetime 被所有 layer allocator 共同持有。
- `SharedKVPageGroup`：plain BF16 的 all-layer page-major GQA 布局；先填已有 chunk，目标 chunk 大小默认 2 MiB，空间/碎片/配额不足时尝试一页的小 chunk；所有层页预留失败时回滚。
- `MultiModelOnlineEnginePool`：统一 `model` 路由、模型列表、返回实际模型身份，以及按 handle 的模型身份路由取消，避免不同 worker 的相同 request ID 混淆。
- `deploy_multi_model.py`：容量预检、顺序加载/warmup、GPU UUID、worker 身份/MPS 注册检查、进程监控、整个 GPU domain 失败退出及按 importer → exporter 的停止顺序。
- 探针和验证器：CUDA IPC memory/event、异构 KV 页、GPU fence、区间重叠提示、独立 KV 与共享 KV 的输出对照、中文 SSE 和取消。

现有 Qwen `PagedKVRuntime` 的 kernel 接口保留。物理 page 从 `offset / half_layer_bytes` 得到稀疏 block ID；各层使用同一个 page ID，层 l 的 K/V tensor 基址分别为 `Arena + 2*l*half_layer_bytes` 和 `Arena + (2*l+1)*half_layer_bytes`。这样与目前复用第 0 层 block table 的 batch metadata 一致。各层 views 的尾部范围均限制在同一 IPC allocation 内，chunk ID 不作为 kernel block ID。

页释放记录所在 compute stream 上的 CUDA event。`RETIRED` 页在 query 完成前不能再次分配；完整页只有所有层都结束引用并完成 fence 后才能重用。全部页变空才把 chunk 归还 owner。owner 不会因超时或 client 消失而直接回收 GPU 正在使用的空间。一个 worker/owner 退出，supervisor 停止整个 GPU domain；重启产生新的 epoch 和 handle。当前控制协议限同 UID、700 权限目录内的可信本地 worker，RPC 不向公网开放。

中文流式输出的 BPE 半字符原先会令 ZMQ JSON 序列化抛异常；新增 UTF-8 边界缓冲，尾部未完成字符在结束时转换为 replacement character，并增加回归测试。

双模型长 prompt 验证发现既有 chunked prefill 的同步缺口：中间块没有采样行时，未经过 sampler 的 D2H 等待，下一步会覆盖仍被 GPU 异步复制使用的 pinned metadata。现已在这条路径等待当前模型的 compute stream，使 staging buffer 可以安全复用。此等待只影响该模型进程；两个 worker 仍能独立提交。后续可改成 metadata staging ring/event 以减少主机等待。

## 编译与环境探测

```bash
cmake -S . -B build-mps \
  -DQWEN2_SUPPORT=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DKUIPER_ENABLE_NCCL=OFF
cmake --build build-mps --target serving_qwen shared_kv_owner mps_kv_probe cuda_ipc_event_probe test_llm -j4
```

本机需提供 nlohmann_json 和 GTest 的 CMake package。实际构建使用 `-Dnlohmann_json_DIR=/home/nowcoder/miniconda3/share/cmake/nlohmann_json -DGTest_DIR=/usr/local/lib/cmake/GTest -DKUIPER_AUTO_ALIGN_CONDA_PACKAGES=OFF`。CMake 将 header-only JSON 的 include root 限制为自身目录，避免 Conda prefix 将不匹配的 Abseil 头文件引入系统 re2/Abseil 库的编译过程。

Python 启动器和探针需要 `pyzmq`；模型导出需要已有的 PyTorch、Transformers。

```bash
python tools/serving/deploy_multi_model.py --config configs/qwen25-dual-mps.json --check
python tools/serving/deploy_multi_model.py --probe --backend process
# 只有 MPS daemon 可用时才运行下面两条：
python tools/serving/deploy_multi_model.py --probe --backend mps_process --start-mps
python tools/serving/deploy_multi_model.py --config configs/qwen25-dual-mps.json --start-mps
```

不加 `--start-mps` 时连接环境变量中指定的已有 daemon；只会停止本次自行启动的 daemon。每次启动使用唯一 runtime 目录，日志与探针 JSON 保留在该目录。无需设置 EXCLUSIVE_PROCESS 才能做预检；工具也不会尝试修改 GPU compute mode。`--check` 是文件与容量静态检查，不能证明 MPS server 可初始化或 CUDA client 可注册；实际环境验收需运行 `--probe`。

`mps_kv_probe` 的 globaltimer 起止时间只表示 kernel 生命周期区间。普通 CUDA context 的抢占/切换也可能使区间重叠，所以结果明确保留 `mps_kernel_overlap_verified=false`。MPS 注册和实际 GPU 时间线是不同验收项；用 Nsight Systems 才能完成时间线检查：

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  -o /tmp/pbe-mps-overlap \
  python tools/serving/deploy_multi_model.py --probe --backend mps_process --start-mps
```

CUDA memory/event 生命周期遵循 [CUDA Runtime IPC API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html)；MPS 环境变量见 [NVIDIA MPS 文档](https://docs.nvidia.com/deploy/mps/appendix-environment-variables.html)。

## 两套 Qwen2.5 的启动与验证

配置已保存于 [qwen25-dual-mps.json](../configs/qwen25-dual-mps.json)。两套下载均已完成，并已导出 BF16：0.5B 的文件为 992259876 字节，1.5B 为 3095817252 字节。权重路径均指向本地目录；下列导出命令用于以后重新生成。

```bash
python tools/export_qwen2.py models/Qwen2.5-0.5B-Instruct/model.bf16.bin \
  --hf models/Qwen2.5-0.5B-Instruct --dtype bf16
python tools/export_qwen2.py models/Qwen2.5-1.5B-Instruct/model.bf16.bin \
  --hf models/Qwen2.5-1.5B-Instruct --dtype bf16
python tools/serving/validate_multi_model.py \
  --config configs/qwen25-dual-mps.json --backend process --test-worker-failure
# 在 MPS 合格环境重复验证：
python tools/serving/validate_multi_model.py \
  --config configs/qwen25-dual-mps.json --backend mps_process --start-mps
```

独立 Transformers 参考诊断（需要含 PyTorch / Transformers 的 Python 环境；逐字差异会返回退出码 2）：

```bash
python tools/serving/compare_transformers.py \
  --config configs/qwen25-dual-mps.json \
  --validation docs/benchmarks/mps-shared-kv-2026-10-11/validation.json \
  --output /tmp/pbe-transformers-reference.json
```

示例总预算：私有资源 1536 + 3584 MiB，共享 Arena 512 MiB，另留 768 MiB 未承诺余量。Arena 只在全局计费一次。启动器从 BF16 header 检查 GQA 几何、上下文与最大请求形状的 chunk 字节数，验证两模型的总 KV 需求能放入 Arena。

`--private-memory-budget-mib` 对 PBE CUDA allocator 的物理分配实施硬上限；缓存中的 allocation 继续计费，缓存复用不重复计费，超过上限的新分配失败。两个 worker warmup 后实测该 allocator 的保留字节分别为 1337288324 和 3661374340，约 1275.34 / 3491.76 MiB，均在配置范围内。context、驱动和绕过该 allocator 的库内部分配不属于这个计数，靠全局余量及 MPS 客户端限制覆盖；这不是普通进程模式下所有 GPU 内存的硬隔离。

supervisor 也使用 owner 的 `cudaMemGetInfo` 检查余量并记录加载前后的空闲显存差值，标记 `private_gpu_delta_is_estimate=true`。WSL 实测第一个 worker 的差值会为 0，所以启动检查同时使用精确 allocator 账目，不能只依赖差值。动态加载后的分配仍受 allocator 上限约束。

示例每模型活跃请求初始设为 1，prompt ≤1024、输出 ≤256、prefill chunk 128。这样先验收 1+1 的跨模型进程并发。`max_active_requests` 可以提高；既有 BF16 mixed batching 本身会因 batch 形状/到达时间改变 greedy 结果，在原独立 KV 路径也复现了代码题的变化。验证器固定每模型 1 个活跃请求，使严格输出对照使用相同 batch 形状；2+2 以上应另外检查 logits 数值与采样决策，不把文本差异直接归咎于共享池或 MPS。

```bash
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen25-1.5b","messages":[{"role":"user","content":"用一句话介绍北京"}],"max_tokens":64}'
```

本机可用的双模型服务启动命令：

```bash
python tools/serving/deploy_multi_model.py \
  --config configs/qwen25-dual-mps.json --backend process
```

`/metrics` 按模型展示 `kv_memory`：全局 Arena 总字节、模型实际租入 chunk 字节与模型容量上限。多个 worker 的 `global_arena_bytes` 指向同一物理分配，不可相加当作总显存。容量上限也不是当前已分配的 KV 字节。

## 验证与当前边界

- RTX 4060 Laptop、WSL2、CUDA 12.6、驱动 566.26 的真实 exec-spawned 双进程已通过 memory IPC、pending IPC event 等待及 4096 字节异步写入可见性测试。
- 两种 GQA 几何（24 层/64 维、28 层/128 维）的 K/V payload 比较与延迟回收探针通过，结束后 `lease_count=0`、Arena 全部归还。
- CPU 所有权测试覆盖不同模型跨布局复用、非二次幂对齐、碎片、硬配额、错误 epoch/身份、重复释放、4000 次随机分配与区间/账目检查。
- 两套指定权重已通过普通 CUDA 双进程共享 KV 验证：6 个并行提交请求覆盖中文、代码与跨多个 prefill 块的长 prompt；输出逐字匹配各模型的 PBE 独立 KV 基线。SSE、取消隔离、未知模型拒绝和 Arena 完全归还均通过。结果见 [双模型验证记录](benchmarks/mps-shared-kv-2026-10-11/validation.json)。
- 在上述检查后强制 `SIGKILL` 一个 worker：supervisor 以失败退出，另一个 worker 和 owner 均已停止；报告 `worker_failure_stops_gpu_domain=true`。此阶段采用整个 GPU domain 退出，由外层重新启动；未实现自动保留请求重试。
- C++ 回归共 53 项：50 项通过、3 项因第二张 GPU／NCCL 缺失跳过；Python 部署契约测试 4 项通过。现有 attention 测试手动销毁由 `CudaConfig` 持有的 stream 导致重复销毁，已去掉多余销毁。验证概要见 [verification.json](benchmarks/mps-shared-kv-2026-10-11/verification.json)。
- IPC 探针结果见 [probe.json](benchmarks/mps-shared-kv-2026-10-11/probe.json)。实际执行重叠指标是生命周期提示，报告没有将其宣称为 MPS kernel 并发。
- 首次验证时本机没有 `nvidia-cuda-mps-control`，严格 MPS 模式预检明确拒绝启动。随后按用户要求补装了官方二进制；当前 server 初始化及 client 注册仍失败，详见下面的补装记录。尚未完成 MPS server 注册、Nsight 时间线和 MPS 收益验收。
- 额外使用本地 Transformers BF16 eager、相同显式 ChatML prompt 和 greedy 24 token 做了参考对照。6 个输出中仅 1 个逐字一致；共享与独立 PBE 路径一致不等于 Transformers 数值一致。参考差异记录见 [transformers-reference.json](benchmarks/mps-shared-kv-2026-10-11/transformers-reference.json)，这项仍需检查 tokenizer、算子与 logits。
- 当前仅 plain BF16、单 GPU、all-layer Full GQA，prefix cache 强制关闭；没有活跃 KV 搬迁、动态 Arena 扩容、跨模型 KV 内容复用、共享权重、Graph capture 或同模型 P/D handoff。
- 全局软份额公平调度与确保全部活跃请求完成的最大输出容量预留尚未接入。hard limit 限制实际租约容量；动态接纳的 free-block 数是模型容量提示，不是无争用保证。首轮配置 1+1、小上下文用于避免这项未完成工作影响验收。

后续优先：在可运行 MPS 的环境完成注册与 profiler 验收；定位 PBE 与 Transformers 的参考输出差异；完善请求级最大输出空间预留与共享池压力接纳；再接入同模型 P/D 的 IPC connector 和更大 batch 的数值/性能测试。

## MPS 工具补装与实测（2026-10-11）

按用户要求，已从 NVIDIA 官方 Ubuntu 22.04 仓库下载 `nvidia-compute-utils-565_565.57.01-0ubuntu1_amd64.deb`，核对仓库 SHA256 后仅提取 `nvidia-cuda-mps-control` 和 `nvidia-cuda-mps-server`。文件保存在 `/opt/pbe-mps/565.57.01/bin`，命令链接位于 `/usr/local/bin`；版权文件和来源清单保存在同一安装目录。没有通过 apt 安装该包及其 Linux 驱动库依赖，dpkg 包清单保持不变，WSL 的 `libcuda` 和 Windows 566.26 驱动保持原样。此方式是独立工具安装，不是完整 Linux 驱动包安装。

安装后实测：

- 控制程序 `-h` 正常；隔离的 control daemon 能启动，`get_server_list` 可交互。
- `--probe --backend mps_process --start-mps` 的普通 CUDA IPC memory/event 测试通过，但两名 KV client 没有注册到 MPS server；严格验收失败，未回退为普通进程的 MPS 成功报告。
- 额外手动执行 `start_server -uid 0`：server 进程退出码 1，server list 仍为空。strace 确认实际执行了已安装的 server，并加载了 WSL 提供的 `libcuda.so.1`；没有证据把该失败归因于找不到 server 二进制。退出前没有输出具体 CUDA 错误码，尚不能仅凭日志确定内部失败 API。
- 所有测试 daemon、探针和 owner 均已停止；之前停止的双模型服务没有重新启动。

因此，工具安装完成，但当前 WSL2/驱动/工具组合仍未具备通过验证的 MPS 运行能力。MPS 客户端逻辑也依赖驱动中的 `libcuda.so`，见 [NVIDIA MPS 架构文档](https://docs.nvidia.com/deploy/mps/latest/architecture.html)。保留 WSL 驱动的安装方式遵循 [NVIDIA WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) 中不要安装 Linux 显示驱动的要求。

证据：[安装与探针结果](benchmarks/mps-shared-kv-2026-10-11/mps-installation.json)、[server 初始化记录](benchmarks/mps-shared-kv-2026-10-11/mps-server-initialization.json)、[server 驱动加载轨迹](benchmarks/mps-shared-kv-2026-10-11/mps-server-driver-trace.log)。首次预检报告保留，反映的是补装前的历史状态。

后续详细诊断已定位具体失败点：两个版本的 MPS server 均成功查询内部导出表，然后进入当前 WSL libcuda 中固定返回 999 的入口，最终退出码 1。普通 CUDA client 可创建 context，但没有连接 MPS socket。完整调用链、GDB 记录、版本对照和限制说明见 [MPS WSL 诊断报告](cuda-mps-wsl-diagnosis-2026-10-11.md)。
