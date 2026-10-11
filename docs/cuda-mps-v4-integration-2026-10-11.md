# V4 基础上的 CUDA MPS 与共享 Chunk KV 集成

## 基线与迁移

V4 基线为 `40afd32553d356816525c2f9b05696b923d6ba64`（`feat: complete V4 multi-role validation`），包含远端六个新增提交。首次 MPS/Chunk 实现 `67faf2cd68ca80da651b53711d35eefa2b5357ad` 基于旧版 `f3597a7`，保留在原分支，仅用于追溯。本分支从 V4 出发迁移该实现，未覆盖 V4 的多模态、数据服务、分角色、恢复依赖、共享权重和架构文档。

解决十处文本冲突后，额外完成以下语义适配：

- Chunk-backed `BlockAllocator` 初始化 V4 的 pool ID、allocation generation、compute/I/O pins 和引用计数。租约释放之前不归还物理页；最后一个引用释放后仍经过 CUDA event fence。物理位置再次分配会更新代次，拒绝旧句柄和跨层句柄。
- 保留 V4 外部固定 KV 池的授权页集合与回收规则。共享 Chunk 页不会进入该池的本地 free queue。
- 保留 V4 显式 blocks-per-layer 和外部池初始化。共享 Chunk 模式拒绝同时配置外部 data-service KV 池、固定 block 数或 host migration，避免两套物理所有权混用。
- ZMQ 返回 request ID 保持 V4 的 `int64_t` 读取类型，避免代次编码 ID 截断。
- CUDA allocator 保留 V4 的显式缓存清理，同时对分配、归还、预算统计及清理加锁；清理后实际占用归零，预算限制继续生效。
- 双模型验证发现显存压力使实际 workspace 缩到 89 token，但显式 128-token 配置仍被 warmup 使用。现在显式/自动调度均受实际 workspace 上限约束；已经由 IPC owner 分配的 KV 不再重复预留私有 KV 容量。
- 补齐 `node_memory_budget.cpp` 的 `<exception>` 直接依赖，支持本机 GCC 11 构建；CMake 显式查找 CUDA driver 库，兼容 WSL 的 `/usr/lib/wsl/lib`，避免裸 `-lcuda` 链接失败。

这次集成保留两种可选后端：V4 data-service 的固定外部 KV 池，以及异模型共用容量的 Chunk owner。它们尚未合并为同一个 data-service allocator；也没有把不同模型的 KV 内容当作可复用数据。新启动器仍使用每模型一个 mixed worker，未将 V4 Vision/独立 P/D/共享权重部署自动接入 MPS。

## 验证方法

重新构建 `serving_qwen`、`shared_kv_owner`、`mps_kv_probe`、`cuda_ipc_event_probe`、`pbe_data_service`、`pbe_vlm_language_role` 和 `test_llm`。使用 CUDA 12.6、SM89、Release、QWEN2_SUPPORT=ON、NCCL=OFF，JSON 头文件单独隔离以避免 Conda Abseil 与系统 re2 ABI 混用。

本机首次配置缺少 OpenSSL 开发文件；已安装 `libssl-dev`。此次构建使用已有 Conda OpenSSL 3.6.2 的 libcrypto 和复制到构建目录的独立 openssl 头文件。模型权重继续使用原工作区下载的两个 BF16 文件。

IPC 探针在真实 exec 子进程中增加 V4 compute/I/O 租约、请求释放后的页保留、跨层句柄拒绝、重新分配后的旧代次拒绝检查。另验证原 V4 external KV serving 的两请求完成和全部页/租约归还，并在新双模型共享路径对照同一 V4 版本的独立 KV 输出。

本机只有一张 RTX 4060 Laptop，无法复现 V4 的双 H20、多模态模型全套验收。MPS 驱动限制仍见 [WSL 诊断](cuda-mps-wsl-diagnosis-2026-10-11.md)，普通多进程验证不代表真正 MPS kernel 并发。旧版 Transformers 逐字对照也不作为新版本数值等价证明。

## 本次结果

证据目录：[v4-mps-shared-kv-2026-10-11](benchmarks/v4-mps-shared-kv-2026-10-11/verification.json)。

- C++ 主回归 `--gtest_filter='*-test_*.*'`：212 项，199 通过，13 跳过，0 失败。随后使用本地 0.5B 权重补跑 checkpoint 恢复测试并通过；合计 200 个不同测试通过，12 个仍受双 GPU/NCCL/VLM fixtures/Python golden 限制。
- Python 部署合同 4 项和 V4 数据层 5 项全部通过。
- 两个真实子进程的异构 KV 布局、CUDA IPC memory/event、V4 compute/I/O pins、旧代次/跨层句柄拒绝及 fence 后完整归还均通过。
- 0.5B/1.5B 六个并行提交请求输出逐字匹配同一 V4 版本的独立 KV 基线；中文 SSE、取消隔离、未知模型拒绝、Arena 全部归还及杀死一个 worker 后整个 GPU domain 退出均通过。
- 原 V4 `serving_external_kv.sh`：2 请求完成、0 失败，最终 64/64 页空闲、0 grant、0 lease。
- 严格 MPS 探针返回 1：两个客户端未注册到同一 MPS server。没有回退为普通 process；探针和测试服务均已停止。

此次 GPU 可用显存低于默认配置的 6400 MiB 预检需求，因此验证配置只将未承诺余量从 768 改为 512 MiB，模型私有预算、512 MiB Arena、128-token 请求配置不变。仓库默认配置保持原值；实际启动应先执行 `--check`。这不是放宽 allocator 硬预算。

独立 `test/cache_core` 目标仍有 V4 原有的 CUDA 头文件及 DataChecksum、RecoveryDependencyGraph、PrefixBuilder 链接依赖缺口，未宣称该独立目标通过；上述相关 CPU 测试在完整 `test_llm` 目标中通过。本次未重跑双 H20/VLM 全套验收或 Transformers 数值对照，也未验证 MPS 实际 kernel 并发。

## 2026-10-12 原生 Linux / RTX 5090 验收补充

在 Ubuntu 22.04、RTX 5090（SM 120）、驱动 610.57.04 上，从远端功能分支 `a8cc255` 开始构建。工作目录为 `/workspace/Paged-Batch-Engine`，模型目录为 `/workspace/pbe-models`。

构建使用 CUDA 12.8.61、SentencePiece 0.2.0、Abseil 20240722.0（C++17），以及系统的 RE2、GTest、Armadillo、OpenSSL、ZeroMQ、nlohmann-json 开发包。系统旧版 SentencePiece/Abseil 组合会发生 `string_view` 冲突，需要配套安装。

本轮同时修正两个可移植性问题：RE2 缺少 CMake config 时回退到 pkg-config；tokenizer 直接遍历 `vocabs.items()`，兼容 Ubuntu 22.04 的 nlohmann-json 3.10.5。这两项修正已纳入本次发布。

```bash
cmake -S . -B build-mps -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -Dabsl_DIR=/usr/local/lib/cmake/absl \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DQWEN2_SUPPORT=ON -DKUIPER_ENABLE_NCCL=OFF
cmake --build build-mps -j 8
ctest --test-dir build-mps --output-on-failure
python3 tools/serving/deploy_multi_model.py \
  --build-dir build-mps --backend mps_process --start-mps --probe
```

验收通过了 8 组默认回归，以及真实 MPS 的同 server 客户端注册、CUDA IPC 内存/事件、不同 KV 几何的数据校验、延迟释放、完整回收和优雅关闭。主测试 210 通过、12 跳过；历史测试 43 通过；补充提供 0.5B 模型后，断点恢复测试单独通过。

最初两次 Nsight Systems 默认 CUDA 跟踪遇到 IPC 探针超时，没有 CUDA kernel 数据。后续改用 `--trace=cuda-sw,nvtx` 已完成探针和真实双模型采集，并补充 kernel 内 SM 执行采样与关闭 MPS 的三轮对照；见[实机并行与异构 KV 复用证据](benchmarks/mps-kernel-evidence-2026-10-12/README.md)。起止时间重叠本身仍不能排除进程时间片切换。NCCL 本轮构建关闭，双 GPU/视觉夹具相关边界仍未验收。

结构化结果与日志：[RTX 5090 验收记录](benchmarks/rtx5090-mps-2026-10-12/report.json)。双模型推理结果以该报告后续记录为准。

### 双模型 MPS 结果

Qwen2.5-0.5B-Instruct 与 Qwen2.5-1.5B-Instruct 的 BF16 权重和 tokenizer 均从已验证的本机文件传输，逐文件 SHA-256 一致。两模型真实 MPS 部署通过了独立基线输出对照、UTF-8 流式输出、路由、取消记录回收、非法请求拒绝和共享池完整回收。5 轮交替测试共 10 个请求，每个请求结束后 `free_bytes=536870912`、`lease_count=0`。

沿用原来的 1536/3584 MiB 私有预算时，第一个 MPS worker 报显存不足。本机验收使用 4096/6144 MiB 私有预算，加上映射的 512 MiB arena 后，MPS 单客户端限额为 4608/6656 MiB。这些值保留余量，不是已测最小预算。不能仅按权重文件大小设置 MPS 限额，还需考虑 CUDA 上下文、工作区与共享映射的计费。

生命周期结果：drain 优雅退出且不要求恢复；测试直接关闭 worker 的 cancel 场景和 Owner 故障场景均停止整个部署域并标记恢复需要，所有 GPU 进程退出。这里的直接 worker 关闭测试与普通请求取消不同；普通请求取消已在双模型功能验收中通过。

远端启动已验证配置：

```bash
cd /workspace/Paged-Batch-Engine
python3 tools/serving/deploy_multi_model.py \
  --config /workspace/pbe-validation/qwen25-mps.json \
  --build-dir build-mps --backend mps_process --start-mps
```

配置默认仅监听 `127.0.0.1:8080`。本轮验收完成后服务保持停止；该命令用于按需启动。
