# MPS 在当前 WSL 环境不可用的具体原因

最新复查：2026-10-12，Windows 驱动升级到 616.92、WSL 升级到 3.0.1 后，MPS 服务入口仍不可用，实测返回 `801 / CUDA_ERROR_NOT_SUPPORTED`。详见文末第 7 节。

检查日期：2026-10-11。结论针对本机 Windows 566.26 驱动映射给 WSL 的 `libcuda`，不推断所有未来 WSL 驱动版本。

**已定位直接失败点：真实 MPS server 从 WSL 驱动取得内部导出表后，调用表中偏移 `0x20` 的入口。该入口在当前驱动中完成进程/线程状态初始化后固定返回 `999`，server 将这个非零结果转为进程退出码 `1`。普通 CUDA 接口正常，但这个 MPS server 入口没有执行可用的服务初始化路径。**

本轮使用 GDB 跟踪真实 server、检查二进制调用点、交叉运行两个版本的工具，并以独立 CUDA 客户端做系统调用跟踪。没有加载 PBE 模型，没有修改驱动或绕过失败分支。

## 1. 实际环境与加载链

| 项目 | 实测 |
| --- | --- |
| 系统 | Ubuntu 22.04.5，WSL2，内核 `6.6.87.2-microsoft-standard-WSL2` |
| GPU | RTX 4060 Laptop，计算能力 8.9，UVA=1 |
| Windows 驱动 | 566.26 |
| nvidia-smi 工具版本 | 565.75，与 Windows 驱动显示的是不同版本字段 |
| CUDA Driver API | 12070，即 12.7；本地 Toolkit 为 12.6 |
| 已安装 MPS 工具 | 565.57.01，server 参数中的接口版本为 12070 |
| 交叉验证工具 | 560.35.05，仅解包到临时目录，接口版本为 12060 |

server 动态加载的是：

```text
/usr/lib/wsl/lib/libcuda.so.1
  → /usr/lib/wsl/drivers/nvlt.inf_amd64_3ab002af0a386850/libcuda.so.1.1
```

后者 SHA256：`095cf4a5b782e2bddb0f7d7ffbd801124fc108916ba3c6981b7a764a413622f8`。这是定位偏移的对应文件；不能把下面的地址直接套用到其他版本。

## 2. 失败发生在哪一步

真实执行顺序：

```text
control daemon 启动、处理 start_server                 成功
  → exec nvidia-cuda-mps-server                       成功
  → dlopen("libcuda.so.1")                            成功
  → dlsym("cuGetExportTable")                         成功
  → cuGetExportTable(内部 MPS 表 UUID)                 返回 0
  → 调用导出表 + 0x20 对应的驱动入口                  返回 999
  → server 检测非零结果，return 1                     退出
```

导出表 UUID 的原始字节是 `a392d3871cf47f4d84b1719b17aeac8f`。这次加载的表大小为 `0x28`；server 确实调用了最后一个函数指针，即表内字节偏移 `0x20`。该函数位于实际 WSL 驱动 ELF 偏移 `0x321b90`。它没有公开符号名称，因此这里按照真实调用点描述，不把 objdump 显示的“最近公开符号 + 偏移”当成该函数名称。

关键反汇编：

```asm
; server 的调用点，565.57.01
0x40185a: call QWORD PTR [rax+0x20]
0x40185d: test eax,eax
; 非零返回进入 return 1

; 当前 WSL libcuda 的被调用入口公共返回路径
0x321bd6: add rsp,0x10
0x321bda: mov eax,0x3e7    ; 十进制 999
...
0x321be3: ret
```

完整函数中的正常返回路径均汇合到这段代码。实测 GDB 输出 `MPS_SERVER_ENTRY_RETURN=999`。公开 `cuGetErrorName(999)` 返回 `CUDA_ERROR_UNKNOWN`，但仅凭这个通用名称无法解释根因；固定失败返回的驱动代码和真实调用链才是关键证据。

用 daemon 拉起 GDB 包装的 server 后，也得到同样结果。控制日志中的短暂 PID 4023 是尚未退出的调试包装进程，随后退出码 1，server list 清空；不能据此声称 MPS server 已进入 ACTIVE。

证据：[server 调用点](benchmarks/mps-shared-kv-2026-10-11/mps-diagnosis/server-callsite.asm)、[驱动入口](benchmarks/mps-shared-kv-2026-10-11/mps-diagnosis/driver-mps-entry.asm)、[daemon 下的 GDB 输出](benchmarks/mps-shared-kv-2026-10-11/mps-diagnosis/server-under-daemon-gdb.log)。

## 3. 已排除的原因

| 排查项 | 结果与含义 |
| --- | --- |
| 程序没有安装或 PATH 错误 | 已确认成功 exec 真正的 server 二进制 |
| 找不到 libcuda 或符号 | dlopen、dlsym 和导出表查询全部成功 |
| PBE、模型、共享 KV 代码 | 独立 server 即可复现，测试没有加载模型 |
| CUDA 整体不可用 | cuInit、创建 context、分配 4096 字节显存全部返回 0 |
| GPU 位数、UVA、计算能力 | x86-64、UVA=1、SM 8.9；不是这些基础条件未满足 |
| 仅因使用了 12.7 工具而 Toolkit 是 12.6 | 12.7 和 12.6 两种 server 均进入相同驱动函数，返回 999；当前驱动本身报告支持 12.7 |
| daemon 与 client 用户或目录不一致 | 本轮同 UID，使用专属 700 权限目录；控制命令交互正常 |
| 残留环境变量干扰 | daemon 对照测试去除了继承的 CUDA_*、LD_PRELOAD 和 LD_LIBRARY_PATH，再设置自己的 MPS 目录 |

没有尝试通过替换 WSL libcuda、安装 Linux 驱动库或修改机器码来绕过返回值。普通 CUDA 通路依赖当前 WSL 驱动桥接，替换文件不能补出缺失的 MPS 服务实现。

## 4. 为什么 CUDA client 能跑，却没有加入 MPS

在控制 daemon 存活时，独立 Python/ctypes 客户端设置了同一个 `CUDA_MPS_PIPE_DIRECTORY` 和 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50`，然后调用 Driver API：

- `cuInit(0)`：成功。
- `cuCtxCreate_v2`：成功。
- `cuMemAlloc_v2(4096)`：成功。
- 当前可见 SM 数：24。
- client context 存活期间，server list 为空。
- strace 跟踪 `connect/openat/ioctl`，没有观察到对本轮 MPS pipe/socket 的访问；实际打开了 `/dev/dxg` 和 WSL 驱动库。
- 释放显存并销毁 context 后正常退出。

这说明本次 CUDA client 使用了普通 WSL CUDA 通路，没有建立 MPS 连接。仅仅 CUDA 推理正常、控制 daemon 存活，或环境变量已经设置，都不能证明 MPS 生效。SM 数是辅助观察，未把它单独作为判定依据。

证据：[独立 CUDA 基线](benchmarks/mps-shared-kv-2026-10-11/mps-diagnosis/cuda-baseline.json)、[受控实验结果](benchmarks/mps-shared-kv-2026-10-11/mps-diagnosis/controlled-result.json)、[client 系统调用](benchmarks/mps-shared-kv-2026-10-11/mps-diagnosis/client.strace)。

## 5. 关于独占模式和可行路径

NVIDIA 官方文档将 `EXCLUSIVE_PROCESS` 描述为使用 MPS 时的推荐设置；它用于确保 GPU 上只有一个仲裁 server，并不是必须先设置才能初始化 MPS 的普遍前提。因此，不能简单用“WSL 无法修改 compute mode”解释本轮失败。这里已经定位到了更直接的驱动入口失败。[NVIDIA GPU Compute Modes](https://docs.nvidia.com/deploy/mps/595/when-to-use-mps.html#gpu-compute-modes)

MPS 的客户端和 GPU 服务逻辑依赖 CUDA 驱动；控制 daemon 主要负责协调启动与连接。[NVIDIA MPS Architecture](https://docs.nvidia.com/deploy/mps/latest/architecture.html)

对当前 PBE 的实际影响：

1. 重新编译 PBE、修改共享池大小、调整 MPS 线程比例，都无法改变这个驱动入口的固定失败返回。
2. 当前已经验证的普通多进程 CUDA IPC 共享 KV 路径仍可使用，但不能据此宣称具备 MPS 跨进程 kernel 并发。
3. 要验收真实 MPS，需要在提供可用 MPS 实现的驱动环境中运行。原生 Linux 是应优先验证的部署路径；换环境后仍需通过 client 注册和 GPU 时间线检查。
4. 本轮只验证了现有 Windows 566.26 驱动，不承诺升级 Windows 驱动即可修复，也不推断未来 WSL 版本永久不支持。

WSL 官方指南要求使用 Windows 提供的驱动，避免在 WSL 中安装 Linux 显示驱动。[NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

## 6. 复现与收尾

当前安装的 565 server 可用以下 CPU GDB 脚本复现导出表和入口返回值：

```bash
gdb --return-child-result -q -batch \
  -x docs/benchmarks/mps-shared-kv-2026-10-11/mps-diagnosis/trace-entry.gdb \
  --args /opt/pbe-mps/565.57.01/bin/nvidia-cuda-mps-server
```

预期关键输出为 `EXPORT_TABLE_RESULT=0`、`MPS_SERVER_ENTRY_RETURN=999`，被调试 server 退出码 1。脚本观察的是这两个已验证版本使用的内部表布局，不是通用或公开的 CUDA API 契约。

测试 daemon、server、调试器及 client 全部退出，双模型服务保持停止。安装的 565 工具保持原状；560 对照包仅解包在临时目录，未替换系统命令。结构化结论与文件哈希见 [diagnosis.json](benchmarks/mps-shared-kv-2026-10-11/mps-diagnosis/diagnosis.json)。

## 7. 2026-10-12：升级 616.92 后的复查

Windows NVIDIA 官方安装器已完成，Windows 设备驱动版本为 `32.0.16.1692`，WSL `nvidia-smi` 报告 `616.92`。WSL 更新到 `3.0.1.0`，运行内核为 `6.18.40.1-microsoft-standard-WSL2`。CUDA Driver API 报告 `13040`。

新的 `/usr/lib/wsl/lib/libcuda.so.1` 是加载桥接库。实际进程映射到 Windows 驱动包中的 `nvltsi.inf_amd64_f0bf3af178442601/libcuda.so.1.1`，不能只凭桥接库的文件大小或 `readlink` 判断驱动是否正确加载。

565.57.01 MPS Server 在新版驱动中仍能成功获取内部导出表，但表内 `+0x20` 入口返回 `801`，公开 CUDA API 将其解释为 `CUDA_ERROR_NOT_SUPPORTED`。该入口本次对应 ELF 偏移 `0x438800`：正常返回路径在 `0x438843` 和 `0x438883` 均执行 `mov eax,0x321`，然后返回。函数未读取 server 传入的版本参数。这里是当前文件的实测结论，不应把地址套到其他驱动版本。

独立控制实验使用同一 UID、专属 pipe/log 目录，并清除继承的 CUDA 与动态库覆盖变量：

- 显式启动 server 后，GDB 再次记录入口返回 801，server 退出码 1。短暂出现在列表中的调试包装进程不是可用 MPS Server。
- 独立客户端创建 CUDA 上下文、分配和释放显存均成功。
- 客户端存活时 server list 为空；strace 没有观察到客户端访问专属 MPS socket。
- 项目真实 MPS 探针因两个客户端未注册到同一个 server 而失败，关闭报告标记 `mps_recovery_required=true`，本轮自建 daemon 已停止。
- 普通 `process` 后端探针通过 CUDA IPC 内存/事件、KV 数据、延迟回收和显存池完整归还验证。这是探针级验证，不等于重新验收完整双模型推理。

因此，升级已成功，但当前驱动提供的这条 WSL MPS 服务入口仍不可用。不能通过修改 PBE 的 KV 分配逻辑、放宽注册检查或把返回码改为成功来解决。

证据目录：[616.92 实测记录](benchmarks/mps-driver-61692-2026-10-12/)，公开驱动环境和两种后端的结构化探针结果；完整 server 调用轨迹与客户端系统调用日志保留在原工作区。

### 新版工具交叉验证

从 NVIDIA Ubuntu 22.04 软件源下载 `libnvidia-compute_615.71.09-2ubuntu1_amd64.deb`，SHA-256 与本机软件源索引一致：`7cd68883edce0fd7591a9abe4351b398d429fc9debeebfbb213bfd955a1204a8`。仅解包到 `/tmp/pbe-mps-615/extracted`，未安装包、未把其中的 Linux CUDA 库放入加载路径、未替换系统 MPS 命令。

| 实验 | 实测结果 |
| --- | --- |
| 565.57.01 server，接口版本 12070 | 驱动入口返回 801，进程退出 1 |
| 615.71.09 server，接口版本 13040 | 同一个驱动入口返回 801，进程退出 1 |
| 615.71.09 control，Legacy v2，前台启动 | `Failed to initialize CUDA: 801`，退出 1 |
| 615.71.09 control，MPS v3，前台启动 | `Failed to initialize device query: 801`，退出 1 |

新版 server 传入的接口版本与当前驱动报告的 `13040` 一致，仍然失败。这样排除了“只是 565 工具太旧”这一解释；选择 MPS v3 也没有绕过驱动限制。615 工具的后台启动命令可能先返回 0，而子进程随即失败，因此必须检查控制日志和实际服务状态。

最终结论：**当前 RTX 4060 Laptop + Windows 616.92 + WSL 3.0.1 的已测 MPS 通路不可用，直接失败原因是驱动返回不支持。** 普通 CUDA 可用并不意味着 MPS 可用。继续调整 PBE 启动参数、升级同一环境中的 CUDA Toolkit、修改 MPS 成功判定，都无法补全这个入口。需要真实 MPS 的验收应移到具备可用 MPS 驱动的原生 Linux 环境，再运行项目的严格探针；这不是对所有未来 WSL 驱动版本的永久判断。

复查结束时诊断 CUDA 客户端、MPS 控制进程和服务进程均已退出，模型服务保持停止。

## 8. 原生 Linux 对照结果

2026-10-12 在原生 Linux 容器环境、RTX 5090、驱动 610.57.04 上，MPS Server 正常启动，两个独立 CUDA 客户端注册到同一 server。PBE 的严格 MPS/共享 KV 探针与 Qwen2.5-0.5B、1.5B 双模型功能验收均通过。这进一步说明本机 WSL 上的“不支持”不是 PBE 必然无法使用 MPS；结论仍以具体平台测试为准。完整边界和资源配置见 [原生 Linux 验收补充](cuda-mps-v4-integration-2026-10-11.md)及 [结构化报告](benchmarks/rtx5090-mps-2026-10-12/report.json)。
