# MPS 启动与停机生命周期补齐（2026-10-11）

本次针对当前“一模型一个 mixed worker + 一个 CUDA IPC KV Owner”的部署补齐生命周期。普通 process 路径已经用两个真实 Qwen 模型验证；本机 WSL 的 MPS server 仍不可用，MPS 命令的成功路径只有模拟测试，不视为真实 MPS 验收。实现使用现有 565 工具对应的 legacy MPS control 接口，未实现 MPS v3 接口适配。

## 正常退出

启动器收到 SIGINT/SIGTERM 后执行：

1. 向 CPU API 进程发送 SIGTERM。信号处理函数仅设置标志；HTTP 主循环关闭监听入口，再等待已接受请求的 handler 完成。已有 SSE 请求可以继续接收最终结果。
2. 等 API 退出后，向每个 worker 发送 `{"type":"shutdown","mode":"drain"}`。worker 不再处理新的 generate，等待 scheduler 线程排空，并 join 所有 token 转发线程。
3. worker 调用 `cudaDeviceSynchronize()`，收集已完成的共享页 fence，再回复 `gpu_synchronized=true`。启动器还会等待 worker 进程真正退出，包含模型销毁和 CUDA IPC 映射关闭。
4. 查询 Owner 的空闲字节和租约数量。使用当前 epoch 发送 Owner shutdown；Owner 拒绝仍有租约的退出请求，只有零租约时才同步、释放 Arena 并确认 `arena_freed=true`。
5. 所有客户端退出后，仅关闭由本次 `--start-mps` 启动的控制 daemon。外部已有 daemon 不由本部署关闭。

token 取消路径原先使用 detach，现在会 join，防止退出后 token 线程继续访问 RPC server 的局部互斥锁或请求状态。最终 RPC 回复使用有界 linger，减少进程退出时丢失停机确认的风险。信号处理函数不执行 CUDA API、RPC 或线程 join。

`shutdown_timeout_s` 默认为 30 秒，可配置为 1–300 秒，是各停机阶段的等待上限，不是整个部署的总时限。超时会进入异常退出路径，不再声称请求完整排空。正常 Ctrl-C/SIGTERM 的 supervisor 退出码可以是 130；应结合 `shutdown.json` 的 `graceful` 判断清理是否完成。

## 故障退出与重建

- 任意 worker、Owner 或 API 意外退出会结束整个部署。API 故障/超时后，存活 worker 使用 `shutdown` 的 `cancel` 模式；仍须等待 GPU 同步确认。
- MPS worker 无法响应时，启动器先查询该 PID 所在的 MPS server，再执行 `terminate_client <server PID> <client PID>`。只有控制命令明确返回 `0`，才记录 GPU 工作已终止；不能仅根据控制程序退出码为 0 判断成功。
- 随后才强制结束对应 OS 进程。若 GPU 终止不能确认，报告会保留 `gpu_termination_unconfirmed` 和 `mps_recovery_required=true`，不会记为优雅退出或允许旧 Arena 继续服务。
- 不会按超时回收死亡 worker 的租约给存活模型。整个故障域退出后，重新部署时新建进程、Arena 和 epoch。当前没有自动重试推理请求，也没有循环自动重启服务。
- 极端情况下 importer 收到 SIGKILL 后仍不能退出，保留并隔离 exporter，不释放它的 Arena，也不关闭本部署的 MPS daemon；报告 `owner_quarantined`，留待外部恢复。这条路径通过模拟测试验证，实际硬件测试没有出现无法退出的进程。

这符合 NVIDIA 对提前终止 MPS 客户端的要求：应先同步 GPU 工作，或使用 MPS 的客户端终止命令完成 GPU 上下文终止，再结束 OS 进程。容器中的 PID 必须与 control daemon 所在 PID namespace 对应；当前启动器采用本机一致的 PID namespace，不能把容器内 PID 当成宿主机 PID。[NVIDIA Client Early Termination](https://docs.nvidia.com/deploy/mps/595/when-to-use-mps.html#client-early-termination)

## 启动检查与资源记录

- CUDA 初始化前设置 MPS pipe、可见 GPU 和每个模型的 active-thread/显存限制。
- 先启动 Owner 并检查它确实加入 MPS，失败则不加载模型权重。之后要求 Owner 和全部 worker 注册到同一个 MPS server。
- 服务运行期间每 5 秒检查这些 PID 仍属于原 server。注册丢失或 server 改变时结束旧故障域，不能沿用旧 epoch。
- `registry.json` 增加 backend、MPS server、可见 SM 数、实际有效 batched-token 上限及配置的 MPS 资源限制，配合已有私有 allocator 统计进行验收。SM 数仅作辅助信息，注册和 GPU 时间线仍是必要证据。
- `process` 后端清理继承的 MPS 限制变量并使用不存在的专属 pipe 路径，避免误连默认 daemon。
- 两模型的 50% active-thread 配置和私有预算保持不变。真实 MPS 的最优比例、内部显存开销和 IPC 导入计费尚未在本机校准；不能把配置值当作实际吞吐或显存隔离结果。[NVIDIA Resource Provisioning and Memory Limits](https://docs.nvidia.com/deploy/mps/595/when-to-use-mps.html)

普通多进程和 MPS 探针也使用 Owner 的正常退出协议；探针异常中断时遵守 importer 先退出、exporter 后退出的顺序。

## 本次实测

证据目录：[mps-lifecycle-2026-10-11](benchmarks/mps-lifecycle-2026-10-11/drain.json)。真实测试使用本地 Qwen2.5-0.5B/1.5B BF16、512 MiB Arena、16-token workspace/prefill 配置，未开启 MPS。

| 场景 | 结果 |
| --- | --- |
| 1.5B 流式请求仍活跃且有 KV 租约时停止 supervisor | 收到 SSE `[DONE]`；API、两个 worker、Owner 顺序退出；两个 worker 均确认 GPU 同步；512 MiB 全部归还；本轮约 3.07 秒完成停机 |
| 活跃请求中直接向 worker 发送 cancel shutdown | 确认 GPU 同步后返回；该 worker 退出触发其余部署退出；最终 Arena 全空 |
| 杀死 Owner（请求空闲时） | supervisor 检测故障，剩余 worker 取消并同步退出，无 GPU 子进程遗留 |
| 杀死一个 worker | 整个 GPU domain 退出；保留故障原因和各阶段清理记录 |
| Owner 仍持有测试租约时请求 shutdown | 明确拒绝 `arena_has_live_leases`，释放租约后正常服务 |
| 重新部署 | 4 次真实部署得到 4 个不同 epoch |
| 双模型生成回归 | 6 个并行提交请求、10 次交替请求均匹配独立 KV 基线；25 次 Chunk 分配/释放匹配，最终零租约 |
| C++ 主回归 | 212 项：200 通过，12 跳过，0 失败 |
| Python 部署 / V4 数据层 | 8 / 5 项通过；包含终止确认、停机顺序、超时及 exporter 隔离模拟测试 |
| CUDA IPC memory/event、租约及旧句柄检查 | 通过，探针 Owner 正常释放 32 MiB Arena 后退出 |

12 个跳过项仍依赖双 GPU、NCCL、VLM fixtures 或 Python golden。此次没有扩大到独立 P/D、多模态部署或多 GPU MPS。

最后实际调用 `--backend mps_process --start-mps`：Owner 没有注册到 MPS，启动器在加载模型前返回失败，完整归还 512 MiB 并关闭本次控制 daemon；没有回退为 process。详见 [mps-unavailable.json](benchmarks/mps-lifecycle-2026-10-11/mps-unavailable.json)。本机驱动限制仍见 [WSL 诊断](cuda-mps-wsl-diagnosis-2026-10-11.md)。所有本次实际启动的进程均已停止。

## 使用和复现

可用 MPS 环境中的启动入口仍为：

```bash
python3 tools/serving/deploy_multi_model.py \
  --config configs/qwen25-dual-mps.json --backend mps_process --start-mps
```

需按目标机器配置模型路径和预算；`--check` 只做配置/资源预检，不证明 MPS server 可用。停止时向 supervisor 发送 SIGTERM 或 Ctrl-C，查阅运行目录的 `shutdown.json`。故障后使用新运行目录重新启动，不能复用旧 registry 或 epoch。

本地验证工具：

```bash
python3 tools/serving/validate_lifecycle.py \
  --config docs/benchmarks/shared-kv-reuse-2026-10-11/config.json \
  --scenario drain --output-dir /tmp/pbe-lifecycle-repeat
python3 -m unittest tools/serving/test_deployment.py
```

另外两个真实场景为 `--scenario cancel` 和 `--scenario owner-failure`，每次使用新输出目录。

仍需在可用 MPS 环境验收：实际 `terminate_client`、server 故障/重建、GPU 时间线上的 kernel 并行，以及不同资源比例的吞吐/尾延迟。当前不能声称这些硬件相关路径已经通过。
