# V3 数据流重构执行记录

日期：2026-09-12

## Checkout 与环境

- HEAD：`f3597a763b6cd5cf60b826baa2dc94395c3d02fd`
- 开始执行前 dirty binary diff SHA-256：`b835cd56561340e90b17149d8e5aff4352d56a326153efe2dffdfebdbcf8e41f`
- 最终定向验收时 tracked binary diff SHA-256：`71942cb3b21367fdd3171139f86d9c84b8befcc44df59ccc9f68c41e0477532b`；新增的 untracked 源文件由下方交付清单和测试命令共同标识，不冒充已进入该 diff。
- 构建：CMake 3.31.8，`RelWithDebInfo`，CUDA，CPM dependencies，tests enabled，NCCL disabled，Qwen2 target disabled。
- 硬件：2 × NVIDIA H20-3e（每卡 143771 MiB），driver 550.127.08，CUDA toolkit 12.8.61。
- 当前 checkout 没有模型权重或 tokenizer fixture；只发现 `tools/config.json`。

## 已实现并接入

### DF0

- checkpoint revision 使用全局单调 revision；旧 prepare 不能覆盖较新的 READY revision。
- `cancel_client` 撤销 PREPARING/READY/RESTORING 全生命周期记录，回收 revision metadata。
- record 和 payload bytes 在 snapshot 分配前联合准入；KV、FP8 scales、SequenceState 动态字段和 outbox text 均计入，并检查整数溢出。
- 显式 `kv_cache_blocks_per_layer` 贯通 benchmark CLI、模型初始化和容量输出。
- checkpoint prepare 不再全局 drain 无关 cache transfer。

### DF1

- PageSchema 支持每层 K/V 与 FP8 scale 的连续 KV-head 分片完整覆盖。
- immutable transfer plan 区分 logical fingerprint 与 source/destination representation fingerprint，支持 ExactDirect、SenderPack、ReceiverUnpack。
- 计划缓存按 entries 和 template bytes 做 LRU 上限；key 不含运行时地址。
- binding 在首笔写入前检查两端 schema、layout、component spans、epoch、bounds、重叠和目标完整覆盖。
- PageMigrationEngine、InProc KV copy 和 CUDA P2P connector 使用长期 planner；真实 connector 先编译和全量绑定，再执行 copy。

### DF2–DF4 的已落地部分

- `append_slots` 先计算 delta 并完成容量准入，再分配，失败不留下部分页。
- Host-only restore 为每个唯一 logical target 预留一次 GPU 目标；32 个共享 waiter 不重复计费，取消与完成会 reconcile。
- TransferScheduler 提供 D2H/H2D/P2P lane、总 active 限制、后台 aging、demand reserve、absolute steady-clock deadline、超时 donation 撤销，以及 typed submit outcome。
- forced `checkpoint` preemption 已接到 decode KV 动态增长失败路径；保存成功后释放 victim，其他 runnable 工作优先，队列清空后恢复。失败显式计数并回退 recompute。
- SequenceState checkpoint 保存 process-local outbox item、连续提交 cursor 与 sampled-token 区间。

## 已运行验收

构建命令：

```bash
cmake -S . -B build-v3 -DUSE_CPM=ON -DKUIPER_BUILD_DEMOS=OFF \
  -DKUIPER_BUILD_TESTS=ON -DQWEN2_SUPPORT=OFF \
  -DKUIPER_ENABLE_NCCL=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-v3 --target test_llm -j16
```

定向综合回归：54 passed，2 skipped。skipped 均为构建时关闭的 NCCL connector：

```bash
./build-v3/test/test_llm \
  --gtest_filter='TransferSchedulerTest.*:SchedulerCheckpointTest.*:KVCacheManagerTest.*:RequestCheckpointTest.*:PDHandoffTest.*:PageSchemaTest.*:CacheTransferPlanTest.*:CacheLayoutTest.*'
```

通过内容包括：10,000 次 checkpoint 生命周期、100,000 次 request generation 生命周期、真实 CUDA FP8 snapshot/restore、真实双卡 CUDA P2P、FP8 K/V/scales connector、head split/merge/reorder、三个 placement 的 CPU reference、32 waiter single-flight、restore target 去重、lane 并行、deadline/cancel、forced pressure checkpoint。

## 未通过、阻塞与未完成

- ASan/UBSan 构建阻塞：系统没有 `libasan`/`libubsan`，链接器报 `cannot find -lasan`；没有把它记录为通过。
- 真实模型 DF6 阻塞：checkout 内没有模型权重和 tokenizer fixture。
- 全量无筛选 `test_llm` 不是绿色：两个既有 SplitKV attention 数值测试失败，随后 CUDA 测试在先前 death-test fork 后报 CUDA error 222 并终止。定向数据流集合在独立进程通过；全量失败仍保留为待处理问题。
- DF2 的通用 `RequestPhysicalDemand` move-only reservation、P−1/P/P+1 COW 和完整分池守恒指标尚未实现。
- DF3 的总 bytes/staging credits、显式依赖环检测和 unknown 换址重试协议尚未完成。
- DF4 的异步有界 save、stop-string/UTF-8 隐藏缓冲统一所有权及 10,000 次真实流式压力尚未完成。
- DF5 成本模型、水位控制器和 cold-prefix 策略未实现；因此 DF6 等价消融和性能结论未执行。
