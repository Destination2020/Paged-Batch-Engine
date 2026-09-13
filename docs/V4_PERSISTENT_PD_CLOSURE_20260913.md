# V4 常驻 Prefill/Decode 部署对照收口

日期：2026-09-13

结论：`V4_CLOSURE_RECHECK_20260913.md` 指出的最后一个阻塞已关闭。新增实验测量的是同一多模态请求在独立常驻 Prefill 进程完成 prompt 后，将 external KV 页表交给独立常驻 Decode 进程继续生成；不再用两个完整模型副本轮询代替角色分离。

## 实现与验收合同

- `pbe_vlm_language_role` 新增 `pd_prefill`、`pd_decode`、`pd_release`。Prefill 对真实 Vision bundle 执行一次完整 prompt 计算，seal 版本化 KV handoff 并持有源请求；Decode 通过 lease 获取 handoff，把 Prefill slot 作为只读共享前缀挂入自己的页表，仅为生成尾部使用自身 grant。
- Decode 使用 `restore_external_shared_request` 增加共享页引用，结束时释放消费者引用而不消耗 provider 基线所有权，因此同一常驻 Prefill slot 集合可以顺序服务多个请求。
- 可执行门禁逐请求检查相同 request/generation、`prefill_pid != decode_pid`、有效 handoff、prefix/private grant 不同、保存全部 prompt token、Decode 实际 prompt 计算为 0、调度 Prefill token 为 0、输出与预热统一路径 oracle 完全一致，以及显式 release。
- worker 回收以模型初始化后的 KV 请求数为基线。Qwen2 本身会常驻一个 `single_seq_request_id_`，所以正确稳态是总 active=1、serving active=0，而不是总 active=0；状态同时暴露三项值以防掩盖泄漏。进程关闭后再检查 Data service 的物理资源守恒。

## 最终 GPU 结果

证据目录：`data_flow_evidence/v4/M9/persistent_pd_deployment_final_gpu0_v4/`。

同一 GPU、同一 Qwen2.5-VL-3B BF16 模型、同一常驻 Vision/Data service、同一图片和到达 seed 下，统一路径与独立 P/D 路径均预热后交错运行 5 次。三个 language 模型进程在观测窗口内同时常驻；两个拓扑各拥有 64 个 KV slot。

| 指标 | 统一 P/D worker | 常驻 Prefill→Decode |
| --- | ---: | ---: |
| client latency 中位数 | 771.15 ms | 772.95 ms |
| 端到端 TTFT 中位数 | 57.06 ms | 59.29 ms |
| 有效 service throughput | 1.2855 req/s | 1.2883 req/s |
| model allocation | 7,327,449,088 bytes | 14,654,898,176 bytes |
| workspace reserved | 329,646,080 bytes | 659,292,160 bytes |

5 次 P/D 请求均由 Prefill PID 2837024 交给 Decode PID 2837151；每次保存 95 个 prompt token，Decode 的 `actual_computed_prompt_tokens=0` 且 `scheduled_prefill_tokens=0`，全部输出匹配统一 oracle。所有 handoff release 后，三个 worker 均回到各自模型 KV 基线，Data service 最终为 `total_slots=128 free_slots=128 active_grants=0`、`leases=0`。

可执行结论位于 `validation.json`，十项检查全部为 true。`persistent_pd_deployment_final_gpu0_v2` 的首次失败暴露共享页所有权不能被首个消费者耗尽；v3 的推理和物理回收成功，但旧门禁错误要求模型总 active request 为 0。失败证据均保留，最终验收只引用 v4。

旧 `persistent_deployment_final_gpu0_v2` 继续作为“单个完整 language 模型副本 vs 两个完整副本轮询”的成本实验，在最终汇总中命名为 `replica_count_cost` 并明确 `counts_as_prefill_decode_handoff_acceptance=false`。
