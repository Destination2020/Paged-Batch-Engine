# V4 最终验收复核

> 后续状态（2026-09-13）：本文记录的是整改前复核，内容保持不改。所列缺口现已逐项修复并通过可执行门禁；完成结论和新证据索引见 [V4 最终完成报告](V4_FINAL_COMPLETION_20260913.md)。

结论：当前不能确认原 V4 全部完成。新的常驻模型、Agent-owned pool、混合 Prefill/Decode、跨卡副本和四分支 COW 证据是实质进展；本次不撤销这些成果。但 M5 数值门禁、M7 设备账本与多模态压力恢复、M8 在线生命周期、M9 原定实验仍有缺口，不能用 10/10 掩盖。

本次读取最新状态、计划、最终 results/manifest、M1 数值门槛及常驻 Coordinator/language 源码。最终 manifest 的 9 个记录文件 hash 全部与当前文件一致。运行了现有 20-case 校验器，没有重新构建 C++ 或重跑完整 GPU 实验；209 passed/57 passed 等属于提供的历史门禁记录，不是本次新增测试。

## 1. M5 数值验收存在实际失败

运行：

```bash
python3 tools/bench/data_flow/validate_persistent_vlm_fixtures.py docs/data_flow_evidence/v4/post_review_persistent_vlm_20_retry/result.json
```

退出码 1，脚本第 27 行 mixed/single 全序列一致断言失败。最新记录确实是 18/20、153/160，报告并未隐瞒分叉，但现有自动门禁未通过。

M1 的 tie 判定有具体证据：对应步骤参考 top-2 margin=0，并在 teacher-forced 同历史下验证 logits/KV。当前两例只记录首次分叉和后续自回归级联，未在审阅的 20-case 结果中找到等价的 logits margin 或同历史误差证据。一个旧样例的 tie 不能证明两个新样例也是 tie。

下一步：记录两例首次分叉的 token、top-k/margin，比较相同输入历史下的 logits、位置和必要 KV；若满足原冻结阈值，再使自动校验器明确支持该数值合同。不能只删除全等断言或把 95.625% token 一致率作为新的宽松门槛。

## 2. M5 常驻执行仍有在线协议缺口

`python/pbe_roles/coordinator.py::run_round` 等待本轮所有 Encoder 返回后才调用一次 language infer。`LanguageProcess.call` 同步等待整批响应；C++ main 在 Infer 返回后才读取下一条 stdin 命令，未提供执行中接收新请求/外部 cancel 的路径。`cancel_after_tokens` 是预先配置的模拟取消触发点，不能替代客户端在 Decode 中途发送取消消息。

deadline 在 Encode 使用 timeout_ms，在 Join 后 language 又取得默认 30000ms，C++ 再以当前时间建立 deadline。它没有延续同一个端到端绝对 deadline。generation 字段出现在请求中，不等于已经实现跨调用的 stale-generation 去重和取消。

下一步：有界异步入口/命令 mailbox 与持续运行的 scheduler owner；传递同一截止时间或严格递减的剩余时间预算。验证推理期间到达的新请求、真实外部取消、迟到回复和多阶段累计超时。当前实现可以准确称为“常驻、按轮批处理的多模态演示”。

## 3. M7 的账本还不是完整设备内存准入

`demo/pbe_vlm_language_role.cpp` 第 190–206 行：weights 按模型文件大小计，workspace 仅按 `512 * hidden_size * 2` 计。后者只相当于单个 embedding 大小，没有覆盖真实 Qwen workspace 中的 Q/K/V、MLP、logits、attention 临时缓冲；Infer 内还动态分配 mRoPE tensor。角色本地账本也不包含同设备 Vision 权重/workspace 和服务端实际保留对象的完整占用。

因此 `budget_invariant=true` 证明这些声明额度内部守恒，不能证明实际显存总量受到约束。bundle 在 acquire 得到副本后才进行这份本地账本 reserve，也不能把它当作服务端分配前准入的证据。

下一步：复用实际 serving workspace profile 并记录 allocator 高水位；明确 weights、IPC owner、import view、副本、bundle、staging 的唯一计费权威与设备归属。各角色初始化和增长必须服从同一设备上限，避免重复计算共享页或漏算独立副本。

## 4. M7/M8 真实多模态压力恢复仍缺证据

本次线上压力实验说明 staging 满时准入 2、拒绝 4，已完成请求释放后可再次准入。这是有价值的背压/拒绝恢复验证，但没有证明运行中的多模态请求被 checkpoint 抢占、重新恢复并保持输出/位置一致。

原计划 M7 第 5–6 项仍要求真实 `recompute|checkpoint|auto`、异步保存、多模态依赖/position state、成本与水位策略。当前 language request 的 positions/rope_delta 在 demo 的本地 Input 结构中，审阅的 SequenceState/RequestCheckpoint 定义未见其完整恢复字段。将旧文本 checkpoint 和 lane 单测引用进最终矩阵不足以补齐这条数据路径。

下一步：真实压力下暂停图文请求，保存/恢复其 feature 依赖、mRoPE state、RNG/outbox；同时让无关请求继续推进。对取消、服务重启、在途 copy 和最后必要数据保留补充该路径的故障测试。不能把“拒绝新请求后重新提交成功”称为完整 checkpoint 恢复。

## 5. M9 仍未覆盖正文承诺的全部实验

已完成的五次 cache off/on 与 process-owned/Agent-owned A/B 可保留。最终 results 中的 merged_vs_separated_speedup=null 是诚实报告，但其理由称该实验不在 V4 scope，与当前计划第 12 节“合并部署/角色分离同预算对照”冲突。

正文还要求 KV share、singleflight、lane off/on 和压力策略比较；引用功能/COW/旧合同测试不是这些公平实验。图片重复率、前缀长度、压力等级等应有实际 workload 记录，不能由 matrix 字段或单个拒绝实验代替。

此外，现有 `itl_ms` 来自每请求 mean_token_gap，跨 5 次重复再取 P95/P99 得到的是“5 个平均值的分位数”，不是 token 级 ITL 尾延迟，应在图表中准确命名。当前 language 返回整批最终 JSON，multimodal_ttft_ms 是 encode 时间加内部首 token 时间；它不是已经实现流式接口并在客户端观测的首 token 到达时间。

下一步：保留现有负收益；补齐同资源、同请求轨迹的缺项，并从逐 token trace 计算 ITL 分布。若明确缩减交付范围，先修改架构/计划的验收合同，列清延期项，称为限定范围版本完成；不能在结果文件中单方面把未测项移出分母。

## 建议的验收顺序

1. 先处理失败的 20-case 自动门禁，明确分叉是否满足冻结数值合同。
2. 完成真正异步在线请求、取消和端到端 deadline；据此重新验收 M5。
3. 实际设备账本与多模态 checkpoint 主路径；据此重新验收 M7。
4. 在上述新路径上完成 M8 生命周期故障矩阵。
5. 补齐 M9 实验或明确修改范围，再更新最终百分比。

本次不重新给出一个主观工作量百分比，也不覆盖原始日志。可沿用已经单独确认的阶段成果，但最终 10/10 当前不能通过独立复核。
