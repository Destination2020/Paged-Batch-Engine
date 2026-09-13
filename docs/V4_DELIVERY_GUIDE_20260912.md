# PBE V4 交付指南

V4 已在 2 × NVIDIA H20-3e、driver 550.127.08 上完成验收。冻结模型为
`Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3`。

## 启动

常驻 Vision 与常驻 PBE language（typed E→Join→mixed Prefill/Decode）演示只有一个命令：

```bash
PBE_RUN_DIR=/tmp/pbe-v4-demo tools/launch/persistent_vlm_online.sh \
  /tmp/Paged-Batch-Engine-build-qwen05 \
  /tmp/Paged-Batch-Engine-models/Qwen2.5-VL-3B-Instruct.pbe-bf16-v1.bin \
  /tmp/Paged-Batch-Engine-models/Qwen2.5-VL-3B-Instruct/tokenizer.json \
  /tmp/Paged-Batch-Engine-models/Qwen2.5-VL-3B-Instruct image.png
```

纯文本回退使用同一数据服务和分页语言核心：

```bash
tools/launch/multi_role_text.sh /tmp/Paged-Batch-Engine-build-qwen05 \
  /tmp/Paged-Batch-Engine-models/Qwen2-0.5B-Instruct.bf16.bin \
  /tmp/Paged-Batch-Engine-models/Qwen2-0.5B-Instruct/tokenizer.json
```

模型准备过程、文件摘要和依赖版本见
[`v4/M1/model_manifest.json`](data_flow_evidence/v4/M1/model_manifest.json)。导出命令是
`tools/models/export_qwen25_vl.py`；运行时必须核对 revision 与导出文件 SHA-256
`526ed2ed568a8d639211d9eb3cd91aaba56f67bbe89017ec2d2bf2bed6a0bf52`。

## 协议与故障边界

数据服务控制协议是固定宽度 little-endian v2；语言 role 使用有界 JSONL v2 typed envelope，reader/mailbox 在推理期间继续接收外部取消与 checkpoint 命令。消息含 owner incarnation、operation ID、
content/representation 摘要、allocation generation 和长度。Host 路径明确计为 copy；同卡
CUDA IPC 映射传输字节为 0，跨卡副本按 payload 全量计费。Tensor bundle 在首笔 compute 前验证
shape、dtype、offset、coverage 和每组件校验和。

正常关闭顺序是停止准入、停止 producer、drain/隔离 copy、释放 importer，最后释放 exporter 与池。
未 seal 的 producer 退出不会发布；一个 consumer 取消不撤销共享物理 flight；未知 CUDA 完成进入
有界 quarantine。Data service 重启会改变 incarnation，旧句柄全部失败；首版没有透明 HA。
进程内 outbox 使用连续 item 序号和提交 cursor，checkpoint 重试不重复入队；这不表示网络 exactly-once。

## 五分钟演示

1. 展示冻结 revision、GPU 和 `M1/model_manifest.json`（30 秒）。
2. 执行上面的单命令图文演示，指出 Vision 与 language 模型各只加载一次，并展示 mixed Prefill/Decode step（约 2 分钟）。
3. 展示 Join 取消、生成中 Decode 取消、统一预算计数，以及 language 退出后的 64/64 页和 0 lease（45 秒）。
4. 打开四分支证据，解释完整页共享、尾页首次 append 的 589,824-byte 全层 COW 和取消一半分支后的引用回稳（45 秒）。
5. 打开 [`M9/final_results.json`](data_flow_evidence/v4/M9/final_results.json)，说明完整公平对照、逐 token ITL、0/50/90% 实际复用矩阵及负收益边界（1 分钟）。

## 面试问答

- **为什么 ContentId、RepresentationId、AllocationHandle 分开？** 前者表示语义内容，中间者表示可执行布局，后者表示一次有 generation 的物理实例；分开后缓存命中不会绕过布局或旧句柄校验。
- **为什么普通 radix 只发布完整页？** 未满页还可能写。分支点通过 BranchSnapshot 保存有效尾页，append 前 COW，避免另一分支看到修改。
- **相同 placeholder 为什么不会复用错图？** radix key 在占位 span 内编码完整 256-bit 图片 ContentId；不同图片至少产生不同 key，token 数和页映射不变。
- **singleflight 与共享缓存是否相同？** singleflight 合并并发生产；缓存复用已完成对象。两者分别开关和计量。
- **取消为何不能立即 free copy buffer？** waiter 生命周期不证明 DMA 已停止；必须等可靠完成、drain 或 quarantine。
- **分角色一定更快吗？** 不一定。低复用会承担进程、权重与传输成本；本交付不宣称超过 SGLang。

## 实测边界

常驻进程内 cache off/on 各随机顺序 5 次：多模态 TTFT 中位数为 2491.70/2483.70 ms，端到端请求时延为
3193.22/3178.80 ms，ITL 为 46.17/46.23 ms。cache off 的 Vision forward 中位数为 37.27 ms，hit 为 0；
processor 与 RPC 仍占主要时间，所以不能把省下的 encoder compute 等同于同量端到端加速。低复用 external KV serving 的
吞吐实测下降 3.26%，本交付不声明吞吐优势。等预算 merged Language 与 separated P/D 的五次对照已完成，墙钟中位数为
23,504.24/24,544.52 ms；它只说明本工作负载中重复模型进程的成本，不外推为通用结论。完整结果见 `M9/final_results.json`。
