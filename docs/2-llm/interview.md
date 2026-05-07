---
title: "模块 2 面试指南"
---

# 模块 2：大语言模型 LLM — 面试指南

> 本指南覆盖 Transformer 注意力机制、LoRA 原理、KV Cache、vLLM PagedAttention 等高频面试题。每道题标注难度和出现频率，方便按优先级复习。

## Transformer 架构

### Q1: Encoder-Only、Decoder-Only、Encoder-Decoder 的区别？

**难度**：⭐⭐⭐ | **频率**：🔥🔥🔥 | **关联**：[Transformer 架构详解](/2-llm/01-transformer-deep-dive)

**答题思路**：三种变体 → 注意力方向 → 代表模型 → 为什么 Decoder-Only 成为主流

**标准答案**：Encoder-Only（BERT）使用双向注意力，适合理解任务。Decoder-Only（GPT/LLaMA）使用因果注意力（只看左边），适合生成任务。Encoder-Decoder（T5）编码器双向 + 解码器因果，适合 seq2seq。现代 LLM 全部采用 Decoder-Only，因为统一范式 + Scaling Laws 表现最好。

**追问**：为什么 BERT 不适合文本生成？Pre-Norm 和 Post-Norm 的区别？

---

## 注意力机制

### Q2: 为什么注意力要除以 √d_k？

**难度**：⭐⭐⭐ | **频率**：🔥🔥🔥 | **关联**：[注意力机制](/2-llm/02-attention-mechanism)

**答题思路**：点积方差 → Softmax 饱和 → 梯度消失

**标准答案**：当 d_k 较大时，Q 和 K 的点积方差为 d_k，值会很大。大的值经过 Softmax 后进入饱和区，梯度接近 0，导致训练困难。除以 √d_k 将方差归一化为 1，使 Softmax 输出更平滑。

**追问**：Flash Attention 的原理？

---

### Q3: KV Cache 的原理和作用？

**难度**：⭐⭐⭐ | **频率**：🔥🔥🔥 | **关联**：[注意力机制](/2-llm/02-attention-mechanism)

**答题思路**：自回归生成 → 重复计算问题 → 缓存 K/V → 复杂度降低

**标准答案**：自回归生成时，每步只生成一个新 Token，但需要与所有历史 Token 计算注意力。KV Cache 缓存已计算的 K 和 V 矩阵，每步只需计算新 Token 的 Q/K/V，将计算复杂度从 O(n²) 降到 O(n)。代价是额外的显存占用。

**追问**：KV Cache 的显存占用如何计算？GQA 如何减少 KV Cache？

---

### Q4: GQA 和 MQA 的区别？

**难度**：⭐⭐⭐ | **频率**：🔥🔥 | **关联**：[注意力机制](/2-llm/02-attention-mechanism)

**答题思路**：MHA → MQA → GQA 的演进 → 各自的 KV Cache 大小

**标准答案**：MHA 每个头有独立 K/V，KV Cache 最大。MQA 所有头共享一组 K/V，Cache 最小但质量可能下降。GQA 将头分组共享 K/V，是折中方案。LLaMA 2/3 和 Qwen2 使用 GQA（如 32 个 Q 头，8 个 KV 头）。

---

## 位置编码

### Q5: RoPE 的核心思想？

**难度**：⭐⭐⭐ | **频率**：🔥🔥🔥 | **关联**：[位置编码](/2-llm/03-position-encoding)

**答题思路**：旋转编码 → 相对位置 → 外推能力

**标准答案**：RoPE 将位置信息编码为旋转角度，对 Q 和 K 向量进行旋转。旋转后 Q·K 的点积自然包含相对位置信息（只依赖 m-n），不需要额外的位置编码向量。优势：支持长序列外推（配合 NTK-aware/YaRN 缩放），是 LLaMA/Qwen/Mistral 的标配。

**追问**：RoPE 和 ALiBi 的区别？如何实现长序列外推？

---

## 训练流程

### Q6: LLM 训练的三个阶段？

**难度**：⭐⭐⭐ | **频率**：🔥🔥🔥 | **关联**：[训练流程](/2-llm/04-training-pipeline)

**答题思路**：预训练 → SFT → 对齐，每个阶段的目标和数据

**标准答案**：(1) 预训练：在 TB 级文本上学习语言知识，任务是下一个 Token 预测。(2) SFT 指令微调：在 10K-100K 条指令数据上训练，让模型学会遵循指令。(3) 对齐（RLHF/DPO）：用人类偏好数据优化，使输出有用、无害、诚实。

**追问**：DPO 和 RLHF 的区别？为什么需要对齐？

---

## LoRA 微调

### Q7: LoRA 的原理？

**难度**：⭐⭐⭐ | **频率**：🔥🔥🔥 | **关联**：[LoRA/QLoRA](/2-llm/07-lora-qlora)

**答题思路**：低秩分解 → 参数效率 → 推理无开销

**标准答案**：LoRA 通过低秩分解近似权重更新：W' = W + B×A，其中 W 冻结，只训练低秩矩阵 A(d_in, r) 和 B(r, d_out)。当 r 远小于 d_in 和 d_out 时，可训练参数量减少 99%+，效果接近全参数微调。推理时可将 LoRA 权重合并回原始权重，无额外开销。

**追问**：秩 r 如何选择？QLoRA 的改进？目标模块如何选择？

---

### Q8: QLoRA 相比 LoRA 的改进？

**难度**：⭐⭐⭐ | **频率**：🔥🔥🔥 | **关联**：[LoRA/QLoRA](/2-llm/07-lora-qlora)

**答题思路**：NF4 量化 → 双重量化 → 分页优化器 → 显存对比

**标准答案**：QLoRA 三大创新：(1) NF4 量化基座模型到 4-bit，显存减少 4x。(2) 双重量化，对量化常数再量化。(3) 分页优化器，显存不足时卸载到 CPU。效果：7B 模型微调只需 ~6GB 显存（vs LoRA ~16GB），效果与全参数微调相当。

---

## 推理部署

### Q9: vLLM 的 PagedAttention 原理？

**难度**：⭐⭐⭐⭐ | **频率**：🔥🔥🔥 | **关联**：[vLLM 推理加速](/2-llm/12-vllm-deployment)

**答题思路**：KV Cache 碎片化 → 分页管理 → 显存利用率

**标准答案**：PagedAttention 借鉴操作系统虚拟内存分页，将 KV Cache 分成固定大小的块（pages），非连续存储，按需分配。解决了传统方案 KV Cache 碎片化问题（利用率 ~50%），提升到 ~95%，吞吐量提升 2-4x。

**追问**：Continuous Batching 的原理？Tensor Parallelism 如何工作？

---

### Q10: 模型量化的原理和作用？

**难度**：⭐⭐⭐ | **频率**：🔥🔥 | **关联**：[量化与 GGUF](/2-llm/11-quantization-gguf)

**答题思路**：精度降低 → 大小减少 → 质量损失 → 量化级别选择

**标准答案**：量化将模型权重从高精度（FP16）转为低精度（INT4/INT8），减少模型大小和显存需求。7B 模型从 14GB（FP16）压缩到 4GB（Q4_K_M），可在消费级 GPU 上运行。Q4_K_M 级别质量损失极小，是最佳性价比选择。

---

## Tokenizer

### Q11: BPE 算法的原理？

**难度**：⭐⭐⭐ | **频率**：🔥🔥 | **关联**：[Tokenizer](/2-llm/15-tokenizer)

**答题思路**：字符级开始 → 合并高频对 → 子词分词

**标准答案**：BPE 从字符级别开始，迭代合并频率最高的相邻 Token 对，直到达到目标词表大小。它是子词分词方法，平衡了字符级（词表小但序列长）和词级（词表大但 OOV 多）的优缺点。GPT 系列使用 Byte-level BPE（tiktoken）。

---

## 复习优先级

| 优先级 | 知识点 | 面试频率 |
|:------:|--------|:--------:|
| P0 | Transformer 三种变体、注意力机制 | 🔥🔥🔥 |
| P0 | KV Cache、GQA | 🔥🔥🔥 |
| P0 | LoRA/QLoRA 原理 | 🔥🔥🔥 |
| P0 | vLLM PagedAttention | 🔥🔥🔥 |
| P0 | RoPE 位置编码 | 🔥🔥🔥 |
| P0 | 训练三阶段（预训练/SFT/RLHF） | 🔥🔥🔥 |
| P1 | 模型量化、GGUF | 🔥🔥 |
| P1 | BPE 分词算法 | 🔥🔥 |
| P1 | Scaling Laws | 🔥🔥 |
| P2 | DPO vs RLHF | 🔥🔥 |
| P2 | 部署方案对比 | 🔥🔥 |
