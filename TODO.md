# TODO 改进项

## 模块 2（大语言模型 LLM）文档增强

> 子代理批量创建的 15 篇文档（docs/2-llm/02-15）内容深度不如早期手动创建的文档，需要后续增强。

### 需要补充 Mermaid 图的文档
- [ ] `02-attention-mechanism.md` — 补充 KV Cache 工作流程图、Flash Attention 对比图
- [ ] `07-lora-qlora.md` — 补充 LoRA 权重分解示意图、QLoRA 架构图
- [ ] `11-quantization-gguf.md` — 补充量化流程图
- [ ] `12-vllm-deployment.md` — 补充 PagedAttention 分页示意图（当前只有文字描述）
- [ ] `15-tokenizer.md` — 补充 BPE 合并过程图

### 需要丰富面试题追问的文档
- [ ] `02-attention-mechanism.md` — 每道面试题补充 2-3 个追问
- [ ] `07-lora-qlora.md` — 补充追问（如 LoRA 与 Adapter 的区别）
- [ ] `12-vllm-deployment.md` — 补充追问（如 vLLM vs TGI 性能对比）
- [ ] `15-tokenizer.md` — 补充追问（如中文分词的挑战）

### 需要丰富实战要点的文档
- [ ] 所有后期文档的"实战要点"从 3-4 条扩展到 6-8 条，增加代码示例和场景分析

### 推荐工具链接补充
- [ ] `12-vllm-deployment.md` — 补充至少 2 个工具链接（当前只有 1 个）
- [ ] `15-tokenizer.md` — 补充工具链接

### 整体质量对齐目标
- 每篇文档 150+ 行（当前部分只有 117 行）
- 每篇文档至少 1 个 Mermaid 图（复杂流程必须有）
- 每道面试题至少 2 个追问
- 实战要点分场景、有代码片段
