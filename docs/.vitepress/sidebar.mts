import type { DefaultTheme } from 'vitepress'

/* ========== 编码线：模块 0-6 ========== */

const prerequisitesSidebar: DefaultTheme.SidebarItem[] = [
  {
    text: '模块 0：前提准备',
    collapsed: true,
    items: [
      { text: '模块概览', link: '/0-prerequisites/' },
      { text: '异步编程', link: '/0-prerequisites/01-async-programming' },
      { text: '错误处理', link: '/0-prerequisites/02-error-handling' },
      { text: '类型注解', link: '/0-prerequisites/03-type-annotations' },
      { text: '包管理', link: '/0-prerequisites/04-package-management' },
      { text: 'NumPy 基础', link: '/0-prerequisites/05-numpy-basics' },
      { text: 'Pandas 基础', link: '/0-prerequisites/06-pandas-basics' },
      { text: 'Git/GitHub', link: '/0-prerequisites/07-git-github' },
      { text: 'Jupyter Notebook', link: '/0-prerequisites/08-jupyter-notebook' },
      { text: '虚拟环境管理', link: '/0-prerequisites/09-virtual-env' },
      { text: '面试指南', link: '/0-prerequisites/interview' },
      { text: '速查卡片', link: '/0-prerequisites/cheatsheet' },
    ],
  },
]

const mlBasicsSidebar: DefaultTheme.SidebarItem[] = [
  {
    text: '模块 1：AI/ML 基础理论',
    collapsed: true,
    items: [
      { text: '模块概览', link: '/1-ml-basics/' },
      { text: '监督学习', link: '/1-ml-basics/01-supervised-learning' },
      { text: '无监督学习', link: '/1-ml-basics/02-unsupervised-learning' },
      { text: '强化学习', link: '/1-ml-basics/03-reinforcement-learning' },
      { text: '常见算法', link: '/1-ml-basics/04-classic-algorithms' },
      { text: '神经网络', link: '/1-ml-basics/05-neural-networks' },
      { text: 'CNN', link: '/1-ml-basics/06-cnn' },
      { text: 'RNN/LSTM', link: '/1-ml-basics/07-rnn-lstm' },
      { text: 'Transformer', link: '/1-ml-basics/08-transformer' },
      { text: '数学基础', link: '/1-ml-basics/09-math-foundations' },
      { text: '评估与调优', link: '/1-ml-basics/10-evaluation-tuning' },
      { text: '损失函数', link: '/1-ml-basics/11-loss-functions' },
      { text: '面试指南', link: '/1-ml-basics/interview' },
      { text: '速查卡片', link: '/1-ml-basics/cheatsheet' },
    ],
  },
]

const llmSidebar: DefaultTheme.SidebarItem[] = [
  {
    text: '模块 2：大语言模型 LLM',
    collapsed: true,
    items: [
      { text: '模块概览', link: '/2-llm/' },
      { text: 'Transformer 架构详解', link: '/2-llm/01-transformer-deep-dive' },
      { text: '注意力机制', link: '/2-llm/02-attention-mechanism' },
      { text: '位置编码', link: '/2-llm/03-position-encoding' },
      { text: '训练流程', link: '/2-llm/04-training-pipeline' },
      { text: 'Scaling Laws', link: '/2-llm/05-scaling-laws' },
      { text: '主流模型对比', link: '/2-llm/06-model-comparison' },
      { text: 'LoRA/QLoRA 微调', link: '/2-llm/07-lora-qlora' },
      { text: '全参数微调', link: '/2-llm/08-full-finetuning' },
      { text: '微调数据准备', link: '/2-llm/09-data-preparation' },
      { text: '微调工具', link: '/2-llm/10-finetuning-tools' },
      { text: '量化与 GGUF', link: '/2-llm/11-quantization-gguf' },
      { text: 'vLLM 推理加速', link: '/2-llm/12-vllm-deployment' },
      { text: 'Ollama 本地部署', link: '/2-llm/13-ollama-local' },
      { text: 'TGI 推理服务', link: '/2-llm/14-tgi-deployment' },
      { text: 'Tokenizer', link: '/2-llm/15-tokenizer' },
      { text: '面试指南', link: '/2-llm/interview' },
      { text: '速查卡片', link: '/2-llm/cheatsheet' },
    ],
  },
]

const aiAppsSidebar: DefaultTheme.SidebarItem[] = [
  {
    text: '模块 3：AI 应用开发 ⭐',
    collapsed: true,
    items: [
      { text: '模块概览', link: '/3-ai-apps/' },
      { text: 'Prompt Engineering', link: '/3-ai-apps/01-prompt-engineering' },
      { text: 'Chain-of-Thought', link: '/3-ai-apps/02-chain-of-thought' },
      { text: 'Few-shot Learning', link: '/3-ai-apps/03-few-shot-learning' },
      { text: '系统提示词设计', link: '/3-ai-apps/04-system-prompt' },
      { text: '文档加载', link: '/3-ai-apps/05-document-loading' },
      { text: '文档切分', link: '/3-ai-apps/06-text-splitting' },
      { text: 'Embedding 模型', link: '/3-ai-apps/07-embedding-models' },
      { text: '向量数据库', link: '/3-ai-apps/08-vector-databases' },
      { text: '检索策略', link: '/3-ai-apps/09-retrieval-strategies' },
      { text: 'Rerank 重排序', link: '/3-ai-apps/10-rerank' },
      { text: 'RAG 优化', link: '/3-ai-apps/11-rag-optimization' },
      { text: 'Function Calling', link: '/3-ai-apps/12-function-calling' },
      { text: 'Tool Use', link: '/3-ai-apps/13-tool-use' },
      { text: 'ReAct 模式', link: '/3-ai-apps/14-react-pattern' },
      { text: 'Multi-Agent', link: '/3-ai-apps/15-multi-agent' },
      { text: 'Agent 记忆', link: '/3-ai-apps/16-agent-memory' },
      { text: 'LangChain', link: '/3-ai-apps/17-langchain' },
      { text: 'LangGraph', link: '/3-ai-apps/18-langgraph' },
      { text: 'LlamaIndex', link: '/3-ai-apps/19-llamaindex' },
      { text: '框架选型对比', link: '/3-ai-apps/20-framework-comparison' },
      { text: 'LangSmith', link: '/3-ai-apps/21-langsmith' },
      { text: 'RAG 评估指标', link: '/3-ai-apps/22-rag-evaluation' },
      { text: '评估框架', link: '/3-ai-apps/23-evaluation-frameworks' },
      { text: '面试指南', link: '/3-ai-apps/interview' },
      { text: '速查卡片', link: '/3-ai-apps/cheatsheet' },
    ],
  },
]

const cvSidebar: DefaultTheme.SidebarItem[] = [
  {
    text: '模块 4：计算机视觉（可选）',
    collapsed: true,
    items: [
      { text: '模块概览', link: '/4-cv/' },
      { text: 'OpenCV 基础', link: '/4-cv/01-opencv-basics' },
      { text: '图像处理', link: '/4-cv/02-image-processing' },
      { text: '颜色空间', link: '/4-cv/03-color-spaces' },
      { text: '图像变换', link: '/4-cv/04-image-transforms' },
      { text: '视频处理', link: '/4-cv/05-video-processing' },
      { text: 'YOLO 目标检测', link: '/4-cv/06-yolo-detection' },
      { text: '模型训练', link: '/4-cv/07-model-training' },
      { text: '模型微调', link: '/4-cv/08-model-finetuning' },
      { text: '模型评估', link: '/4-cv/09-model-evaluation' },
      { text: '模型导出', link: '/4-cv/10-model-export' },
      { text: 'Diffusion Model', link: '/4-cv/11-diffusion-model' },
      { text: 'Stable Diffusion', link: '/4-cv/12-stable-diffusion' },
      { text: 'Diffusers 库', link: '/4-cv/13-diffusers-library' },
      { text: 'LLaVA 多模态', link: '/4-cv/14-llava-multimodal' },
      { text: '视觉-语言模型', link: '/4-cv/15-vision-language' },
      { text: '语义分割', link: '/4-cv/16-semantic-segmentation' },
      { text: 'SAM', link: '/4-cv/17-sam' },
      { text: '面试指南', link: '/4-cv/interview' },
      { text: '速查卡片', link: '/4-cv/cheatsheet' },
    ],
  },
]

const aiEngineeringSidebar: DefaultTheme.SidebarItem[] = [
  {
    text: '模块 5：AI 工程化',
    collapsed: true,
    items: [
      { text: '模块概览', link: '/5-ai-engineering/' },
      { text: 'MLOps 训练流水线', link: '/5-ai-engineering/01-mlops-pipeline' },
      { text: '实验追踪', link: '/5-ai-engineering/02-experiment-tracking' },
      { text: '模型版本管理', link: '/5-ai-engineering/03-model-registry' },
      { text: 'MLOps 成熟度', link: '/5-ai-engineering/04-mlops-maturity' },
      { text: 'vLLM 推理服务', link: '/5-ai-engineering/05-vllm-serving' },
      { text: 'TGI 推理服务', link: '/5-ai-engineering/06-tgi-serving' },
      { text: 'API 网关', link: '/5-ai-engineering/07-api-gateway' },
      { text: '负载均衡', link: '/5-ai-engineering/08-load-balancing' },
      { text: '缓存策略', link: '/5-ai-engineering/09-caching-strategies' },
      { text: 'GPU 选型', link: '/5-ai-engineering/10-gpu-selection' },
      { text: '显存优化', link: '/5-ai-engineering/11-memory-optimization' },
      { text: '混合精度训练', link: '/5-ai-engineering/12-mixed-precision' },
      { text: '批处理策略', link: '/5-ai-engineering/13-batching-strategies' },
      { text: '数据标注', link: '/5-ai-engineering/14-data-labeling' },
      { text: '数据清洗', link: '/5-ai-engineering/15-data-cleaning' },
      { text: '数据增强', link: '/5-ai-engineering/16-data-augmentation' },
      { text: '合成数据', link: '/5-ai-engineering/17-synthetic-data' },
      { text: '成本优化', link: '/5-ai-engineering/18-cost-optimization' },
      { text: 'LangSmith 监控', link: '/5-ai-engineering/19-langsmith-monitoring' },
      { text: 'Prometheus + Grafana', link: '/5-ai-engineering/20-prometheus-grafana' },
      { text: '日志管理', link: '/5-ai-engineering/21-logging' },
      { text: '告警策略', link: '/5-ai-engineering/22-alerting' },
      { text: '面试指南', link: '/5-ai-engineering/interview' },
      { text: '速查卡片', link: '/5-ai-engineering/cheatsheet' },
    ],
  },
]

const aiFrontierSidebar: DefaultTheme.SidebarItem[] = [
  {
    text: '模块 6：AI 前沿与趋势',
    collapsed: true,
    items: [
      { text: '模块概览', link: '/6-ai-frontier/' },
      { text: 'MCP 协议原理', link: '/6-ai-frontier/01-mcp-protocol' },
      { text: 'MCP Server 开发', link: '/6-ai-frontier/02-mcp-server-dev' },
      { text: 'Agent 间通信', link: '/6-ai-frontier/03-mcp-agent-comm' },
      { text: 'GitHub Copilot', link: '/6-ai-frontier/04-copilot' },
      { text: 'Cursor', link: '/6-ai-frontier/05-cursor' },
      { text: 'Kiro', link: '/6-ai-frontier/06-kiro' },
      { text: 'Trae', link: '/6-ai-frontier/07-trae' },
      { text: 'IDE 选型对比', link: '/6-ai-frontier/08-ide-comparison' },
      { text: 'Harness 工程', link: '/6-ai-frontier/09-harness-engineering' },
      { text: 'Archon 框架', link: '/6-ai-frontier/10-archon' },
      { text: 'OpenClaw', link: '/6-ai-frontier/11-openclaw' },
      { text: 'Agent 安全', link: '/6-ai-frontier/12-agent-security' },
      { text: 'Kiro Skills', link: '/6-ai-frontier/13-kiro-skills' },
      { text: '能力扩展模式', link: '/6-ai-frontier/14-plugin-patterns' },
      { text: 'Prompt Injection', link: '/6-ai-frontier/15-prompt-injection' },
      { text: 'Bias 检测', link: '/6-ai-frontier/16-bias-detection' },
      { text: '红队测试', link: '/6-ai-frontier/17-red-teaming' },
      { text: '对抗攻击', link: '/6-ai-frontier/18-adversarial-attacks' },
      { text: '多模态 API 实战', link: '/6-ai-frontier/19-multimodal-fusion' },
      { text: '跨模态趋势', link: '/6-ai-frontier/20-cross-modal-trends' },
      { text: 'Vibe Coding', link: '/6-ai-frontier/21-vibe-coding' },
      { text: '零代码 AI 工具', link: '/6-ai-frontier/22-zero-code-ai' },
      { text: '面试指南', link: '/6-ai-frontier/interview' },
      { text: '速查卡片', link: '/6-ai-frontier/cheatsheet' },
    ],
  },
]

/* ========== 使用线：模块 7 ========== */

const aiToolsSidebar: DefaultTheme.SidebarItem[] = [
  {
    text: '7.1 AI 效率工具',
    collapsed: true,
    items: [
      { text: '概览', link: '/7-ai-tools/7.1-efficiency/' },
      { text: 'AI 对话助手', link: '/7-ai-tools/7.1-efficiency/ai-chat' },
      { text: 'AI 搜索', link: '/7-ai-tools/7.1-efficiency/ai-search' },
      { text: 'AI 写作', link: '/7-ai-tools/7.1-efficiency/ai-writing' },
      { text: 'AI 办公', link: '/7-ai-tools/7.1-efficiency/ai-office' },
      { text: 'AI 阅读/学习', link: '/7-ai-tools/7.1-efficiency/ai-reading' },
      { text: 'AI 编程辅助', link: '/7-ai-tools/7.1-efficiency/ai-coding' },
      { text: 'Prompt 技巧', link: '/7-ai-tools/7.1-efficiency/prompt-tips' },
    ],
  },
  {
    text: '7.2 AIGC 内容创作',
    collapsed: true,
    items: [
      { text: '概览', link: '/7-ai-tools/7.2-aigc/' },
      { text: 'AI 图像生成', link: '/7-ai-tools/7.2-aigc/image-generation' },
      { text: 'AI 视频生成', link: '/7-ai-tools/7.2-aigc/video-generation' },
      { text: 'AI 短剧制作', link: '/7-ai-tools/7.2-aigc/short-drama' },
      { text: 'AI 音频/语音', link: '/7-ai-tools/7.2-aigc/audio-voice' },
      { text: 'AI 数字人', link: '/7-ai-tools/7.2-aigc/digital-human' },
      { text: 'ComfyUI 进阶', link: '/7-ai-tools/7.2-aigc/comfyui-advanced' },
    ],
  },
  {
    text: '7.3 AI 商业变现',
    collapsed: true,
    items: [
      { text: '概览', link: '/7-ai-tools/7.3-business/' },
      { text: 'AI 自媒体', link: '/7-ai-tools/7.3-business/ai-media' },
      { text: 'AI 副业', link: '/7-ai-tools/7.3-business/ai-side-hustle' },
      { text: 'AI 产品思维', link: '/7-ai-tools/7.3-business/ai-product' },
      { text: 'AI 工具选型', link: '/7-ai-tools/7.3-business/ai-tool-selection' },
    ],
  },
]

/* ========== 面试汇总 ========== */

const interviewSidebar: DefaultTheme.SidebarItem[] = [
  {
    text: '面试汇总',
    collapsed: true,
    items: [
      { text: '面试总入口', link: '/interview/' },
      { text: '按岗位分类', link: '/interview/by-position' },
      { text: '按公司分类', link: '/interview/by-company' },
      { text: '按难度分类', link: '/interview/by-difficulty' },
      { text: '知识图谱', link: '/interview/knowledge-map' },
    ],
  },
]

/* ========== 学习路径 ========== */

const learningPathsSidebar: DefaultTheme.SidebarItem[] = [
  {
    text: '学习路径',
    collapsed: true,
    items: [
      { text: '4 个月速成', link: '/learning-paths/fast-track' },
      { text: '6-8 个月全栈', link: '/learning-paths/full-stack' },
      { text: 'AI 工具使用者', link: '/learning-paths/tool-user' },
    ],
  },
]

/* ========== 导出侧边栏配置 ========== */

export const sidebar: DefaultTheme.Sidebar = {
  '/guide/': [
    {
      text: '指南',
      items: [
        { text: '快速开始', link: '/guide/getting-started' },
        { text: '使用指南', link: '/guide/how-to-use' },
        { text: 'GPU 环境配置', link: '/guide/gpu-setup' },
      ],
    },
  ],
  '/0-prerequisites/': prerequisitesSidebar,
  '/1-ml-basics/': mlBasicsSidebar,
  '/2-llm/': llmSidebar,
  '/3-ai-apps/': aiAppsSidebar,
  '/4-cv/': cvSidebar,
  '/5-ai-engineering/': aiEngineeringSidebar,
  '/6-ai-frontier/': aiFrontierSidebar,
  '/7-ai-tools/': aiToolsSidebar,
  '/interview/': interviewSidebar,
  '/learning-paths/': learningPathsSidebar,
}
