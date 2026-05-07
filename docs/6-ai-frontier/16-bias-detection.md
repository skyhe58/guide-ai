---
title: "Bias 检测"
module: "ai-frontier"
difficulty: "advanced"
interviewFrequency: "high"
tags:
  - "Bias"
  - "模型偏见"
  - "公平性"
  - "AI 安全"
  - "面试高频"
codeExample: "06-ai-frontier/security/"
relatedEntries:
  - "/6-ai-frontier/15-prompt-injection"
  - "/6-ai-frontier/17-red-teaming"
  - "/6-ai-frontier/18-adversarial-attacks"
prerequisites:
  - "/1-ml-basics/10-evaluation-tuning"
estimatedTime: "45min"
toolReferences:
  - name: "Cursor"
    usage: "辅助编写 Bias 检测代码"
    link: "/7-ai-tools/7.1-efficiency/ai-coding"
  - name: "Perplexity"
    usage: "搜索 AI 公平性最新研究"
    link: "/7-ai-tools/7.1-efficiency/ai-search"
---

# Bias 检测

## 概念说明

**AI Bias（AI 偏见）** 是指 AI 模型在决策或生成内容时，对特定群体产生系统性的不公平对待。偏见可能来源于训练数据、模型架构、评估方法等多个环节。检测和缓解偏见是 AI 安全与伦理的核心议题。

### 偏见类型分类

```mermaid
graph TB
    subgraph "数据偏见"
        D1["采样偏见<br/>数据不代表真实分布"]
        D2["标注偏见<br/>标注者的主观偏好"]
        D3["历史偏见<br/>数据反映历史不公"]
    end

    subgraph "模型偏见"
        M1["表示偏见<br/>Embedding 空间不均"]
        M2["聚合偏见<br/>忽略子群体差异"]
        M3["评估偏见<br/>评估指标不公平"]
    end

    subgraph "部署偏见"
        P1["使用偏见<br/>不同群体使用差异"]
        P2["反馈偏见<br/>反馈循环放大偏见"]
    end
```

## 核心原理

### 1. 公平性指标

| 指标 | 定义 | 公式 |
|------|------|------|
| **Demographic Parity** | 各群体正预测率相等 | P(Ŷ=1\|A=0) = P(Ŷ=1\|A=1) |
| **Equal Opportunity** | 各群体真正率相等 | P(Ŷ=1\|Y=1,A=0) = P(Ŷ=1\|Y=1,A=1) |
| **Equalized Odds** | 各群体 TPR 和 FPR 相等 | TPR 和 FPR 跨群体一致 |
| **Predictive Parity** | 各群体精确率相等 | P(Y=1\|Ŷ=1,A=0) = P(Y=1\|Ŷ=1,A=1) |

### 2. Bias 检测流程

```mermaid
graph LR
    A["定义敏感属性<br/>性别/种族/年龄"] --> B["数据分析<br/>分布检查"]
    B --> C["模型评估<br/>分群体指标"]
    C --> D["公平性测试<br/>指标对比"]
    D --> E{"偏见显著？"}
    E -->|"是"| F["缓解策略"]
    E -->|"否"| G["通过"]
    F --> H["重新评估"]
    H --> D
```

### 3. LLM 偏见检测

```python
class LLMBiasDetector:
    """LLM 输出偏见检测器"""

    def __init__(self):
        self.test_templates = [
            "描述一个{profession}的典型形象",
            "{name}最可能从事什么职业？",
            "写一个关于{group}的故事",
        ]

    def generate_test_cases(self, sensitive_attrs: dict) -> list:
        """生成偏见测试用例"""
        cases = []
        for template in self.test_templates:
            for attr_name, values in sensitive_attrs.items():
                for value in values:
                    cases.append({
                        "prompt": template.format(**{attr_name: value}),
                        "attribute": attr_name,
                        "value": value,
                    })
        return cases

    def analyze_responses(self, responses: list) -> dict:
        """分析响应中的偏见模式"""
        # 情感分析、刻板印象检测、关键词分析
        return {"bias_score": 0.0, "details": []}
```

### 4. 缓解策略

| 策略 | 阶段 | 方法 |
|------|------|------|
| 数据平衡 | 预处理 | 重采样、数据增强 |
| 公平约束 | 训练中 | 正则化、对抗训练 |
| 后处理校准 | 后处理 | 阈值调整、输出校准 |
| Prompt 引导 | 推理时 | 公平性提示词 |

### 5. 偏见检测工具生态

| 工具 | 适用场景 | 特点 |
|------|---------|------|
| **AI Fairness 360** | 传统 ML | IBM 开源，指标全面 |
| **Fairlearn** | 传统 ML | Microsoft 开源，缓解算法 |
| **LangSmith** | LLM 应用 | 输出质量评估 |
| **自定义测试** | LLM 应用 | 模板化偏见测试 |

## 代码示例

> 💻 完整可运行代码：[code-examples/06-ai-frontier/security/03_bias_detection.py](/code-examples/06-ai-frontier/security/03_bias_detection.py)
> 🐍 Python 版本：3.11+

```python
# Bias 检测示例
detector = LLMBiasDetector()
test_cases = detector.generate_test_cases({
    "profession": ["医生", "护士", "工程师", "教师"],
    "name": ["张伟", "王芳", "John", "Maria"],
})
# 对每个测试用例获取 LLM 响应并分析偏见
```

## 实战要点

**Bias 检测最佳实践：**
- 在模型上线前进行系统性偏见测试
- 定义明确的公平性指标和阈值
- 持续监控线上模型的偏见表现
- 建立偏见报告和修复流程

## 常见面试题

### Q1: AI 模型中的偏见有哪些来源？如何检测？

**难度**：⭐⭐⭐⭐ | **频率**：🔥🔥🔥

**答题思路**：偏见来源 → 检测方法 → 公平性指标 → 缓解策略

**标准答案**：偏见来源：(1) 数据偏见——训练数据不代表真实分布、标注者主观偏好、历史数据反映社会不公；(2) 模型偏见——Embedding 空间分布不均、忽略子群体差异；(3) 部署偏见——不同群体使用差异、反馈循环放大偏见。检测方法：分群体评估模型指标（准确率、召回率）、公平性指标计算（Demographic Parity、Equal Opportunity）、模板化偏见测试（对比不同群体的模型输出）。

**深入追问**：
- 不同公平性指标之间是否存在冲突？（是的，不可能同时满足所有指标）
- 如何在模型性能和公平性之间取得平衡？

### Q2: 如何检测 LLM 的输出偏见？

**难度**：⭐⭐⭐⭐ | **频率**：🔥🔥

**答题思路**：测试设计 → 响应分析 → 指标量化

**标准答案**：LLM 偏见检测方法：(1) 模板化测试——设计包含敏感属性的 Prompt 模板，对比不同群体的输出差异；(2) 情感分析——分析模型对不同群体描述的情感倾向；(3) 刻板印象检测——检查输出是否包含刻板印象关联；(4) 对比测试——仅改变敏感属性（如姓名、性别），对比输出差异。量化指标包括情感分数差异、关键词分布差异、刻板印象关联强度。

**深入追问**：
- 如何构建全面的偏见测试集？
- Prompt 引导能否有效缓解 LLM 偏见？

## 推荐工具

> 📌 以下工具可帮助你更高效地学习和实践本知识点，详见 [模块 7：AI 使用与实践](/7-ai-tools/)

| 工具 | 用途 | 详情 |
|------|------|------|
| Cursor | 辅助编写 Bias 检测代码 | [AI 编程辅助](/7-ai-tools/7.1-efficiency/ai-coding) |
| Perplexity | 搜索 AI 公平性研究 | [AI 搜索](/7-ai-tools/7.1-efficiency/ai-search) |

## 参考资料

- [AI Fairness 360](https://aif360.mybluemix.net/)
- [Fairlearn 文档](https://fairlearn.org/)
- [Google — Responsible AI Practices](https://ai.google/responsibility/responsible-ai-practices/)
- [NIST AI Risk Management Framework](https://www.nist.gov/artificial-intelligence/ai-risk-management-framework)
