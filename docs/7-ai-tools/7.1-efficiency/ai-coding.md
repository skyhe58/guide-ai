---
title: "AI 编程辅助"
module: "ai-tools"
tags:
  - "GitHub Copilot"
  - "Cursor"
  - "Kiro"
  - "Trae"
  - "AI 编程"
estimatedTime: "30min"
toolReferences:
  - name: AI 对话助手
    usage: 复杂问题先用对话助手分析思路，再用编程工具实现
    link: /7-ai-tools/7.1-efficiency/ai-chat
  - name: Prompt 技巧
    usage: 掌握编程场景的 Prompt 写法
    link: /7-ai-tools/7.1-efficiency/prompt-tips
  - name: AI 工具选型
    usage: 根据团队和项目需求选择编程工具
    link: /7-ai-tools/7.3-business/ai-tool-selection
  - name: AI 产品思维
    usage: 用 AI 编程工具快速验证产品原型
    link: /7-ai-tools/7.3-business/ai-product
---

# AI 编程辅助

## 概念说明

AI 编程辅助工具通过大语言模型为开发者提供代码补全、代码生成、Bug 修复、代码解释等能力。本节从日常使用视角介绍主流 AI 编程工具的使用技巧，帮助开发者在日常工作中提升编码效率。

> 📌 如需了解这些工具的技术原理和架构设计，请参考 [模块 6：AI 前沿与趋势](/6-ai-frontier/)。

### AI 编程工具发展阶段

```mermaid
graph LR
    A["代码补全<br/>Copilot 初代"] --> B["对话式编程<br/>Copilot Chat/Cursor"]
    B --> C["项目级理解<br/>Cursor Composer"]
    C --> D["Spec 驱动开发<br/>Kiro"]
    D --> E["自主编程 Agent<br/>Devin/Copilot Agent"]
```

## 主流工具详解

### GitHub Copilot

GitHub Copilot 是最早商业化的 AI 编程助手，拥有最大的用户基数。

**日常使用技巧：**

| 功能 | 快捷键 | 使用场景 |
|------|--------|----------|
| 代码补全 | Tab 接受 | 编写代码时自动建议 |
| Copilot Chat | Ctrl+I | 侧边栏对话式编程 |
| Inline Chat | 选中代码 → Ctrl+I | 对选中代码提问或修改 |
| 解释代码 | 选中 → /explain | 理解他人代码 |
| 修复代码 | 选中 → /fix | 修复 Bug |
| 生成测试 | 选中 → /tests | 自动生成单元测试 |
| 生成文档 | 选中 → /doc | 生成代码注释和文档 |

**高效使用策略：**
```
# 1. 写好注释，引导补全
# 计算两个日期之间的工作日天数，排除周末和法定节假日
def count_business_days(start_date, end_date, holidays=None):
    # Copilot 会根据注释生成完整实现

# 2. 使用 @workspace 提供项目上下文
@workspace 这个项目的数据库连接是怎么配置的？

# 3. 使用 #file 引用特定文件
#file:src/models/user.py 请为这个 User 模型添加邮箱验证方法
```

### Cursor

Cursor 是 AI-first 的代码编辑器，基于 VS Code 构建，以项目级 AI 能力著称。

**核心功能使用：**

| 功能 | 说明 | 使用方式 |
|------|------|----------|
| **Tab 补全** | 智能代码补全 | 编码时自动触发 |
| **Cmd+K** | 内联代码生成/编辑 | 选中代码或空行按 Cmd+K |
| **Chat** | 侧边栏对话 | Cmd+L 打开 |
| **Composer** | 多文件编辑 | Cmd+Shift+I 打开 |
| **@引用** | 引用文件/文档/代码 | 在对话中使用 @ 符号 |

**Composer 多文件编辑实操：**
```
# 场景：添加用户认证功能
打开 Composer（Cmd+Shift+I），输入：

请为这个 FastAPI 项目添加 JWT 用户认证功能：
1. 创建 auth/models.py — 用户模型
2. 创建 auth/routes.py — 登录/注册接口
3. 创建 auth/dependencies.py — 认证依赖
4. 修改 main.py — 注册路由
5. 添加必要的依赖到 requirements.txt

@file:main.py @file:requirements.txt

# Cursor 会同时修改多个文件，生成完整的认证模块
```

**Cursor Rules 配置：**
```
# .cursorrules 文件
你是一位资深的 Python 后端工程师。
项目使用 FastAPI + SQLAlchemy + PostgreSQL。
代码风格要求：
- 使用类型注解
- 遵循 PEP 8
- 中文注释
- 异步优先
- 错误处理完善
```

### Kiro

Kiro 是 AWS 推出的 AI IDE，核心特色是 Spec 驱动开发。

**核心功能使用：**

| 功能 | 说明 | 使用方式 |
|------|------|----------|
| **Spec 驱动** | 需求 → 设计 → 任务 → 代码 | 创建 Spec 文件 |
| **Steering** | 项目级规则和约束 | .kiro/steering/ 目录 |
| **Hooks** | 自动化触发动作 | .kiro/hooks/ 配置 |
| **Agent 模式** | 自主完成多步骤任务 | 对话中描述任务 |
| **Vibe 模式** | 快速原型开发 | 自由对话编码 |

**Spec 驱动开发流程：**
```
# 1. 创建 Spec — 描述需求
"我需要一个用户管理模块，支持注册、登录、个人信息修改"

# 2. Kiro 自动生成
- requirements.md — 需求文档（验收标准）
- design.md — 技术设计文档
- tasks.md — 实现任务列表

# 3. 逐个执行任务
Kiro 按任务列表逐步实现代码，每个任务完成后可审查

# 4. Steering 文件引导
在 .kiro/steering/ 中定义项目规范，Kiro 会遵循这些规则
```

**Steering 文件示例：**
```markdown
# coding-standards.md
## 代码规范
- 使用 Python 3.11+ 语法
- 所有函数必须有类型注解
- 使用 Pydantic 进行数据验证
- 异步函数使用 async/await
- 错误处理使用自定义异常类
```

### Trae

Trae 是字节跳动推出的 AI IDE，以中文交互和 Builder 模式著称。

**核心功能使用：**

| 功能 | 说明 | 使用方式 |
|------|------|----------|
| **Builder 模式** | 从描述生成完整项目 | 描述需求，自动生成 |
| **Chat** | 对话式编程 | 侧边栏对话 |
| **代码补全** | 智能补全 | 编码时自动触发 |
| **中文优化** | 中文理解和交互 | 自然中文对话 |

**Builder 模式实操：**
```
# 场景：快速生成一个 Web 应用
在 Builder 模式中输入：

请帮我创建一个待办事项 Web 应用：
- 前端：React + Tailwind CSS
- 后端：FastAPI
- 数据库：SQLite
- 功能：增删改查、标记完成、按优先级排序

# Trae 会自动生成完整的前后端项目
```

## AI 编程工具选型对比表

| 维度 | Copilot | Cursor | Kiro | Trae |
|------|---------|--------|------|------|
| **基础 IDE** | VS Code 插件 | VS Code Fork | VS Code Fork | VS Code Fork |
| **代码补全** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **对话编程** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **多文件编辑** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **项目理解** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Spec 驱动** | ❌ | ❌ | ✅ 核心特色 | ❌ |
| **中文支持** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **免费额度** | 有限 | 有限 | 充足 | 充足 |
| **付费价格** | $10-19/月 | $20/月 | 免费/Pro | 免费/Pro |
| **模型选择** | GPT-4o/Claude | 多模型可选 | Claude | 多模型可选 |
| **适合人群** | 所有开发者 | 高效率追求者 | 工程化团队 | 中文开发者 |

## 实战要点

### 场景化选型建议

| 使用场景 | 推荐工具 | 理由 |
|----------|----------|------|
| VS Code 用户、轻量辅助 | Copilot | 无缝集成、学习成本低 |
| 追求极致效率 | Cursor | Composer 多文件编辑能力强 |
| 团队协作、工程化项目 | Kiro | Spec 驱动确保代码质量 |
| 中文开发者、快速原型 | Trae | 中文交互好、Builder 模式快 |
| 预算有限 | Trae / Kiro | 免费额度充足 |

### 通用使用技巧

1. **写好注释和文档**：AI 根据注释理解意图，好的注释 = 好的代码生成
2. **提供上下文**：引用相关文件、说明技术栈、描述约束条件
3. **小步迭代**：不要一次生成太多代码，分步骤逐步完善
4. **审查生成代码**：AI 生成的代码必须审查，特别是安全相关逻辑
5. **建立项目规则**：使用 .cursorrules 或 Steering 文件定义项目规范

### 常见使用误区

| 误区 | 正确做法 |
|------|----------|
| 完全信任 AI 生成的代码 | 始终审查，特别是安全和业务逻辑 |
| 用 AI 写所有代码 | 核心架构和设计决策由人做 |
| 不提供上下文 | 引用文件、说明需求、提供示例 |
| 一次生成整个项目 | 分模块、分步骤逐步生成 |
| 忽略测试 | 让 AI 同时生成测试代码 |

## Prompt 模板库

以下是 AI 编程场景中常用的 Prompt 模板，可直接复制使用。

### 代码生成模板

```
请用 [语言/框架] 实现以下功能：
功能描述：[具体功能]
技术要求：
- [要求 1，如：使用 async/await]
- [要求 2，如：包含类型注解]
- [要求 3，如：包含错误处理]
输入输出：
- 输入：[参数说明]
- 输出：[返回值说明]
请同时生成单元测试。
```

### 代码审查模板

```
请审查以下代码，从以下维度分析：
1. 代码质量（命名、结构、可读性）
2. 性能问题（时间/空间复杂度）
3. 安全隐患（注入、越权、数据泄露）
4. 最佳实践（设计模式、SOLID 原则）
对每个问题给出严重程度（高/中/低）和修复建议。
```

### Bug 修复模板

```
以下代码出现了 [错误描述]：
[粘贴代码]
错误信息：[粘贴错误日志]
期望行为：[描述期望结果]
请分析根因并给出修复方案。
```

### 重构模板

```
请重构以下代码，目标：
- 提高可读性和可维护性
- 消除重复代码
- 遵循 [语言] 最佳实践
- 保持功能不变
请解释每处修改的理由。
```

## 注意事项

- **代码安全**：不要让 AI 处理包含密钥、密码的代码
- **代码版权**：了解 AI 生成代码的版权和许可问题
- **技能退化**：保持手写代码的能力，不要过度依赖 AI
- **隐私保护**：注意公司代码是否允许上传到 AI 服务

## 参考资料

- [GitHub Copilot 文档](https://docs.github.com/copilot)
- [Cursor 官方文档](https://docs.cursor.com)
- [Kiro 官方文档](https://kiro.dev/docs)
- [Trae 官方网站](https://www.trae.ai)
