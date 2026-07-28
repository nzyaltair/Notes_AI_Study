# Agent系统

## 1. 概述

Agent是大语言模型应用的高级形态——能够自主感知环境、规划任务、使用工具、观察反馈并迭代执行直至完成目标的智能体。Agent代表了AI从"回答问题"到"完成任务"的范式转变。

Agent系统的核心公式：**Agent = LLM + 规划 + 工具 + 记忆 + 反馈循环**。LLM作为"大脑"进行推理决策，工具扩展行动能力，记忆维持上下文，反馈循环实现自我修正。

## 2. 发展历史

| 时间 | 里程碑 | 意义 |
|------|--------|------|
| 2022.10 | ReAct框架（Yao et al.） | 推理与行动结合，Agent范式雏形 |
| 2022.10 | Reflexion（Shinn et al.） | 反思机制，Agent自我改进 |
| 2023.03 | AutoGPT / BabyAGI | 自主Agent引发公众关注热潮 |
| 2023.03 | LangChain Agent模块 | Agent开发框架化 |
| 2023.04 | Generative Agents（斯坦福小镇） | 记忆流与社交互动Agent |
| 2023.05 | Toolformer（Schick et al.） | LLM自学习工具使用 |
| 2023.10 | AutoGen v2 | 多Agent对话框架 |
| 2023.10 | MemGPT（Packer et al.） | 操作系统式记忆管理 |
| 2024.01 | CrewAI | 角色化多Agent协作 |
| 2024.03 | LangGraph | 状态图工作流，Agent工程化 |
| 2024.09 | OpenAI o1推理模型 | 推理模型增强Agent规划能力 |
| 2024.10 | Computer Use（Anthropic） | Agent可直接操作GUI界面 |
| 2025 | OpenAI Agents SDK / Agent平台化 | Agent工程化走向成熟 |

## 3. 核心概念

### Agent核心循环

```
Thought（推理）→ Action（行动/工具调用）→ Observation（观察结果）→ 循环 → Final Answer
```

### Agent与LLM的区别

| 维度 | LLM（聊天机器人） | Agent |
|------|-------------------|-------|
| 交互模式 | 单轮问答 | 多轮自主执行 |
| 行动能力 | 仅生成文本 | 调用工具、执行操作 |
| 状态管理 | 无状态 | 有记忆和上下文 |
| 目标导向 | 被动响应 | 主动规划完成目标 |
| 错误处理 | 无法修正 | 可反思和重试 |
| 执行时长 | 秒级 | 分钟到小时级 |

### 关键术语

- **Planning**：将复杂任务分解为可执行的子任务序列
- **Tool Use**：调用外部工具（API、代码执行器、数据库等）
- **Memory**：存储和检索历史信息，维持上下文
- **Reflection**：对执行结果的反思和自我评估
- **Autonomy**：Agent自主决策的程度
- **Grounding**：Agent与外部环境（API、文件系统、浏览器等）的连接

## 4. 技术原理

### 规划（Planning）

| 方法 | 原理 | 适用场景 | 代表论文 |
|------|------|---------|---------|
| ReAct | 推理与行动交替执行 | 通用Agent | Yao et al., 2022 |
| Plan-and-Execute | 先制定完整计划再执行 | 结构化复杂任务 | |
| Tree of Thoughts | 树形搜索推理路径，可回溯 | 探索性任务 | Yao et al., 2023 |
| Reflexion | 失败后反思再重试 | 调试类任务 | Shinn et al., 2023 |
| LATS | 语言Agent树搜索 | 复杂决策 | Zhou et al., 2023 |

**ReAct循环形式化**：

$$\text{Thought}_t \to \text{Action}_t \to \text{Observation}_t \to \text{Thought}_{t+1}$$

终止条件：$\text{Action}_t = \text{Finish}[\text{answer}]$

### 记忆（Memory）

| 类型 | 特点 | 实现方式 |
|------|------|---------|
| 短期记忆 | 容量有限，访问快，易遗忘 | 对话上下文窗口、滑动窗口 |
| 工作记忆 | 动态更新，任务导向 | 临时Scratchpad、笔记 |
| 长期记忆 | 容量大，相对稳定 | `向量数据库` 、知识图谱 |
| 情景记忆 | 特定事件、交互记录 | 有时间戳的事件存储 |
| 语义记忆 | 事实知识、概念关系 | 结构化知识库 |
| 程序记忆 | 技能、操作流程 | 可执行脚本/工具定义 |

详见 `Agent记忆` 。

### 工具使用（Tool Use）

Agent通过Function Calling调用外部工具：

- **信息获取**：搜索引擎、 `RAG` 检索、数据库查询
- **代码执行**：Python沙箱、代码解释器
- **文件操作**：读写文件、目录管理
- **API调用**：外部服务集成
- **GUI操作**：浏览器自动化、Computer Use

详见 `工具调用与MCP协议` 。

### 多Agent协作

| 模式 | 原理 | 适用场景 | 代表框架 |
|------|------|---------|---------|
| 层级式 | 主管Agent分配任务给子Agent | 项目管理 | AutoGen |
| 对话式 | Agent间对话讨论 | 代码审查、辩论 | AutoGen GroupChat |
| 竞争式 | 多Agent各自解决，取最优 | 数学求解、创意生成 | |
| 流水线式 | Agent串联各负责一步 | 内容生产 | CrewAI |
| 市场式 | Agent竞标任务，最优者执行 | 动态任务分配 | |

### 反思机制

```
执行任务 → 评估结果 → 如果失败：
  → 分析失败原因（Why did it fail?）
  → 提取经验教训（What should I do differently?）
  → 生成改进策略（How to fix it?）
  → 重试（with new strategy）
```

## 5. 关键方法与模型

### 核心算法

| 算法 | 核心思想 | 论文 |
|------|---------|------|
| ReAct | 推理与行动交替循环 | Yao et al., 2022 |
| Reflexion | 基于反思的自我改进 | Shinn et al., 2023 |
| Tree of Thoughts | 树搜索推理路径 | Yao et al., 2023 |
| Plan-and-Execute | 先规划后执行 | |
| Multi-Agent Debate | 多Agent辩论提升推理质量 | Du et al., 2023 |
| CICERO | 社交互动Agent | Meta, 2022 |

### 代表性Agent系统

| 系统 | 特点 | 贡献 |
|------|------|------|
| AutoGPT | 首个开源自主Agent | 引发Agent热潮 |
| Generative Agents | 斯坦福小镇Agent | 记忆流与社交模拟 |
| MemGPT | 操作系统式记忆管理 | 分层记忆管理范式 |
| Devin | AI软件工程师 | 编程Agent的里程碑 |
| Claude Code | 终端Agent编程助手 | 实用Agent工具 |
| Computer Use | GUI操作Agent | Agent操作真实界面 |

## 6. 优势与局限

### 优势

- **自主性**：无需人工干预即可完成复杂任务
- **通用性**：通过工具组合可处理多种类型任务
- **可扩展性**：新工具即新能力，无需重训模型
- **持续改进**：通过反思机制不断优化执行策略

### 局限

- **可靠性不足**：Agent可能在多步执行中累积错误
- **成本高昂**：多轮LLM调用+工具执行，成本和延迟显著
- **评估困难**：Agent行为具有非确定性，难以系统评估
- **安全风险**：自主执行能力带来安全隐患（误操作、数据泄露）
- **上下文管理复杂**：长任务需要有效的记忆和上下文管理
- **工具选择错误**：工具数量多时，LLM可能选择错误工具

## 7. 应用场景

- **编程助手**：Cursor、Claude Code、Devin
- **数据分析**：自主查询数据库、生成分析报告
- **研究助手**：文献检索、综述生成、实验设计
- **自动化办公**：邮件处理、日程管理、文档生成
- **Web Agent**：浏览器操作、网页信息提取
- **多模态Agent**：结合视觉能力的UI操作
- **客服自动化**：工单分类→检索→回复→质检
- **科研自动化**：实验设计、数据分析、论文撰写

## 8. 与其他技术关系

- Agent以 `大语言模型` 为核心推理引擎
- `思维链` 和ReAct是Agent规划的基础
- `工具调用与MCP协议` 是Agent的行动能力
- `RAG` 是Agent的知识来源
- `提示工程` 定义Agent行为
- `多模态模型` 扩展Agent感知范围
- `Agent框架` 提供Agent开发的基础设施
- `Agent记忆` 维持Agent的上下文和知识

## 9. 前沿发展

- **推理模型驱动Agent**：o1/o3等推理模型从根本上改变Agent规划方式，模型内部推理替代外部CoT提示
- **Computer Use Agent**：Agent直接操作GUI界面，实现通用自动化（Anthropic Computer Use、OpenAI Operator）
- **多Agent系统**：从单Agent走向多Agent协作，模拟人类团队分工
- **Agent操作系统**：将Agent作为操作系统级原语，统一调度与管理（AIOS等）
- **端侧Agent**：在手机/PC本地运行的Agent，隐私保护与低延迟（Apple Intelligence等）
- **Agent可观测性**：Agent行为的追踪、调试和评估工具成熟
- **Agent安全**：Agent权限控制、沙箱执行、行为审计等安全技术
- **长程任务Agent**：支持小时级甚至天级的长时间任务执行

## 相关知识

- 前置：[[../../10_大语言模型核心架构/09_推理模型与思维链|推理模型与思维链]]、[[../05_工具调用与MCP协议|工具调用与MCP协议]]
- 平级：[[../06_自动化工作流|自动化工作流]]、[[../03_RAG系统/00_RAG系统|RAG系统]]
- 延伸：[[../../10_大语言模型核心架构/00_大语言模型核心架构_综述|大语言模型核心架构综述]]
