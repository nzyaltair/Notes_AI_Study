---
tags:
  - LLM应用工程
  - 综述
  - 大模型时代
  - Agent
created: 2025-07-10
updated: 2025-07-10
---

# 大语言模型应用工程

## 1. 概述

LLM应用工程（LLM Application Engineering）是以大语言模型为推理核心，通过提示工程、检索增强生成（RAG）、Agent架构、工具调用和结构化输出等技术手段，构建端到端智能应用系统的工程方法论。

传统软件工程以确定性逻辑为基础，而LLM应用工程的核心特征是**以概率性语言模型为推理引擎**，围绕其能力边界设计系统架构。它解决的核心问题是：如何让LLM从"聊天机器人"进化为能够可靠完成实际任务的"智能系统"。

LLM应用工程的技术栈可概括为五大支柱：

- **提示工程**：LLM行为的"软编程"接口
- **结构化输出**：连接自由文本与可编程数据的桥梁
- **RAG系统**：赋予LLM外部知识与事实准确性
- **Agent系统**：实现自主规划、工具使用与任务执行
- **工具调用与MCP**：扩展LLM的行动能力边界

## 2. 发展历史

| 时间 | 里程碑 | 意义 |
|------|--------|------|
| 2020 | GPT-3发布，Few-Shot Learning | 提示工程开端，证明规模带来涌现能力 |
| 2020 | RAG提出（Lewis et al.） | 检索增强生成范式确立 |
| 2020 | DPR密集检索（Karpukhin et al.） | 向量检索取代传统关键词检索 |
| 2022 | Chain-of-Thought（Wei et al.） | 推理引导突破，LLM可逐步推理 |
| 2022 | ReAct框架（Yao et al.） | 推理与行动结合，Agent范式雏形 |
| 2023.03 | GPT-4发布，Function Calling | LLM原生工具调用能力 |
| 2023.03 | LangChain爆发，AutoGPT引发Agent热潮 | Agent框架生态形成 |
| 2023.06 | Toolformer（Schick et al.） | LLM自学习工具使用 |
| 2023.10 | DSPy框架发布 | 提示工程从手工调参走向自动优化 |
| 2023.10 | Self-RAG（Asai et al.） | LLM自评估检索需求 |
| 2024.03 | LangGraph发布 | 状态图工作流，Agent工程化 |
| 2024.09 | ReAct→o1推理模型发布 | 推理模型增强Agent规划能力 |
| 2024.11 | MCP协议发布（Anthropic） | 工具调用标准化，解耦工具与应用 |
| 2024.10 | Computer Use（Anthropic） | Agent可直接操作GUI界面 |
| 2025 | Agent平台化与生产运维 | Agent工程化走向成熟 |

整体演进脉络：**提示工程探索期（2020-2022）→ RAG与工具调用集成期（2023）→ Agent自主执行期（2024至今）**。

## 3. 核心概念

### LLM作为推理核心

LLM应用系统的本质是将LLM作为"大脑"，通过外部组件弥补其不足：

| LLM固有局限 | 工程解决方案 |
|-------------|-------------|
| 知识时效性 | RAG外部知识库 |
| 事实幻觉 | RAG + 来源引用 |
| 无法执行操作 | 工具调用 / Agent |
| 输出不可控 | 结构化输出 |
| 上下文有限 | 记忆系统 / 长上下文扩展 |
| 无持续状态 | Agent记忆与状态管理 |

### 核心范式

```
用户请求 → [Agent规划] → [知识检索(RAG)] → [工具选择与调用] → [结果整合] → [结构化输出] → 最终回答
```

### 关键术语

- **Prompt**：输入给LLM的指令文本，是控制LLM行为的主要接口
- **Context Window**：LLM一次能处理的最大token数量
- **Embedding**：将文本映射为高维向量表示
- **Retrieval**：从知识库中查找与查询相关的内容
- **Tool/Function**：LLM可调用的外部API或函数
- **Agent**：能自主规划、使用工具、观察反馈并迭代执行的智能体
- **Grounding**：将LLM输出锚定到外部可信信息源

## 4. 技术原理

### 技术架构总览

```
┌─────────────────────────────────────────────────┐
│              用户交互层                           │
│        (自然语言输入 / 结构化接口)                  │
├─────────────────────────────────────────────────┤
│              Agent编排层                          │
│   规划(Planning) → 行动(Action) → 观察(Observe)   │
├──────────┬──────────┬──────────┬────────────────┤
│ 提示工程  │ RAG系统  │ 工具调用  │  结构化输出     │
│ Prompt   │ Retriever│ Tool Use │ Structured IO  │
├──────────┴──────────┴──────────┴────────────────┤
│              基础设施层                           │
│  向量数据库 │ 嵌入模型 │ LLM推理引擎 │ 记忆系统     │
└─────────────────────────────────────────────────┘
```

### 五大技术支柱

1. **提示工程**：通过Zero/Few-Shot、CoT、ReAct等技术引导LLM行为
2. **结构化输出**：通过JSON Schema约束、Constrained Decoding等确保输出可解析
3. **RAG系统**：检索器 + 生成器，从外部知识库注入事实信息
4. **Agent系统**：规划→行动→观察循环，配合记忆与反思实现自主执行
5. **工具调用与MCP**：Function Calling + MCP协议，扩展LLM行动能力

### 核心形式化

**RAG检索-生成范式**：

$$P(a|q) = \sum_{d \in \mathcal{R}(q)} P(a|q, d) \cdot P(d|q)$$

其中 $\mathcal{R}(q)$ 为检索器返回的文档集合，$P(d|q)$ 为检索相关性，$P(a|q,d)$ 为生成器基于查询和文档生成答案的概率。

**ReAct循环**：

$$\text{Thought}_t \to \text{Action}_t \to \text{Observation}_t \to \text{Thought}_{t+1}$$

终止条件：$\text{Action}_t = \text{Finish}[\text{answer}]$

## 5. 关键方法与模型

### 核心算法

| 领域 | 关键方法 | 代表论文 |
|------|----------|----------|
| 提示工程 | CoT, Self-Consistency, ToT, ReAct | Wei 2022; Wang 2022; Yao 2022/2023 |
| 结构化输出 | Constrained Decoding, Grammar-based | Outlines; Guidance |
| RAG | HyDE, RRF融合, Self-RAG, CRAG | Gao 2022; Asai 2023; Yan 2024 |
| Agent | ReAct, Reflexion, Plan-and-Execute | Yao 2022; Shinn 2023 |
| 工具调用 | Function Calling, MCP, Toolformer | OpenAI 2023; Anthropic 2024 |

### 代表论文

- **Lewis et al. (2020)**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" — RAG范式奠基
- **Wei et al. (2022)**: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" — CoT推理
- **Yao et al. (2022)**: "ReAct: Synergizing Reasoning and Acting in Language Models" — Agent核心范式
- **Schick et al. (2023)**: "Toolformer: Language Models Can Teach Themselves to Use Tools" — 自学习工具使用
- **Asai et al. (2023)**: "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" — 自适应检索
- **Anthropic (2024)**: "Model Context Protocol" — 工具调用标准化协议

### 主流框架

| 框架 | 定位 | 核心能力 |
|------|------|----------|
| LangChain | 通用LLM应用框架 | 模块化工具链，组件齐全 |
| LangGraph | 状态图工作流引擎 | 复杂决策流程，循环控制 |
| LlamaIndex | RAG增强框架 | 文档处理与检索优化 |
| AutoGen | 多Agent对话框架 | 多Agent协作 |
| CrewAI | 角色化Agent协作 | 流程化团队任务 |
| DSPy | 声明式提示优化 | 自动化提示调优 |
| OpenAI Agents SDK | 官方Agent框架 | 快速部署与托管 |

## 6. 优势与局限

### 优势

- **灵活性**：自然语言交互，无需硬编码逻辑
- **泛化性**：单一模型可处理多种任务
- **快速迭代**：修改提示即可调整行为，无需重新训练
- **知识可扩展**：通过RAG动态更新知识，无需重训模型

### 局限

- **不确定性**：LLM输出具有概率性，同一输入可能产生不同输出
- **延迟较高**：相比传统API，LLM推理延迟显著
- **成本问题**：大规模调用API成本高昂
- **可控性有限**：提示工程无法100%保证输出格式和行为
- **安全风险**：Prompt注入、数据泄露、越狱攻击等安全隐患
- **评估困难**：缺乏标准化评估方法，难以量化系统质量

## 7. 应用场景

- **智能问答与知识库**：企业内部知识检索与问答系统
- **编程助手**：Cursor、Claude Code等AI编程工具
- **数据分析自动化**：自然语言查询数据库并生成分析报告
- **客服自动化**：工单分类、知识检索、自动回复
- **内容生产**：研究→写作→审核的自动化流水线
- **Web Agent**：浏览器操作、网页信息提取与自动化
- **多模态应用**：图文问答、视频内容理解与检索
- **科研助手**：文献检索、论文综述、实验设计辅助

## 8. 与其他技术关系

- 以 [[../10_大语言模型核心架构/00_大语言模型核心架构_综述|大语言模型]] 为核心推理引擎
- [[../10_大语言模型核心架构/09_推理模型与思维链|思维链]] 推理是Agent规划的基础
- [[../14_多模态AI/00_多模态AI|多模态模型]] 扩展Agent的感知范围
- [[../18_模型部署与工程化/00_模型部署与工程化|模型部署与工程化]] 为LLM应用提供推理基础设施
- [[../12_大模型推理与优化/01_KV-Cache机制|KV-Cache机制]] 直接影响LLM应用响应延迟
- [[../12_大模型推理与优化/02_量化技术|量化技术]] 降低LLM应用的部署成本
- [[../12_大模型推理与优化/06_长上下文扩展|长上下文扩展]] 影响RAG与Agent的上下文设计策略

## 9. 前沿发展

- **推理模型驱动Agent**：o1/o3等推理模型从根本上改变Agent的规划方式，从外部提示引导走向模型内部推理
- **Agentic RAG**：RAG从静态检索进化为Agent式动态检索，LLM自主决定何时检索、如何检索
- **多Agent系统**：从单Agent走向多Agent协作，模拟人类团队分工
- **Computer Use Agent**：Agent直接操作GUI界面，实现通用自动化
- **MCP生态化**：工具调用标准化后，社区共享MCP Server，形成工具复用生态
- **LLM应用可观测性**：从传统APM走向LLM专用可观测性（LangSmith、Langfuse）
- **端侧Agent**：在手机/PC本地运行的Agent，隐私保护与低延迟
- **Agent操作系统**：将Agent作为操作系统级原语，统一调度与管理

## 笔记导航

| 笔记 | 主题 |
|------|------|
| [[01_提示工程]] | 提示设计、CoT、ReAct、工程化 |
| [[02_结构化输出]] | JSON Schema、Constrained Decoding |
| [[03_RAG系统/00_RAG系统|RAG系统]] | 检索增强生成（基础/进阶/多模态/向量数据库） |
| [[04_Agent系统/00_Agent系统|Agent系统]] | Agent架构、框架、记忆、工作流 |
| [[05_工具调用与MCP协议]] | Function Calling、MCP、Computer Use |
| [[06_自动化工作流]] | 编排框架、多Agent协作、容错与可观测性 |
| [[07_LLM应用评估与可观测性]] | 评估方法、监控、追踪 |
| [[08_LLM应用安全]] | Prompt注入、数据安全、防护策略 |

## 学习路径

1. **入门**：提示工程 → 结构化输出 → 理解LLM能力边界
2. **进阶**：RAG系统（检索器+生成器） → 工具调用与MCP
3. **高级**：Agent系统（规划→行动→观察） → 自动化工作流
4. **工程化**：评估与可观测性 → 安全防护 → 生产部署

## References

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020) — RAG范式奠基
- Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* (2022) — CoT推理
- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022) — Agent核心范式
- Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools* (2023) — 自学习工具使用
- Asai et al., *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection* (2023) — 自适应检索
- Anthropic, *Model Context Protocol* (2024) — [官方文档](https://modelcontextprotocol.io)
- [LangChain 官方文档](https://python.langchain.com)
- [LlamaIndex 官方文档](https://docs.llamaindex.ai)
