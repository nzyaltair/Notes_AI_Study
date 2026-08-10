---
tags:
  - 工具调用
  - Function Calling
  - MCP
  - API集成
  - 代码执行
  - Agent核心
created: 2026-07-28
updated: 2026-07-28
---

# 工具调用与 MCP 综述：Agent 与外部世界的接口

## 一句话理解

工具调用是 Agent 从"纯语言空间"进入"真实世界"的关键接口——Function Calling 让模型结构化调用 API，MCP 协议标准化工具发现与调用流程，使 Agent 能搜索网页、执行代码、操作文件，将"说"变成"做"。

## 1. 领域定义

### 工具调用 (Tool Use / Function Calling)

让 LLM 在生成文本时，选择调用预定义的外部工具（API、代码执行器、数据库等），并将工具返回结果整合到后续推理中。

### MCP (Model Context Protocol)

Anthropic 提出的开放协议，标准化 LLM 应用与外部数据源/工具的连接方式。

## 2. 核心方法

### 2.1 Function Calling

| 阶段 | 过程 | 说明 |
|------|------|------|
| 工具注册 | 定义工具 schema | JSON Schema 描述参数 |
| 决策 | 模型选择调用哪个工具 | 训练/提示驱动 |
| 执行 | 外部运行工具并返回结果 | 沙箱/容器环境 |
| 整合 | 结果融入上下文继续推理 | 观察结果后行动 |

### 2.2 MCP 协议

| 组件 | 功能 | 说明 |
|------|------|------|
| MCP Host | 发起连接 | LLM 应用 (如 Claude Desktop) |
| MCP Client | 与 Server 通信 | 协议客户端 |
| MCP Server | 提供工具/资源 | 第三方服务 |
| Transport | 通信机制 | stdio / SSE |

MCP 提供三种能力：
- **Tools**：可调用的函数
- **Resources**：可读取的数据
- **Prompts**：可复用的提示模板

### 2.3 代码执行

| 方式 | 特点 | 代表 |
|------|------|------|
| Python REPL | 交互式执行 | Code Interpreter |
| 沙箱执行 | 隔离环境 | Docker / Jupyter |
| 代码生成+执行 | 生成代码后运行 | PAL, Code-as-Reasoning |
| 浏览器操作 | 网页交互 | WebArena, BrowserUse |

### 2.4 检索增强 (RAG as Tool)

- 将检索系统作为工具调用
- 向量搜索 + 重排序
- 与 Agent 的 Memory 系统融合

## 3. 发展历史

1. 2023：OpenAI Function Calling 推出
2. 2023：AutoGPT / BabyAGI 展示工具链 Agent
3. 2024：Anthropic 发布 MCP 协议
4. 2024：Coding Agent (Cursor, Devin) 成为主流工具
5. 2025-26：MCP 生态成熟，工具调用成为 Agent 标配

## 4. 与 AGI 的关系

- **Agent 的手脚**：没有工具调用，Agent 只能"说话"不能"行动"
- **能力扩展**：工具让 LLM 突破训练数据的局限
- **安全挑战**：工具调用是 Agent 安全风险的主要来源
- **AGI 阶梯**：工具调用是 Agent 阶梯的核心能力之一

## 5. 学习路径

1. 理解 Function Calling 的工作流程
2. 学习 MCP 协议的核心概念
3. 实践代码执行工具集成
4. 研究工具调用的安全约束
5. 探索工具自动发现与组合

## 相关知识

- [[../../01_智能体理论基础/00_智能体理论基础_综述|智能体理论基础]]：工具调用是 Agent 的核心组件
- [[../../02_LLM应用工程/00_LLM应用工程_综述|LLM应用工程]]：RAG 和提示工程的基础
- [[../../../../C_基础模型与通用智能/04_大模型训练与对齐/00_大模型训练与对齐_综述|大模型训练与对齐]]：工具调用能力需要对齐训练

## References

- Anthropic, *Model Context Protocol (MCP)* Specification
- OpenAI, *Function Calling* API Documentation
- Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools*
- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*
