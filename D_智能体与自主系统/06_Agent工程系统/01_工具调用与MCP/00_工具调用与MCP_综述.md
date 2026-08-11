---
tags:
  - 工具调用
  - Function Calling
  - MCP
  - Agent核心
created: 2026-07-28
updated: 2026-07-28
---

# 工具调用与 MCP 综述：Agent 与外部世界的接口

## 一句话理解

工具调用是 Agent 从"纯语言空间"进入"真实世界"的关键接口——Function Calling 让模型结构化调用 API，MCP 协议标准化工具发现与调用流程，使 Agent 能搜索网页、执行代码、操作文件，将"说"变成"做"。

## 领域定义

**工具调用（Tool Use / Function Calling）**：让 LLM 在生成文本时，识别何时需要外部能力、选择合适的工具、生成结构化调用参数，并将工具执行结果整合回后续推理，形成"思考-行动-观察"的闭环。

**MCP（Model Context Protocol）**：Anthropic 于 2024 年提出的开放协议，标准化 LLM 应用与外部工具/数据源的连接方式，解决"每个应用都要为每个工具重新集成"的 N×M 碎片化问题。

两者的关系是：Function Calling 是模型能力（模型有没有能力生成结构化调用），MCP 是通信协议（工具以什么标准接口被发现和调用）。完整的技术细节、代码示例与最佳实践见 [[01_工具调用与MCP协议|工具调用与MCP协议]]。

## 为什么会出现

GPT-3 时代的 LLM 只能基于训练数据生成文本，无法获取实时信息（今天的天气、最新股价）、无法执行精确计算（大数乘法）、无法操作真实系统（发邮件、查数据库）。早期开发者尝试用 Prompt 让模型"假装"调用工具并解析自由文本输出，但格式不稳定、错误率高。2023 年 OpenAI 推出原生 Function Calling，让模型直接输出符合 JSON Schema 的结构化调用请求，从根本上解决了这个问题。随着 Agent 需要接入的工具（搜索、代码执行、数据库、第三方 SaaS）数量爆炸式增长，"每接入一个工具就要为每个 LLM 应用重新写一套集成代码"的 N×M 问题日益严重，这催生了 MCP 这样的统一协议——工具开发者只需实现一次 MCP Server，即可被所有支持 MCP 的 LLM 应用调用。

## 发展历史

| 时间 | 里程碑 | 意义 |
|:---|:---|:---|
| 2023.03 | OpenAI Function Calling | LLM 原生结构化工具调用能力 |
| 2023.05 | Toolformer（Schick et al.） | LLM 自监督学习何时、如何使用工具 |
| 2023.08 | Anthropic Tool Use | Claude 原生工具调用支持 |
| 2023.10 | 并行工具调用 | 单次返回多个工具调用，提升执行效率 |
| 2024.06 | OpenAI Structured Output | 严格 Schema 约束，保证 100% 格式合规 |
| 2024.11 | MCP 协议发布（Anthropic） | 工具调用标准化，解耦工具与应用 |
| 2024.10 | Computer Use（Anthropic） | Agent 可直接通过截图+坐标操作 GUI 界面 |
| 2025 | MCP 生态爆发 | OpenAI、Google 等厂商跟进支持，社区共享 MCP Server |

## 核心问题

1. 模型如何知道"现在需要调用工具"而不是直接回答？
2. 如何让模型生成的调用参数严格符合外部系统要求的格式？
3. 工具集成如何避免为每个应用重复开发？
4. 工具调用带来的安全风险（越权、误操作）如何约束？

## 技术演进路线

```
自由文本工具调用（Prompt 解析，脆弱）
  → Function Calling（原生结构化调用，2023）
    → 并行工具调用（效率提升）
      → Structured Output（严格 Schema 保证）
        → MCP 协议（连接标准化，2024）
          → Computer Use（GUI 操作能力扩展）
            → MCP 生态化（工具即插即用）
```

## 重要分支

- **Function Calling 机制**：模型如何决策与生成调用参数
- **MCP 协议架构**：Host / Client / Server 三方通信模型
- **代码执行**：将解释器/沙箱作为特殊工具，解决计算与逻辑验证问题
- **Computer Use**：以截图-坐标操作 GUI，突破 API 覆盖不到的场景
- **RAG as Tool**：将检索系统封装为可调用工具，与 Agent 记忆系统融合

## 学习路径

1. 理解 [[01_工具调用与MCP协议|Function Calling 的工作流程]]与结构化参数生成
2. 学习 MCP 协议的三方架构（Host/Client/Server）与三种能力（Tools/Resources/Prompts）
3. 实践代码执行、RAG 检索等工具类型的集成
4. 研究工具调用的安全约束，参见 [[../07_LLM应用安全|LLM应用安全]]
5. 探索 Computer Use 等 GUI 操作类工具的前沿实践

## 当前发展状态

- MCP 已成为跨厂商工具接入的事实标准，主流 Agent 框架（LangChain、Claude Desktop 等）均已原生支持
- Function Calling 准确率随模型能力提升持续改善，但工具数量过多（>20）时选择准确率仍会下降
- Computer Use 类技术尚在早期，操作精度和速度低于原生 API 调用，但覆盖了无 API 场景

## 未来趋势

- **工具生态化**：类似 npm/pip 的 MCP Server 包管理与发现机制
- **自主工具发现**：Agent 无需预定义工具列表，能动态发现并学习使用新工具
- **多模态工具调用**：工具参数从纯文本扩展到图像、音频等多模态输入

## 相关方向

- [[../01_Agent系统与框架|Agent系统与框架]]：工具调用是 Agent 核心循环的行动环节
- [[../06_结构化输出|结构化输出]]：工具参数生成依赖的底层机制
- [[../03_RAG与检索增强/00_RAG系统|RAG系统]]：检索可作为工具被 Agent 调用
- [[../07_LLM应用安全|LLM应用安全]]：工具调用是 Agent 安全风险的主要来源

## References

- Anthropic. *Model Context Protocol (MCP) Specification*. https://modelcontextprotocol.io/
- OpenAI. *Function Calling* API Documentation.
- Schick, T. et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. arXiv:2302.04761.
- Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv:2210.03629.