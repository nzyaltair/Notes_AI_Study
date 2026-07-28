---
tags:
  - Agent
  - 智能体
  - LLM-Agent
  - 工具调用
  - 多智能体
  - AGI阶梯
created: 2026-07-28
updated: 2026-07-28
---

# Agent 系统综述：AGI 阶梯的当前台阶

## 一句话理解

Agent 系统是能够感知环境、制定计划、调用工具、执行动作并根据反馈调整行为的智能体，是 2026 年 AGI 演进阶梯的当前台阶——从语言模型到持续学习之间的关键一步。

## 1. 领域定义

Agent（智能体）是一个能够自主感知环境、做出决策并采取行动以实现目标的人工系统。LLM-based Agent 以大语言模型为推理核心，结合记忆、工具和规划能力完成复杂任务。

Agent 不是单一技术，而是一种**系统架构范式**：LLM 提供"大脑"，记忆系统提供"经验"，工具提供"手脚"，规划模块提供"策略"。

## 2. AGI 阶梯定位

根据 DeepSeek 梁文锋的技术路线：

```
语言模型 -> CoT思维链 -> [Agent] -> 持续学习 -> 奇点(自我迭代) -> 具身智能
                          当前阶段
```

- Agent 基于 CoT（思维链），CoT 基于语言模型——每一步都依赖前一步
- Agent 之后要解决的核心问题是**持续学习**
- Agent 的能力范围比 CoT 更大，智能上限更高

## 3. 核心架构

### 3.1 经典 Agent 架构（ReAct 循环）

```
观察(Observation) -> 思考(Thought) -> 行动(Action) -> 观察 -> ...
```

### 3.2 Agent 核心组件

| 组件 | 功能 | 典型实现 |
|------|------|----------|
| 推理核心 | 思考、规划、决策 | LLM (GPT-4, Claude, DeepSeek) |
| 记忆系统 | 存储经验、上下文、知识 | 向量数据库、对话历史、长期记忆 |
| 工具调用 | 与外部世界交互 | API、MCP协议、代码执行 |
| 规划模块 | 分解任务、制定策略 | CoT、ToT、Plan-and-Solve |
| 反馈机制 | 评估结果、调整策略 | 自我反思、人工反馈 |

### 3.3 Agent 范式

| 范式 | 核心思想 | 代表 |
|------|----------|------|
| ReAct | 推理与行动交替 | Yao et al. 2023 |
| Plan-and-Execute | 先规划后执行 | HuggingGPT |
| Reflexion | 自我反思改进 | Shinn et al. 2023 |
| LATS | 语言Agent树搜索 | Zhou et al. 2023 |
| 多Agent | 协作与分工 | AutoGen, CrewAI |

## 4. 发展历史

1. 2022：CoT (Wei et al.) 证明 LLM 可进行多步推理
2. 2023：ReAct 统一推理与行动；AutoGPT 引爆 Agent 潮流
3. 2024：MCP 协议标准化工具调用；Coding Agent (Cursor, Devin) 落地
4. 2025-26：Agent 成为 AGI 阶梯核心台阶；多 Agent 协作框架成熟

## 5. 核心能力与局限

### 5.1 能力

- 多步推理与规划
- 工具调用与外部系统交互
- 自我反思与错误修正
- 长程任务分解与执行

### 5.2 局限（指向持续学习瓶颈）

- **无法持续学习**：每次会话从零开始，不积累经验
- **上下文窗口有限**：长程任务容易遗忘
- **幻觉影响行动**：符号空间的错误可能导致实际损害
- **可靠性不足**：无法保证在关键场景中稳定执行

## 6. 与其他方向的关系

| 关系 | 方向 | 说明 |
|------|------|------|
| 前置 | LLM + CoT | Agent 的推理基础 |
| 核心依赖 | 强化学习 | Agent 优化与对齐 |
| 核心依赖 | 工具调用/MCP | Agent 与外部交互 |
| 下一步 | 持续学习 | Agent 之后要解决的核心瓶颈 |
| 物理延伸 | 具身智能 | Agent + 物理执行 = 具身 Agent |
| 协作基础 | 群体智能 | 多 Agent 协作的理论基础 |

## 7. 学习路径

1. 理解 ReAct 循环：观察-思考-行动
2. 实践工具调用（Function Calling / MCP）
3. 学习 Agent 记忆系统设计
4. 研究规划算法（CoT - ToT - Plan-and-Solve）
5. 理解多 Agent 协作框架
6. 分析 Agent 的局限性 - 引向持续学习

## 8. 未来趋势

- Agent + 持续学习 = 真正自适应智能体
- Agent 自我迭代（用 Agent 开发 Agent）
- 具身 Agent（Agent + 机器人控制）
- Agent 安全与对齐（Agent 行为约束）
- 从单 Agent 到多 Agent 生态系统

## 相关知识

- [[../../C_大语言模型与语言智能/02_大语言模型核心架构/09_推理模型与思维链|推理模型与思维链]]：Agent 的推理基础
- [[../01_强化学习/00_强化学习_综述|强化学习]]：Agent 优化与对齐的理论基础
- [[../05_持续学习与元学习/00_持续学习与元学习_综述|持续学习与元学习]]：Agent 的下一个瓶颈
- [[../../H_具身智能/01_具身智能/00_具身智能_综述|具身智能]]：Agent 的物理延伸

## References

- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*
- Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*
- Wu et al., *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*
- Anthropic, *Model Context Protocol (MCP)* Specification
- DeepSeek 梁文锋投资者交流会，2026-05-20
