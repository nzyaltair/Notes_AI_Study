---
title: "LLM Multi-Agent 与评测"
domain: "D-07"
level: "核心"
status: "稳定"
last_reviewed: 2026-08-11
review_cycle: 180
evidence_level: "综述"
source_of_truth: true
tags:
  - LLM多Agent
  - AutoGen
  - MetaGPT
  - 协作评测
created: 2026-08-11
updated: 2026-08-11
---

# LLM Multi-Agent 与评测

## 一句话理解

LLM 多 Agent 是经典 MAS 理论在 LLM 时代的新实现——用自然语言替代结构化协议，用 LLM 推理替代预设策略；但 LLM 多 Agent 的实际价值来自信息、工具或评估视角的互补，而不是让同一模型重复对话。

## 概述

2023 年以来，AutoGen、MetaGPT、ChatDev 等框架将多个 LLM 实例组织为多 Agent 系统，通过角色分工和自然语言对话协作完成任务。这种范式继承了经典 MAS 的组织结构思想（[[../01_理论与通信/01_MAS理论与通信|MAS 理论与通信]]），但用 LLM 的灵活推理替代了预设的规则和协议。

然而，LLM 多 Agent 也带来了新的问题：所有 Agent 共享 LLM 的先验知识，可能导致"多数幻觉"；自然语言对话的成本高、效率低；难以证明协作效果优于单 Agent 加更好的工具。[[../00_Multi-Agent系统_综述|Multi-Agent 系统综述]] 提到"LLM 多 Agent 的实际价值来自信息、工具或评估视角的互补"，本笔记聚焦 LLM 多 Agent 的架构模式、失效模式和评测方法。

## 发展历史

| 时间 | 里程碑 | 意义 |
|:---|:---|:---|
| 2023.06 | AutoGen（Wu et al.） | 首个通用 LLM 多 Agent 对话框架 |
| 2023.07 | ChatDev（Qian et al.） | 软件工程 LLM 多 Agent 协作 |
| 2023.08 | MetaGPT（Hong et al.） | 角色化 LLM 多 Agent，模拟软件公司 |
| 2023.10 | AgentVerse（Chen et al.） | 动态 Agent 组管理 |
| 2024 | 多 Agent 评测标准化 | 端到端成功率、协作增益、通信成本的系统度量 |

## 核心概念

### 四种主流架构

**主管-执行者（Supervisor-Executor）**：

```text
主管 Agent：理解任务、分解为子任务、分配给执行者、汇总结果
  ├── 执行者 Agent 1：负责子任务 A
  ├── 执行者 Agent 2：负责子任务 B
  └── 执行者 Agent 3：负责子任务 C
```

**辩论（Debate）**：多个 Agent 从不同角度分析同一问题，通过辩论达成共识。适合需要多方视角的任务（如决策分析、风险评估）。

**流水线（Pipeline）**：Agent 按固定顺序处理任务，前一个 Agent 的输出是后一个 Agent 的输入。适合步骤明确的流程（如软件开发：架构师→开发者→测试）。

**并行候选（Parallel Candidate）**：多个 Agent 独立尝试完成任务，选择最优结果。适合搜索和探索类任务。

### 角色系统设计

LLM 多 Agent 中，每个 Agent 的角色通过系统提示定义：

| 角色 | 系统提示示例 | 核心能力 |
|:---|:---|:---|
| 主管 | "你是一个项目主管，负责分解任务并分配" | 任务分解、分配、汇总 |
| 搜索专家 | "你是一个搜索专家，擅长信息检索" | 搜索、筛选、信息提取 |
| 代码审查者 | "你是一个代码审查者，负责发现潜在问题" | 代码分析、安全问题识别 |

## 技术原理

### 协作收益的条件

LLM 多 Agent 的协作收益来自以下三个条件之一：

1. **信息互补**：不同 Agent 有不同数据源或 API 访问权限，信息覆盖面更广
2. **工具互补**：不同 Agent 有不同工具集，可以实现"搜索+计算+验证"的完整流程
3. **评估视角互补**：不同 Agent 从不同角度评估同一结果，减少单一视角的盲区

### 常见失效模式

| 失效模式 | 原因 | 表现 | 缓解方案 |
|:---|:---|:---|:---|
| 多数幻觉 | 所有 Agent 共享 LLM 先验 | 错误结论被多方"确认" | 引入外部验证工具 |
| 角色漂移 | Agent 忘记自己的角色 | 主管开始写代码，开发开始管进度 | 定期角色确认 |
| 讨论发散 | 对话无终止条件 | 无限循环讨论，成本失控 | 预设对话轮次上限 |
| 协调者瓶颈 | 主管 Agent 成为瓶颈 | 所有决策等待主管，执行效率低 | 分布式决策 |
| 归因困难 | 任务失败时不知道谁的责任 | 无法定位失败根因 | 结构化日志 |

### 评测框架

LLM 多 Agent 的评测不应只看端到端成功率，还需要测量以下指标：

| 指标 | 含义 | 测量方法 |
|:---|:---|:---|
| 协作增益 | 多 Agent 相对单 Agent 基线 | 多 Agent 成功率 - 单 Agent 成功率 |
| 通信效率 | 单位协作增益的通信成本 | 总 Token 消耗 / 协作增益 |
| 任务分配正确率 | 子任务是否分配给了最合适的 Agent | 人工评估或基线对比 |
| 冲突解决率 | Agent 间冲突的解决效率 | 冲突次数 / 解决次数 |
| 并发效率 | 并行 Agent 的加速比 | 单 Agent 耗时 / 多 Agent 耗时 |
| 鲁棒性 | Agent 故障时的系统表现 | 移除一个 Agent 后成功率变化 |
| 安全违规 | Agent 协作中的越权或误操作 | 审计日志分析 |

## 关键方法与模型

| 框架 | 架构 | 特点 |
|:---|:---|:---|
| AutoGen | 对话式，可扩展 | 灵活的 Agent 对话管理，支持人类介入 |
| MetaGPT | 角色化流水线 | 模拟软件公司，结构化产出物 |
| ChatDev | 角色化流水线 | 定向开发流程，角色分工明确 |
| AgentVerse | 动态组管理 | Agent 组动态创建和销毁 |
| CrewAI | 声明式角色定义 | 角色+工具+任务的声明式组合 |

## 优势与局限

**LLM 多 Agent 的优势**：角色分工使系统可以处理需要多种能力的复杂任务；多视角交叉验证减少单一 Agent 的盲区；系统容错性更好（一个 Agent 失败不影响整体）。

**LLM 多 Agent 的局限**：所有 Agent 共享 LLM 的先验知识，互补性不如预期；自然语言通信成本高（Token 消耗大）；"多数幻觉"等失效模式可能抵消协作收益；评测困难，难以客观衡量协作的真实增益。

## 应用场景

- **软件工程**：架构师+开发者+测试的协作开发流程
- **信息调研**：搜索 Agent + 阅读 Agent + 总结 Agent 的协作调研
- **内容创作**：策划 Agent + 写作 Agent + 编辑 Agent 的协作创作
- **辩论分析**：正方 Agent + 反方 Agent + 裁判 Agent 的辩论评估

## 与其他技术关系

- LLM 多 Agent 是 [[../01_理论与通信/01_MAS理论与通信|MAS 理论]] 在 LLM 时代的具体实现
- 角色系统提示与 [[../06_Agent工程系统/05_提示工程|提示工程]] 直接相关
- 评测方法与 [[../06_Agent工程系统/04_评测与可观测性/00_LLM应用评估与可观测性|LLM 应用评估]] 的方法论一致

## 前沿发展

- **动态角色发现**：Agent 不再被预设角色限死，而是根据任务动态发现和承担角色
- **学习型通信协议**：Agent 通过 RL 学习何时、与谁、如何通信，而非使用固定的自然语言格式
- **结构化 LLM 多 Agent**：在自然语言对话中嵌入结构化数据（JSON Schema、共享状态），兼顾灵活性和效率
- **可证明的协作增益**：开发理论框架，精确预测在什么条件下多 Agent 优于单 Agent

## 常见问题

**Q: 多 Agent 总是比单 Agent 好？**
不是。如果所有 Agent 共享相同的 LLM 和数据源，多 Agent 的协作增益为负——因为增加了通信成本但没有带来新的信息或视角。多 Agent 只有在信息、工具或评估视角确实互补时才有效。

**Q: LLM 多 Agent 和经典 MAS 的主要区别？**
经典 MAS 使用结构化协议（FIPA-ACL）和精确的 Agent 能力模型，通信可靠但灵活性差；LLM 多 Agent 使用自然语言对话，灵活但不可靠。两者的发展方向是融合——在 LLM 多 Agent 中引入结构化通信，在经典 MAS 中引入 LLM 的灵活推理。

## 相关知识

- 前置：[[../01_理论与通信/01_MAS理论与通信|MAS 理论与通信]]、[[01_协作_竞争与任务分配|协作、竞争与任务分配]]
- 平级：[[../06_Agent工程系统/01_Agent系统与框架|Agent 系统与框架]]
- 延伸：[[../06_Agent工程系统/04_评测与可观测性/00_LLM应用评估与可观测性|LLM 应用评估]]

## References

- Wu, Q. et al. (2023). *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*. arXiv:2308.08155.
- Hong, S. et al. (2023). *MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework*. arXiv:2308.00352.
- Qian, C. et al. (2023). *Communicative Agents for Software Development*. arXiv:2307.07924 (ChatDev).
- Chen, W. et al. (2023). *AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors*. ICLR 2024.
- Liu, X. et al. (2023). *AgentBench: Evaluating LLMs as Agents*. arXiv:2308.03688.