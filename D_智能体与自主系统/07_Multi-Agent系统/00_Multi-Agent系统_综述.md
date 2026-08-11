---
title: "Multi-Agent 系统 Overview"
domain: "D-07"
level: "综述"
status: "稳定"
last_reviewed: 2026-08-11
review_cycle: 180
evidence_level: "综述"
source_of_truth: true
tags:
  - Multi-Agent
  - MAS
  - 协作
  - 综述
created: 2026-07-28
updated: 2026-08-11
---

# Multi-Agent 系统 Overview Overview

## 领域定义

Multi-Agent 系统（MAS）研究多个具有局部观察、局部目标和局部行动能力的智能体如何在共享环境中通过通信、协作、竞争和制度设计来实现全局目标。MAS 的核心假设是：**没有任何一个 Agent 拥有解决问题所需的全部信息、能力或视角**——因此必须通过交互来弥补各自的局限。LLM 多 Agent 只是 MAS 的一个实现分支，而非 MAS 的全部。

## 为什么会出现

单一 Agent 的能力再强，也有其边界：信息视野有限（只能看到局部环境）、工具集有限（只能调用自身可访问的工具）、认知视角有限（可能产生盲区）。在搜索、编程、问答等复杂任务中，多个 Agent 分工协作可以覆盖更广的信息来源、使用更多样的工具、从多个角度交叉验证结果。但 MAS 也带来了新的问题：通信成本、协调复杂度、冲突解决——拆分 Agent 的前提是信息、工具或评估视角确实互补，且协作收益大于通信开销。

## 发展历史

| 年代 | 里程碑 | 意义 |
|:---|:---|:---|
| 1979 | 合同网协议（Smith） | 首个任务分配协议，MAS 工程化的起点 |
| 1995 | Wooldridge & Jennings 系统化 MAS 理论 | 智能体理论的分支，MAS 成为独立学科 |
| 2000 | FIPA 通信标准 | 定义了 Agent 通信语言的标准协议 |
| 2003 | Wooldridge 教材 | MAS 理论的系统化教材 |
| 2010s | 多智能体强化学习（MARL） | 从"手动设计协议"到"学习协作策略" |
| 2017 | QMIX / MADDPG | 深度 MARL 的代表性算法 |
| 2023 | AutoGen / ChatDev / MetaGPT | LLM 多 Agent 协作框架兴起 |
| 2024 | 多 Agent 协作评测 | 协作收益度量、失效模式分析成为研究重点 |

## 核心问题

1. **通信**：Agent 之间如何高效、可靠地交换信息？自然语言对话是否足够？
2. **协作**：如何让多个 Agent 为共同目标协同工作，而非互相干扰？
3. **任务分配**：如何将全局任务分解为子任务，分配给最合适的 Agent？
4. **冲突解决**：Agent 之间产生冲突时，如何仲裁和解决？
5. **评测**：如何衡量多 Agent 系统相对于单 Agent 的真实增益？

## 重要分支

- **MAS 理论与通信**：组织结构（层级/对等/黑板/市场）、通信协议、共享状态与共识（[[01_理论与通信/01_MAS理论与通信]]）
- **协作、竞争与任务分配**：合同网、协同规划、博弈、MARL（[[02_协作与评测/01_协作_竞争与任务分配]]）
- **LLM Multi-Agent 与评测**：角色系统、协作收益度量、失效模式分析（[[02_协作与评测/02_LLM_Multi_Agent与评测]]）

## 概念边界

- **MAS vs. 分布式系统**：MAS 中的 Agent 是**自主的**——有自身的目标和决策能力，不是简单的"分布式计算节点"；分布式系统关注"如何把任务分给多台机器"，MAS 关注"如何让多个自主决策者协同工作"。
- **LLM 多 Agent vs. 经典 MAS**：LLM 多 Agent 用自然语言通信替代了 FIPA 等结构化协议，灵活但不可靠；经典 MAS 强调通信协议的形式化保证（如 FIPA-ACL 的语义），但灵活性不足。

## 学习路径

1. [[01_理论与通信/01_MAS理论与通信]]：理解 MAS 的组织结构、通信协议和共识机制
2. [[02_协作与评测/01_协作_竞争与任务分配]]：掌握任务分配、协同规划和 MARL 的核心方法
3. [[02_协作与评测/02_LLM_Multi_Agent与评测]]：了解 LLM 多 Agent 的实践、协作收益和失效模式

## 当前发展状态

- **经典 MAS 理论成熟**：组织结构、协议、博弈论基础已非常完善，但工业应用有限
- **LLM 多 Agent 实践领先理论**：AutoGen、MetaGPT 等框架快速迭代，但协作收益的定量分析仍不充分
- **评测标准不统一**：缺乏公认的跨领域多 Agent 评测基准，难以公平比较不同方案

## 未来趋势

- **从固定角色到动态分工**：Agent 角色不再预先分配，而是根据任务动态协商
- **从对话到结构化通信**：自然语言对话被结构化消息（含证据、置信度、来源）替代
- **可证明的协作增益**：开发理论框架，在何种条件下多 Agent 必然优于单 Agent
- **安全与治理**：多 Agent 系统的权限管理、审计、事故归因

## 相关方向

- [[../01_智能体理论基础/00_智能体理论基础_综述|D-01 智能体理论基础]]：Agent 的理性定义是 MAS 行为的基础
- [[../04_强化学习/09_多智能体强化学习|D-04 多智能体强化学习]]：MARL 是 MAS 在 RL 中的实现
- [[../06_Agent工程系统/00_Agent工程系统_综述|D-06 Agent工程系统]]：多 Agent 部署的工程基础

## References

- Wooldridge, M. (2009). *An Introduction to MultiAgent Systems* (2nd ed.). Wiley.
- Smith, R. G. (1980). *The Contract Net Protocol: High-Level Communication and Control in a Distributed Problem Solver*. IEEE T-C.
- Rashid, T. et al. (2018). *QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning*. ICML.
- Wu, Q. et al. (2023). *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*. arXiv:2308.08155.
- Hong, S. et al. (2023). *MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework*. arXiv:2308.00352.