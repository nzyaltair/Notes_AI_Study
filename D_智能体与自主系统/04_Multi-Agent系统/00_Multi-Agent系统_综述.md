---
tags:
  - 多智能体
  - Multi-Agent
  - Agent协作
  - 群体智能
  - Agent框架
  - Agent通信
created: 2026-07-28
updated: 2026-07-28
---

# 多 Agent 系统综述：从单兵到协作群体

## 一句话理解

多 Agent 系统让多个专精智能体通过分工、通信和协作完成单个 Agent 难以处理的复杂任务——从简单的主从分配到涌现式群体智能，多 Agent 是 Agent 从"个体智能"迈向"集体智能"的关键一步。

## 1. 领域定义

多 Agent 系统 (Multi-Agent System, MAS) 是由多个自主智能体组成的系统，各智能体可以独立决策，通过通信和协作共同完成任务。

### 核心问题

| 问题 | 描述 | 挑战 |
|------|------|------|
| 分工 | 如何分配任务给不同 Agent | 能力匹配、负载均衡 |
| 通信 | Agent 之间如何交换信息 | 协议设计、信息损失 |
| 协调 | 如何避免冲突、达成一致 | 竞争与合作、一致性 |
| 涌现 | 群体行为是否超出个体能力之和 | 可控性与涌现性的平衡 |

## 2. 核心架构

### 2.1 组织结构

| 结构 | 特点 | 代表 |
|------|------|------|
| 主从式 | 一个主 Agent 协调 | AutoGen GroupChat Manager |
| 层级式 | 多级指挥链 | 组织模拟 |
| 对等式 | 无中心，平等协作 | ChatDev |
| 黑板式 | 共享信息空间 | 黑板系统 |
| 市场式 | 竞标机制分配任务 | Agent Bidding |

### 2.2 通信模式

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| 直接消息 | Agent 间直接通信 | 小规模协作 |
| 广播 | 一对多通知 | 任务分配 |
| 共享黑板 | 读写共享状态 | 解耦通信 |
| 事件驱动 | 发布-订阅模式 | 异步协作 |

### 2.3 LLM Multi-Agent 框架

| 框架 | 特点 | 适用场景 |
|------|------|----------|
| AutoGen | 灵活对话式 | 通用协作 |
| CrewAI | 角色驱动 | 团队任务 |
| ChatDev | 软件开发流水线 | 代码开发 |
| CAMEL | 角色扮演 | 创意协作 |
| MetaGPT | SOP 驱动 | 工程化流水线 |

## 3. 多 Agent 强化学习

| 问题 | 方法 | 代表 |
|------|------|------|
| 完全合作 | 共享奖励 | QMIX, MAPPO |
| 竞争对抗 | 零和博弈 | AlphaGo, OpenAI Five |
| 混合动机 | 部分合作竞争 | Negotiation Agent |
| 通信学习 | 学习通信协议 | CommNet, TarMAC |

## 4. 发展历史

1. 1980s：分布式 AI 和多 Agent 系统概念
2. 1990s：Agent 通信语言 (KQML, FIPA)
3. 2017：OpenAI Five (Dota 2) 多 Agent RL
4. 2023：AutoGPT / AutoGen 引爆 Multi-Agent 潮流
5. 2024：CrewAI, ChatDev 等框架成熟
6. 2025-26：Multi-Agent 从原型走向生产

## 5. 与 AGI 的关系

- **集体智能**：AGI 可能不是单一超级智能，而是多智能体生态系统
- **分工与专精**：不同 Agent 专精不同领域，组合后更强
- **安全与对齐**：多 Agent 的对齐问题更复杂（联盟形成、策略串通）
- **群体智能基础**：多 Agent 系统是群体智能的计算实现

## 6. 学习路径

1. 理解多 Agent 系统的组织结构
2. 学习 AutoGen / CrewAI 等框架
3. 研究多 Agent RL（QMIX, MAPPO）
4. 探索多 Agent 通信协议设计
5. 思考多 Agent 对齐与安全问题

## 相关知识

- [[../01_Agent理论综述/00_Agent理论综述|Agent理论综述]]：多 Agent 是 Agent 的群体扩展
- [[../../03_进化与群体智能/00_进化与群体智能_综述|进化与群体智能]]：群体智能是多 Agent 的理论来源
- [[../../01_强化学习/00_强化学习_综述|强化学习]]：多 Agent RL 是 MARL 的基础

## References

- Wu et al., *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*
- Hong et al., *MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework*
- Park et al., *Generative Agents: Interactive Simulacra of Human Behavior*
- Rashid et al., *QMIX: Monotonic Value Function Factorisation* (Multi-Agent RL)
