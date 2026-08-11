---
title: "STRIPS、PDDL 与 HTN"
domain: "D-03"
level: "核心"
status: "稳定"
last_reviewed: 2026-08-11
review_cycle: 180
evidence_level: "理论"
source_of_truth: true
tags:
  - STRIPS
  - PDDL
  - HTN
  - 经典规划
created: 2026-08-11
updated: 2026-08-11
---

# STRIPS、PDDL 与 HTN

## 一句话理解

STRIPS 和 PDDL 将"规划问题"形式化为"初始状态、目标状态和一组动作"的数学问题，让 AI 可以自动搜索动作序列；HTN 则从"如何分解任务"的角度规划，将高层目标递归分解为可执行的子任务——这两种经典规划范式至今仍是机器人、航天和游戏 AI 中的重要工具。

## 概述

在 LLM Agent 时代，"规划"常被理解为 CoT 或 ReAct 式的语言推理。但更早的规划研究——STRIPS、PDDL、HTN——提供了一套完全不同的规划范式：**基于逻辑状态和动作模型的形式化规划**。它不是"让模型生成计划"，而是"让求解器在已知状态空间中搜索可行计划"。[[00_规划与推理_综述|规划与推理]] 中提到了"符号规划提供清晰前置条件"，本笔记正是对这一范式的详细展开。理解经典规划，对于理解"LLM 规划能做什么、不能做什么"至关重要。

## 发展历史

| 年代 | 里程碑 | 意义 |
|:---|:---|:---|
| 1969 | STRIPS（Shakey 机器人项目） | 首个实际 AI 规划系统，用前置条件/增加/删除列表建模动作 |
| 1971 | ABSTRIPS | 层次化 STRIPS，引入抽象层级 |
| 1988 | HTN 形式化 | 将"任务分解"作为规划的核心机制 |
| 1998 | PDDL 标准发布 | 统一规划领域描述语言，推动国际规划竞赛（IPC） |
| 2000-2010 | IPC 驱动的 PDDL 扩展 | 时间规划、数值规划、不确定性规划等扩展 |
| 2010s | 规划与学习的结合 | 从经典规划走向"学习动作模型 + 规划" |
| 2022- | LLM 作为规划器 | LLM 从自然语言生成计划，但经典规划的可验证性仍不可替代 |

## 核心概念

### STRIPS

STRIPS（Stanford Research Institute Problem Solver）将规划世界建模为：

- **状态**：一组逻辑原子（谓词）的合取，如 `At(Robot, Room1) ∧ Clean(Room1)`
- **动作**：由三个列表定义——前置条件（Precondition）、增加列表（Add-list）、删除列表（Delete-list）
- **目标**：一组要满足的逻辑原子，如 `At(Robot, Room2) ∧ Clean(Room2)`

一个动作的形式化表示：

```
Action: Move(r, from, to)
  Precondition: At(r, from) ∧ Connected(from, to)
  Effect: At(r, to) ∧ ¬At(r, from)
```

### PDDL

PDDL（Planning Domain Definition Language）将规划问题分为两部分：

- **领域文件（Domain）**：定义谓词和动作的通用模板，可跨问题复用
- **问题文件（Problem）**：定义具体的初始状态和目标状态

```pddl
;; 领域文件示例
(define (domain robot)
  (:predicates (at ?r ?loc) (connected ?from ?to))
  (:action move
    :parameters (?r ?from ?to)
    :precondition (and (at ?r ?from) (connected ?from ?to))
    :effect (and (at ?r ?to) (not (at ?r ?from)))))

;; 问题文件示例
(define (problem move-robot)
  (:domain robot)
  (:init (at robot room1) (connected room1 room2))
  (:goal (at robot room2)))
```

### HTN（层次任务网络）

HTN（Hierarchical Task Network）与 STRIPS/PDDL 的根本区别在于：HTN 不是从"初始状态→目标状态"搜索，而是从"高层任务→底层动作"分解。

- **任务（Task）**：需要完成的事情，分为原始任务（Primitive Task，可直接执行）和复合任务（Compound Task，需要分解）
- **方法（Method）**：将复合任务分解为子任务网络的方式
- **规划过程**：从顶层任务开始，递归选择方法分解，直到全部为原始任务

```text
任务: "准备旅行"
  → 方法: ① 预订交通 ② 预订住宿 ③ 打包行李
    → 子任务: "预订交通"
      → 方法: ① 选择交通方式 ② 查询时间 ③ 支付
```

## 技术原理

### 三种规划范式的对比

| 维度 | STRIPS/PDDL | HTN |
|:---|:---|:---|
| 问题视角 | 状态空间搜索 | 任务分解 |
| 动作模型 | 前置条件+效果（显式） | 方法+子任务（层次化） |
| 搜索方向 | 前向/后向搜索 | 自上而下分解 |
| 可验证性 | 高（精确的状态谓词） | 中（依赖分解方法的正确性） |
| 人工建模成本 | 需要定义完整的动作模型 | 需要定义方法和任务知识 |
| 适合场景 | 状态空间小、动作模型明确 | 任务结构已知、层次化场景 |

### 经典规划的局限

1. **封闭世界假设**：经典规划假设世界是封闭的——所有相关信息都在初始状态中描述。这在开放世界中不成立
2. **动作模型获取困难**：需要人工定义每个动作的前置条件和效果，复杂度高
3. **确定性假设**：经典规划假设动作有确定效果，不处理随机或不完全信息
4. **可扩展性**：状态空间随问题规模指数增长，需要高效的启发式搜索

### LLM 与经典规划的关系

LLM 和经典规划可以互补：

| LLM 的优势 | 经典规划的优势 |
|:---|:---|
| 从自然语言理解任务 | 精确的可验证性 |
| 自动生成候选计划 | 保证计划可行（如果模型正确） |
| 处理开放式、非结构化任务 | 给出最优性或可行性证明 |

结合方式：LLM 生成候选计划，经典规划器验证其可行性；或 LLM 从自然语言中提取 PDDL 动作模型，交给规划器求解。

## 关键方法与模型

| 方法 | 核心思想 | 特点 |
|:---|:---|:---|
| STRIPS | 前置条件+增加/删除列表 | 最简单的规划形式化 |
| PDDL | STRIPS 的标准化 | 国际规划竞赛标准语言 |
| HTN | 任务分解 + 方法库 | 适合已知过程的任务 |
| FastDownward | 前向搜索 + 启发式 | IPC 获奖规划器 |
| FF（Fast Forward） | 贪婪前向搜索 | 高效启发式规划 |

## 优势与局限

**优势**：提供精确的可验证性——如果动作模型正确，规划器可以保证找到的计划可行且最优；PDDL 规划器在特定领域（如航天、机器人、推理任务）中已有成熟应用；HTN 的层次分解与人类认知方式接近，易于理解和设计。

**局限**：动作模型获取困难，在开放世界中几乎不可行；规划器对状态/动作空间的大小敏感，大规模问题需要高效启发式；LLM 时代，大多数 Agent 系统不采用经典规划，而是使用更灵活但不可验证的 CoT/ReAct 范式。

## 应用场景

- **机器人任务规划**：在已知环境中，用 PDDL 规划器生成机器人动作序列
- **航天任务调度**：NASA 使用 PDDL 规划器调度航天器活动
- **游戏 AI 规划**：游戏中的 NPC 任务规划和角色行为决策
- **LLM 计划验证**：用 LLM 生成计划，用 PDDL 规划器验证可行性

## 与其他技术关系

- 经典规划是 [[02_搜索与MCTS|搜索算法]] 在规划问题中的直接应用——规划器本质上是状态空间搜索
- HTN 的层次分解是 [[04_层次规划与规划_RL结合|层次规划]] 的理论基础
- 经典规划的可验证性弥补了 [[03_LLM规划与长程任务|LLM 规划]] 的不可靠性
- 动作模型可以视为 [[../02_环境与世界模型/04_状态表示与环境预测|环境预测模型]] 的一种符号化形式

## 前沿发展

- **自动动作模型学习**：从观察数据中自动学习 STRIPS/PDDL 动作模型，减少人工建模成本
- **LLM 辅助规划**：LLM 从自然语言中提取 PDDL 领域定义，或生成候选计划供规划器验证
- **概率规划**：扩展 PDDL 以处理不确定性（PPDDL、MDP 规划）
- **规划与学习的统一**：神经网络学习启发式，传统规划器使用启发式做搜索

## 常见问题

**Q: 为什么现在还在研究 50 年前的 STRIPS？**
STRIPS 的核心思想——用前置条件和效果表示动作——仍然是规划领域最简洁的形式化，也是所有后续规划系统（从 PDDL 到 LLM 规划）的基础。理解 STRIPS 是理解一切规划系统的起点。

**Q: LLM 规划会取代经典规划吗？**
不会。LLM 规划灵活但不可验证，经典规划严格但需要人工建模。两者是互补关系：LLM 处理"规划什么"（从自然语言理解任务），经典规划处理"如何规划"（在已知模型中搜索可行路径）。

## 相关知识

- 前置：[[02_搜索与MCTS|搜索算法]]（规划器本质上是搜索器）
- 平级：[[../01_智能体理论基础/02_决策理论与BDI|BDI 决策理论]]（意图与计划的关系）
- 延伸：[[03_LLM规划与长程任务|LLM 规划]]、[[04_层次规划与规划_RL结合|层次规划]]

## References

- Fikes, R. E. & Nilsson, N. J. (1971). *STRIPS: A New Approach to the Application of Theorem Proving to Problem Solving*. Artificial Intelligence.
- Ghallab, M. et al. (2004). *Automated Planning: Theory and Practice*. Morgan Kaufmann.
- Erol, K. et al. (1994). *HTN Planning: Complexity and Expressivity*. AAAI.
- McDermott, D. et al. (1998). *PDDL: The Planning Domain Definition Language*. Yale Center for Computational Vision and Control.