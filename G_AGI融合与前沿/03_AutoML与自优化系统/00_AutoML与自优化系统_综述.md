---
tags:
  - AutoML
  - NAS
  - 超参优化
  - 自优化
  - 递归自我改进
  - AGI奇点
created: 2026-07-28
updated: 2026-07-28
---

# AutoML 与自优化系统综述：从自动化到自我改进

## 一句话理解

AutoML 让机器自动寻找最优模型架构与超参数，自优化系统让 AI 能够改进自身——当这种改进形成递归闭环时，就接近了 AGI 阶梯中的"奇点"阶段：AI 自主开发下一版 AI。

## 1. 领域定义

### AutoML

自动化机器学习的全流程：数据预处理、特征工程、模型选择、超参优化和模型评估。目标是让非专家也能获得高质量模型。

### 自优化系统

AI 系统能够自动评估自身性能、识别弱点并进行针对性改进。当改进过程可以递归进行（改进后的系统更能改进自身），就形成了"递归自我改进"。

### 与 AGI 奇点的关系

```
Agent -> 持续学习 -> 奇点(自我迭代) -> 具身智能
                       |
               AutoML + 自优化 + 递归改进
```

梁文锋："当模型能够持续学习后，它已经能够自己开发自己的版本，能够自己再研究，然后开发自己的下一个版本。"

## 2. 核心内容

### 2.1 神经架构搜索（NAS）

| 方法 | 搜索策略 | 代表 |
|------|----------|------|
| 基于强化学习 | RL 控制器生成架构 | Zoph & Le 2017 |
| 基于进化 | 遗传算法搜索 | AmoebaNet |
| 可微分 | 梯度优化连续架构 | DARTS |
| One-Shot | 训练超网 + 采样子网 | Once-for-All |

### 2.2 超参优化（HPO）

| 方法 | 特点 | 代表 |
|------|------|------|
| 网格/随机搜索 | 基线方法 | Grid, Random |
| 贝叶斯优化 | 概率代理模型 | TPE, SMAC |
| Hyperband | 早早淘汰差配置 | ASHA |
| 多保真度 | 低精度评估加速 | Successive Halving |

### 2.3 自动数据与特征工程

- 自动特征工程（AutoFeat）
- 自动数据增强（AutoAugment, RandAugment）
- 数据清洗与标签噪声处理

### 2.4 递归自我改进

| 概念 | 描述 | 状态 |
|------|------|------|
| AI 辅助 AI 研究 | LLM 辅助写代码/设计实验 | 已实现（Cursor, Devin） |
| AI 优化 AI 模型 | 自动搜索最优架构/训练策略 | 部分实现（NAS, HPO） |
| AI 开发下一版 AI | 递归闭环的自我改进 | 远期目标（奇点） |
| DeepSeek 路线 | 模型首先对我们自己有用 | 明确方向 |

## 3. 发展历史

1. 2013：Auto-WEKA 开创 AutoML
2. 2017：Zoph NAS（RL搜索）、Google AutoML
3. 2018：DARTS（可微分 NAS）
4. 2020：AutoML 从学术走向工业
5. 2023：LLM 辅助编程（AI 改进 AI 的雏形）
6. 2025-26：AI 辅助 AI 研究成为常态

## 4. 与 AGI 的关系

- **奇点路径**：递归自我改进是 AGI 奇点的核心技术
- **当前阶段**：AI 辅助 AI 研究已经是现实（Coding Agent）
- **未来方向**：从"辅助"到"自主"的跨越需要持续学习
- **安全挑战**：自我改进系统需要严格的安全约束

## 5. 学习路径

1. 理解 AutoML 全流程
2. 学习 NAS 核心方法（DARTS 最重要）
3. 学习贝叶斯超参优化
4. 理解 AI 辅助 AI 研究的现状
5. 思考递归自我改进的安全约束

## 相关知识

- [[../../A_基础与范式/02_优化理论与方法/00_优化理论与方法_综述|优化理论]]：NAS/HPO 的优化基础
- [[../../D_行为主义与智能体/03_进化与群体智能/00_进化与群体智能_综述|进化与群体智能]]：进化 NAS 的理论来源
- [[../../D_行为主义与智能体/05_持续学习与元学习/00_持续学习与元学习_综述|持续学习与元学习]]：自优化需要持续学习能力
- [[../../I_AGI安全与治理/01_AI安全与对齐/00_AI安全与对齐_综述|AI安全与对齐]]：自我改进系统需要安全约束

## References

- Zoph & Le, *Neural Architecture Search with Reinforcement Learning*
- Liu et al., *DARTS: Differentiable Architecture Search*
- Feurer et al., *Efficient and Robust Automated Machine Learning* (Auto-sklearn)
- Borgeaud et al., *Improving Language Models by Retrieving from Trillions of Tokens*
- DeepSeek 梁文锋投资者交流会，2026-05-20
