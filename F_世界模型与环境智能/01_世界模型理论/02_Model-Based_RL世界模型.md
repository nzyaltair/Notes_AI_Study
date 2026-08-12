---
tags:
  - Model-Based RL
  - 想象学习
  - 世界模型
updated: 2026-08-12
---

# Model-Based RL 世界模型

## 一句话理解

模型化强化学习（Model-Based RL）先学习或给定环境模型，再利用模型在"想象"中推演行动后果并优化策略，从而用更少的真实交互达到目标。

## 概述

相比直接从真实试错学习策略（model-free），MBRL 可以复用每条交互数据做大量内部推演，是样本效率的主要来源；代价是模型误差会在推演中累积，甚至被优化器主动放大。

## 发展历史

- **Dyna（1991）**：Sutton 提出用学得的模型生成"计划经验"，与真实经验混合更新策略，确立"真实数据 + 模型想象"的基本范式。
- **像素空间模型（2018–2019）**：World Models、SimPLe 等在像素或表征空间预测未来，验证了深度生成模型用于 RL 的可行性，但像素预测浪费容量。
- **潜在空间模型（2019–2023）**：PlaNet/Dreamer 在潜空间做想象式 actor-critic；MuZero 学习隐式模型直接预测价值、奖励与策略；TD-MPC 系列面向连续控制的短时域潜在 MPC。
- **通用性与规模化（2023–）**：DreamerV3 以单一配置在 150+ 任务上超越专用方法，并首次从零学会 Minecraft 钻石；MBRL 开始与离线 RL、生成式模拟器结合。

## 基本闭环

```text
收集真实转移 (o, a, r, o')
        ↓
学习动力学、奖励与终止模型
        ↓
在模型中搜索或生成想象轨迹
        ↓
优化行动序列 / 价值函数 / 策略
        ↓
真实执行并用新数据校正模型
```

## 主要范式

| 范式 | 代表 | 特点 |
|---|---|---|
| 显式模型 + 搜索 | PETS、MPC | 模型集成与在线轨迹优化，解释直接 |
| 潜在模型 + 想象 | Dreamer | 在低维潜空间训练 actor-critic |
| 隐式模型 + 搜索 | MuZero | 预测价值/奖励/策略，无需复原观测 |
| 短时域潜在 MPC | TD-MPC/TD-MPC2 | 目标是控制相关潜状态，强调实时性 |

## 关键风险

优化器会主动选择模型最错误、却被预测为高回报的区域，称为 **exploitation of model errors**。应以不确定性惩罚、模型集成、行为约束、短滚动时域和频繁现实再规划进行缓解。模型与策略的分布偏移（策略访问训练分布外状态）是误差累积的另一来源，需用真实数据周期校正。

## 优势与局限

- **优势**：样本效率高；可在执行前比较候选方案；支持反事实评估与安全沙箱测试；可复用离线数据。
- **局限**：模型学习本身需要数据与调参；在环境高维、强随机且模型难以校准时，model-free 方法可能更稳健；模型偏差的闭环放大难以完全消除。

## 何时适用

真实交互昂贵、风险高或样本稀缺时，世界模型最有价值；环境高维、强随机且模型难以校准时，模型自由方法可能更稳健。实际系统常采用混合方案（模型规划 + 无模型策略兜底 + 规则安全层）。

## 相关知识

- **前置**：[[../../D_智能体与自主系统/04_强化学习/00_强化学习_综述|强化学习]]
- **平级**：[[03_Latent_Dynamics_Model|潜在动力学模型]]、[[../05_世界模型驱动智能体/01_基于世界模型的规划|基于世界模型的规划]]
- **延伸**：[[../../07_世界模型评估与前沿/00_世界模型评估与前沿_综述|评估与前沿]]

## References

- Sutton, R. S. (1991). *Dyna, an Integrated Architecture for Learning, Planning, and Reacting*. ACM SIGART Bulletin.
- Kaiser, L. et al. (2019). *Model-Based Reinforcement Learning for Atari* (SimPLe). arXiv:1903.00374.
- Hafner, D. et al. (2020). *Dream to Control: Learning Behaviors by Latent Imagination* (Dreamer). ICLR.
- Hafner, D. et al. (2023). *Mastering Diverse Domains through World Models* (DreamerV3). arXiv:2301.04104.
- Schrittwieser, J. et al. (2020). *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model*. Nature.
- Hansen, N. et al. (2023). *TD-MPC2: Scalable, Robust World Models for Continuous Control*. arXiv:2310.16828.
