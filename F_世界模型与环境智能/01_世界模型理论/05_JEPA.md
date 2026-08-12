---
tags:
  - JEPA
  - 预测表征
  - 自监督学习
  - 世界模型
updated: 2026-08-12
---

# JEPA 与联合嵌入预测

## 一句话理解

JEPA（Joint-Embedding Predictive Architecture）在表示空间预测目标，而不是重建目标的全部原始细节，从而把建模容量集中在对象、布局、运动与事件等稳定规律上。

## 概述

上下文编码器从可见区域产生上下文表示，预测器据此推断目标区域/未来的目标编码器表示。因为只预测抽象表征，模型无需为纹理、光照等不可预测细节付出生成代价；这是 LeCun 提出的自主智能路线（"A Path Towards Autonomous Machine Intelligence"）的核心组件之一。

## 发展历史

- **I-JEPA（2023）**：图像上以掩码区域为预测目标，验证了"预测表征而非像素"的自监督路线，无需数据增强也可学习语义特征。
- **V-JEPA（2024）**：视频上预测未来/掩码时空区域的表征，学到时序结构；与 LWM 等视频预测模型对比，更强调表征泛化。
- **动作条件 JEPA**：为服务控制与机器人，把动作纳入预测条件，向可行动世界模型靠拢（前沿方向）。

## 核心价值

- 避免为不可预测细节付出生成代价。
- 鼓励对象、布局、时序和语义等抽象规律。
- 可通过视频和多模态数据进行大规模自监督预训练。

## 关键约束

目标编码器常使用 stop-gradient 或动量更新；配合掩码策略和方差/协方差正则，防止所有表示收敛为常数（表征塌缩）。**预测目标的选择决定模型会学到什么**：若目标与行动、奖励无关，则学到的是表征而非世界模型。

## 与可行动世界模型的差距

V-JEPA 等视频表征模型可学习时序结构，但机器人或 Agent 还需要动作条件、可供性、奖励/安全约束、长时域不确定性和规划接口。因此 JEPA 可以是世界模型的表征骨架，而非自动等同完整世界模型。

## 优势与局限

- **优势**：样本与计算效率高；表征可迁移到下游任务；天然支持多模态对齐。
- **局限**：不生成观测，难以直接用作模拟器；表示空间预测可能丢失控制相关细节；动作条件、奖励与闭环验证仍需另行补足。

## 相关知识

- **前置**：[[04_Predictive_Learning|预测学习]]
- **平级**：[[03_Latent_Dynamics_Model|潜在动力学模型]]、[[../../03_生成式世界模型/00_生成式世界模型_综述|生成式世界模型]]
- **延伸**：[[../../05_世界模型驱动智能体/00_世界模型驱动智能体_综述|世界模型驱动智能体]]

## References

- LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*. OpenReview.
- Assran, M. et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA). CVPR.
- Bardes, A. et al. (2024). *Revisiting Feature Prediction for Learning Visual Representations from Video* (V-JEPA). arXiv:2404.08471.
