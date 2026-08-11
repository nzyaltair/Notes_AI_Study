# 缩放规律（Scaling Laws）

## 一句话理解

缩放规律是在固定模型族、数据分布和训练配方下，拟合损失与参数量、数据量或计算量之间常出现的经验幂律关系，可用于训练预算分配而非能力保证。

## 基本形式

常见拟合形式为：

$$L(N)\approx L_\infty+aN^{-\alpha},$$

其中 $N$ 可表示参数、数据或计算量；指数和不可约损失都依赖实验设置。计算最优训练通常要平衡模型大小与训练 token，而非只扩参数。

## 使用原则

外推前检查数据质量、去重、模型架构和训练稳定性是否与拟合区间一致。损失缩放不等价于所有基准、推理成本、安全性或涌现能力同步缩放。

## 相关知识

[[05_过参数化]] · [[10_涌现能力]] · [[../../C_基础模型与通用智能/01_基础模型理论/00_基础模型理论_综述\|基础模型理论]]

## References

- Kaplan et al. Scaling Laws for Neural Language Models. *arXiv*, 2020.
- Hoffmann et al. Training Compute-Optimal Large Language Models. *arXiv*, 2022.
