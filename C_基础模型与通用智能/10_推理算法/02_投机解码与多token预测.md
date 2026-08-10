# 投机解码与多token预测

## 一句话理解

投机解码（Speculative Decoding）使用小模型草拟候选 token、大模型并行验证，在不影响生成质量的前提下将解码延迟降低 30-50%；多 token 预测（Medusa/EAGLE/Lookahead）进一步突破逐 token 串行限制，通过并行预测多个未来 token 实现更大加速。

## 概述

自回归解码的天然瓶颈是逐 token 串行 —— 每步只能生成一个 token，无法利用 GPU 并行计算能力。投机解码通过引入"草稿 → 验证"两阶段打破这一瓶颈，其核心思想是：**用廉价的小模型做猜测，用昂贵的大模型做验证**。

本笔记聚焦投机解码的算法原理与变体，工程实现与系统集成见 [[../../K_AI工程化/03_推理工程/03_投机解码|K-03 投机解码]]。

## 发展历史

| 年代 | 里程碑 | 意义 |
|------|--------|------|
| 2023 | Speculative Decoding（Leviathan et al.）| 开创投机解码框架 |
| 2023 | Blockwise Parallel Decoding（Stern et al.）| 并行解码前身 |
| 2024 | Medusa（Cai et al.）| 多头预测，无需独立草稿模型 |
| 2024 | EAGLE（Li et al.）| 基于隐藏状态的高质量草拟 |
| 2024 | Lookahead Decoding（Fu et al.）| 无草稿模型、无训练的并行解码 |
| 2024 | Self-Speculative Decoding | 跳过部分层作为草稿模型 |

## 核心概念

### 草稿模型（Draft Model）

一个更小、更快的模型，用于生成候选 token 序列。可以是：
- 独立的小模型（如 7B 草稿 + 70B 目标）
- 目标模型的剪枝/层跳过版本
- 目标模型的浅层（如跳过后半层）

### 接受率（Acceptance Rate）

草稿模型生成的 token 被目标模型接受的比例。接受率越高，加速比越大。

### 无损保证

投机解码通过拒绝采样保证输出分布与原始自回归解码完全一致，**不损失生成质量**。

## 技术原理

### 标准投机解码流程

1. 草稿模型自回归生成 $k$ 个候选 token
2. 目标模型并行计算这 $k$ 个 token 的 logits
3. 逐 token 比较草稿模型和目标模型的概率分布
4. 接受匹配的 token，在首个不匹配处重新采样

### 接受/拒绝规则

设 $\pi_{\text{draft}}$ 为草稿模型分布，$\pi_{\text{target}}$ 为目标模型分布：

- 若 $\pi_{\text{draft}}(y_i) \leq \pi_{\text{target}}(y_i)$：**接受** token $y_i$
- 若 $\pi_{\text{draft}}(y_i) > \pi_{\text{target}}(y_i)$：以 $\frac{\pi_{\text{target}}(y_i)}{\pi_{\text{draft}}(y_i)}$ 概率接受

拒绝后，从截断分布 $\text{norm}(\max(0, \pi_{\text{target}} - \pi_{\text{draft}}))$ 中重新采样。

### 加速比分析

$$\text{加速比} = \frac{1}{(1 - \alpha) + \alpha / \beta}$$

其中 $\alpha$ 为接受率，$\beta$ 为草稿/目标模型速度比。

## 关键方法与模型

| 方法 | 草稿来源 | 训练需求 | 加速比 | 特点 |
|------|---------|---------|--------|------|
| Speculative Decoding | 独立小模型 | 需训练草稿模型 | 2-3x | 通用框架，质量无损 |
| Medusa | 目标模型头部添加预测头 | 需微调预测头 | 2-3x | 无需独立草稿模型 |
| EAGLE | 目标模型隐藏状态 | 需训练轻量 MLP | 3-4x | 接受率最高（70-85%） |
| Lookahead Decoding | 无草稿模型 | 无需训练 | 1.5-2x | Jacobi 迭代并行解码 |
| Self-Speculative | 目标模型层跳过 | 无需训练 | 1.5-2x | 通过层跳过作为草稿 |

## 优势与局限

**优势**：
- 无质量损失，输出分布与原始解码完全一致
- 可组合使用任意解码策略
- 适用于各类自回归模型

**局限**：
- 加速比依赖草稿模型质量
- 需要额外显存加载草稿模型（部分方法）
- 草稿模型训练成本
- 短序列场景加速比有限

## 应用场景

- **实时对话系统**：降低感知延迟
- **批量推理服务**：提升吞吐量
- **长文本生成**：解码阶段加速
- **Agent 推理**：减少工具调用链的等待时间

## 与其他技术关系

- 工程实现参见 [[../../K_AI工程化/03_推理工程/03_投机解码|K-03 投机解码]]
- 解码策略基础见 [[01_解码与采样策略]]
- 推理时扩展见 [[04_推理时扩展与计算分配]]

## 前沿发展

- **EAGLE-2/3**：持续提升接受率
- **多模态投机解码**：扩展到视觉语言模型
- **自适应草稿长度**：动态调整候选 token 数量
- **投机解码 + 推理时搜索**：组合使用提高质量与效率

## 相关知识

- [[01_解码与采样策略]]
- [[04_推理时扩展与计算分配]]
- [[../../K_AI工程化/03_推理工程/03_投机解码]]

## References

- Leviathan, Y. et al. (2023). *Fast Inference from Transformers via Speculative Decoding*. ICML.
- Cai, T. et al. (2024). *Medusa: Simple LLM Inference Acceleration Framework*. arXiv.
- Li, Y. et al. (2024). *EAGLE: Speculative Sampling with Extrapolation*. arXiv.
- Fu, Y. et al. (2024). *Lookahead Decoding: Better LLM Fast Inference via Parallel Token Prediction*. ACL.