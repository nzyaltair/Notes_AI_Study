# Scaling Laws 与计算最优训练

## 1. 概述

Scaling Laws（缩放定律）描述了语言模型性能与模型规模、数据量、计算量之间的幂律关系。这一发现由 Kaplan et al. (2020) 首次系统提出，后经 Hoffmann et al. (2022, Chinchilla) 修正和完善。Scaling Laws 为大模型训练提供了理论指导——在有限计算预算下如何最优分配参数量和数据量，是推动 LLM 规模化的关键理论基础。

## 2. 发展历史

| 时间 | 里程碑 | 核心发现 |
|------|--------|---------|
| 2017 | Hestness et al. | 早期观察到深度学习的幂律关系 |
| 2020.01 | **Kaplan et al.** (OpenAI) | 系统提出 Scaling Laws |
| 2020.05 | GPT-3 发布 | 验证规模化带来能力涌现 |
| 2021 | Hoffmann et al. 初步修正 | 指出 Kaplan 低估了数据重要性 |
| 2022.03 | **Chinchilla** (DeepMind) | 计算最优比例：参数与数据等比例缩放 |
| 2022.06 | Emergent Abilities (Wei et al.) | 系统描述涌现能力现象 |
| 2023 | Chinchilla Scaling 被广泛采用 | LLaMA 等开源模型遵循 Chinchilla 比例 |
| 2023+ | 逆 Scaling / U形 Scaling | 部分任务随规模先变差再变好 |

## 3. 核心概念

### 幂律关系

模型损失（交叉熵）与关键因素呈幂律关系：

$$L(x) = \left(\frac{x_c}{x}\right)^{\alpha} + L_\infty$$

其中 $x$ 可以是参数量 $N$、数据量 $D$ 或计算量 $C$，$L_\infty$ 是不可约损失（数据本身的熵），$x_c$ 和 $\alpha$ 是经验常数。

### 三个维度

| 维度 | 符号 | 含义 | 缩放关系 |
|------|------|------|---------|
| 参数量 | $N$ | 模型可训练参数数 | $L(N) \propto N^{-\alpha_N}$ |
| 数据量 | $D$ | 训练 token 数 | $L(D) \propto D^{-\alpha_D}$ |
| 计算量 | $C$ | 总 FLOPs | $L(C) \propto C^{-\alpha_C}$ |

### 涌现能力（Emergent Abilities）

某些能力在模型规模超过阈值后**突然出现**，而非渐进提升：

- **少样本推理**：~10^22 FLOPs 后出现
- **多步算术**：~10^22 FLOPs 后出现
- **符号操作**：~10^23 FLOPs 后出现

> 注意：2023 年研究（Schaeffer et al.）指出，部分"涌现"可能是评估指标的非线性造成的假象，使用连续指标后能力可能是渐进提升的。

## 4. 技术原理

### 4.1 Kaplan Scaling Laws (OpenAI, 2020)

Kaplan et al. 的核心发现：

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050$$

**联合缩放公式**：

$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}$$

**Kaplan 的最优分配建议**：
- 在固定计算预算 $C$ 下，最优策略是**优先增大模型参数量**
- 最优分配：$N \propto C^{0.73}$，$D \propto C^{0.27}$
- 即：大部分计算预算应投入更大的模型，而非更多数据
- 大模型应**在远未收敛时停止训练**（early stopping）

**推导最优分配**：

给定 $C \approx 6ND$（Transformer 前向+反向 FLOPs 估算），最小化 $L(N, D)$：

$$\min_{N, D} L(N, D) \quad \text{s.t.} \quad C = 6ND$$

使用拉格朗日乘数法求解。

### 4.2 Chinchilla Scaling (DeepMind, 2022)

Hoffmann et al. 通过训练 400+ 个模型（70M 到 16B 参数，5B 到 500B tokens）发现 Kaplan **低估了数据量的重要性**。

**修正后的缩放指数**：

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

其中 $E$ 是不可约损失，拟合参数：
- $\alpha \approx 0.34$
- $\beta \approx 0.28$
- $A \approx 406.4$, $B \approx 410.7$

**Chinchilla 的最优分配**：

$$N^* \propto C^{0.50}, \quad D^* \propto C^{0.50}$$

**关键结论**：参数和数据应**等比例缩放**，最优数据量约为参数量的 20 倍：

$$D^* / N^* \approx 20$$

即每个参数约需 20 个 token 的训练数据。

**与 Kaplan 的对比**：

| 维度 | Kaplan (2020) | Chinchilla (2022) |
|------|--------------|-------------------|
| $N$ 的指数 | 0.73 | 0.50 |
| $D$ 的指数 | 0.27 | 0.50 |
| 数据/参数比 | ~1.7 (偏低) | ~20 (最优) |
| 训练策略 | 大模型+少数据 | 均衡增长 |

### 4.3 Chinchilla 验证

DeepMind 用相同计算预算训练了 Chinchilla（70B 参数，1.4T tokens），对比 Gopher（280B 参数，300B tokens）：

| 模型 | 参数量 | 训练数据 | MMLU 准确率 |
|------|--------|---------|------------|
| Gopher | 280B | 300B tokens | ~60% |
| GPT-3 | 175B | 300B tokens | ~43.9% |
| **Chinchilla** | **70B** | **1.4T tokens** | **67.5%** |

**结论**：用同样计算量，更小的模型 + 更多数据 = 更好的性能。大量已发布的 LLM 都**严重训练不足**。

### 4.4 计算量估算

Transformer 的 FLOPs 估算：

**前向传播**（每个 token）：
$$C_{\text{forward}} \approx 2ND$$

其中 $N$ 是参数量，$D$ 是 token 数。系数 2 来源于矩阵乘法（每个参数对应一次乘法和一次加法）。

**反向传播**：约为前向的 2 倍。

**总计**：
$$C_{\text{total}} \approx 6ND$$

**示例**：
- GPT-3 (175B 参数, 300B tokens): $C \approx 6 \times 175 \times 10^9 \times 300 \times 10^9 \approx 3.15 \times 10^{23}$ FLOPs
- Chinchilla (70B 参数, 1.4T tokens): $C \approx 6 \times 70 \times 10^9 \times 1.4 \times 10^{12} \approx 5.9 \times 10^{23}$ FLOPs

### 4.5 过拟合与关键批次大小

**过拟合边界**：当数据量不足时，模型开始过拟合。

$$D_{\text{crit}} \approx 5.85 \times N^{0.76}$$

当 $D < D_{\text{crit}}$ 时，验证损失开始高于训练损失。

**关键批次大小**（Critical Batch Size）：

$$B_{\text{crit}} \approx \frac{C}{10}$$

- 当 batch size < $B_{\text{crit}}$ 时，增大 batch size 可加速训练且不损失效率
- 当 batch size > $B_{\text{crit}}$ 时，增大 batch size 的收益递减

## 5. 实践意义

### 训练预算规划

给定计算预算 $C$，计算最优参数量和数据量：

$$N^* = \frac{C}{6 \times 20} = \frac{C}{120}, \quad D^* = 20 N^*$$

| 计算预算 (FLOPs) | 最优 $N$ | 最优 $D$ | 典型模型 |
|-----------------|---------|---------|---------|
| $10^{20}$ | ~0.8B | ~16B | 小型实验 |
| $10^{21}$ | ~8B | ~160B | 中型模型 |
| $10^{22}$ | ~80B | ~1.6T | GPT-3 级别 |
| $10^{23}$ | ~800B | ~16T | GPT-4 级别 |
| $10^{24}$ | ~8T | ~160T | 未来模型 |

### 现代模型的数据策略

遵循 Chinchilla 比例的模型（2023+）：

| 模型 | 参数量 | 训练数据 | $D/N$ 比 |
|------|--------|---------|---------|
| LLaMA-1 | 7B-65B | 1T-1.4T tokens | ~15-20 |
| LLaMA-2 | 7B-70B | 2T tokens | ~30-285 |
| LLaMA-3 | 8B-70B | 15T tokens | ~1875-214 |
| Qwen-2 | 0.5B-72B | 12T tokens | 高 |
| Chinchilla | 70B | 1.4T tokens | 20 |

> 注意：LLaMA-2/3 等模型已远超 Chinchilla 比例（$D/N \gg 20$），这被称为"过度训练"（Overtraining），目的是在推理阶段获得更好的性能-成本比（推理次数远多于训练次数）。

## 6. 优势与局限

### 优势
- 为大模型训练提供理论指导，避免盲目试错
- 预测模型性能，降低训练失败风险
- 指导计算资源最优分配

### 局限
- **幂律在极大规模下可能失效**：外推到未验证的规模有风险
- **忽略了数据质量**：公式假设数据质量均匀，实际差异巨大
- **未考虑架构创新**：MoE、新注意力机制等改变了缩放行为
- **涌现不可预测**：Scaling Laws 描述平均损失，不预测特定能力涌现
- **Chinchilla 比例可能过时**：现代模型（LLaMA-3）已大量"过度训练"

## 7. 应用场景

- **训练规划**：计算最优模型大小和数据量
- **性能预测**：在小模型上拟合缩放曲线，外推大模型性能
- **资源分配**：在有限 GPU 预算下最大化模型性能
- **推理优化**：过度训练以降低推理成本（小模型 + 多数据 → 推理更便宜）
- **模型选择**：根据部署场景选择参数-数据的最优平衡

## 8. 与其他技术关系

- Scaling Laws 指导 [[11_大模型训练与对齐/01_预训练流程/01_预训练流程|预训练流程]] 中的资源分配
- 涌现能力是 [[10_大语言模型核心架构|大语言模型]] 规模化的核心驱动力
- 计算量估算与 [[12_大模型推理与优化/04_推理加速与并行/04_推理加速与并行|推理加速与并行]] 中的效率优化相关
- 过度训练策略与 [[12_大模型推理与优化/02_量化技术/02_量化技术|量化技术]]、[[12_大模型推理与优化/03_剪枝与稀疏化/03_剪枝与稀疏化|剪枝与稀疏化]] 等推理优化互补
- Chinchilla 比例影响开源模型的训练策略（LLaMA、Qwen 等）

## 9. 前沿发展

- **过度训练**：LLaMA-2/3 远超 Chinchilla 比例，牺牲训练效率换取推理效率
- **MoE 缩放**：混合专家模型的 Scaling Laws 与稠密模型不同
- **数据质量缩放**：研究数据质量与数据量的交互作用
- **逆 Scaling**：某些任务随规模变差（如社会偏见），需专门研究
- **多模态缩放**：Scaling Laws 在视觉-语言模型中的适用性
- **推理时缩放**：o1 等推理模型在推理阶段投入更多计算，扩展了缩放维度
- **Chinchilla 修正**：2024 年研究指出 Chinchilla 的某些估计可能仍有偏差
