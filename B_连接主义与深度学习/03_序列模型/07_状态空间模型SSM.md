# 状态空间模型（SSM）

## 1. 概述

状态空间模型（State Space Model, SSM）源于控制论和信号处理中的连续时间线性系统理论，在深度学习中被重新发掘为一种高效的序列建模框架。SSM 通过一阶微分（或差分）方程描述系统的内部状态如何随输入和当前状态演化，以**线性复杂度**处理序列数据，成为近年来挑战 Transformer 注意力机制的最有潜力方向之一。

- **解决的问题**：Transformer 自注意力的 $\mathcal{O}(T^2)$ 计算复杂度在长序列（如 DNA 序列百万级、高分辨率视频）上成为瓶颈。SSM 以 $\mathcal{O}(T)$ 或 $\mathcal{O}(T \log T)$ 的复杂度实现长程依赖建模。
- **核心模型**：S4（Structured State Space for Sequences）、Mamba（Selective SSM）、Mamba-2（SSD 框架）。
- **关键意义**：SSM 展现了"不依赖注意力也能有效建模长序列"的可能性，是后 Transformer 时代最重要的架构探索方向之一。

## 2. 发展历史

| 年代 | 里程碑 | 核心贡献 |
|:---|:---|:---|
| 1960s | Kalman 滤波器 | 现代状态空间模型和最优估计理论的起源 |
| 2020 | HiPPO 矩阵理论 (Gu et al.) | 从函数逼近的角度推导出最优状态转移矩阵 $A$，为 SSM 在深度学习中的应用奠定了数学基础 |
| 2021 | LSSL (Gu et al.) | 首次将 HiPPO 线性状态空间层应用于序列建模，展示了 SSM 处理长序列的潜力 |
| 2022 | S4 (Gu et al.) | 引入结构化对角加低秩矩阵，使 SSM 的计算和存储从 $\mathcal{O}(T^2)$ 降至 $\mathcal{O}(T \log T)$，在 Long Range Arena 基准上超越所有现有方法 |
| 2023 | S5 / H3 / Hyena | S5 提出并行扫描；H3（Hungry Hungry Hippos）将 SSM 与门控结合；Hyena 用长卷积替代注意力 |
| 2023.12 | Mamba (Gu & Dao) | 引入**选择性 SSM**：使 $B, C, \Delta$ 依赖于输入 $x$，打破了 LTI 假设。结合硬件感知的并行扫描算法，实现了线性时间的训练和推理 |
| 2024.05 | Mamba-2 / SSD (Dao & Gu) | 提出 SSD（State Space Duality）框架，统一了结构化 SSM 和线性注意力的矩阵变换视角。通过矩阵乘法的结构化分解，相比 Mamba 提速 2~8 倍 |
| 2024 | Jamba (AI21 Labs) | SSM + Transformer 混合架构，交替使用 Mamba 层和注意力层，在长上下文任务中平衡效率和质量 |
| 2024 | Mamba 扩展到视觉和基因组 | Vision Mamba (Vim)、VMamba、DNA Mamba 等，将 SSM 应用到多模态和科学计算领域 |

## 3. 核心概念

### 3.1 连续时间状态空间系统

$$h'(t) = A h(t) + B x(t)$$
$$y(t) = C h(t) + D x(t)$$

- $h(t) \in \mathbb{R}^N$：$N$ 维隐藏状态（系统的内部记忆）
- $x(t) \in \mathbb{R}$：输入信号（标量，实际使用时扩展到 $D$ 个独立通道）
- $y(t) \in \mathbb{R}$：输出信号
- $A \in \mathbb{R}^{N \times N}$：状态转移矩阵（最关键的参数，决定记忆的衰减模式）
- $B \in \mathbb{R}^{N \times 1}$：输入投影向量
- $C \in \mathbb{R}^{1 \times N}$：输出投影向量

### 3.2 HiPPO 理论

HiPPO（High-order Polynomial Projection Operator）回答了"什么样的 $A$ 矩阵能最好地记住输入历史"：

$$\text{HiPPO-LegS: } A_{nk} = -\begin{cases} (2n+1)^{1/2}(2k+1)^{1/2} & n > k \\ n+1 & n = k \\ 0 & n < k \end{cases}$$

HiPPO-LegS 矩阵使隐藏状态在线性函数空间的 Legendre 多项式投影下，最优地"记住"随时间衰减的历史信号。

### 3.3 离散化（Discretization）

将连续时间 SSM 转换为可计算的离散递推关系：

**零阶保持 (ZOH)**：
$$\bar{A} = \exp(\Delta \cdot A)$$
$$\bar{B} = (\Delta A)^{-1} (\exp(\Delta \cdot A) - I) \cdot \Delta B$$

**离散递推**：
$$h_t = \bar{A} h_{t-1} + \bar{B} x_t$$
$$y_t = C h_t$$

其中 $\Delta$ 是步长参数，控制输入信号的采样粒度——$\Delta$ 越小，$\bar{A}$ 越接近单位矩阵，系统"记住"得越久。

### 3.4 选择性机制（Selection Mechanism）

传统 SSM（S4）是**线性时不变（LTI）** 系统——参数 $A, B, C, \Delta$ 对所有输入固定不变。Mamba 的关键突破是**选择性 SSM**：

$$B_t = s_B(x_t), \quad C_t = s_C(x_t), \quad \Delta_t = \tau_{\Delta}(\text{Parameter} + s_{\Delta}(x_t))$$

其中 $s_B, s_C, s_{\Delta}$ 是将输入 $x_t$ 映射到参数空间的小型线性投影。

- **意义**：使模型能够根据当前输入内容，动态决定"忽略什么"和"记住什么"——这与 LSTM 的门控机制精神一致，但通过完全不同的数学路径实现
- **代价**：LTI 系统的卷积计算优势消失，需要新的并行计算方法

### 3.5 并行扫描（Parallel Scan）

选择性 SSM 不再可以通过 FFT 卷积计算（因为 $B_t, C_t$ 依赖于 $x_t$）。并行扫描是解决此问题的关键算法：

给定序列操作 $x_1 \oplus x_2 \oplus \cdots \oplus x_T$，并行扫描在前缀和（Prefix Sum）的框架下高效计算所有中间状态。对于 SSM 的逐时间步递推，可将 $h_t$ 的计算转化为高度并行的前缀扫描操作，在 GPU 上实现接近 $\mathcal{O}(\log T)$ 的并行度。

## 4. 技术原理

### 4.1 S4 的卷积模式

对于 LTI 系统，可将递推形式转化为卷积：

$$y = x * \bar{K}, \quad \bar{K} = (C\bar{B}, C\bar{A}\bar{B}, C\bar{A}^2\bar{B}, \ldots, C\bar{A}^{T-1}\bar{B})$$

$$\bar{y}_t = \sum_{s=0}^{t} \bar{K}_s \cdot x_{t-s}$$

$\bar{K}$ 称为 SSM 的卷积核。通过 FFT 可在 $\mathcal{O}(T \log T)$ 时间内完成整个序列的并行计算。S4 的核心贡献是将非结构化的 $A$ 矩阵参数化为对角加低秩（DPLR）形式，使 $\bar{K}$ 的计算高效稳定。

### 4.2 Mamba 的选择性 SSM 架构

Mamba 块的设计：

1. **输入投影**：$x \to \text{Linear}(x)$，映射到更高维度
2. **1D 卷积**：短卷积（kernel size = 4）提取局部特征
3. **SiLU 激活**
4. **选择性 SSM**：$B(x), C(x), \Delta(x)$ 基于当前输入动态生成
5. **离散化与递推**：并行扫描计算 SSM 输出
6. **门控**：SSM 输出与 SiLU 激活后的残差路径逐元素相乘
7. **输出投影**：线性投影回原始维度

整体结构类似于 Transformer 块（归一化 → 混合器 → 残差），但混合器从自注意力替换为 SSM。

### 4.3 Mamba-2 的 SSD 框架

SSD（State Space Duality）的核心理念：SSM 和线性注意力是**同一类矩阵变换**的两种视图：

- **SSM 视图**：递推计算 $y = \text{SSM}(A, B, C)(x)$
- **注意力视图**：$Y = (L \circ (Q K^T)) \cdot V$，其中 $L$ 是下三角掩码矩阵，$Q, K, V$ 与 $B(x), C(x), x$ 相关

Mamba-2 将选择性 SSM 重新表述为结构化矩阵乘法，使其能利用 GPU 张量核心（Tensor Cores）的矩阵乘法硬件加速。相比 Mamba-1 在训练时可提速 2~8 倍，同时保持推理时的线性复杂度。

### 4.4 时间复杂度对比

| 模型 | 训练复杂度 | 推理复杂度（逐 token） | 长程依赖路径 |
|:---|:---|:---|:---|
| **Transformer (dense)** | $\mathcal{O}(T^2 d)$ | $\mathcal{O}(T d)$ | $\mathcal{O}(1)$ |
| **RNN/LSTM** | $\mathcal{O}(T d^2)$ | $\mathcal{O}(d^2)$ | $\mathcal{O}(T)$ |
| **S4** | $\mathcal{O}(T \log T)$ | $\mathcal{O}(d^2)$ | $\mathcal{O}(\log T)$ |
| **Mamba** | $\mathcal{O}(T d)$ | $\mathcal{O}(d)$ | $\mathcal{O}(1)$ |
| **Mamba-2** | $\mathcal{O}(T d)$ | $\mathcal{O}(d)$ | $\mathcal{O}(1)$ |

## 5. 关键模型与论文

| 论文 | 核心贡献 | 影响 |
|:---|:---|:---|
| Gu et al., *HiPPO* (NeurIPS 2020) | 从函数逼近推导最优 $A$ 矩阵 | 为所有后续 SSM 工作提供了数学基础 |
| Gu et al., *Efficiently Modeling Long Sequences with Structured State Spaces* (ICLR 2022) | S4：结构化对角加低秩矩阵，$\mathcal{O}(T \log T)$ 计算 | Long Range Arena (LRA) 基准上超越所有现有方法 |
| Smith et al., *Simplified State Space Layers for Sequence Modeling* (ICLR 2023) | S5：简化 SSM 架构，引入并行扫描 | 使 SSM 更易于实现和理解 |
| Fu et al., *Hungry Hungry Hippos* (ICLR 2023) | H3：SSM + 门控，在语言建模上接近 Transformer | 证明了 SSM 在语言建模上的竞争力 |
| Gu & Dao, *Mamba* (2023) | 选择性 SSM + 硬件感知并行扫描 | 首次实现线性时间 SSM 在语言建模上匹配 Transformer |
| Dao & Gu, *Transformers are SSMs* (2024) | Mamba-2 / SSD：统一 SSM 与注意力的矩阵框架 | 理论突破 + 实际 2~8x 加速 |
| AI21 Labs, *Jamba* (2024) | SSM-Transformer 混合架构 | 展示了混合架构在长上下文任务上的实际优势 |

## 6. 优势与局限

### 优势
1. **线性计算复杂度**：训练和推理均为 $\mathcal{O}(T)$ 或 $\mathcal{O}(T \log T)$，适合超长序列（DNA、长视频）
2. **理论优雅**：SSM 与控制论和信号处理有深刻的数学联系，提供了比注意力机制更完备的理论解释
3. **长程记忆能力**：HiPPO 矩阵保证了状态空间对历史信号的最优压缩
4. **推理效率**：逐 token 推理仅需 $\mathcal{O}(1)$ 的状态更新，类似 RNN 但无梯度消失问题
5. **硬件友好**：Mamba-2 的矩阵乘法形式能充分利用 GPU 张量核心

### 局限
1. **较年轻**：SSM 生态（预训练模型、框架支持、优化方案）远不如 Transformer 成熟
2. **上下文学习能力待验证**：SSM 在 in-context learning 等需要"直接从上下文提取模式"的能力上，是否及如何超越注意力机制，仍在研究中
3. **选择性机制的成本**：输入依赖的参数化增加了计算开销，Mamba-2 通过矩阵重构缓解
4. **在部分任务上仍有差距**：在某些短上下文生成任务上，SSM 的生成质量仍低于同规模 Transformer

## 7. 应用场景

| 领域 | 应用 | 模型 |
|:---|:---|:---|
| **长序列语言建模** | 百万级 Token 上下文 | Mamba, Jamba, Mamba-2 |
| **DNA 序列建模** | 基因组分析与突变预测 | HyenaDNA, DNA Mamba |
| **音频处理** | 长音频分类与生成 | S4, Mamba-Audio |
| **时间序列预测** | 长时间跨度预测 | S4, Mamba-TS |
| **视觉** | 图像分类、视频理解 | Vision Mamba (Vim), VMamba |
| **混合架构** | SSM + Attention 交替 | Jamba, Zamba |

## 8. 与其他技术关系

- **与 RNN/LSTM 的关系**：SSM 可被视为一种"具有数学保证的 RNN"——状态空间递推 $h_t = \bar{A} h_{t-1} + \bar{B} x_t$ 在形式上就是线性 RNN。但 SSM 通过精心设计的 $A$ 矩阵（HiPPO）和结构化参数化，避免了 RNN 的梯度消失问题
- **与 Transformer/Attention 的关系**：Mamba-2 的 SSD 框架证明了 SSM 和线性注意力的矩阵等价性，揭示了"注意力不需要 Softmax"的可能性。SSM 是 Transformer 在长序列场景中最有潜力的替代者
- **与卷积的关系**：LTI SSM 可以通过卷积（FFT）高效计算，S4 本质上是一种全局卷积。选择性 SSM 打破了这种等价性
- **与门控机制的关系**：H3 和 Mamba 在 SSM 层外引入了独立的门控分支（SiLU 门控），Gate + Mixer 成为通用设计模式

## 9. 前沿发展

### 9.1 大规模语言模型的 SSM 替代

Mamba-2 和 Jamba 证明了 SSM 可扩展到数十亿参数规模。研究焦点包括：MoE + SSM（MoE-Mamba）、SSM 在 Mixture-of-Depths（MoD）中的应用、SSM 的 Scaling Laws 研究。

### 9.2 统一框架理论

Mamba-2 的 SSD 框架揭示的 SSM-注意力二相性正在催生新的混合架构设计空间。研究探索了如何在统一框架中"调配"注意力和 SSM 的组合比例，以针对不同任务自适应调节。

### 9.3 硬件协同设计

Mamba-2 的矩阵乘法重新表述开启了 SSM 与 GPU 张量核心的深度结合。未来的 SSM 架构设计将越来越多地考虑硬件特性（如 HBM vs SRAM 带宽、张量核心利用率）。

### 9.4 多模态 SSM

从 Vision Mamba 到 Mamba-Audio，SSM 正在被扩展到视觉、音频、视频等多模态领域。线性复杂度使其在高分辨率视频（每帧 = 百万像素 × 数千帧）和长音频（小时级录音）中具有天然优势。

### 9.5 SSM 与 Test-Time Training

TTT（Test-Time Training）等最新工作在推理时对 SSM 的状态进行局部更新，使模型能在测试时适应新的上下文分布，结合了两者的动态适应能力。

## 相关知识

- 前置：[[01_循环神经网络RNN]]、[[00_序列模型_综述]]
- 平级：[[04_Seq2Seq与注意力机制]]
- 延伸：[[../10_大语言模型核心架构/02_注意力机制]]、[[../08_Transformer与注意力机制/05_高效注意力机制]]
