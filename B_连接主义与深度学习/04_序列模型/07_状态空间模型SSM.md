# 状态空间模型（SSM）与 Mamba

## 1. 概述

状态空间模型（State Space Model, SSM）源于控制论和信号处理中的连续时间线性系统理论，在深度学习中被重新发掘为一种高效的序列建模框架。SSM 通过一阶微分（或差分）方程描述系统的内部状态如何随输入和当前状态演化，以**线性复杂度**处理序列数据，成为近年来挑战 Transformer 注意力机制的最有潜力方向之一。Mamba 是 SSM 家族中最重要的里程碑——它引入**选择性机制**，使 SSM 首次在语言建模质量上匹配 Transformer，因此本文将 S4（原型 SSM）与 Mamba（选择性 SSM）作为同一技术脉络统一讲解，而非割裂为两篇笔记。

- **解决的问题**：Transformer 自注意力的 $\mathcal{O}(T^2)$ 计算复杂度在长序列（如 DNA 序列百万级、高分辨率视频）上成为瓶颈。SSM 以 $\mathcal{O}(T)$ 或 $\mathcal{O}(T \log T)$ 的复杂度实现长程依赖建模。
- **核心模型谱系**：S4（Structured State Space for Sequences，线性时不变）→ Mamba（Selective SSM，输入依赖）→ Mamba-2（SSD 框架，统一 SSM 与线性注意力）。
- **关键意义**：SSM 展现了"不依赖注意力也能有效建模长序列"的可能性，是后 Transformer 时代最重要的架构探索方向之一，其优势依赖具体任务、硬件和训练配置，不应被视为 Transformer 的普遍替代。

## 2. 发展历史

| 年代 | 里程碑 | 核心贡献 |
|:---|:---|:---|
| 1960s | Kalman 滤波器 | 现代状态空间模型和最优估计理论的起源 |
| 2020 | HiPPO 矩阵理论 (Gu et al.) | 从函数逼近的角度推导出最优状态转移矩阵 $A$，为 SSM 在深度学习中的应用奠定了数学基础 |
| 2021 | LSSL (Gu et al.) | 首次将 HiPPO 线性状态空间层应用于序列建模，展示了 SSM 处理长序列的潜力 |
| 2022 | S4 (Gu et al.) | 引入结构化对角加低秩矩阵，使 SSM 的计算和存储从 $\mathcal{O}(T^2)$ 降至 $\mathcal{O}(T \log T)$，在 Long Range Arena 基准上超越所有现有方法 |
| 2023 | S5 / H3 / Hyena | S5 提出并行扫描；H3（Hungry Hungry Hippos）将 SSM 与门控结合；Hyena 用长卷积替代注意力 |
| 2023.12 | **Mamba** (Gu & Dao) | 引入**选择性 SSM**：使 $B, C, \Delta$ 依赖于输入 $x$，打破了 LTI 假设。结合硬件感知的并行扫描算法，实现了线性时间的训练和推理，首次使 SSM 在语言建模上匹配 Transformer |
| 2024.05 | **Mamba-2 / SSD** (Dao & Gu) | 提出 SSD（State Space Duality）框架，统一了结构化 SSM 和线性注意力的矩阵变换视角。通过矩阵乘法的结构化分解，相比 Mamba 提速 2~8 倍 |
| 2024.06 | Gated DeltaNet (Yang et al.) | 在线性注意力递归上加双重门控（遗忘/输入）与 DeltaRule 投影更新，表达力更强 |
| 2024 | Jamba (AI21 Labs) | SSM + Transformer 混合架构，交替使用 Mamba 层和注意力层，在长上下文任务中平衡效率和质量 |
| 2024 | Mamba 扩展到视觉和基因组 | Vision Mamba (Vim)、VMamba、DNA Mamba 等，将 SSM 应用到多模态和科学计算领域 |
| 2025 | **混合架构进入大规模前沿模型** | MiniMax-M1（7:1 线性注意力）、Qwen3-Next（3:1 Gated DeltaNet）、Nemotron 3（Mamba-2 轻量层）等开源模型验证了线性时间层与注意力混合的可行性 |

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

### 3.4 选择性机制（Selection Mechanism）：从 S4 到 Mamba 的关键跃迁

传统 SSM（S4）是**线性时不变（LTI）** 系统——参数 $A, B, C, \Delta$ 对所有输入固定不变。Mamba 的关键突破是**选择性 SSM**：

$$B_t = s_B(x_t), \quad C_t = s_C(x_t), \quad \Delta_t = \tau_{\Delta}(\text{Parameter} + s_{\Delta}(x_t))$$

其中 $s_B, s_C, s_{\Delta}$ 是将输入 $x_t$ 映射到参数空间的小型线性投影。

- **意义**：使模型能够根据当前输入内容，动态决定"忽略什么"和"记住什么"——这与 LSTM 的门控机制精神一致，但通过完全不同的数学路径实现
- **代价**：LTI 系统的卷积计算优势消失，需要新的并行计算方法（见 3.5）

### 3.5 并行扫描（Parallel Scan）

选择性 SSM 不再可以通过 FFT 卷积计算（因为 $B_t, C_t$ 依赖于 $x_t$）。并行扫描是解决此问题的关键算法：

给定序列操作 $x_1 \oplus x_2 \oplus \cdots \oplus x_T$，并行扫描在前缀和（Prefix Sum）的框架下高效计算所有中间状态。对于 SSM 的逐时间步递推，可将 $h_t$ 的计算转化为高度并行的前缀扫描操作，在 GPU 上实现接近 $\mathcal{O}(\log T)$ 的并行度。这是 Mamba 能够以线性复杂度训练的工程关键。

### 3.6 从线性注意力视角理解现代 SSM：门控设计法则

Mamba-2、Gated DeltaNet 等现代 SSM 虽然源自状态空间理论推导，但机制上可以视为**线性注意力的递归形式加上门控扩展**（CS336 Lecture 4 的统一叙事）：

1. **线性注意力**：去掉 softmax 后利用结合律，$\phi(K)^TV$ 成为固定大小状态 $S$，可增量更新——得到「训练用密集矩阵乘法、推理用递归」的并行/递归二重性（见 [[../08_Transformer与注意力机制/05_高效注意力机制|高效注意力机制]] §4.4）
2. **Mamba-2**：在线性注意力递归上加**输入门控 $\gamma_t$**（只依赖当前输入、不依赖状态），控制多少历史状态传递到未来——解决线性注意力"信息只能一直累加、不能遗忘"的问题
3. **Gated DeltaNet**：再加第二个门控 $\beta_t$（遗忘/写入门），并用 DeltaRule 投影替换简单的累加写入（见 4.5 节）

> **门控设计法则**：只要递归结构中新增的门控项**只依赖输入、不依赖状态**，并行训练与串行推理的二重性就得以保持。这是从 LSTM 时代"知道何时遗忘"的经验到可并行化架构的桥梁——大量被验证有效的线性时间方法最终都收敛为这种「类 LSTM」的简单递归形式。

## 4. 技术原理

### 4.1 S4 的卷积模式（LTI 系统）

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

### 4.3 Mamba-2 的 SSD 框架：SSM 与注意力的统一

SSD（State Space Duality）的核心理念：SSM 和线性注意力是**同一类矩阵变换**的两种视图：

- **SSM 视图**：递推计算 $y = \text{SSM}(A, B, C)(x)$
- **注意力视图**：$Y = (L \circ (Q K^T)) \cdot V$，其中 $L$ 是下三角掩码矩阵，$Q, K, V$ 与 $B(x), C(x), x$ 相关

Mamba-2 将选择性 SSM 重新表述为结构化矩阵乘法，使其能利用 GPU 张量核心（Tensor Cores）的矩阵乘法硬件加速。相比 Mamba-1 在训练时可提速 2~8 倍，同时保持推理时的线性复杂度。这一结果揭示了"注意力不需要 Softmax"的可能性，是 SSM 与 Transformer 两条技术路线在理论上的重要会师点。

从线性注意力视角看（见 3.6 节），SSD 揭示的正是并行/递归二重性：训练时用稠密（分块）矩阵乘法形式，推理时切换到递归形式获得固定状态——历史上 SSM/LSTM 相对注意力的最大劣势（无法高效并行训练）由此被消除，剩下的才是纯粹的表达能力权衡。

### 4.4 Gated DeltaNet：门控增量更新

Gated DeltaNet（Yang et al., 2024）是目前应用最广的"现代 SSM/线性注意力"层之一，在 Mamba-2 风格的状态更新上引入两个改动：

**1. 双重门控**：
- $\gamma_t$（衰减门，即 Mamba-2 的门控）：控制历史状态向未来传递多少
- $\beta_t$（输入门）：控制当前时间步写入多少信息——$\beta_t = 0$ 意味着当前 token 完全不写入状态（让人想起 LSTM 的输入门/遗忘门）

**2. DeltaRule 投影更新**（继承自 DeltaNet）：写入新信息的同时，**擦除状态中当前 key 方向上已有的内容**：

$$S_t = \gamma_t \, S_{t-1} \, (I - \beta_t k_t k_t^\top) + \beta_t\, v_t k_t^\top$$

直观理解：为当前 key $k_t$ 写入信息前，先用 $(I - \beta_t k_t k_t^\top)$ 把状态里与 $k_t$ 同方向的旧分量投影出去——"先删除同 key 旧内容，再写入新内容"，而非简单累加（未做单位归一化，因此只是近似的投影器）。这解决了纯累加写入的冗余问题，等价于在状态空间中求解**在线最小二乘**问题。

**有意思的再发明史**：同样的投影更新在快速权重编程（fast weight programmers）和测试时训练（TTT）领域基于完全不同的设计原则被独立得出（见 9.5 节）。

**大规模验证**：Qwen3-Next 及其后续的 Qwen3.5 采用 3:1（Gated DeltaNet : 完整注意力）混合架构，解码吞吐量远高于 Qwen3，且性能几乎无损——是目前最优秀的开源模型之一。

### 4.5 时间复杂度对比

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
| Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2023) | 选择性 SSM + 硬件感知并行扫描 | 首次实现线性时间 SSM 在语言建模上匹配 Transformer |
| Yang et al., *Gated Delta Networks* (2024) | 双重门控 + DeltaRule 投影更新 | 被 Qwen3-Next 等大规模混合架构采用 |
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
3. **状态容量与上下文长度的权衡（没有免费午餐）**：固定大小状态要压缩整个上下文，状态规模与上下文长度之比决定信息损失——想要极小的状态就很难把大上下文的所有信息都压进去；状态做到与上下文一样大虽然可行，但计算开销也会随之上升。softmax 注意力的 all-to-all 连接则在表示能力上依然更强、更好训练
4. **在部分任务上仍有差距**：在某些短上下文生成任务上，SSM 的生成质量仍低于同规模 Transformer；长上下文精确检索（如 needle-in-a-haystack）对固定状态架构尤其困难
5. **不应视为普遍替代**：SSM 相对 Transformer 的优势高度依赖任务类型（长序列 vs 短序列）、硬件（是否有张量核心加速）和训练配置

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

### 9.1 大规模语言模型的 SSM 替代：混合架构落地

Mamba-2 和 Jamba 证明了 SSM 可扩展到数十亿参数规模。2025 年后混合架构进入开源前沿模型：MiniMax-M1（7:1 线性注意力混合）、Qwen3-Next/Qwen3.5（3:1 Gated DeltaNet 混合）、Nemotron 3（Mamba-2 轻量层与 softmax 注意力交替，性能与 Qwen3-Thinking、GPT-OSS 同档）。但**完全不含 softmax 注意力的纯线性架构尚未有人在大规模上跑通**。

字节 Seed 与 UC Santa Cruz 的对照研究给出了混合比例的经验结论：低比例替换 RNN 层几乎无性能损失，超过某个临界点后长上下文性能明显下降，全部替换为 RNN 层时退化显著——单键检索与 QA 等任务均呈现稳定下降趋势。研究焦点还包括：MoE + SSM（MoE-Mamba）、SSM 在 Mixture-of-Depths（MoD）中的应用、SSM 的 Scaling Laws 研究。

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
- 平级：[[04_Seq2Seq与交叉注意力]]
- 延伸：[[../08_Transformer与注意力机制/05_高效注意力机制|高效注意力机制]]（SSM 与线性注意力的等价性）、[[08_长上下文与外部记忆|长上下文与外部记忆]]

## References

- Gu, Goel & Ré, *Efficiently Modeling Long Sequences with Structured State Spaces* (S4), ICLR 2022
- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023
- Dao & Gu, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2), ICML 2024
