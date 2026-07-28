# Transformer与注意力机制综述

## 1. 概述

Transformer 是 2017 年由 Vaswani 等人在论文《Attention Is All You Need》中提出的序列建模架构。它完全摒弃了 RNN 的递归计算，仅依赖**自注意力机制（Self-Attention）** 实现序列中任意两个位置之间的直接交互，从而支持完全并行化训练。Transformer 不仅在机器翻译上取得了当时最优结果，更成为此后自然语言处理、计算机视觉、语音和多模态 AI 的统一基础架构，催生了 BERT、GPT、ViT、AlphaFold 2 等里程碑模型。

- **解决的问题**：RNN/LSTM 的串行计算导致训练无法并行化；长程依赖在梯度消失中退化；固定长度上下文向量造成信息瓶颈。Transformer 通过自注意力实现 $O(1)$ 路径长度的全局依赖建模，同时支持序列维度的全并行训练。
- **核心价值**：Transformer 是现代深度学习的"通用序列架构"——从 NLP 到 CV，从蛋白质结构预测到代码生成，几乎一切序列建模任务都可以用 Transformer 统一处理。理解 Transformer 是理解 BERT/GPT 等预训练语言模型和所有现代大语言模型的前提。
- **知识体系范围**：注意力机制原理（从 Bahdanau 到 Self-Attention）、缩放点积注意力与多头注意力、Transformer 编码器-解码器架构、位置编码、高效注意力变体（FlashAttention / 稀疏 / 线性 / GQA）、Transformer 架构变体（BERT / GPT / T5 / ViT）、工程实践、前沿发展。

## 2. 发展历史

Transformer 的发展可划分为"注意力机制诞生 → Transformer 革命 → 预训练时代 → 效率优化时代 → 架构多元化"五个阶段。

| 年代 | 里程碑 | 作者 | 核心意义 |
|:---|:---|:---|:---|
| 2014 | Bahdanau 注意力 | Bahdanau, Cho & Bengio | 在 Seq2Seq 中引入加性注意力，打破固定上下文瓶颈——注意力机制的开端 |
| 2015 | Luong 注意力 | Luong, Pham & Manning | 提出乘性注意力（dot/general/concat），丰富注意力计算方式 |
| 2017 | **Transformer** | Vaswani et al. | 自注意力全面取代 RNN，实现并行化序列建模——本方向的起点 |
| 2018 | BERT | Devlin et al. | 双向编码器预训练，11 项 NLP 基准刷新 |
| 2018 | GPT-1 | Radford et al. | 自回归解码器预训练，开启生成式预训练路线 |
| 2019 | T5 | Raffel et al. | 编码器-解码器统一框架，将所有任务转化为 text-to-text |
| 2019 | Transformer-XL | Dai et al. | 相对位置编码 + 段级循环，支持长序列建模 |
| 2020 | ViT | Dosovitskiy et al. | 将 Transformer 引入计算机视觉，图像分块序列化 |
| 2020 | Reformer | Kitaev et al. | LSH 注意力 + 可逆层，将复杂度降至 $O(n \log n)$ |
| 2021 | RoPE | Su et al. | 旋转位置编码，通过旋转矩阵实现相对位置编码 |
| 2021 | ALiBi | Press et al. | 线性偏置位置编码，原生支持长度外推 |
| 2021 | Linformer | Wang et al. | 低秩近似 K/V 矩阵，线性复杂度 |
| 2022 | FlashAttention | Dao et al. | IO 感知的分块注意力计算，2-4 倍加速 |
| 2022 | FlashAttention-2 | Dao | 进一步优化并行度和 warp 级效率 |
| 2023 | GQA | Ainslie et al. | 分组查询注意力，平衡 MHA 质量与 MQA 效率 |
| 2023 | RetNet | Sun et al. | Retention 机制，训练并行 + 推理线性 |
| 2023 | Mamba | Gu & Dao | 选择性 SSM，线性时间的序列建模 |
| 2024 | FlashAttention-3 | Shah et al. | Hopper 架构优化，FP8 支持，1.5-2 倍加速 |
| 2024 | MLA | DeepSeek-AI | 多头潜在注意力，低秩压缩 KV-Cache |
| 2024 | Jamba | AI21 Labs | SSM + 注意力混合架构 |
| 2024 | Mamba-2 | Dao & Gu | 统一 SSM 与线性注意力的理论框架 |

## 3. 核心概念

### 3.1 注意力机制（Attention Mechanism）

注意力机制是一种**软寻址（Soft Addressing）**过程：给定查询（Query）、键（Key）和值（Value），通过计算 Query 与 Key 的相似度得到注意力权重，再对 Value 加权求和。这使模型能够动态地关注输入中最相关的部分。参见 [[01_注意力机制原理]]。

### 3.2 自注意力（Self-Attention）

自注意力是注意力的特例：Query、Key、Value 均来自同一输入序列。序列中每个位置直接与所有位置交互，捕捉全局依赖关系。自注意力是 Transformer 的核心创新。参见 [[02_Self-Attention与多头注意力]]。

### 3.3 多头注意力（Multi-Head Attention）

将 QKV 投影到 $h$ 个不同的子空间分别计算注意力后拼接，使模型能同时关注不同的表示子空间（如语法关系、语义关系、共指关系等）。

### 3.4 位置编码（Position Encoding）

自注意力本身是排列不变的（Permutation Invariant），无法区分词序。位置编码通过向输入注入位置信息，使模型感知序列顺序。从正弦编码到 RoPE 到 ALiBi，位置编码经历了从绝对到相对、从固定到可外推的演进。参见 [[04_位置编码]]。

### 3.5 编码器-解码器架构

Transformer 由编码器和解码器组成。编码器使用双向自注意力对输入进行全局编码；解码器使用因果自注意力（防止信息泄露）和交叉注意力（关注编码器输出）自回归地生成输出。BERT 使用纯编码器，GPT 使用纯解码器，T5 使用编码器-解码器。参见 [[03_Transformer架构详解]]。

### 3.6 高效注意力（Efficient Attention）

标准注意力的 $O(n^2)$ 复杂度限制了长序列应用。高效注意力变体通过 IO 优化（FlashAttention）、稀疏化（Sparse Attention）、低秩近似（Linformer）、核分解（线性注意力）和 KV 共享（GQA/MQA/MLA）等策略降低计算和内存开销。参见 [[05_高效注意力机制]]。

## 4. 核心公式速览

### 缩放点积注意力

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- 缩放因子 $\sqrt{d_k}$：当 $Q, K$ 元素方差为 1 时，$QK^T$ 元素方差为 $d_k$，除以 $\sqrt{d_k}$ 使方差归一化为 1，防止 softmax 饱和。
- 时间复杂度 $O(n^2 d)$，空间复杂度 $O(n^2)$。

### 多头注意力

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

总参数量 $= 4d^2$，与单头注意力相同。

### RoPE（旋转位置编码）

$$q_m' = R_m q_m, \quad k_n' = R_n k_n$$

$$q_m'^T k_n' = q_m^T R_m^T R_n k_n = q_m^T R_{n-m} k_n$$

注意力分数仅依赖相对位置 $m - n$。

## 5. 代表论文与里程碑

| 论文 | 作者 | 年份 | 核心贡献 |
|:---|:---|:---|:---|
| *Attention Is All You Need* | Vaswani et al. | 2017 | 提出 Transformer 架构，自注意力取代 RNN |
| *BERT* | Devlin et al. | 2018 | 双向编码器预训练，掩码语言建模 |
| *Improving Language Understanding by Generative Pre-Training* | Radford et al. | 2018 | GPT-1，自回归预训练 |
| *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* | Raffel et al. | 2019 | T5，统一 text-to-text 框架 |
| *An Image is Worth 16x16 Words* | Dosovitskiy et al. | 2020 | ViT，视觉 Transformer |
| *RoFormer: Enhanced Transformer with Rotary Position Embedding* | Su et al. | 2021 | RoPE 旋转位置编码 |
| *Train Short, Test Long: Attention with Linear Biases* | Press et al. | 2021 | ALiBi 长度外推 |
| *FlashAttention: Fast and Memory-Efficient Exact Attention* | Dao et al. | 2022 | IO 感知的精确注意力加速 |
| *GQA: Training Generalized Multi-Query Transformer Models* | Ainslie et al. | 2023 | 分组查询注意力 |
| *DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model* | DeepSeek-AI | 2024 | MLA 多头潜在注意力 |

## 6. 与其他方向关系

- **与序列模型的关系**：Transformer 直接取代了 [[06_序列模型|RNN/LSTM]] 在 NLP 领域的主导地位。Seq2Seq 的注意力机制是 Transformer 自注意力的直接前身。参见 [[06_序列模型/04_Seq2Seq与注意力机制]]。
- **与预训练语言模型的关系**：BERT、GPT、T5 等预训练模型均基于 Transformer 架构。参见 [[09_预训练语言模型]]。
- **与大语言模型核心架构的关系**：现代 LLM（LLaMA、Qwen、DeepSeek）在 Transformer 基础上引入了 SwiGLU、RMSNorm、GQA、MLA 等优化。参见 [[10_大语言模型核心架构]]。
- **与大模型推理优化的关系**：注意力的 $O(n^2)$ 复杂度和 KV-Cache 是推理优化的核心问题。参见 [[12_大模型推理与优化]]。
- **与计算机视觉的关系**：ViT 将图像分块序列化后用 Transformer 处理，开创了视觉 Transformer 方向。参见 [[15_计算机视觉]]。
- **与多模态 AI 的关系**：CLIP、Flamingo 等多模态模型使用交叉注意力连接视觉和语言模态。参见 [[14_多模态AI]]。

## 笔记导航

- [[01_注意力机制原理]] — 注意力机制的本质、QKV 框架、从 Bahdanau 到 Self-Attention
- [[02_Self-Attention与多头注意力]] — 缩放点积注意力、多头机制、因果掩码
- [[03_Transformer架构详解]] — 编码器-解码器、FFN、残差连接、层归一化
- [[04_位置编码]] — 正弦编码、可学习编码、相对位置编码、RoPE、ALiBi、长度外推
- [[05_高效注意力机制]] — FlashAttention、稀疏注意力、线性注意力、GQA/MQA/MLA
- [[06_Transformer变体与架构演进]] — BERT、GPT、T5、ViT 及架构设计选择
- [[07_Transformer工程实践]] — PyTorch 实现、训练技巧、框架与工具
- [[08_Transformer前沿发展]] — 混合架构、SSM、线性 RNN、架构搜索

## 学习资源

### 经典论文
- Vaswani et al., *Attention Is All You Need* (2017) — 必读原论文
- Dosovitskiy et al., *An Image is Worth 16x16 Words* (2020) — ViT
- Dao et al., *FlashAttention* (2022) — 高效注意力

### 教程与博客
- The Annotated Transformer（Harvard NLP）— 逐行注释的 PyTorch 实现
- Jay Alammar: *The Illustrated Transformer* — 最经典的可视化教程
- Andrej Karpathy: *Let's build GPT from scratch* — 从零构建 GPT
- Lilian Weng: *Transformer Family* — Transformer 变体综述

### 实践任务
1. 从零实现 Self-Attention 和 Multi-Head Attention
2. 实现并对比不同位置编码（正弦 / RoPE / ALiBi）的外推性能
3. 用 PyTorch 实现完整的 Transformer 编码器-解码器
4. 在 WMT 机器翻译数据上训练一个小型 Transformer
5. 实现 Grouped-Query Attention 并测量推理加速
6. 分析注意力矩阵的秩与信息瓶颈

## References

- Vaswani et al., *Attention Is All You Need* (2017) — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) — Transformer 原始论文
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2019) — [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (2021) — [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) — ViT
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022) — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) — FlashAttention
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023) — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) — GQA
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021) — [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) — RoPE
- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2023) — [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) — Mamba
- Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (2020) — [arXiv:1910.10683](https://arxiv.org/abs/1910.10683) — T5
