# Transformer 架构详解

## 1. 概述

Transformer 是 2017 年 Vaswani 等人在论文《Attention Is All You Need》中提出的序列建模架构。它完全摒弃了 RNN 的递归计算，仅依赖 [[02_Self-Attention与多头注意力|自注意力机制]] 实现序列中任意位置之间的直接交互。Transformer 由编码器（Encoder）和解码器（Decoder）两部分组成，通过堆叠多层自注意力、前馈网络、残差连接和层归一化构建深层网络。

- **解决的问题**：RNN 的串行计算导致训练无法并行化；长程依赖在梯度消失中退化。Transformer 实现了 $O(1)$ 路径长度的全局依赖建模和完全并行化训练。
- **核心价值**：Transformer 是现代深度学习的通用序列架构——从机器翻译到文本生成，从图像分类到蛋白质结构预测，几乎所有序列建模任务都可以用 Transformer 统一处理。它是 BERT、GPT、LLaMA 等所有现代预训练语言模型的架构基础。

## 2. 发展历史

| 年代 | 里程碑 | 意义 |
|:---|:---|:---|
| 2017 | Transformer（Vaswani et al.） | 自注意力取代 RNN，并行化序列建模 |
| 2018 | BERT（Devlin et al.） | 编码器预训练，掩码语言建模 |
| 2018 | GPT-1（Radford et al.） | 解码器预训练，自回归语言建模 |
| 2019 | T5（Raffel et al.） | 编码器-解码器统一 text-to-text 框架 |
| 2020 | GPT-3（Brown et al.） | 175B 参数，规模化验证 |
| 2020 | ViT（Dosovitskiy et al.） | Transformer 进入计算机视觉 |
| 2022 | PaLM（Chowdhery et al.） | 540B，SwiGLU + MQA |
| 2023 | LLaMA 2 / Mistral | 开源 LLM 标准化 GQA + SwiGLU + RMSNorm |
| 2024 | LLaMA 3 / Qwen 2 / DeepSeek-V3 | 现代 LLM 架构成熟 |

## 3. 核心概念

### 3.1 编码器-解码器架构

原始 Transformer 为编码器-解码器结构，用于机器翻译：

```
输入序列 → [Embedding + 位置编码] → 编码器 ×N → 编码表示
                                                    ↓
输出序列 → [Embedding + 位置编码] → 解码器 ×N → 输出概率
```

- **编码器**：双向自注意力，每个位置看到所有位置，适合理解类任务
- **解码器**：因果自注意力 + 交叉注意力，自回归生成，适合生成类任务
- **交叉注意力**：解码器的 Query 查询编码器的 Key/Value，实现源-目标对齐

### 3.2 三种架构变体

| 架构 | 使用部分 | 注意力类型 | 代表模型 | 适合任务 |
|:---|:---|:---|:---|:---|
| **Encoder-only** | 编码器 | 双向自注意力 | BERT、RoBERTa | 理解类（分类、NER、问答） |
| **Decoder-only** | 解码器 | 因果自注意力 | GPT、LLaMA | 生成类（文本生成、代码） |
| **Encoder-Decoder** | 编码器+解码器 | 自注意力+交叉注意力 | T5、BART | 序列转换（翻译、摘要） |

> Decoder-only 已成为现代 LLM 的主流选择（GPT-4、LLaMA、Qwen、DeepSeek 均为 Decoder-only），因为自回归语言建模天然适合 zero-shot 生成。

### 3.3 子层结构

每个 Transformer 层由两个子层组成（解码器为三个）：

1. **多头注意力子层**：[[02_Self-Attention与多头注意力|多头自注意力]]
2. **前馈网络子层**：逐位置的非线性变换
3. （解码器额外）**交叉注意力子层**：Query 来自解码器，K/V 来自编码器

每个子层都带有残差连接和层归一化：

$$\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x)) \quad \text{(Post-LN)}$$

$$\text{output} = x + \text{SubLayer}(\text{LayerNorm}(x)) \quad \text{(Pre-LN)}$$

## 4. 技术原理

### 4.1 编码器层

编码器层的前向传播：

```
x → Multi-Head Self-Attention → Add & Norm → Feed-Forward → Add & Norm → output
```

$$z = \text{LayerNorm}(x + \text{MHA}(x))$$

$$\text{output} = \text{LayerNorm}(z + \text{FFN}(z))$$

- 自注意力无掩码（双向），每个位置可关注所有位置
- 典型层数：BERT-base 12 层，BERT-large 24 层

### 4.2 解码器层

解码器层的前向传播：

```
x → Masked Multi-Head Self-Attention → Add & Norm
  → Cross-Attention(K,V from encoder) → Add & Norm
  → Feed-Forward → Add & Norm → output
```

$$z_1 = \text{LayerNorm}(x + \text{MaskedMHA}(x))$$

$$z_2 = \text{LayerNorm}(z_1 + \text{CrossAttn}(z_1, enc\_out, enc\_out))$$

$$\text{output} = \text{LayerNorm}(z_2 + \text{FFN}(z_2))$$

- 因果掩码确保位置 $t$ 只关注 $t' \leq t$
- 交叉注意力将编码器输出注入解码器

### 4.3 前馈神经网络（FFN）

FFN 对每个位置独立地进行非线性变换（位置间不交互）：

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

- 输入维度 $d_{model}$，中间维度 $d_{ff}$（通常 $d_{ff} = 4d_{model}$）
- 两层线性变换 + ReLU 激活
- FFN 的作用：注意力子层负责信息路由（哪些位置相关），FFN 负责信息变换（对相关信息做什么计算）

**现代变体**：

| 激活函数 | 公式 | 代表模型 |
|:---|:---|:---|
| ReLU | $\max(0, x)$ | 原始 Transformer |
| GELU | $x \cdot \Phi(x)$ | BERT、GPT-2 |
| SwiGLU | $(\text{Swish}(xW_1) \odot xW_3) W_2$ | LLaMA、PaLM、Qwen |

SwiGLU 引入门控机制和第三个权重矩阵 $W_3$，中间维度通常缩减为 $\frac{2}{3} \times 4d_{model}$ 以保持参数量不变。

### 4.4 残差连接与层归一化

**残差连接**：

$$y = x + \text{SubLayer}(x)$$

- 解决深层网络的梯度消失问题
- 使信息可以"跳过"子层直接传递
- 现代 LLM 中残差连接携带了大部分信息流（LayerNorm 之后的子层输出相对残差流很小）

**层归一化位置**：

| 方案 | 公式 | 特点 |
|:---|:---|:---|
| **Post-LN** | $\text{LayerNorm}(x + \text{SubLayer}(x))$ | 原始 Transformer，训练不稳定，需 warmup |
| **Pre-LN** | $x + \text{SubLayer}(\text{LayerNorm}(x))$ | 现代 LLM 标配，训练稳定，无需 warmup |
| **Sandwich-LN** | $x + \text{SubLayer}(\text{LayerNorm}(x))$ 后再 LN | 某些模型使用，进一步稳定训练 |

**RMSNorm**（Root Mean Square Normalization）：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \cdot \gamma$$

- 去除 LayerNorm 中的均值减法，仅做缩放归一化
- 计算量减少约 7%-64%，性能与 LayerNorm 相当
- 现代 LLM 标配（LLaMA、Qwen、DeepSeek）

### 4.5 输出层

解码器输出经线性投影到词表维度，再经 softmax 得到下一个 token 的概率分布：

$$P(y_t | y_{<t}, \mathbf{x}) = \text{softmax}(W_{out} \cdot h_t + b_{out})$$

其中 $h_t$ 是解码器在位置 $t$ 的输出，$W_{out} \in \mathbb{R}^{d_{model} \times V}$，$V$ 是词表大小。

### 4.6 原始 Transformer 的超参数

原始论文（base / big）的配置：

| 超参数 | Base | Big |
|:---|:---|:---|
| 层数 $N$ | 6 | 6 |
| $d_{model}$ | 512 | 1024 |
| 头数 $h$ | 8 | 16 |
| $d_k = d_v$ | 64 | 64 |
| $d_{ff}$ | 2048 | 4096 |
| 参数量 | ~65M | ~213M |

### 4.7 现代 LLM 的典型架构

以 LLaMA 为代表的现代 Decoder-only LLM 架构：

| 组件 | 原始 Transformer | 现代 LLM（LLaMA 等） |
|:---|:---|:---|
| 架构 | Encoder-Decoder | Decoder-only |
| 归一化 | Post-LayerNorm | Pre-RMSNorm |
| 激活函数 | ReLU | SwiGLU |
| 位置编码 | 正弦编码（绝对） | RoPE（相对） |
| 注意力 | MHA | GQA |
| 偏置项 | 有 | 无（去除 bias） |

## 5. 关键方法/模型

### 5.1 原始 Transformer（2017）

6 层编码器 + 6 层解码器，$d_{model}=512$，8 头注意力。在 WMT 英德/英法翻译上取得 SOTA，且训练速度远超基于 RNN 的模型。

### 5.2 BERT（2018）

Encoder-only，12/24 层。预训练任务：掩码语言建模（MLM）+ 下一句预测（NSP）。双向注意力适合理解类任务。参见 [[09_预训练语言模型]]。

### 5.3 GPT（2018-2020）

Decoder-only，自回归语言建模。从 GPT-1（117M）到 GPT-3（175B），验证了"自回归预训练 + 规模化"的路线。参见 [[09_预训练语言模型]]。

### 5.4 T5（2019）

Encoder-Decoder，将所有 NLP 任务统一为 text-to-text 格式。使用相对位置偏置（T5 Bias）。

### 5.5 ViT（2020）

将图像分割为 $16 \times 16$ 的 patch，展平为序列后输入 Transformer 编码器。开创了视觉 Transformer 方向。参见 [[15_计算机视觉]]。

## 6. 优势与局限

### 优势

1. **并行训练**：序列维度无递归依赖，训练效率远超 RNN
2. **全局建模**：自注意力实现 $O(1)$ 路径长度，长程依赖无退化
3. **统一架构**：同一架构适用于 NLP、CV、语音、多模态等各领域
4. **可扩展性**：参数量从百万到万亿均有效，符合 Scaling Laws
5. **迁移学习**：预训练 + 微调范式在下游任务上表现优异

### 局限

1. **二次复杂度**：$O(n^2)$ 限制长序列应用（需 [[05_高效注意力机制|高效注意力]] 变体解决）
2. **推理串行性**：自回归生成仍需逐 token 计算
3. **位置编码依赖**：自注意力排列不变，需额外位置编码
4. **数据饥饿**：大规模模型需要海量训练数据
5. **能耗问题**：训练和推理的计算成本高昂

## 7. 应用场景

| 领域 | 应用 | 代表模型 |
|:---|:---|:---|
| 机器翻译 | 文本翻译 | 原始 Transformer、Moses+Transformer |
| 文本理解 | 分类、NER、问答 | BERT、RoBERTa |
| 文本生成 | 对话、写作、代码 | GPT、LLaMA、Claude |
| 计算机视觉 | 图像分类、检测 | ViT、DETR、Swin Transformer |
| 语音 | 语音识别、合成 | Whisper、Conformer |
| 多模态 | 图文理解、视频理解 | CLIP、Flamingo、LLaVA |
| 科学 | 蛋白质结构预测 | AlphaFold 2 |
| 代码 | 代码补全、生成 | Codex、Code LLaMA |

## 8. 与其他技术关系

- **与 [[01_注意力机制原理|注意力机制]] 的关系**：Transformer 的核心是自注意力，是注意力机制的直接应用
- **与 [[02_Self-Attention与多头注意力|多头注意力]] 的关系**：多头注意力是 Transformer 的核心子层
- **与 [[04_位置编码|位置编码]] 的关系**：Transformer 需要位置编码注入顺序信息
- **与 [[06_序列模型|序列模型]] 的关系**：Transformer 取代了 RNN/LSTM 在 NLP 领域的主导地位
- **与 [[09_预训练语言模型|预训练语言模型]] 的关系**：BERT、GPT、T5 均基于 Transformer
- **与 [[10_大语言模型核心架构|大语言模型核心架构]] 的关系**：现代 LLM 在 Transformer 基础上引入了 GQA、SwiGLU、RMSNorm 等优化
- **与 [[05_高效注意力机制|高效注意力]] 的关系**：FlashAttention、GQA 等技术优化了 Transformer 的注意力计算

## 9. 前沿发展

- **架构简化**：现代 LLM 趋向 Decoder-only，去除编码器-解码器结构的复杂性
- **归一化演进**：从 Post-LN → Pre-LN → Pre-RMSNorm，训练稳定性持续提升
- **激活函数演进**：ReLU → GELU → SwiGLU，表达力和效率持续优化
- **混合架构**：Jamba（SSM + Attention）、Mamba-2 等探索非纯注意力的架构路径
- **架构搜索**：通过 NAS 或 LLM 辅助搜索更优的 Transformer 变体
- **硬件协同设计**：FlashAttention-3 针对 Hopper 架构优化，未来架构设计越来越依赖硬件特性
