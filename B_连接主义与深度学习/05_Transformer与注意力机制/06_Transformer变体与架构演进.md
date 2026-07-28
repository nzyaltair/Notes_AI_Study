# Transformer 变体与架构演进

## 1. 概述

自 2017 年原始 Transformer 提出以来，围绕不同任务需求和设计理念，衍生出了大量架构变体。最重要的分化是**编码器-only**（BERT 路线）、**解码器-only**（GPT 路线）和**编码器-解码器**（T5 路线）三种范式。此外，Transformer 还扩展到了计算机视觉（ViT）、语音（Conformer）、科学计算（AlphaFold）等领域。本笔记梳理 Transformer 架构变体的设计选择、演进脉络和适用场景。

- **解决的问题**：原始 Transformer 为机器翻译设计，不同任务（理解 vs 生成、NLP vs CV）对架构有不同要求，需要针对性优化。
- **核心价值**：理解架构变体的设计取舍，有助于在不同场景选择合适的模型架构。

## 2. 发展历史

| 年代 | 模型 | 架构 | 核心创新 |
|:---|:---|:---|:---|
| 2017 | Transformer | Encoder-Decoder | 自注意力 |
| 2018 | BERT | Encoder-only | 双向预训练 + MLM |
| 2018 | GPT-1 | Decoder-only | 自回归预训练 |
| 2019 | T5 | Encoder-Decoder | 统一 text-to-text |
| 2019 | XLNet | 混合 | 排列语言建模 |
| 2019 | ALBERT | Encoder-only | 参数共享 + SOP |
| 2019 | Transformer-XL | Decoder+循环 | 相对位置 + 段级循环 |
| 2020 | GPT-3 | Decoder-only | 175B 规模化 |
| 2020 | ViT | Encoder-only | 图像 Patch 序列化 |
| 2020 | DEiT | Encoder-only | ViT 蒸馏优化 |
| 2021 | Swin Transformer | Encoder-only | 层级结构 + 滑动窗口 |
| 2021 | DeBERTa | Encoder-only | 解耦注意力 |
| 2021 | BART | Encoder-Decoder | 去噪自编码器 |
| 2022 | PaLM | Decoder-only | SwiGLU + MQA + 540B |
| 2023 | LLaMA 2 | Decoder-only | GQA + RMSNorm + RoPE |
| 2023 | Mistral | Decoder-only | 滑动窗口 + GQA |
| 2023 | Mixtral | Decoder-only MoE | 8x7B 稀疏 MoE |
| 2024 | LLaMA 3 | Decoder-only | 128K 上下文 |
| 2024 | DeepSeek-V3 | Decoder-only MoE | MLA + 细粒度 MoE |

## 3. 核心概念

### 3.1 三种架构范式

| 范式 | 注意力类型 | 预训练目标 | 优势 | 适合任务 |
|:---|:---|:---|:---|:---|
| **Encoder-only** | 双向自注意力 | 掩码语言建模（MLM） | 全局双向理解 | 分类、NER、问答 |
| **Decoder-only** | 因果自注意力 | 自回归语言建模（LM） | 生成 + zero-shot | 生成、对话、推理 |
| **Encoder-Decoder** | 自注意力 + 交叉注意力 | 去噪/-span corruption | 源-目标对齐 | 翻译、摘要 |

### 3.2 预训练目标

| 目标 | 公式 | 架构 | 代表模型 |
|:---|:---|:---|:---|
| **掩码语言建模（MLM）** | 预测被 `[MASK]` 替换的 token | Encoder | BERT |
| **自回归语言建模（LM）** | $P(y_t \| y_{<t})$ | Decoder | GPT |
| **Span Corruption** | 预测被替换的连续 span | Encoder-Decoder | T5 |
| **去噪自编码（DAE）** | 重建被噪声破坏的文本 | Encoder-Decoder | BART |
| **排列语言建模（PLM）** | 在随机排列上自回归 | 混合 | XLNet |

### 3.3 架构设计维度

Transformer 架构的关键设计选择：

| 维度 | 选项 | 现代趋势 |
|:---|:---|:---|
| 架构类型 | Encoder / Decoder / Enc-Dec | Decoder-only |
| 归一化 | Post-LN / Pre-LN / RMSNorm | Pre-RMSNorm |
| 激活函数 | ReLU / GELU / SwiGLU | SwiGLU |
| 位置编码 | 正弦 / 可学习 / 相对 / RoPE / ALiBi | RoPE |
| 注意力 | MHA / MQA / GQA / MLA | GQA 或 MLA |
| 偏置项 | 有 / 无 | 无 |
| 深度-宽度比 | $d_{model}$ vs 层数 | 趋向更深 |

## 4. 技术原理

### 4.1 BERT：编码器路线

**论文**：Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2018)

**架构**：Transformer 编码器，12 层（base）/ 24 层（large），双向自注意力。

**预训练任务**：
1. **掩码语言建模（MLM）**：随机掩码 15% 的 token，模型预测被掩码的 token
2. **下一句预测（NSP）**：判断两个句子是否相邻（后续研究表明 NSP 效果有限，RoBERTa 去除了 NSP）

**特点**：
- 双向注意力使每个 token 能看到上下文的所有位置
- 适合理解类任务（分类、NER、问答）
- 不适合生成任务（无自回归机制）

**微调范式**：在 `[CLS]` token 的输出上接分类头，或在每个 token 的输出上接标注头。

### 4.2 GPT：解码器路线

**论文**：Radford et al., *Improving Language Understanding by Generative Pre-Training* (2018)

**架构**：Transformer 解码器（去除交叉注意力），因果自注意力。

**预训练任务**：自回归语言建模 $P(x_t | x_{<t})$

**演进**：
- GPT-1（117M）：验证预训练 + 微调范式
- GPT-2（1.5B）：zero-shot 任务迁移
- GPT-3（175B）：few-shot / in-context learning
- GPT-4：多模态 + RLHF

**特点**：
- 因果掩码使训练可并行（所有位置同时计算），推理自回归
- 天然支持生成任务和 zero-shot
- Decoder-only 已成为现代 LLM 的主流架构

### 4.3 T5：编码器-解码器路线

**论文**：Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (2012019)

**架构**：标准 Transformer 编码器-解码器。

**核心思想**：将所有 NLP 任务统一为 text-to-text 格式：
- 翻译：`translate English to German: [文本]` → `[译文]`
- 分类：`classify: [文本]` → `[标签]`
- 摘要：`summarize: [文本]` → `[摘要]`

**预训练任务**：Span Corruption — 随机选择文本中的 span，替换为哨兵 token（如 `<extra_id_0>`），模型预测原始 span。

**特点**：
- 统一框架，一个模型处理所有任务
- 编码器双向理解输入，解码器自回归生成输出
- 适合序列转换任务（翻译、摘要）

### 4.4 ViT：视觉 Transformer

**论文**：Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (2020)

**架构**：Transformer 编码器。

**核心创新**：将图像转换为序列：
1. 将 $H \times W$ 图像分割为 $P \times P$ 的 patch（如 $16 \times 16$）
2. 每个 patch 展平为 $P^2 \times C$ 的向量
3. 线性投影到 $d_{model}$ 维
4. 添加可学习位置嵌入
5. 添加 `[CLS]` token 用于分类

**关键发现**：
- 在中等数据集（ImageNet-21k）上，ViT 不如 CNN
- 在大规模数据集（JFT-300M）上，ViT 超越 CNN
- Transformer 在视觉领域也需要规模化才有效

### 4.5 Swin Transformer

**论文**：Liu et al., *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows* (2021)

**创新**：
- **层级结构**：类似 CNN 的多尺度设计，通过 patch merging 逐层降低分辨率
- **移位窗口注意力**：在局部窗口内计算注意力（$O(n)$），通过窗口移位实现跨窗口信息交流
- 兼顾了 Transformer 的全局建模和 CNN 的局部效率

### 4.6 现代 LLM 架构（LLaMA 范式）

以 LLaMA 为代表的现代 Decoder-only LLM 的标准架构：

```
Token → Embedding (无 bias)
  → RoPE 位置编码
  → Transformer Block × N:
      Pre-RMSNorm → GQA (因果) → 残差
      Pre-RMSNorm → SwiGLU FFN → 残差
  → RMSNorm
  → Linear (无 bias) → softmax → 词表分布
```

| 设计选择 | LLaMA 2 | LLaMA 3 | Qwen 2 | DeepSeek-V3 |
|:---|:---|:---|:---|:---|
| 架构 | Decoder-only | Decoder-only | Decoder-only | Decoder-only MoE |
| 注意力 | GQA | GQA | GQA | MLA |
| 归一化 | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm |
| 激活 | SwiGLU | SwiGLU | SwiGLU | SwiGLU |
| 位置编码 | RoPE | RoPE | RoPE | RoPE |
| 训练长度 | 4K | 8K | 32K | 4K |
| 最大长度 | 4K | 128K | 128K | 128K |
| 外推方法 | - | YaRN | NTK-aware | YaRN |

### 4.7 MoE Transformer

将 FFN 层替换为多个专家（Expert）网络，通过路由器（Router）选择 Top-K 专家：

$$y = \sum_{i \in \text{TopK}} \text{softmax}(\text{TopK}(W_g x, k))_i \cdot E_i(x)$$

- **稀疏激活**：总参数量大但每次推理只激活部分参数
- **代表模型**：Mixtral 8x7B、DeepSeek-V3（671B 总参数，37B 激活）
- 详见 [[10_大语言模型核心架构/06_混合专家模型]]

## 5. 关键方法/模型

### 5.1 编码器路线演进

```
BERT (2018) → RoBERTa (2019, 去NSP+更多数据) → ALBERT (2019, 参数共享)
    → DeBERTa (2021, 解耦注意力) → RoBERTa-2 / 现代编码器
```

### 5.2 解码器路线演进

```
GPT-1 (2018) → GPT-2 (2019) → GPT-3 (2020) → PaLM (2022)
    → LLaMA (2023) → LLaMA 2/3 (2023-2024) → DeepSeek-V3 (2024)
```

### 5.3 编码器-解码器路线演进

```
Transformer (2017) → MASS (2019) → T5 (2019) → BART (2020)
    → mT5 (2021) → Flan-T5 (2022)
```

### 5.4 视觉 Transformer 演进

```
ViT (2020) → DEiT (2021) → Swin Transformer (2021)
    → CSWin Transformer (2022) → MaxViT (2022)
```

## 6. 优势与局限

### 优势

1. **架构统一性**：同一 Transformer 架构适用于 NLP、CV、语音等多领域
2. **可扩展性**：参数量从百万到万亿均有效
3. **迁移学习**：预训练 + 微调 / in-context learning 范式高效
4. **生态成熟**：丰富的预训练模型和工具链

### 局限

1. **Decoder-only 的局限**：双向理解能力不如 Encoder-only（但规模化后差距缩小）
2. **计算成本**：大模型训练和推理成本高昂
3. **数据依赖**：需要海量高质量训练数据
4. **CV 领域的局限**：ViT 对数据量要求高，小数据集上不如 CNN

## 7. 应用场景

| 任务类型 | 推荐架构 | 代表模型 |
|:---|:---|:---|
| 文本分类 / NER / 问答 | Encoder-only | BERT、RoBERTa |
| 文本生成 / 对话 / 代码 | Decoder-only | GPT、LLaMA |
| 机器翻译 / 摘要 | Encoder-Decoder | T5、BART |
| 图像分类 | Encoder-only (ViT) | ViT、Swin |
| 目标检测 | Encoder + 检测头 | DETR |
| 语音识别 | Encoder-Decoder | Whisper |
| 多模态 | 各类混合 | CLIP、LLaVA |
| 蛋白质结构 | 自定义 | AlphaFold 2 |

## 8. 与其他技术关系

- **与 [[03_Transformer架构详解|Transformer 架构]] 的关系**：本笔记是架构在不同方向的变体应用
- **与 [[09_预训练语言模型|预训练语言模型]] 的关系**：BERT、GPT、T5 的预训练策略详见该方向
- **与 [[10_大语言模型核心架构|大语言模型核心架构]] 的关系**：现代 LLM 的架构设计选择详见该方向
- **与 [[15_计算机视觉|计算机视觉]] 的关系**：ViT、Swin、DETR 详见该方向
- **与 [[14_多模态AI|多模态 AI]] 的关系**：CLIP、Flamingo 等多模态 Transformer 详见该方向
- **与 [[05_高效注意力机制|高效注意力]] 的关系**：架构变体中的注意力优化方案

## 9. 前沿发展

- **Decoder-only 统一**：Encoder-only 和 Encoder-Decoder 路线逐渐被 Decoder-only 取代，现代 LLM 几乎全部采用 Decoder-only
- **MoE 普及**：DeepSeek-V3、Mixtral 等验证了 MoE 在大规模 LLM 中的可行性
- **混合架构**：Jamba（SSM + Attention）、Mamba-2 等探索非纯 Transformer 架构
- **多模态统一**：单一 Transformer 架构处理文本、图像、音频、视频（如 Gemini）
- **架构简化**：去除不必要的组件（bias、NSP 等），追求极致简洁
- **LLM 辅助架构设计**：使用 LLM 搜索更优的 Transformer 变体
