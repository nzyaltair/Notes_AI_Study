# 向量表示与 Embedding（LLM 架构视角）

## 一句话理解

在大语言模型架构中，Embedding 是将离散 token id 映射为连续向量的查找表层（$E \in \mathbb{R}^{V \times d}$），是模型唯一的输入接口；其设计要点是词表划分、维度选择、缩放与权重 tying。

## 概述

本笔记聚焦**大语言模型架构中的 token embedding 层**这一组件。广义的向量表示（Word2Vec / GloVe 的词向量历史、SimCSE / BGE 等检索嵌入模型、RAG 向量召回）不属于 LLM 核心架构，见：

- 词向量与早期表示学习（Word2Vec / GloVe / FastText / ELMo）→ [[01_词向量与早期表示学习]]（09_预训练语言模型 方向）
- 检索嵌入模型与语义召回（SimCSE / E5 / BGE / Matryoshka / Late Chunking）→ [[19_LLM应用工程/03_RAG系统]]（19_LLM应用工程 方向）

## 核心概念

### Token Embedding 层

LLM 的输入是 token id 序列，embedding 层是一个可训练的查找表：

$$E \in \mathbb{R}^{V \times d}, \quad x_t = E[\text{token}_t]$$

其中 $V$ 是词表大小，$d$ 是模型维度。embedding 层是模型唯一的离散→连续接口，之后所有计算都在连续空间进行。

### Embedding 与位置编码的组合

LLM 的输入表示 = token embedding + position encoding：

$$h_0 = \text{Embed}(\text{token}) + \text{PE}(\text{pos})$$

- 原始 Transformer / GPT-2：直接相加
- 现代LLM（LLaMA / Qwen / DeepSeek）：使用 RoPE，**不与 embedding 相加**，而是在注意力计算中旋转注入（见 [[03_位置编码]]）

### Embedding 缩放

原始 Transformer 对 embedding 乘以 $\sqrt{d_{model}}$ 缩放，使 embedding 与位置编码量纲匹配：

$$h_0 = \text{Embed}(\text{token}) \cdot \sqrt{d_{model}} + \text{PE}(\text{pos})$$

现代 LLM 多不再使用此缩放（因改用 RoPE 且有 RMSNorm 稳定数值），但 GPT-2 等仍保留。

## 技术细节

### 权重 Tying（Embedding Sharing）

将**输入 embedding 矩阵**与**输出投影矩阵**（lm_head）共享同一参数：

$$\text{logits} = h \cdot E^T$$

- **优势**：大幅减少参数量（$V \times d$ 在小模型中占比可观）
- **代价**：输入表示与输出表示被强制同一空间，可能损失少量质量
- **实践**：GPT-2 使用 tying；LLaMA / Qwen 等现代 LLM 通常**不 tying**（untied），输入输出分别学习，质量更优

### 词表大小与维度选择

| 模型 | 词表大小 $V$ | 模型维度 $d$ | 备注 |
|:---|:---|:---|:---|
| GPT-2 | 50,257 | 768-1600 | BPE |
| LLaMA 2 | 32,000 | 4096-8192 | SentencePiece BPE |
| LLaMA 3 | 128,256 | 4096-16384 | 扩大词表以支持多语言与代码 |
| Qwen 2 | 151,646 | 3584-8192 | 大词表覆盖多语言 |
| DeepSeek-V3 | 129,280 | 7168 | tiktoken 风格 |

要点：

- **词表越大**：单 token 信息密度高、序列更短，但 embedding 参数量 $V \times d$ 增大（LLaMA 3 的 128K 词表 × 4096 ≈ 5亿参数）。
- **词表越小**：序列更长、embedding 参数少，但压缩率高、对分词器质量敏感。
- 现代趋势是扩大词表（100K+）以更好支持多语言、代码与 Unicode。

### Embedding 与归一化

现代 LLM 在 embedding 之后立即接 RMSNorm（而非 LayerNorm）：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$$

RMSNorm 去掉 LayerNorm 的减均值操作，计算更高效，LLaMA / Qwen / DeepSeek 均采用。见 [[10_大语言模型核心架构/01_LLM整体架构]]。

### 上下文嵌入（Contextual Embedding）

LLM 每一层的隐藏状态都是"上下文感知的 token 嵌入"——同一 token 在不同上下文中得到不同向量。这是 LLM 相对静态词向量（Word2Vec）的根本优势。从 LLM 隐藏层提取嵌入用于下游任务（检索、分类）的方法见 [[19_LLM应用工程/03_RAG系统]]。

## 应用场景

- **LLM 输入接口**：所有 token 先经 embedding 层进入模型
- **迁移学习**：冻结 embedding 层微调上层（参数高效）
- **多模态扩展**：视觉 token / 音频 token 通过各自 embedding 层对齐到语言模型维度（见 [[00_多模态AI_综述]]）

## 优势与局限

### 优势

- **简洁**：单一查找表完成离散→连续映射
- **可学习**：端到端学习 token 的最优表示

### 局限

- **参数量大**：大词表（128K）下 embedding 参数可达数亿，对小模型占比显著
- **OOV 依赖分词器**：embedding 质量受分词器（BPE/SentencePiece）影响
- **静态性**：同一 token 的 embedding 在所有上下文相同，依赖后续层注入上下文

## 常见问题

- **要不要做权重 tying？** 小模型为省参数常 tying（GPT-2）；现代大模型多 untied，输入输出分别学习质量更优（LLaMA/Qwen/DeepSeek）。
- **词表越大越好吗？** 不一定。大词表单 token 信息密度高、序列短，但 embedding 参数 $V\times d$ 增大（LLaMA 3 的 128K×4096≈5亿）；需在多语言/代码覆盖与参数预算间权衡。
- **embedding 缩放 $\sqrt{d}$ 还需要吗？** 用绝对位置编码相加时需要（量纲匹配）；改用 RoPE 注入 + RMSNorm 后通常去掉。
- **LLM 的 embedding 和 RAG 向量是一回事吗？** 不是。本笔记的 embedding 是模型输入查找表；RAG 检索向量是从隐藏层提取的语义嵌入，见 [[19_LLM应用工程/03_RAG系统]]。

## 相关知识

- [[01_词向量与早期表示学习]] — 词向量历史（Word2Vec/GloVe/FastText/ELMo）
- [[03_位置编码]] — LLM 位置编码视角
- [[10_大语言模型核心架构/01_LLM整体架构]] — LLM 整体架构
- [[19_LLM应用工程/03_RAG系统]] — 检索嵌入模型与语义召回
- [[00_多模态AI_综述]] — 多模态 token 对齐

## References

- Vaswani et al., *Attention Is All You Need*, 2017.（embedding 缩放 $\sqrt{d_{model}}$）
- Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, 2023.（RMSNorm、untied embedding）
- Press & Wolf, *Using the Output Embedding to Improve Language Models*, 2016.（权重 tying）
