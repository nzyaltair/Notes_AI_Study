# T5 与 Encoder-Decoder 路线

## 1. 概述

T5（Text-to-Text Transfer Transformer）由 Google 于 2019 年提出，将所有 NLP 任务统一为"文本到文本"（Text-to-Text）格式。T5 采用 Encoder-Decoder 架构和 Span Corruption 预训练目标，提供了一种将理解与生成任务统一在单一框架下的方案。虽然 Decoder-only 架构在 LLM 时代逐渐统一，但 Encoder-Decoder 路线在特定场景（如翻译、摘要）仍有独特优势。

## 2. 发展历史

| 时间 | 模型 | 核心改进 |
|------|------|---------|
| 2019.10 | **T5** | 统一文本到文本框架，Span Corruption |
| 2019.10 | **BART** | 噪声去除预训练（多种噪声组合） |
| 2019.10 | **mT5** | 多语言 T5，101 种语言 |
| 2020 | **T5-11B** | T5 最大版本 |
| 2021 | **FLAN-T5** | 指令微调 T5 |
| 2021 | **mT5-MTL** | 多任务学习 |
| 2023+ | Encoder-Decoder 逐渐被 Decoder-only 取代 |

## 3. 核心概念

### 文本到文本统一框架

T5 的核心思想：**将所有 NLP 任务转化为文本到文本的生成问题**。

```
翻译:   "translate English to German: That is good." → "Das ist gut."
摘要:   "summarize: [文章内容]" → "[摘要内容]"
分类:   "cola sentence: The coin is big." → "acceptable"
问答:   "question: ... context: ..." → "[答案]"
回归:   "stsb sentence1: ... sentence2: ..." → "3.8"
```

这一设计的关键优势：
- **统一架构**：一个模型处理所有任务，无需任务特定头
- **统一训练**：所有任务共享相同的损失函数（CLM on target）
- **灵活性强**：新任务只需设计输入格式，无需改架构

### Encoder-Decoder 架构

```
编码器：双向自注意力（理解输入）
     ↓
交叉注意力（Decoder的Q关注Encoder的KV）
     ↓
解码器：因果自注意力 + 交叉注意力（生成输出）
```

与纯 Decoder-only 的区别：Encoder-Decoder 在编码阶段可以双向关注全部输入，在解码阶段通过交叉注意力访问编码器输出。

## 4. 技术原理

### 4.1 Span Corruption 预训练

T5 使用 Span Corruption 作为预训练目标：随机遮盖连续的文本片段，用 sentinel token 标记，让模型生成被遮盖的内容。

**示例**：

```
原始文本: "Thank you for inviting me to your birthday party"
遮盖后:   "Thank you <X> me to <Y>"
目标:     "<X> for inviting <Y> your birthday party"
```

**Span Corruption 参数**：
- 遮盖比例：15%（与 BERT 相同）
- 平均 Span 长度：3 tokens
- Sentinel token：`<extra_id_0>`, `<extra_id_1>`, ...

**与 BERT MLM 的区别**：
- BERT：预测被遮盖的原始 token（分类）
- T5：生成被遮盖的 span 序列（自回归生成）
- T5 的目标与微调一致（都是文本生成）

### 4.2 T5 模型结构

| 配置 | 层数(Enc/Dec) | d_model | 头数 | 参数量 |
|------|-------------|---------|------|--------|
| T5-Small | 6/6 | 512 | 8 | 60M |
| T5-Base | 12/12 | 768 | 12 | 220M |
| T5-Large | 24/24 | 1024 | 16 | 770M |
| T5-3B | 24/24 | 1024 | 32 | 2.8B |
| T5-11B | 24/24 | 1024 | 128 | 11B |

**与原始 Transformer 的差异**：
- 使用相对位置编码（而非正弦编码）
- 编码器和解码器不共享参数
- LayerNorm 在残差之前（Pre-LN）
- 简化的 FFN：只使用一个权重矩阵（Gated Linear Unit变体）

### 4.3 BART 预训练

BART（Bidirectional and Auto-Regressive Transformers）使用**噪声去除**作为预训练目标，组合多种噪声：

| 噪声类型 | 描述 |
|---------|------|
| Token Masking | 随机遮盖 token（同 BERT） |
| Token Deletion | 随机删除 token |
| Text Infilling | 遮盖连续 span，可能遮盖0个token（插入噪声） |
| Sentence Permutation | 打乱句子顺序 |
| Document Rotation | 旋转文档（选一个点，前后交换） |
| Token Rotation | 旋转 token 序列 |

BART 的优势：预训练目标与微调任务（文本生成）更一致，适合摘要等 Seq2Seq 任务。

### 4.4 多语言 T5（mT5）

mT5 在 101 种语言的 Common Crawl 数据（mC4）上预训练，覆盖低资源语言。

关键挑战：
- **语言不平衡**：高资源语言（英语、中文）数据远多于低资源语言
- **分词器设计**：需要覆盖多语言词表（通常使用 SentencePiece）
- **跨语言迁移**：高资源语言的能力是否能迁移到低资源语言

### 4.5 FLAN-T5：指令微调

FLAN-T5 在 T5 基础上进行指令微调（Instruction Tuning），在 1,800+ 任务上微调：

```
任务格式:
"Read the following sentence and determine if it is grammatically correct:
The cat sat on the mat.
Answer:"
→ "yes"
```

FLAN-T5 证明：指令微调后的 Encoder-Decoder 模型在零样本和少样本设置下也能表现优异，且参数效率高于 GPT-3。

## 5. 三大架构路线对比

| 维度 | Encoder-only (BERT) | Decoder-only (GPT) | Encoder-Decoder (T5) |
|------|--------------------|--------------------|---------------------|
| 注意力 | 双向 | 单向（因果） | 编码双向+解码因果 |
| 预训练目标 | MLM | CLM | Span Corruption |
| 理解任务 | ★★★★★ | ★★★☆ | ★★★★ |
| 生成任务 | ★★ | ★★★★★ | ★★★★ |
| Seq2Seq任务 | ★★ | ★★★ | ★★★★★ |
| 推理效率 | 高 | KV-Cache加速 | 中等 |
| 参数效率 | 中 | 高 | 中 |
| LLM时代地位 | 式微 | 统一主流 | 特定场景使用 |

## 6. 优势与局限

### 优势
- **任务统一**：Text-to-Text 框架将所有 NLP 任务统一为生成
- **编码优势**：Encoder 双向注意力对理解输入有优势
- **生成分离**：编码与解码分离，各自优化
- **适合 Seq2Seq**：翻译、摘要等任务天然适配

### 局限
- **参数效率低**：Encoder 和 Decoder 各一套参数，同等能力需要更多参数
- **规模化不如 Decoder-only**：Scaling Laws 实验显示 Decoder-only 更优
- **推理复杂**：需要同时维护 Encoder 和 Decoder 的 KV-Cache
- **LLM 时代式微**：主流大模型几乎全部采用 Decoder-only

## 7. 应用场景

- **机器翻译**：Encoder-Decoder 是翻译任务的经典架构
- **文本摘要**：BART 在摘要任务上表现优异
- **多语言任务**：mT5 覆盖 100+ 种语言
- **问答系统**：生成式问答
- **信息抽取**：通过 Text-to-Text 格式统一抽取任务
- **代码生成**：CodeT5 等领域特定变体

## 8. 与其他技术关系

- T5 基于 [[08_Transformer与注意力机制|Transformer与注意力机制]] 的完整 Encoder-Decoder 架构
- 预训练原理详见 [[02_预训练核心原理|预训练核心原理]]
- 与 [[03_BERT与编码器路线|BERT与编码器路线]] 和 [[04_GPT与解码器路线|GPT与解码器路线]] 构成三大架构路线
- FLAN-T5 的指令微调思想详见 [[11_大模型训练与对齐/03_监督微调/03_监督微调|监督微调]]
- 多语言扩展与 [[16_自然语言处理|自然语言处理]] 的跨语言研究相关
- CodeT5 等变体在 [[19_LLM应用工程|LLM应用工程]] 中有应用

## 9. 前沿发展

- **Decoder-only 统一**：主流 LLM 不再使用 Encoder-Decoder，翻译/摘要也由 Decoder-only 完成
- **领域变体**：CodeT5（代码）、SciFive（生物医学）、ClinicalT5（临床）
- **指令微调延续**：FLAN-T5 的思想延续到 FLAN-PaLM 等更大模型
- **多语言效率**：研究如何在有限参数下覆盖更多语言
- **轻量化**：T5 的蒸馏与量化版本用于边缘部署
- **小模型复兴**：2023-2024 年 Encoder-Decoder 在小参数量场景重新受到关注（如 Flan-T5 在 Agent 工作流中的应用）
