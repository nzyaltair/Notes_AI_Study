# BERT 与编码器路线

## 1. 概述

BERT（Bidirectional Encoder Representations from Transformers）是 Google 于 2018 年提出的预训练语言模型，通过掩码语言模型（MLM）实现真正的双向预训练。BERT 的发布标志着 NLP 进入预训练时代，在 11 项 NLP 任务上取得 SOTA，深刻影响了后续所有预训练模型的设计。编码器路线以双向注意力为核心，擅长文本理解类任务。

## 2. 发展历史

| 时间 | 模型 | 核心改进 |
|------|------|---------|
| 2018.10 | **BERT** | MLM + NSP 双向预训练 |
| 2019.07 | **XLNet** | 排列语言建模，解决BERT的预训练-微调不一致 |
| 2019.07 | **RoBERTa** | 去NSP、动态Masking、更大批量更多数据 |
| 2019.09 | **ALBERT** | 参数压缩：因式分解嵌入 + 跨层参数共享 |
| 2020.03 | **ELECTRA** | RTD判别式预训练，效率提升 |
| 2020.06 | **DeBERTa** | 解耦注意力 + 增强型Mask解码器 |
| 2021.11 | **DeBERTaV3** | 替换为ELECTRA式RTD目标 |
| 2023+ | 编码器路线式微 | LLM时代Decoder-only架构统一 |

## 3. 核心概念

### 双向注意力

BERT 使用 Encoder 架构，每个 token 可以同时关注左侧和右侧的所有 token：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

与 GPT 的因果掩码不同，BERT 不使用任何掩码限制注意力方向。

### [CLS] token

BERT 在每个输入序列开头添加特殊的 `[CLS]` token，其最终隐藏状态用于分类任务：

$$h_{[\text{CLS}]} = \text{BERT}(\text{Input})_0$$

分类头：$P(y) = \text{softmax}(W h_{[\text{CLS}]})$

### [SEP] token

用 `[SEP]` 分隔句子对，配合段嵌入（Segment Embedding）区分输入的两句话。

## 4. 技术原理

### 4.1 BERT 预训练

BERT 采用两个预训练任务的联合训练：

**任务一：掩码语言模型（MLM）**

随机选择 15% 的 token 进行处理：
- 80% 概率替换为 `[MASK]`
- 10% 概率替换为随机 token
- 10% 概率保持不变

$$\mathcal{L}_{MLM} = -\sum_{i \in \mathcal{M}} \log P(x_i | x_{\backslash\mathcal{M}})$$

混合策略的原因：强迫模型在所有位置维持分布式表示，而非只依赖 `[MASK]` 标记。

**任务二：下一句预测（NSP）**

判断句子 B 是否紧跟句子 A（50% 正例，50% 随机负例）：

$$\mathcal{L}_{NSP} = -\log P(\text{IsNext} | A, B)$$

> **注意**：RoBERTa 实验证明 NSP 任务无效，移除后性能反而提升。后续模型基本不再使用 NSP。

### 4.2 BERT 模型结构

| 配置 | 层数 | 隐藏维度 | 注意力头数 | 参数量 |
|------|------|---------|-----------|--------|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |

每层结构：
```
输入 → 多头自注意力 → Add & LayerNorm → FFN → Add & LayerNorm → 输出
```

### 4.3 BERT 微调

不同任务使用不同的微调方式：

| 任务类型 | 输入表示 | 输出方式 |
|---------|---------|---------|
| 单句分类 | `[CLS] sentence [SEP]` | `[CLS]` 的隐藏状态 → softmax |
| 句对分类 | `[CLS] A [SEP] B [SEP]` | `[CLS]` 的隐藏状态 → softmax |
| 问答（抽取式） | `[CLS] question [SEP] passage [SEP]` | 每个 token → start/end logits |
| 序列标注（NER） | `[CLS] sentence [SEP]` | 每个 token → 标签 logits |

### 4.4 关键变体原理

#### RoBERTa（Robustly Optimized BERT）

RoBERTa 的核心贡献是**训练优化**，而非架构创新：

1. **移除 NSP**：实验证明 NSP 无效
2. **动态 Masking**：每次训练对同一序列使用不同的 mask 模式（BERT 是静态的）
3. **更大批量**：batch size 从 256 增至 8192
4. **更多数据**：16GB → 160GB（CC-News、OpenWebText、Stories）
5. **更长训练**：从 100K 步增至 500K 步
6. **使用 Byte-Pair Encoding**：字符级 BPE 而非 WordPiece

#### ALBERT（A Lite BERT）

两个参数压缩技术：

**嵌入因式分解**：将 $V \times H$ 的嵌入矩阵分解为 $V \times E$ 和 $E \times H$（$E \ll H$）

$$V \times H \rightarrow V \times E + E \times H$$

BERT-large: $V=30000, H=1024$ → $30.7M$ 参数；ALBERT: $E=128$ → $3.8M + 0.13M$ 参数。

**跨层参数共享**：所有 Transformer 层共享同一组参数，大幅减少参数量。

#### ELECTRA

引入**生成器-判别器**架构：

1. **生成器**（小MLM模型）：替换部分 token
2. **判别器**（主模型）：判断每个 token 是否被替换（二分类）

$$\mathcal{L}_{Disc} = -\sum_{i=1}^T [x_i \text{被替换}] \log D(h_i, 1) + [x_i \text{未被替换}] \log D(h_i, 0)$$

优势：所有 token 都有训练信号（BERT 仅 15%），训练效率提升约 4 倍。

#### DeBERTa（Decoding-enhanced BERT with disentangled attention）

**解耦注意力**：将 token 的内容嵌入和位置嵌入分离，分别计算注意力：

$$\text{Attention} = \text{softmax}\left(\frac{(Q_c + Q_r)(K_c + K_r)^T}{\sqrt{d}}\right)(V_c)$$

展开后包含四项：内容-内容、内容-位置、位置-内容、位置-位置。关键创新是**位置-内容**项，使模型能感知相对位置关系。

**增强型 Mask 解码器（EMD）**：在最后一层使用绝对位置信息，缓解底层缺少绝对位置信息的问题。

## 5. 关键模型对比

| 模型 | 参数量 | 预训练目标 | 关键创新 | GLUE分数 |
|------|--------|-----------|---------|---------|
| BERT-large | 340M | MLM + NSP | 双向预训练 | 80.5 |
| RoBERTa | 355M | MLM | 训练优化 | 88.5 |
| ALBERT-xxlarge | 12M(共享后) | MLM + SOP | 参数压缩 | 89.4 |
| ELECTRA-large | 330M | RTD | 判别式预训练 | 89.0 |
| DeBERTa-large | 400M | MLM | 解耦注意力 | 90.3 |
| DeBERTaV3-large | 434M | RTD | 解耦+判别 | 91.1 |

## 6. 优势与局限

### 优势
- 双向注意力充分利用上下文信息，理解类任务表现优异
- 微调简单：只需添加一个输出层
- 生态成熟：Hugging Face 提供大量预训练权重

### 局限
- **不适合生成任务**：MLM 预训练目标与自回归生成不匹配
- **预训练-微调不一致**：预训练有 `[MASK]`，微调没有
- **固定长度输入**：通常限制为 512 token
- **计算成本**：双向注意力无法像自回归模型那样利用 KV-Cache 加速推理

## 7. 应用场景

- **文本分类**：情感分析、意图识别、内容审核
- **命名实体识别（NER）**：序列标注
- **问答系统**：抽取式 QA（SQuAD）
- **语义相似度**：句对匹配
- **信息抽取**：关系抽取、事件抽取
- **句子嵌入**：Sentence-BERT 用于语义检索

## 8. 与其他技术关系

- BERT 基于 [[08_Transformer与注意力机制|Transformer与注意力机制]] 的 Encoder 架构
- 预训练原理详见 [[02_预训练核心原理|预训练核心原理]]
- 词向量是 BERT 输入层的基础：[[01_词向量与早期表示学习|词向量与早期表示学习]]
- 编码器路线与 [[04_GPT与解码器路线|GPT与解码器路线]] 形成对比
- Sentence-BERT 催生了稠密检索技术（[[19_LLM应用工程/03_RAG系统/00_RAG系统|RAG系统]]）
- BERT 的微调策略演化为 [[11_大模型训练与对齐/05_参数高效微调/00_参数高效微调|参数高效微调]]

## 9. 前沿发展

- **Decoder-only 统一**：LLM 时代，理解任务通过指令微调转为生成任务，编码器路线逐渐式微
- **高效推理**：DistilBERT（知识蒸馏）、TinyBERT 等轻量化方向
- **长文本扩展**：Longformer、BigBird 引入稀疏注意力处理长序列
- **多语言**：mBERT、XLM-R 扩展到 100+ 种语言
- **领域适配**：BioBERT（生物医学）、SciBERT（科学文献）、LegalBERT（法律）
- **Sentence Embedding**：SimCSE、E5 等，从 token 级到句子级表示
