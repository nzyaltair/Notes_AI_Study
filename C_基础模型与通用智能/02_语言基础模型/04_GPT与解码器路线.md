# GPT 与解码器路线

## 1. 概述

GPT（Generative Pre-trained Transformer）系列由 OpenAI 开发，采用 Decoder-only 架构和因果语言模型（CLM）预训练目标。GPT 系列的发展历程——从 GPT-1 的预训练微调到 GPT-3 的上下文学习——展现了"规模化"如何让自回归语言模型从特定任务工具进化为通用能力基座。Decoder-only 架构在 LLM 时代逐渐统一了理解和生成任务，成为现代大语言模型的主流选择。

## 2. 发展历史

| 时间 | 模型 | 参数量 | 核心创新 |
|------|------|--------|---------|
| 2018.06 | **GPT-1** | 1.17亿 | 预训练+微调范式验证 |
| 2019.02 | **GPT-2** | 15亿 | 零样本学习（Zero-shot），WebText数据 |
| 2020.05 | **GPT-3** | 1750亿 | 上下文学习（In-Context Learning），少样本 |
| 2021.07 | **Codex** | 120亿 | 代码预训练，GitHub数据 |
| 2022.01 | **InstructGPT** | — | 指令微调 + RLHF |
| 2022.11 | **ChatGPT** | — | 对话优化 + RLHF，引爆LLM时代 |
| 2023.03 | **GPT-4** | 未公开 | 多模态、推理能力飞跃 |

## 3. 核心概念

### 自回归生成

GPT 采用因果语言模型，逐 token 生成文本：

$$P(x_{1:T}) = \prod_{t=1}^T P(x_t | x_{<t})$$

训练目标：

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^T \log P(x_t | x_{<t})$$

### 因果掩码（Causal Masking）

通过下三角掩码确保位置 $t$ 只能关注位置 $\leq t$：

```python
mask = torch.tril(torch.ones(seq_len, seq_len))
scores = scores.masked_fill(mask == 0, float('-inf'))
```

### 上下文学习（In-Context Learning, ICL）

GPT-3 的核心创新：不更新模型参数，通过在输入中提供示例引导模型完成新任务。

**Zero-shot**：仅给指令
```
Translate to French: cheese →
```

**One-shot**：给一个示例
```
Translate to French: cheese → fromage
Translate to French: hello →
```

**Few-shot**：给多个示例
```
Translate to French: cheese → fromage
Translate to French: water → eau
Translate to French: hello →
```

## 4. 技术原理

### 4.1 GPT 架构

GPT 使用 Transformer Decoder，每层结构：

```
输入 → 掩码多头自注意力 → Add & LayerNorm → FFN → Add & LayerNorm → 输出
```

与 BERT 的关键区别：
- **因果掩码**：只允许关注左侧 token（BERT 是双向的）
- **GELU 激活**：FFN 使用 GELU 而非 ReLU
- **可学习位置编码**：而非正弦编码

### 4.2 GPT-1：预训练微调范式

GPT-1 验证了"无监督预训练 + 有监督微调"在生成任务上的有效性。

**预训练**：在 BooksCorpus（~7000本书）上进行 CLM 训练。

**微调**：对特定任务添加线性分类头，联合优化：

$$\mathcal{L} = \mathcal{L}_{CLM} + \lambda \mathcal{L}_{task}$$

微调时将不同 NLP 任务统一为序列生成格式：
- **分类**：`[text] [extract]` → 预测标签
- **蕴含**：`[premise] [hypothesis]` → 预测关系
- **相似度**：两次前向（text1→text2, text2→text1）
- **QA**：`[passage] [question]` → 生成答案

### 4.3 GPT-2：零样本学习

GPT-2 的核心飞跃是**零样本学习**：无需微调，模型直接通过 prompt 完成任务。

关键改进：
1. **更大规模**：参数量从 1.17 亿增至 15 亿（10 倍+）
2. **更大数据**：WebText（~40GB，Reddit 高赞链接，karma≥3）
3. **零样本任务适配**：模型理解任务指令本身

```
零样本翻译示例:
"In the year 1969, humans first landed on the moon. Summarize this in Chinese:"
→ 1969年，人类首次登上月球。
```

**为什么零样本成为可能**：规模化使模型在预训练中隐式学习了"执行指令"的能力。

### 4.4 GPT-3：规模化与涌现

GPT-3 的核心贡献是证明了**规模化（Scaling）**带来的能力质变。

| 配置 | 层数 | d_model | 头数 | 参数量 |
|------|------|---------|------|--------|
| GPT-3 Small | 12 | 768 | 12 | 125M |
| GPT-3 Medium | 24 | 1024 | 16 | 350M |
| GPT-3 Large | 24 | 1536 | 16 | 760M |
| GPT-3 XL | 24 | 2048 | 24 | 1.3B |
| GPT-3 6.7B | 32 | 4096 | 32 | 6.7B |
| GPT-3 175B | 96 | 12288 | 96 | 175B |

**涌现能力（Emergent Abilities）**：
- 少样本翻译、算术推理、代码生成
- 模型规模超过一定阈值后，某些能力突然出现
- 小模型完全不具备这些能力

**训练数据**：
- Common Crawl（过滤后 410B tokens）
- WebText2（19B tokens）
- Books1（12B tokens）
- Books2（55B tokens）
- Wikipedia（3B tokens）

**关键设计决策**：
- 使用交替稠密与稀疏注意力（Sparse Transformer）
- 上下文窗口 2048 tokens
- Few-shot 优于 Zero-shot 和 Fine-tuning（在多数任务上）

### 4.5 从 GPT-3 到 ChatGPT

GPT-3 是强大的基座模型，但存在以下问题：
- 不遵循指令，生成风格不可控
- 可能产生有害、偏见或虚假内容
- 交互体验不佳

解决路径：

```
GPT-3（基座模型）
  → SFT（Supervised Fine-Tuning，指令微调）
  → RLHF（Reinforcement Learning from Human Feedback）
  → ChatGPT（对话优化）
```

**InstructGPT（GPT-3.5）三阶段训练**：

1. **SFT**：人工标注指令-回复对，监督微调
2. **奖励模型（RM）**：训练一个打分模型，学习人类偏好排序
3. **PPO 强化学习**：用 RM 的奖励信号优化 SFT 模型

详见 [[11_大模型训练与对齐/03_监督微调/03_监督微调|监督微调]] 和 [[11_大模型训练与对齐/04_偏好对齐方法/00_偏好对齐方法|偏好对齐方法]]。

## 5. 关键模型对比

| 模型 | 参数量 | 预训练数据 | 核心能力 | 意义 |
|------|--------|-----------|---------|------|
| GPT-1 | 117M | BooksCorpus | 微调后任务适配 | 验证Decoder预训练 |
| GPT-2 | 1.5B | WebText | 零样本学习 | 规模带来零样本能力 |
| GPT-3 | 175B | 500B tokens | 上下文学习 | 规模化涌现 |
| Codex | 12B | 代码+文本 | 代码生成 | 领域预训练 |
| InstructGPT | 1.3-6B | +指令数据 | 指令跟随 | RLHF对齐 |

## 6. 优势与局限

### 优势
- **生成能力强**：自回归预训练天然适合开放式生成
- **训练效率高**：CLM 每个位置都是训练信号
- **任务统一**：所有任务可转化为"续写"
- **推理高效**：可利用 KV-Cache 加速自回归生成
- **规模化友好**：Decoder-only 架构的 Scaling Laws 更优

### 屺限
- **单向注意力**：不能直接利用右侧上下文（理解任务理论上不如 BERT）
- **幻觉问题**：自回归生成可能产生看似合理但错误的内容
- **固定上下文长度**：超出窗口的历史信息丢失
- **推理成本高**：逐 token 生成，延迟随长度增长
- **可控性差**：基座模型不天然遵循指令（需RLHF解决）

## 7. 应用场景

- **对话与问答**：ChatGPT、Copilot
- **文本生成**：摘要、续写、创意写作
- **代码生成**：GitHub Copilot、Codex
- **翻译**：少样本机器翻译
- **推理**：数学推理、逻辑推理（需思维链）
- **信息抽取**：通过指令转化为生成任务
- **Agent 系统**：作为推理与规划的核心引擎

## 8. 与其他技术关系

- GPT 基于 [[08_Transformer与注意力机制|Transformer与注意力机制]] 的 Decoder 架构
- 预训练原理详见 [[02_预训练核心原理|预训练核心原理]]
- 与 [[03_BERT与编码器路线|BERT与编码器路线]] 形成架构对比
- Scaling Laws 详见 [[07_Scaling_Laws与计算最优训练]]
- 从零实现详见 [[06_从零构建GPT]]
- 微调与对齐详见 [[11_大模型训练与对齐/03_监督微调/03_监督微调|监督微调]]、[[11_大模型训练与对齐/04_偏好对齐方法/00_偏好对齐方法|偏好对齐方法]]
- GPT 的演进直接催生了 [[10_大语言模型核心架构|大语言模型]] 时代
- 位置编码详见 [[08_Transformer与注意力机制/04_位置编码|位置编码]]，分词详见 [[10_大语言模型核心架构/04_分词与Tokenization|分词]]

## 9. 前沿发展

- **架构统一**：Decoder-only 成为 LLM 事实标准（LLaMA、Qwen、Mistral 等）
- **推理模型**：OpenAI o1/o3 系列，结合思维链与强化学习
- **多模态扩展**：GPT-4V、GPT-4o 支持图像、音频输入
- **Agent 化**：GPT 作为工具调用和自主规划的核心
- **长上下文**：从 2K 扩展到 128K+ tokens
- **效率优化**：推测解码（Speculative Decoding）、量化（[[12_大模型推理与优化/02_量化技术/02_量化技术|量化技术]]）
- **开源生态**：LLaMA 系列推动 Decoder-only 架构的开源普及
