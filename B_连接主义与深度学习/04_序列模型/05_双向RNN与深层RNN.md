# 双向 RNN 与深层 RNN

## 1. 概述

标准 RNN 在每个时间步仅利用"过去"的信息（从 $t=1$ 到当前）。然而在许多序列任务中，当前位置的预测同时依赖于"上文"和"下文"——例如命名实体识别中，判断"苹果"是公司还是水果需要看前后文。双向 RNN（Bidirectional RNN, BiRNN）通过同时运行正向和反向两个 RNN，使每个位置的输出都能利用完整的双向上下文。深层 RNN（Stacked/Deep RNN）则通过堆叠多个 RNN 层，构建更深层次的特征表示。

- **解决的问题**：单向 RNN 对"未来"信息的盲视（BiRNN 解决）；浅层 RNN 表达能力有限（深层 RNN 解决）。
- **核心思想**：BiRNN = 正向 RNN + 反向 RNN，输出拼接；深层 RNN = 下层输出作为上层输入的层级堆叠。

## 2. 发展历史

| 年代 | 里程碑 | 意义 |
|:---|:---|:---|
| 1997 | Schuster & Paliwal 提出双向 RNN | 首次引入双向处理机制，语音和 NLP 任务取得显著提升 |
| 2005 | Graves & Schmidhuber 提出 BiLSTM + CTC | 双向 LSTM 在语音识别上超越传统 HMM 系统 |
| 2014 | Sutskever et al. 使用多层 LSTM 做 NMT | 4 层 LSTM 编码器 + 4 层 LSTM 解码器，深层 RNN 在机器翻译上展现优势 |
| 2016 | GNMT 使用 8 层 BiLSTM + 残差连接 | 证明了残差连接对深层 RNN 训练稳定性的关键作用 |
| 2018 | ELMo 使用深层 BiLSTM 做上下文词向量 | 双向深层 LSTM 在 NLP 下游任务中广泛使用（后被 BERT 取代） |

## 3. 核心概念

### 3.1 双向 RNN 的信息流

BiRNN 由两个独立的 RNN 构成：
- **正向 RNN**：从 $t=1$ 到 $t=T$ 处理，隐藏状态 $\overrightarrow{h}_t$ 编码 $x_{1:t}$ 的信息
- **反向 RNN**：从 $t=T$ 到 $t=1$ 处理，隐藏状态 $\overleftarrow{h}_t$ 编码 $x_{t:T}$ 的信息
- **最终输出**：$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$（拼接），包含 $t$ 位置的完整双向上下文

正向和反向 RNN 的权重**不共享**，是两个独立的网络同时训练。

### 3.2 深层 RNN 的层级结构

第 1 层 RNN 接收原始输入 $x_t$，输出隐藏状态 $h_t^{(1)}$；第 2 层 RNN 接收 $h_t^{(1)}$ 作为输入，输出 $h_t^{(2)}$……以此类推：

$$h_t^{(l)} = \text{RNN}^{(l)}(h_t^{(l-1)}, h_{t-1}^{(l)})$$

其中 $h_t^{(0)} = x_t$。不同层的 RNN 具有独立的权重参数，各层可以学到不同时间尺度的抽象特征——底层捕捉局部模式（如短语），高层捕捉全局语义（如句子意图）。

### 3.3 双向 + 深层的组合

深层 BiRNN 是最强大的 RNN 配置，同时实现：
- **维度一（层数）**：从底层到高层，逐步抽象
- **维度二（方向）**：每层内部双向建模

最常见的 NLP 架构：2-4 层 BiLSTM，底层捕捉词法和局部句法，高层捕捉语义。

## 4. 技术原理

### 4.1 双向 RNN 的数学形式

设 $\overrightarrow{f}$ 和 $\overleftarrow{f}$ 分别为正向和反向 RNN 的转移函数：

$$\overrightarrow{h}_t = \overrightarrow{f}(\overrightarrow{W}_{hh} \overrightarrow{h}_{t-1} + \overrightarrow{W}_{xh} x_t + \overrightarrow{b}_h)$$
$$\overleftarrow{h}_t = \overleftarrow{f}(\overleftarrow{W}_{hh} \overleftarrow{h}_{t+1} + \overleftarrow{W}_{xh} x_t + \overleftarrow{b}_h)$$
$$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$$

最终 $h_t \in \mathbb{R}^{2d_h}$（维度翻倍）。输出层可基于 $h_t$ 进行预测：

$$y_t = \text{softmax}(W_{hy} h_t + b_y)$$

其中 $W_{hy} \in \mathbb{R}^{d_{\text{out}} \times 2d_h}$。

### 4.2 BPTT 在双向 RNN 中的实现

正向和反向 RNN 的梯度计算是**完全独立的**：
- 正向部分：标准 BPTT，从 $t=T$ 反向传播到 $t=1$
- 反向部分：标准 BPTT，从 $t=1$ 反向传播到 $t=T$

两者没有交叉依赖，可以并行计算。

### 4.3 深层 RNN 的梯度流挑战

深度增加带来的梯度传输路径变长：

$$\frac{\partial \mathcal{L}}{\partial \theta^{(1)}} = \frac{\partial \mathcal{L}}{\partial h^{(L)}} \cdot \prod_{l=2}^{L} \frac{\partial h^{(l)}}{\partial h^{(l-1)}} \cdot \prod_{t=T}^{1} \frac{\partial h^{(1)}_t}{\partial h^{(1)}_{t-1}} \cdot \frac{\partial h^{(1)}}{\partial \theta^{(1)}}$$

梯度需要同时穿越**时间维度**和**深度维度**，双重连乘使训练变得更加困难。这就是为什么深层 RNN 几乎总是需要残差连接。

### 4.4 残差连接的引入

在深层 RNN 的层间加入残差连接（Residual Connection）：

$$h_t^{(l)} = \text{RNN}^{(l)}(h_t^{(l-1)}, h_{t-1}^{(l)}) + h_t^{(l-1)}$$

这条直接加法通路为梯度提供了"高速公路"，使深层 RNN 的训练稳定性大幅提升。Google 的 GNMT（8 层 LSTM）和 ELMo（2 层 BiLSTM）均使用了残差连接。

### 4.5 正则化技巧

- **Recurrent Dropout**（Gal & Ghahramani, 2016）：基于变分推断的 Dropout，在前向传播的循环连接中应用固定的 Dropout 掩码（而非每步随机变化），保持时序一致性
- **Layer Normalization**：在每层 RNN 的门控输入上应用 LayerNorm，稳定深层训练。BiLSTM + LayerNorm 是 ELMo 等模型的标准配置
- **Zoneout**（Krueger et al., 2017）：随机保持上一时刻的隐藏状态不变（而非置零），作为 Dropout 的时序替代

## 5. 关键变体与模型

| 模型/配置 | 结构 | 典型应用 |
|:---|:---|:---|
| **BiRNN** | 1 层双向简单 RNN | 简单序列标注 |
| **BiLSTM** | 1~2 层双向 LSTM | NER、分词、语音识别 |
| **BiLSTM-CRF** | BiLSTM 提取特征 + CRF 建模标签转移 | 序列标注标准方案 |
| **Deep BiLSTM (ELMo)** | 2 层 BiLSTM + 残差 | 上下文词向量预训练 |
| **GNMT** | 8 层 LSTM + 残差 | 工业级机器翻译 |
| **CNN-BiLSTM** | CNN 提取局部特征 + BiLSTM 建模序列 | 文本分类、音频分类 |
| **BiGRU** | 双向 GRU | 轻量级双向场景 |

## 6. 优势与局限

### 优势
- **双向信息利用**：BiRNN 显著提升了序列标注等需要上下文感知的任务性能
- **多层抽象**：深层 RNN 可以学到底层局部特征到高层全局语义的层次化表征
- **残差连接稳定训练**：解决了深层 RNN 训练困难的问题，8 层以上深度成为可能

### 局限
- **非因果性限制**：双向 RNN 需要完整的输入序列才能运行，不适用于在线/流式场景和自回归生成
- **参数量增长**：BiRNN 参数量翻倍（两个独立 RNN），深层 RNN 参数量线性增加
- **推理延迟**：双向 RNN 必须等待完整输入，深层 RNN 增加每步计算量
- **被 Transformer 双向替代**：BERT 的自注意力在双向建模上效果更好且训练并行

## 7. 应用场景

| 场景 | 推荐配置 | 原因 |
|:---|:---|:---|
| **命名实体识别（NER）** | BiLSTM-CRF | 双向上下文 + 标签转移约束 |
| **词性标注（POS Tagging）** | BiLSTM | 判断词性需要完整的句子上下文 |
| **语音识别** | Deep BiLSTM + CTC | 语音帧的双向依赖 |
| **文本分类** | BiLSTM + Max Pooling | 关键短语可能出现在任意位置 |
| **机器翻译编码器** | Deep BiLSTM | 需要理解整个输入句子 |
| **上下文词向量（ELMo）** | Deep BiLSTM (L=2) | 提取上下文相关的词表示 |

## 8. 与其他技术关系

- **与单向 RNN 的关系**：BiRNN 是单向 RNN 的双倍配置，适用场景互补——单向 RNN 适用于需要因果性（自回归生成、在线预测）的场景，BiRNN 适用于完整序列可用的场景
- **与 LSTM/GRU 的关系**：BiLSTM/BiGRU 分别是 LSTM/GRU 的双向版本，BiRNN 的核心增益来源于方向扩充而非模型选择
- **与 Transformer 的关系**：BERT 的自注意力本质上提供了一种比 BiRNN 更强的双向建模方式——自注意力直接连接任意两个位置，而 BiRNN 需要通过多层递推才能获取远距离的双向信息
- **与 ELMo 的关系**：ELMo 是深层 BiLSTM 在预训练语言模型中的经典应用，证明了双向深层 LSTM 在 NLP 上的威力（后被 BERT 的 Transformer 双向替代）
- **与 CRF 的关系**：BiLSTM-CRF 是序列标注的经典组合——BiLSTM 负责特征提取，CRF 负责建模输出标签之间的转移约束

## 9. 前沿发展

- **Transformer 时代的双向性**：双向 RNN 的"双向性"在 Transformer 中被更优雅地实现——通过 Attention Mask 控制信息流向（双向 Encoder、因果 Decoder），不再需要训练两个独立网络
- **ELMo 与 BERT 的对比启示**：ELMo（BiLSTM）和 BERT（Transformer）的目标一致（上下文词向量），但 BERT 证明了自注意力在双向建模上优于 BiLSTM。这一对比是理解"为什么 Transformer 取代 RNN"的关键案例
- **非自回归序列标注**：现代序列标注（如 SpanBERT、依存句法分析）已不再依赖 BiLSTM 提取特征，转而使用 Transformer 的全自注意力编码
- **残差连接的普遍化**：深层 RNN 中的残差连接被 Transformer 延续并发展为 Pre-LN / Post-LN 等精细化变体，成为所有深层架构的标准组件

## 相关知识

- 前置：[[01_循环神经网络RNN]]、[[02_长短期记忆网络LSTM]]
- 平级：[[06_CTC与序列标注]]
- 延伸：[[../16_自然语言处理/05_序列标注模型]]
