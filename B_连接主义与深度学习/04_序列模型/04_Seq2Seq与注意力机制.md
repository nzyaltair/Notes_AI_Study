# Seq2Seq 与注意力机制

## 1. 概述

Seq2Seq（Sequence-to-Sequence）是一个通用的序列转换框架，由编码器（Encoder）和解码器（Decoder）两部分组成。编码器将可变长度的输入序列压缩为固定维度的上下文向量（Context Vector），解码器从该向量出发自回归地生成输出序列。注意力机制（Attention Mechanism）作为 Seq2Seq 的关键增强，使解码器在每一步生成时能够动态地关注输入序列的不同部分，解决了固定长度上下文向量的信息瓶颈问题。

- **解决的问题**：传统 RNN 要求输入输出等长，无法处理机器翻译（中文 5 词 → 英文 8 词）、文本摘要（长文章 → 短摘要）等序列转换任务。Seq2Seq 通过编码器-解码器架构实现了任意长度映射；注意力机制进一步解决了信息压缩损失的问题。
- **历史意义**：注意力机制是 Seq2Seq 向 Transformer 过渡的桥梁——理解注意力为什么出现以及如何运作，是理解 Transformer 自注意力机制的关键前置知识。

## 2. 发展历史

| 年代 | 里程碑 | 作者 | 意义 |
|:---|:---|:---|:---|
| 2014 | Seq2Seq 论文（两篇） | Sutskever et al. & Cho et al. | 提出编码器-解码器架构，开创了现代神经机器翻译范式 |
| 2014 | Bahdanau 注意力 | Bahdanau, Cho & Bengio | 在 Seq2Seq 中引入加性注意力，翻译性能大幅超越无注意力基线 |
| 2015 | Luong 注意力 | Luong, Pham & Manning | 提出乘性注意力（dot/general/concat），丰富了注意力计算方式 |
| 2015 | 基于注意力的语音识别 | Chorowski et al. | 注意力机制从机器翻译扩展到语音识别 |
| 2016 | GNMT (Google NMT) | Wu et al. | 工业级 NMT 系统，使用深层 LSTM + 注意力 + 残差连接 |
| 2017 | Transformer | Vaswani et al. | 自注意力彻底取代 RNN+注意力组合 |

## 3. 核心概念

### 3.1 编码器（Encoder）

编码器处理输入序列 $\mathbf{x} = (x_1, x_2, \ldots, x_T)$，将其编码为一个向量序列（Annotations）。最简单的方式是取 RNN/LSTM 最后一个时间步的隐藏状态作为上下文向量：

$$\mathbf{c} = h_T^{\text{enc}}$$

注意力机制的增强版本则是保留所有时刻的编码器隐藏状态 $(h_1^{\text{enc}}, h_2^{\text{enc}}, \ldots, h_T^{\text{enc}})$ 供解码器动态查询。

### 3.2 解码器（Decoder）

解码器自回归地生成输出序列。在时刻 $t$，解码器基于：
- 上一时刻的隐藏状态 $s_{t-1}$
- 上一时刻生成的输出 $y_{t-1}$（训练时可用真实标签 Teacher Forcing）
- 上下文向量 $c_t$（注意力机制计算）

来计算当前隐藏状态 $s_t$ 并生成 $y_t$。

### 3.3 注意力机制三要素

注意力机制的本质是一个**软寻址（Soft Addressing）** 过程：

1. **Query（查询）**：解码器当前状态 $s_{t-1}$（或 $s_t$），代表"我现在需要什么信息"
2. **Key（键）**：编码器各时刻的隐藏状态 $h_i^{\text{enc}}$，代表"每个位置有什么信息"
3. **Value（值）**：通常与 Key 相同，即 $h_i^{\text{enc}}$

通过计算 Query 和所有 Key 的相似度（注意力分数），得到加权权重，再对 Value 加权求和得到上下文向量。

### 3.4 注意力类型对比

| 类型 | 计算公式 | 特点 | 提出者 |
|:---|:---|:---|:---|
| **Additive (Bahdanau)** | $e_{t,i} = v_a^T \tanh(W_a s_{t-1} + U_a h_i)$ | 参数量大，性能泛化好 | Bahdanau 2014 |
| **Dot-Product (Luong)** | $e_{t,i} = s_t^T h_i$ | 零额外参数，速度快 | Luong 2015 |
| **General (Luong)** | $e_{t,i} = s_t^T W_a h_i$ | 带可学习线性变换 | Luong 2015 |
| **Scaled Dot-Product** | $e_{t,i} = s_t^T h_i / \sqrt{d}$ | 缩放防梯度爆炸 | Transformer 2017 |

### 3.5 Teacher Forcing

训练时，解码器每一步不接收上一步的预测结果，而是接收**真实的上一个 token** 作为输入。这避免了错误级联（一步错步步错），加速收敛。代价是训练-推理不一致（Exposure Bias）。

## 4. 技术原理

### 4.1 Seq2Seq 数学框架

**编码**：
$$h_t^{\text{enc}} = \text{RNN}(x_t, h_{t-1}^{\text{enc}})$$

**注意力计算（Bahdanau 风格）**：
$$e_{t,i} = v_a^T \tanh(W_a s_{t-1} + U_a h_i^{\text{enc}})$$
$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^T \exp(e_{t,j})}$$
$$c_t = \sum_{i=1}^T \alpha_{t,i} \cdot h_i^{\text{enc}}$$

**解码**：
$$s_t = \text{RNN}(y_{t-1}, s_{t-1}, c_t)$$
$$p(y_t | y_{<t}, \mathbf{x}) = \text{softmax}(W_o [s_t, c_t] + b_o)$$

### 4.2 Beam Search 解码

训练时使用 Teacher Forcing 并行计算所有时刻的损失。推理时需自回归生成，简单的贪婪解码（每步选概率最大的 token）可能得到次优序列。Beam Search 通过维护 $k$ 个最优候选序列来搜索更好的输出：

1. 初始化：$k$ 个候选序列，每个只包含 `<SOS>` 标记
2. 每一步：对每个候选序列，计算其下一个 token 的概率分布，取 top-$k$ 扩展
3. 直到所有候选生成 `<EOS>` 或达到最大长度
4. 选择总分最高的序列

**Beam Size** 越大，搜索空间越大，但计算成本也线性增加。机器翻译中 $k=4\sim8$ 常用。

### 4.3 注意力可视化的意义

注意力权重 $\alpha_{t,i}$ 可以被可视化为一组热力图，展示了"生成第 $t$ 个输出词时，模型在关注哪些输入词"。这不仅提供了模型可解释性，也在实践中帮助研究人员诊断翻译错误。

### 4.4 信息瓶颈与注意力

无注意力的 Seq2Seq 将所有输入信息压缩到单个向量 $c = h_T^{\text{enc}}$ 中。对于长序列，这一瓶颈导致信息丢失严重——翻译质量随输入长度增加而急剧下降（Bahatari 等人验证，超过 30 词后 BLEU 骤降）。

注意力机制的本质是：**放弃单一瓶颈，改为动态构建每个解码步骤所需的定制化上下文**。这是深度学习架构设计中的一个通用原则——"保留所有信息，按需查询"比"预先压缩"更有效。

## 5. 关键模型与论文

| 论文 | 核心贡献 | 影响 |
|:---|:---|:---|
| Sutskever et al., *Sequence to Sequence Learning with Neural Networks* (2014) | 提出用深层 LSTM 构建编码器-解码器，在 WMT 英法翻译上取得突破 | NMT 范式确立 |
| Cho et al., *Learning Phrase Representations using RNN Encoder–Decoder* (2014) | 使用 GRU 构建 Seq2Seq，提出 GRU 架构 | Seq2Seq + GRU 标准方案 |
| Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate* (2014) | 首次在 NMT 中引入加性注意力 | 注意力机制的开端 |
| Luong et al., *Effective Approaches to Attention-based NMT* (2015) | 系统化注意力变体，提出输入馈送（Input Feeding） | 注意力方法的标准化 |
| Wu et al., *Google's Neural Machine Translation System* (2016) | 工业级 GNMT，残差连接 + 深层 LSTM + 注意力 + 分段训练 | 工业落地标杆 |

## 6. 优势与局限

### 优势
1. **任意长度映射**：打破了固定维度输入输出的限制
2. **注意力带来的质量飞跃**：在长句翻译上，注意力版本的 BLEU 相比无注意力版本提升 5-10 分
3. **可解释性**：注意力权重提供了直观的输入-输出对齐可视化
4. **架构灵活性**：编码器和解码器可以选用任意 RNN 变体（LSTM/GRU/双向 RNN）

### 局限
1. **训练串行性**：编码器和解码器均依赖 RNN，训练无法沿序列维度并行
2. **解码器自回归限制**：推理时仍须逐 token 生成，长输出延迟高
3. **注意力复杂度**：标准注意力计算量为 $\mathcal{O}(T_{\text{src}} \cdot T_{\text{tgt}})$，长序列仍面临效率问题
4. **Exposure Bias**：训练时的 Teacher Forcing 导致模型从未见过自己的错误，推理时一旦出错容易级联
5. **被 Transformer 替代**：自注意力机制的全局建模、并行训练和更好的长程依赖捕获能力使其完全取代了 RNN+注意力范式

## 7. 应用场景

| 任务 | 配置 |
|:---|:---|
| **机器翻译** | BiLSTM Encoder + LSTM Decoder + Bahdanau Attention |
| **文本摘要** | 同上，输入长文章，输出短摘要 |
| **对话生成** | Encoder 编码对话历史，Decoder 生成回复 |
| **代码生成** | 输入自然语言描述，输出代码 |
| **图像描述（Image Captioning）** | CNN 作为编码器 + RNN 作为解码器 + 注意力 |

## 8. 与其他技术关系

- **与 RNN/LSTM/GRU 的关系**：Seq2Seq 是构建于 RNN 之上的架构框架，注意力机制是锦上添花的增强模块
- **与 Transformer 的关系**：Transformer 的自注意力可以理解为 "Query=Key=Value=同一序列的所有位置" 的特例。Seq2Seq 的注意力是"解码器查询编码器"，而自注意力是"序列内各位置互相查询"
- **与 CNN 的关系**：可以用 CNN 替代编码器中的 RNN（如 Image Captioning），构成跨模态 Seq2Seq
- **与 Beam Search 的关系**：Seq2Seq 的标准推理策略，在高精度场景下仍优于简单的贪婪解码

## 9. 前沿发展

- **自注意力的进化**：Transformer 将 Seq2Seq 中的"跨序列注意力"推广为"序列内自注意力"和"跨序列交叉注意力"两个独立组件，构成了现代 LLM 的核心
- **非自回归解码**：CTC、NAT（Non-Autoregressive Transformer）等工作试图打破自回归的限制，实现并行生成
- **推测解码（Speculative Decoding）**：用小型模型快速生成草稿，大型模型并行验证，大幅加速自回归推理
- **检索增强（RAG）的注意力**：在 Seq2Seq 框架中引入检索到的外部知识，注意力池从固定编码器输出扩展到动态检索数据库

## 相关知识

- 前置：[[01_循环神经网络RNN]]、[[02_长短期记忆网络LSTM]]
- 平级：[[07_状态空间模型SSM]]
- 延伸：[[../08_Transformer与注意力机制/00_Transformer与注意力机制_综述]]、[[../08_Transformer与注意力机制/01_注意力机制原理]]
