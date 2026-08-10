# 长短期记忆网络（LSTM）

## 1. 概述

长短期记忆网络（Long Short-Term Memory, LSTM）是由 Hochreiter 和 Schmidhuber 于 1997 年提出的循环神经网络变体，专为解决简单 RNN 的梯度消失问题而设计。LSTM 通过引入**门控机制（Gating Mechanism）** 和独立的**细胞状态（Cell State）**，在结构上实现了梯度在长序列中的稳定传递，是序列模型发展史上最重要的突破。

- **解决的问题**：简单 RNN 因梯度连乘导致的指数衰减或增长，使得模型在实践中几乎无法学习超过 10 步的依赖关系。LSTM 通过加法更新机制，使梯度可以在细胞状态中近乎无损地跨时间步流动。
- **核心创新**：遗忘门 + 输入门 + 输出门的三门结构 + 独立于隐藏状态的细胞状态通道。

## 2. 发展历史

| 年代 | 里程碑 | 贡献者 | 意义 |
|:---|:---|:---|:---|
| 1991 | 发现梯度消失问题 | Hochreiter | 毕业论文中证明 RNN 在长序列上的学习失败源于梯度指数衰减 |
| 1997 | LSTM 原始论文 | Hochreiter & Schmidhuber | 提出 8 参数版本的 LSTM，包含输入门、输出门和常量误差旋转单元（CEC） |
| 1999 | 引入遗忘门 | Gers, Schmidhuber & Cummins | 在原始 LSTM 上增加遗忘门，使 LSTM 具备了主动"忘记"不相关信息的能力——这是 LSTM 成为标准的关键改进 |
| 2000 | 窥视孔连接（Peephole） | Gers & Schmidhuber | 让门控单元也能"看到"细胞状态的当前值，提供更精细的控制 |
| 2005 | BiLSTM + CTC | Graves & Schmidhuber | 将双向 LSTM 与 CTC 损失结合，在 TIMIT 语音数据集上取得当时最优结果 |
| 2013 | LSTM 在现代语音识别中突破 | Graves, Mohamed & Hinton | Deep LSTM + CTC 在语音识别上大幅超越传统 HMM-GMM 系统 |
| 2014 | Seq2Seq LSTM | Sutskever et al. | 用多层 LSTM 构建 Seq2Seq 翻译模型 |
| 2015 | Google 使用 LSTM 改进语音搜索、翻译 | Google | LSTM 成为工业级序列建模的标准工具 |
| 2017 | Transformer 出现 | Vaswani et al. | 自注意力机制逐渐取代 LSTM 在 NLP 领域的主导地位 |

## 3. 核心概念

### 3.1 细胞状态（Cell State）

细胞状态 $C_t$ 是 LSTM 最核心的设计。它是一条贯穿整个时间序列的"信息高速公路"——信息可以通过加法操作（$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$）跨越任意长度的时间步传播，避免了乘法连乘导致的梯度衰减。

### 3.2 三门结构

LSTM 通过三个 Sigmoid 门控单元精确控制信息流动：
- **遗忘门（Forget Gate）** $f_t$：决定丢弃 $C_{t-1}$ 中的哪些信息。$f_t = \sigma(\cdots)$ 输出 0~1 之间的值，1 表示"完全保留"，0 表示"完全忘记"
- **输入门（Input Gate）** $i_t$：决定将候选细胞状态 $\tilde{C}_t$ 中的哪些信息写入 $C_t$
- **输出门（Output Gate）** $o_t$：决定从 $C_t$ 中提取哪些信息生成当前隐藏状态 $h_t$

### 3.3 梯度高速公路（Gradient Highway）

LSTM 中细胞状态的梯度传递：

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

当遗忘门接近 1（模型选择"记住"）时，梯度几乎无衰减地反向传播。即使 $f_t$ 值小于 1，由于它是加法更新中的一项系数（而非纯乘法链中的因子），梯度消失的速度也比 RNN 慢得多。

### 3.4 变体关系

- **标准 LSTM**：含遗忘门、输入门、输出门的三门版本（当前最通用）
- **窥视孔 LSTM（Peephole）**：门控不仅接收 $h_{t-1}$ 和 $x_t$，还接收 $C_{t-1}$（或 $C_t$）
- **Coupled LSTM**：遗忘门和输入门耦合为 $f_t = 1 - i_t$，减少参数
- **GRU**：LSTM 的简化变体，合并遗忘门和输入门，去掉独立细胞状态

## 4. 技术原理

### 4.1 完整的 LSTM 前向传播

设 $d_h$ 为隐藏状态维度，$d_x$ 为输入维度：

**遗忘门**：
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**输入门**：
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**候选细胞状态**：
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**细胞状态更新**：
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**输出门**：
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**隐藏状态输出**：
$$h_t = o_t \odot \tanh(C_t)$$

其中 $W_{f,i,C,o} \in \mathbb{R}^{d_h \times (d_h + d_x)}$，$b_{f,i,C,o} \in \mathbb{R}^{d_h}$。

### 4.2 梯度流分析

**步骤一：细胞状态梯度**

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

这是一个逐元素的对角矩阵。关键在于：
- RNN：$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\tanh'(z_t)) \cdot W_{hh}$——包含权重矩阵乘法
- LSTM：$\frac{\partial C_t}{\partial C_{t-1}} = f_t$——不含权重矩阵，仅含门控值

**步骤二：隐藏状态梯度与细胞状态梯度的关系**

$$\frac{\partial h_t}{\partial C_t} = \text{diag}(o_t \odot \tanh'(C_t))$$

**步骤三：从 $t$ 到 $k$ 的长距离梯度**

$$\frac{\partial \mathcal{L}_t}{\partial C_k} = \frac{\partial \mathcal{L}_t}{\partial C_t} \cdot \prod_{i=k+1}^{t} f_i$$

关键洞察：$f_i \in (0, 1)$，当 $f_i \approx 1$ 时梯度得以保留；即使 $f_i$ 略小于 1，乘积 $\prod f_i$ 的衰减速度也远慢于 RNN 中 $\|W_{hh}\|^{t-k}$ 的指数衰减。更重要的是，对于每个 $i$，$f_i$ 是**由模型学习**的——模型可以学会在需要时将 $f_i$ 设置为接近 1。

### 4.3 参数量分析

标准 LSTM 的参数量：

$$4 \times [d_h \times (d_h + d_x) + d_h] = 4 \times d_h \times (d_h + d_x + 1)$$

其中 4 对应四个线性变换（遗忘门、输入门、候选状态、输出门）。例如 $d_h = 512, d_x = 128$ 时，LSTM 单层约有 1.3M 参数。

### 4.4 训练技巧

- **遗忘门偏置初始化**：将 $b_f$ 初始化为较大正数（如 1~3），使遗忘门初始值接近 1，鼓励模型在训练初期"记住"更多信息
- **梯度裁剪**：虽然 LSTM 缓解了梯度消失，但梯度爆炸仍可能发生，梯度裁剪（阈值通常 5~10）是标准做法
- **Dropout 在 RNN 中的应用**：标准 Dropout 不宜直接应用于循环连接（会破坏时序依赖），应使用基于变分推断的 recurrent dropout（Gal & Ghahramani, 2016），仅在输入-隐藏/隐藏-输出连接间应用 Dropout

## 5. 关键变体

| 变体 | 特点 | 参数量 | 适用场景 |
|:---|:---|:---|:---|
| **标准 LSTM** | 遗忘门 + 输入门 + 输出门 | $4d_h(d_h + d_x)$ | 通用场景 |
| **Peephole LSTM** | 门控额外接收 $C_{t-1}$ 或 $C_t$ | $4d_h(d_h + d_x + d_h)$ | 需精确时序控制的场景 |
| **Coupled LSTM** | $f_t = 1 - i_t$，减少一个门 | $3d_h(d_h + d_x)$ | 轻量级部署 |
| **LayerNorm LSTM** | 在每个门的输入上添加 LayerNorm | 同标准 + 少量 LN 参数 | 深层 LSTM 训练稳定 |
| **LSTM with Projection** | 在输出端添加线性投影降低维度 | 额外 $d_h \times d_{proj}$ | 大 $d_h$ 时的维度压缩 |

## 6. 优势与局限

### 优势
1. **长期依赖建模能力**：通过加法型细胞状态更新从根本上缓解了梯度消失，实践中可处理数百步的依赖
2. **灵活的信息控制**：遗忘门和输入门的独立控制使模型能精确管理记忆的写入和擦除
3. **训练稳定性**：相比简单 RNN，LSTM 的训练过程更稳定，对超参数更鲁棒
4. **广泛验证**：是过去 20 年最成功的序列模型架构，拥有大量的工程经验和优化方案

### 局限
1. **参数冗余**：参数量是简单 RNN 的 4 倍，在小数据集上容易过拟合
2. **计算开销**：每个时间步需要 4 次大矩阵乘法，计算量约为 RNN 的 4 倍
3. **串行训练瓶颈**：与所有 RNN 变体一样，训练无法沿时间轴并行化
4. **Transformer 的超越**：在 NLP 大规模预训练场景下，LSTM 的记忆能力和并行效率被 Transformer 全面超越
5. **长程退化**：虽然显著优于 RNN，但在超长序列（>1000 步）上 LSTM 仍然面临信息退化

## 7. 应用场景

| 应用领域 | 具体任务 | 配置 |
|:---|:---|:---|
| 语音识别 | 端到端语音转文字 | Deep BiLSTM + CTC |
| 机器翻译（历史） | 序列到序列翻译 | Multi-layer LSTM Seq2Seq |
| 时间序列预测 | 销售预测、电力负荷 | Stacked LSTM + 注意力 |
| 自然语言处理 | NER、情感分析、文本分类 | BiLSTM + CRF / Attention |
| 手写识别 | 在线手写轨迹识别 | MDLSTM + CTC |
| 音乐生成 | 旋律/和弦序列生成 | LSTM + 自回归解码 |

## 8. 与其他技术关系

- **与 RNN/GRU 的关系**：LSTM 是 RNN 框架中解决梯度消失问题的最成熟方案。GRU 在参数效率上更优，但 LSTM 在需要最强记忆能力的场景中仍有优势
- **与 Transformer 的关系**：LSTM 是 Transformer 出现之前序列建模的事实标准。Transformer 的自注意力机制在全局依赖建模和训练并行化方面超越了 LSTM，但 LSTM 的低成本逐 token 推理模式在边缘设备上仍有价值
- **与 Attention 机制的关系**：Seq2Seq LSTM + Attention 的组合曾是 NLP 序列转换任务的黄金管道。理解这一组合是理解"为什么需要 Transformer"的关键
- **与状态空间模型的关系**：Mamba 等 SSM 可以理解为 LSTM 的"结构化替代者"——它们都以线性复杂度处理序列并维护内部状态，但 SSM 通过结构化设计和并行扫描实现了训练并行化

## 9. 前沿发展

尽管 LSTM 的影响力已被 Transformer 超越，新一轮研究正在探索 LSTM 核心思想的现代化实现：

- **xLSTM**（Beck et al., 2024）：Sepp Hochreiter 团队提出的"The xLSTM"，引入指数门控（Exponential Gating）和矩阵记忆（Matrix Memory），将 LSTM 的设计思想扩展到大规模语言模型场景，展示了与传统 LSTM 截然不同的性能潜力
- **mLSTM**：xLSTM 的矩阵记忆变体，用矩阵代替向量作为细胞状态，大幅提升记忆容量，结合并行扫描实现高效训练
- **LSTM 在硬件级实现**：Google 的 TPU 和专用 AI 芯片中内置的 LSTM 硬件加速单元，使 LSTM 在推理延迟敏感的场景（如设备端语音助手）中仍具竞争力
- **与 Transformer 的混合架构**：在某些语音任务中，LSTM 层仍被用作 Transformer 的前端特征提取器，利用其对局部时序模式的敏感度

## 相关知识

- 前置：[[01_循环神经网络RNN]]、[[00_序列模型_综述]]
- 平级：[[03_门控循环单元GRU]]
- 延伸：[[04_Seq2Seq与注意力机制]]、[[../08_Transformer与注意力机制/03_Transformer架构详解]]
