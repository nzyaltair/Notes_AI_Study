# CTC 与序列标注

## 1. 概述

连接时序分类（Connectionist Temporal Classification, CTC）是由 Alex Graves 等人于 2006 年提出的损失函数和输出层设计，专为解决序列标注任务中输入-输出对齐关系未知的问题。CTC 在标签集中引入空白符（blank token），通过动态规划对输入帧到输出标签之间的所有合法对齐路径进行边际化求和，实现端到端的无需预对齐标注的训练。

- **解决的问题**：传统语音和手写识别需要人工标注每帧对应的字素（音素/字符），成本极高且对齐主观。CTC 使模型可以仅用序列级标签（"这句话说的是什么"）进行训练，由模型和算法自动推断帧-标签的对应关系。
- **核心思想**：在概率层面对所有可能的对齐路径求边际似然，训练的梯度自动分配最可能的对齐。

## 2. 发展历史

| 年代 | 里程碑 | 作者 | 意义 |
|:---|:---|:---|:---|
| 2006 | CTC 原始论文 | Graves et al. | 提出 CTC 框架，在 TIMIT 音素识别上超越当时的 HMM 基线 |
| 2008 | CTC 在手写识别上的突破 | Graves et al. | 多向 LSTM (MDLSTM) + CTC 在 IAM 手写数据库上达到 state-of-the-art |
| 2011 | CTC 在深度语音识别中的应用 | Graves, Mohamed & Hinton | Deep BiLSTM + CTC 在语音识别上取得突破 |
| 2014 | Deep Speech (百度) | Hannun et al. | 工业级端到端语音识别系统，展示了 CTC 在大规模语音识别上的可行性 |
| 2015 | Baidu Deep Speech 2 | Amodei et al. | 大规模多 GPU 训练 CTC 语音模型，识别准确率接近人类水平 |
| 2018 | LAS + CTC 联合训练 | 混合注意力与 CTC 的语音识别 | Attention + CTC 成为语音识别标准配置 |
| 2022 | Whisper (OpenAI) | 大规模弱监督语音识别 | 用 CTC 作为辅助损失训练 Transformer 编码器-解码器 |

## 3. 核心概念

### 3.1 序列标注的对齐问题

给定输入序列 $\mathbf{x} = (x_1, x_2, \ldots, x_T)$（如语音的 $T$ 帧声学特征）和目标标签序列 $\mathbf{y} = (y_1, y_2, \ldots, y_U)$（如 $U$ 个字符），其中 $U \leq T$。

核心困难：我们不知道 $x_i$ 对应 $y_j$ 中的哪一个（或对应空白/重复）。最常见的情形是一个字符可能对应多个输入帧（如"a"可能持续 200ms = 20 帧）。

### 3.2 CTC 的输出空间

CTC 在原始标签集 $\mathcal{L}$ 的基础上引入空白符 $\epsilon$（blank）：

$$\mathcal{L}' = \mathcal{L} \cup \{\epsilon\}$$

输入帧 $x_t$ 的神经网络输出是 $|\mathcal{L}'|$ 维的 softmax 概率分布。

### 3.3 对齐路径（Alignment Path）

一条对齐路径 $\pi = (\pi_1, \pi_2, \ldots, \pi_T)$ 是长度为 $T$ 的 $\mathcal{L}'$-序列。通过映射函数 $\mathcal{B}$（去除重复标签和空白符），$\pi$ 映射到目标标签序列：

- $\mathcal{B}(aa\epsilon ab) = aab$（连续重复合并，空白符跳过）
- $\mathcal{B}(\epsilon a\epsilon\epsilon b) = ab$
- 多条 $\pi$ 可以映射到同一个 $\mathbf{y}$

### 3.4 前向-后向算法

给定 $\mathbf{y}$，所有映射到它的对齐路径的概率和为：

$$p(\mathbf{y} | \mathbf{x}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{y})} p(\pi | \mathbf{x}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{y})} \prod_{t=1}^T p(\pi_t | x_t)$$

直接枚举所有 $\pi$ 是 $\mathcal{O}(2^T)$ 的指数复杂度。前向-后向算法通过动态规划将其降至 $\mathcal{O}(TU)$。

## 4. 技术原理

### 4.1 扩展标签序列

为处理重复标签和空白符，在 $\mathbf{y}$ 的每两个标签之间以及首尾插入空白符 $\epsilon$，得到扩展序列 $\mathbf{y}'$：

$$\mathbf{y}' = (\epsilon, y_1, \epsilon, y_2, \epsilon, \ldots, \epsilon, y_U, \epsilon)$$

$|\mathbf{y}'| = 2U + 1$。

### 4.2 前向变量递推

定义前向变量 $\alpha_t(s)$：在时刻 $t$，所有对齐路径到达 $\mathbf{y}'$ 中位置 $s$ 的概率和。

**初始化**：
$$\alpha_1(1) = p(\epsilon | x_1)$$
$$\alpha_1(2) = p(y_1 | x_1)$$
$$\alpha_1(s) = 0, \quad s > 2$$

**递推**（从 $t$ 到 $t+1$）：位置 $s$ 的前向概率可以从位置 $s$（保持）和 $s-1$（前进）转移而来。如果 $s$ 处的标签不同于 $s-2$ 处（非相同标签间的 blank），还可以从 $s-2$ 跳跃：

$$\alpha_t(s) = 
\begin{cases}
(\alpha_{t-1}(s) + \alpha_{t-1}(s-1)) \cdot p(\mathbf{y}'_s | x_t) & \text{if } \mathbf{y}'_s = \epsilon \text{ or } \mathbf{y}'_s = \mathbf{y}'_{s-2} \\
(\alpha_{t-1}(s) + \alpha_{t-1}(s-1) + \alpha_{t-1}(s-2)) \cdot p(\mathbf{y}'_s | x_t) & \text{otherwise}
\end{cases}$$

**最终概率**：
$$p(\mathbf{y} | \mathbf{x}) = \alpha_T(2U) + \alpha_T(2U+1)$$

### 4.3 后向变量递推

定义后向变量 $\beta_t(s)$：在时刻 $t$，从 $\mathbf{y}'$ 中位置 $s$ 出发，所有对齐路径完成剩余序列的概率和。

递推对称于前向变量（从 $T$ 逆向到 $1$），可用于高效计算梯度。

### 4.4 CTC 损失与梯度

CTC 损失为负对数似然：

$$\mathcal{L}_{\text{CTC}} = -\log p(\mathbf{y} | \mathbf{x}) = -\log \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{y})} p(\pi | \mathbf{x})$$

对输出 logit $z_t^k$（对应标签 $k$ 的未归一化分数）的梯度：

$$\frac{\partial \mathcal{L}_{\text{CTC}}}{\partial z_t^k} = p(k | x_t) - \frac{1}{p(\mathbf{y} | \mathbf{x})} \sum_{s: \mathbf{y}'_s = k} \alpha_t(s) \beta_t(s)$$

这里的关键直觉：梯度将 softmax 概率分布引导到支持目标标签 $\mathbf{y}$ 的对齐上。

### 4.5 CTC 解码

**贪婪解码**：
$$\hat{y}_t = \arg\max_k p(k | x_t)$$
然后应用 $\mathcal{B}$ 去除重复和空白。简单快速，但忽略了标签间的转移依赖。

**Beam Search 解码**：
维护 $k$ 个最佳前缀路径，每步为每个候选扩展所有可能的标签。通常结合语言模型（如 KenLM）进行重打分（Rescoring），显著提升准确率：

$$\hat{\mathbf{y}} = \arg\max_{\mathbf{y}} \left( \log p_{\text{CTC}}(\mathbf{y} | \mathbf{x}) + \alpha \log p_{\text{LM}}(\mathbf{y}) + \beta |\mathbf{y}| \right)$$

其中 $\alpha$ 和 $\beta$ 是平衡超参数（语言模型权重和长度惩罚）。

## 5. 关键模型与论文

| 论文 | 贡献 | 影响 |
|:---|:---|:---|
| Graves et al., *Connectionist Temporal Classification* (2006) | CTC 框架的原始提出 | 序列标注无需对齐的理论基础 |
| Graves et al., *A Novel Connectionist System for Unconstrained Handwriting Recognition* (2008) | MDLSTM + CTC 在手写识别上达到领先水平 | 证明了 CTC 在二维序列上的适用性 |
| Graves et al., *Speech Recognition with Deep RNNs* (2013) | Deep BiLSTM + CTC 在语音识别上的突破 | 开启了深度序列模型在语音识别中的大规模应用 |
| Hannun et al., *Deep Speech* (2014) | 工业级端到端语音识别 | CTC 工业化落地的标杆 |
| Amodei et al., *Deep Speech 2* (2015) | 大规模 GPU 集群训练的 CTC 语音模型 | 展示了 CTC 的大规模扩展性 |
| OpenAI, *Whisper* (2022) | CTC 辅助损失 + Transformer | CTC 在现代大模型中的辅助角色 |

## 6. 优势与局限

### 优势
1. **无需逐帧标注**：只需序列级标签，大幅降低标注成本
2. **端到端训练**：特征提取和序列对齐由梯度驱动联合优化
3. **概率解释**：输出 $p(\mathbf{y} | \mathbf{x})$ 是明确定义的边际概率
4. **与 RNN 天然结合**：CTC 输出层 + BiLSTM/Transformer 编码器是标准语音识别配置

### 局限
1. **条件独立性假设**：CTC 假设各帧的预测条件是独立的（$p(\pi | \mathbf{x}) = \prod p(\pi_t | x_t)$），忽略了输出标签间的依赖关系
2. **不能建模标签转移**：与 CRF 不同，CTC 自身不建模标签间的转移概率（如 "B-PER → I-PER" 的合理性）
3. **峰值分布（Spiky Distribution）**：训练易导致高度"spiky"的概率分布——大部分帧被分配给 blank，有效预测集中在少数帧
4. **长度约束**：要求 $T \geq U$，总帧数不能小于标签数
5. **被 Attention 和 Transducer 竞争**：Seq2Seq Attention 和 RNN-Transducer 在某些场景下表现更好

## 7. 应用场景

| 领域 | 任务 | 配置 |
|:---|:---|:---|
| **语音识别 (ASR)** | 语音转文字（端到端） | Deep BiLSTM/Transformer + CTC，或 CTC + Attention 联合 |
| **手写识别 (HWR)** | 在线/离线手写文字识别 | CNN-BiLSTM + CTC |
| **光学字符识别 (OCR)** | 场景文字识别 | CRNN (CNN + BiLSTM + CTC) |
| **关键词识别 (KWS)** | 唤醒词检测（"Hey Siri"） | 轻量级 LSTM + CTC |
| **手势识别** | 骨架序列 → 手势标签 | BiLSTM + CTC |
| **视频动作分割** | 未裁剪视频 → 动作序列 | 时间卷积 + CTC |

## 8. 与其他技术关系

- **与 RNN 的关系**：CTC 通常以 BiLSTM 作为骨干网络提取上下文特征，两者的结合是语音识别的经典配置
- **与 CRF 的关系**：两者都用于序列标注，但 CRF 在已知分割（已知每个位置的标签）时建模标签转移，CTC 在分割未知时同时推断对齐和标签。BiLSTM-CRF 用于 NLP 序列标注（已知边界），CTC 用于语音/手写（未知边界）
- **与 Attention 的关系**：CTC 和 Attention 是两种不同的序列对齐机制——CTC 通过边际化所有路径，Attention 通过软加权。现代语音识别中两者常联合使用（Hybrid CTC/Attention）
- **与 Transformer 的关系**：Whisper 等现代语音系统使用 Transformer Encoder + CTC 辅助损失，利用 CTC 的对齐能力辅助注意力机制的训练收敛
- **与 RNN-Transducer 的关系**：两者都是端到端 ASR 的主流框架。CTC 假设帧间条件独立，Transducer 引入预测网络建模标签间的依赖关系

## 9. 前沿发展

- **CTC + Transformer 融合**：Whisper、Conformer 等现代 ASR 架构将 CTC 作为辅助损失或联合解码组件，利用其严格的单调对齐约束来规范注意力机制
- **CTC 的改进版本**：RNN-Transducer、GTC（Graphical TC）、Star-CTC 等工作试图在保持 CTC 效率的同时引入标签间依赖建模
- **CTC 在自监督学习中**：wav2vec 2.0 等自监督语音预训练模型使用 CTC 作为微调目标，展示了 CTC 在迁移学习中的价值
- **与 SSM 的结合**：Mamba-ASR 等工作探索了用状态空间模型替代 Transformer/RNN 作为 CTC 的编码器骨干

## 相关知识

- 前置：[[01_循环神经网络RNN]]、[[05_双向RNN与深层RNN]]
- 平级：[[08_序列模型工程实践]]
- 延伸：[[../16_自然语言处理/05_序列标注模型]]
