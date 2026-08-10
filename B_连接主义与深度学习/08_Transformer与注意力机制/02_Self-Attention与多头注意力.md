# Self-Attention 与多头注意力

## 一句话理解

自注意力让序列中每个位置以 $O(1)$ 路径长度直接关注所有其他位置（Q/K/V 同源），多头机制则将其投影到多个子空间并行计算后拼接，使模型同时学习多种注意力模式——二者共同构成 Transformer 的核心计算单元。

## 1. 概述

自注意力（Self-Attention）是 [[01_注意力机制原理|注意力机制]] 的特例：Query、Key、Value 均来自同一输入序列。它使序列中每个位置能够直接与所有其他位置交互，捕捉全局依赖关系，路径长度为 $O(1)$。多头注意力（Multi-Head Attention, MHA）是自注意力的扩展：将 QKV 投影到多个子空间分别计算注意力后拼接，使模型能同时学习多种注意力模式。二者共同构成 Transformer 的核心计算单元。

- **解决的问题**：RNN 通过隐藏状态逐步传递信息，路径长度 $O(n)$ 导致长程依赖退化且无法并行。自注意力实现 $O(1)$ 路径长度和完全并行化。
- **核心价值**：自注意力 + 多头机制使 Transformer 能同时建模局部和全局依赖、语法和语义关系，是现代深度学习最通用的序列建模原语。

## 2. 发展历史

| 年代 | 里程碑 | 意义 |
|:---|:---|:---|
| 2014 | Bahdanau 注意力 | 注意力作为 Seq2Seq 的辅助模块 |
| 2017 | 自注意力 + 多头注意力 | Vaswani et al. 将注意力推广为序列内建模的通用机制 |
| 2019 | 多头功能分析 | Clark et al. 发现 BERT 注意力头具有可解释的功能 |
| 2020 | 注意力头剪枝 | Michel et al. 证明大量注意力头是冗余的 |
| 2023 | GQA | Ainslie et al. 在多头和单头之间找到平衡点 |
| 2024 | MLA | DeepSeek 通过低秩压缩进一步优化多头注意力 |

## 3. 核心概念

### 3.1 自注意力（Self-Attention）

在自注意力中，输入序列 $X \in \mathbb{R}^{n \times d}$ 通过三个权重矩阵映射为 Q、K、V：

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

其中 $W_Q, W_K \in \mathbb{R}^{d \times d_k}$，$W_V \in \mathbb{R}^{d \times d_v}$。由于 Q、K、V 均来自同一输入 $X$，序列中每个位置都能"看到"所有其他位置。

### 3.2 多头注意力（Multi-Head Attention）

单头注意力只有一个注意力分布，可能无法同时捕捉多种关系（如语法依赖和语义关联）。多头注意力将输入投影到 $h$ 个子空间，每个头独立计算注意力：

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

拼接后通过输出投影：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

### 3.3 因果掩码（Causal Mask）

在自回归生成中，位置 $t$ 不应看到位置 $t' > t$ 的信息。因果掩码通过将注意力矩阵的上三角部分设为 $-\infty$ 实现：

$$\text{mask}_{i,j} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

经 softmax 后，未来位置的注意力权重为 0。

## 4. 技术原理

### 4.1 缩放点积注意力

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**逐步展开**：

1. **相似度计算**：$S = QK^T \in \mathbb{R}^{n \times n}$，$S_{ij}$ 是位置 $i$ 的 Query 与位置 $j$ 的 Key 的点积
2. **缩放**：$\hat{S} = S / \sqrt{d_k}$
3. **归一化**：$A = \text{softmax}(\hat{S})$，每行归一化为概率分布
4. **加权求和**：$O = AV \in \mathbb{R}^{n \times d_v}$

**缩放因子推导**：当 $Q, K$ 元素为均值 0、方差 1 的独立随机变量时：

$$\text{Var}(S_{ij}) = \text{Var}\left(\sum_{l=1}^{d_k} Q_{il} K_{jl}\right) = d_k$$

softmax 对大值敏感，方差 $d_k$ 导致输出趋近 one-hot（梯度趋零）。除以 $\sqrt{d_k}$ 使方差归一化为 1。

**复杂度**：
- 时间：$O(n^2 d_k)$（$QK^T$ 和 $AV$ 各为 $n \times n \times d_k$）
- 空间：$O(n^2)$（注意力矩阵 $A$）

### 4.2 多头注意力的参数分析

每个头的投影矩阵：$W_i^Q \in \mathbb{R}^{d \times d_k}$, $W_i^K \in \mathbb{R}^{d \times d_k}$, $W_i^V \in \mathbb{R}^{d \times d_v}$

输出投影：$W^O \in \mathbb{R}^{hd_v \times d}$

当 $d_k = d_v = d/h$ 时，总参数量：

$$h \times (d \cdot d_k + d \cdot d_k + d \cdot d_v) + hd_v \cdot d = h \times 3 \times \frac{d^2}{h} + d^2 = 4d^2$$

**关键结论**：多头注意力的总参数量与单头注意力（$d_k = d$）相同，但表达能力更强——多个低维注意力可以捕捉更多样的模式。

### 4.3 多头注意力的实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # QKV 投影（合并为单个矩阵提高效率）
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape  # Batch, SeqLen, d_model
        
        # 投影并 reshape 为多头
        qkv = self.qkv_proj(x)  # (B, T, 3*d_model)
        q, k, v = qkv.split(self.d_model, dim=-1)
        
        # (B, T, num_heads, d_k) -> (B, num_heads, T, d_k)
        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 因果掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # softmax 和加权求和
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, num_heads, T, d_k)
        
        # 拼接多头并投影
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)
```

### 4.4 因果掩码的实现

```python
# 下三角掩码，禁止位置 i 关注位置 j > i
seq_len = 10
mask = torch.tril(torch.ones(seq_len, seq_len))
# mask[i][j] = 1 if j <= i else 0

# 在注意力分数中应用
scores = scores.masked_fill(mask == 0, float('-inf'))
# softmax 后，mask==0 位置的权重为 0
```

因果掩码使自注意力可用于自回归生成（GPT、LLaMA），训练时所有位置可并行计算，推理时配合 KV-Cache 逐 token 生成。

### 4.5 注意力头的功能分析

研究表明，Transformer 的不同注意力头自发地学习到不同的功能（Clark et al., 2019; Olsson et al., 2022）：

| 注意力模式 | 功能 | 典型层 |
|:---|:---|:---|
| 上一 token 注意力 | 复制前一个 token | 浅层 |
| `[CLS]` / `[BOS]` 注意力 | 聚合全局信息用于分类 | 多层 |
| 句法依赖注意力 | 关注语法相关的 token（如主谓） | 中层 |
| 间接宾语注意力 | 关注间接宾语名词 | 中深层 |
| 指代消解注意力 | 关联代词与其先行词 | 深层 |

### 4.6 注意力矩阵的秩

注意力矩阵 $A = \text{softmax}(QK^T / \sqrt{d_k})$ 的秩分析揭示了自注意力的信息瓶颈：

- 理论最大秩为 $\min(n, d_k)$
- 实际中注意力矩阵往往是低秩的（大部分信息集中在前几个主成分）
- 这为低秩近似（Linformer）和线性注意力提供了理论依据

## 5. 关键方法/模型

### 5.1 标准多头注意力（MHA）

原始 Transformer 方案，每个头独立拥有 Q、K、V。参数量 $4d^2$，KV-Cache 为 $n \times h \times d_k \times 2$。代表模型：BERT、GPT-2、LLaMA 7B。

### 5.2 多查询注意力（MQA）

**论文**：Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (2019)

所有 Query 头共享一组 K/V。KV-Cache 降至 $n \times d_k \times 2$（减少 $h$ 倍）。推理速度大幅提升，但质量有一定下降。代表模型：PaLM、StarCoder。

### 5.3 分组查询注意力（GQA）

**论文**：Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* (2023)

将 $h$ 个 Query 头分为 $g$ 组，每组共享一组 K/V。当 $g=1$ 时退化为 MQA，$g=h$ 时退化为 MHA。GQA 在质量和效率间取得最优平衡。代表模型：LLaMA 2 70B、Mistral、LLaMA 3。

### 5.4 多头潜在注意力（MLA）

**论文**：DeepSeek-AI, *DeepSeek-V2* (2024)

通过低秩压缩将 K/V 压缩为潜在向量 $c_K, c_V$，推理时只缓存压缩向量。KV-Cache 缩减为 $n \times d_c \times 2$（$d_c \ll d \cdot h$），大幅降低推理内存。代表模型：DeepSeek-V2/V3。

### 5.5 KV-Cache 共享策略对比

| 变体 | 原理 | KV-Cache 大小 | 质量 | 代表模型 |
|:---|:---|:---|:---|:---|
| **MHA** | 每头独立 K/V | $n \times h \times d_k \times 2$ | 最高 | GPT-2、LLaMA 7B |
| **MQA** | 所有头共享 K/V | $n \times d_k \times 2$ | 下降 | PaLM、StarCoder |
| **GQA** | 分组共享 K/V | $n \times g \times d_k \times 2$ | 接近 MHA | LLaMA 2 70B、Mistral |
| **MLA** | 低秩压缩 K/V | $n \times d_c \times 2$ | 接近 MHA | DeepSeek-V2/V3 |

## 6. 优势与局限

### 优势

1. **全局依赖**：任意两个位置直接交互，路径长度 $O(1)$，远优于 RNN 的 $O(n)$
2. **完全并行**：序列维度无递归依赖，训练时可完全并行计算
3. **多头表达力**：多个子空间并行关注不同模式，无需增加参数量
4. **统一架构**：自注意力、交叉注意力、因果注意力统一于 QKV 框架

### 局限

1. **二次复杂度**：$O(n^2)$ 的时间和空间复杂度限制长序列应用
2. **位置无关**：自注意力本身是排列不变的，需要 [[04_位置编码|位置编码]]
3. **内存瓶颈**：注意力矩阵 $n \times n$ 在长序列下占用大量显存
4. **注意力头冗余**：Michel et al. (2019) 证明可移除大量头而不影响性能
5. **推理串行性**：自回归生成时仍需逐 token 计算，KV-Cache 随序列增长

## 7. 应用场景

| 场景 | 注意力配置 | 说明 |
|:---|:---|:---|
| 编码器（BERT） | 双向 MHA | 无掩码，每个位置看到所有位置 |
| 解码器（GPT） | 因果 MHA | 下三角掩码，自回归生成 |
| 编码器-解码器（T5） | 因果 MHA + 交叉 MHA | 解码器先自注意力再交叉注意力 |
| 多模态融合 | 交叉注意力 | Q 来自文本，K/V 来自图像/音频 |
| 长序列 | 滑动窗口 + GQA | 局部窗口注意力 + 分组共享 KV |

## 8. 与其他技术关系

- **与 [[01_注意力机制原理|注意力机制原理]] 的关系**：自注意力是注意力机制的特例（Q=K=V 来自同一序列）
- **与 [[03_Transformer架构详解|Transformer 架构]] 的关系**：多头注意力是 Transformer 的核心子层
- **与 [[04_位置编码|位置编码]] 的关系**：自注意力排列不变，位置编码为其注入顺序信息
- **与 [[05_高效注意力机制|高效注意力]] 的关系**：GQA、MLA、FlashAttention 均为优化多头注意力的方法
- **与 KV-Cache 的关系**：多头注意力的 K/V 缓存是推理优化的核心问题，参见 [[12_大模型推理与优化/01_KV-Cache机制]]

## 9. 前沿发展

- **注意力头功能定位**：机械可解释性研究正在系统性地识别每个注意力头的精确功能（如 Indirect Object Identification、copy head、previous-token head）
- **注意力头剪枝**：基于重要性评分自动移除冗余头，减少推理开销
- **动态头数**：根据输入复杂度动态调整激活的注意力头数量
- **KV-Cache 压缩**：MLA 之后，量化 KV-Cache（如 KIVI、KVQuant）进一步降低推理内存
- **注意力与 SSM 融合**：Jamba 等混合架构交替使用注意力层和 SSM 层，兼顾全局建模和线性效率

## 常见问题

- **为什么缩放因子是 $\sqrt{d_k}$ 而不是 $d_k$？** 当 $d_k$ 较大时，$QK^T$ 的方差约为 $d_k$，点积数量级变大导致 softmax 进入饱和区（梯度趋零）。除以 $\sqrt{d_k}$ 使方差恢复为 1，梯度保持健康。除以 $d_k$ 会过度缩小，使分布过于平坦、损失区分度。
- **头数越多越好吗？** 不是。头数 $h$ 增加则每头维度 $d_k = d_{model}/h$ 减小，表达能力下降；头数过少则子空间不足。经验上 $d_k$ 在 64 左右效果较好（如 $d_{model}=512, h=8$）。
- **MQA/GQA/MLA 会不会损失质量？** MQA 质量下降较明显，GQA 几乎无损（LLaMA 2 70B 验证），MLA 通过低秩压缩+上投影恢复甚至超越 MHA 质量（DeepSeek-V2）。权衡的是 KV-Cache 显存与质量。
- **多头能否用一次大矩阵乘替代？** 实现上确实将 $h$ 个头的投影拼成大矩阵 $W^Q/W^K/W^V \in \mathbb{R}^{d \times d}$ 一次计算，再 reshape 成多头。多头的"多"体现在 reshape 后分组计算注意力的逻辑结构。

## 相关知识

- [[01_注意力机制原理]] — 注意力基础与类型
- [[10_大语言模型核心架构/02_注意力机制]] — LLM 视角（MQA/GQA/MLA）
- [[04_位置编码]] — 注意力的位置信息补充
- [[05_高效注意力机制]] — 高效注意力变体
- [[12_大模型推理与优化/01_KV-Cache机制]] — 推理时的 K/V 缓存

## References

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017. — 自注意力与多头注意力的提出
- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, 2019. — MQA
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, EMNLP 2023. — GQA
- DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, 2024. — MLA
- Elhage et al., *A Mathematical Framework for Transformer Circuits*, Anthropic 2021. — 注意力头功能与机械可解释性
