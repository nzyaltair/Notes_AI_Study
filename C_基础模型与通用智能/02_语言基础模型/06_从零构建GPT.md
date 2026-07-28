# 从零构建 GPT

## 1. 概述

本笔记整理自 Andrej Karpathy 的系列教程（nanoGPT 项目），从最简单的二元语言模型逐步构建到完整的 GPT Transformer 架构。通过渐进式构建，深入理解 Transformer 各组件的作用机制。这是理解 [[10_大语言模型核心架构|大语言模型]] 内部原理的最佳实践路径。

**核心目标**：从验证损失 2.45 的 Bigram 模型，逐步进化到验证损失 1.48 的完整 GPT。

## 2. 发展历史

| 阶段 | 核心内容 | 来源 |
|------|---------|------|
| 2022 | Micrograd | 自动微分引擎从零实现 |
| 2022 | makemore | 字符级语言模型系列 |
| 2023 | Let's build GPT | 从 Bigram 到 Transformer 完整教程 |
| 2023 | nanoGPT | 最简 GPT 实现（~300行模型 + ~300行训练） |
| 2024 | llm.c | CUDA C 实现的 GPT-2 训练 |

## 3. 核心概念

### 语言模型的本质

语言模型是对序列概率分布的建模：

$$P(x_1, x_2, \dots, x_T) = \prod_{t=1}^T P(x_t | x_1, \dots, x_{t-1})$$

**Transformer 的本质**是**通信（自注意力）与计算（前馈网络）的交替堆叠**：
- 自注意力：让序列中每个位置直接交互
- 因果掩码：确保自回归生成的方向性
- 多头注意力：捕获不同子空间的模式
- 残差连接与层归一化：稳定深层训练

### 语言模型演进路径

```
二元语法模型（Bigram）
  → 多层感知机（MLP + 字符嵌入）
  → 激活函数与梯度（初始化、BatchNorm）
  → 手动反向传播
  → WaveNet（分层网络）
  → Transformer（GPT）
```

## 4. 技术原理

### 4.1 Bigram 模型（起点）

最简单的语言模型：仅根据前一个字符预测当前字符。

$$P(x_t | x_{t-1}) = \text{softmax}(E[x_{t-1}])$$

- 嵌入表 $E \in \mathbb{R}^{V \times V}$，直接将 token ID 映射为 logits
- 无隐藏层，无注意力
- 验证损失：~2.45（字符级 Shakespeare）

### 4.2 MLP 语言模型

在 Bigram 基础上添加隐藏层：

```python
# 输入: 前n个字符的嵌入拼接
x = embedding[input_ids]  # (batch, block_size, embed_dim)
x = x.view(batch, -1)     # 拼接
x = tanh(x @ W1 + b1)     # 隐藏层
logits = x @ W2 + b2      # 输出
```

- 引入上下文窗口（block_size > 1）
- 验证损失：~2.08

### 4.3 GPT 核心组件

完整 GPT 由以下组件构成：

#### 4.3.1 Token 嵌入 + 位置嵌入

```python
self.token_emb = nn.Embedding(vocab_size, n_embd)
self.pos_emb = nn.Embedding(block_size, n_embd)

x = token_emb(idx) + pos_emb(positions)
```

#### 4.3.2 自注意力

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

```python
Q = x @ W_Q  # (batch, seq, d_k)
K = x @ W_K
V = x @ W_V
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
```

#### 4.3.3 因果掩码

```python
# 下三角掩码，禁止位置 i 关注位置 j > i
mask = torch.tril(torch.ones(seq_len, seq_len))
scores = scores.masked_fill(mask == 0, float('-inf'))
attn = F.softmax(scores, dim=-1)
out = attn @ V
```

#### 4.3.4 多头注意力

```python
# 多头并行：将大维度拆分为 h 个子空间
Q = x @ W_Q  # (batch, seq, n_embd)
Q = Q.view(batch, seq, n_heads, head_dim).transpose(1, 2)
# 每个头独立计算注意力
```

#### 4.3.5 残差连接 + LayerNorm

```python
# Pre-LN 结构
x = x + self.attn(self.ln1(x))
x = x + self.ffn(self.ln2(x))
```

#### 4.3.6 前馈网络（FFN）

```python
self.ffn = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),
    nn.GELU(),
    nn.Linear(4 * n_embd, n_embd),
)
```

#### 4.3.7 完整 Block

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln1(x))    # 通信
        x = x + self.ffn(self.ln2(x))     # 计算
        return x
```

### 4.4 完整 GPT 模型

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(...) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
```

### 4.5 训练实践

| 配置 | 值 |
|------|---|
| 数据 | Tiny Shakespeare (~1MB, 65个字符) |
| 模型参数 | ~1000万 |
| 上下文长度 | 256 |
| Batch size | 64 |
| 学习率 | 3e-4 |
| 训练步数 | 5000 |
| 验证损失 | 1.48 |

### 4.6 分词器：BPE

BPE（Byte Pair Encoding）分词原理：

```
1. 初始化词表为所有字符
2. 统计所有相邻字节对的频率
3. 合并频率最高的字节对为新token
4. 重复直到词表达到目标大小
```

- 从字符级到子词级的过渡
- 平衡词表大小和序列长度
- 能处理未登录词（拆分为已知子词）

详见 [[10_大语言模型核心架构/04_分词与Tokenization|分词]]。

## 5. 关键阶段对比

| 阶段 | 模型 | 验证损失 | 核心改进 |
|------|------|---------|---------|
| 1 | Bigram | 2.45 | 仅 token 嵌入 |
| 2 | MLP | 2.08 | 上下文窗口 + 隐藏层 |
| 3 | GPT (nano) | 1.48 | 完整 Transformer |

## 6. 优势与局限

### 优势
- **最佳教学路径**：从简到繁，每个组件的作用清晰可见
- **极简实现**：nanoGPT 仅 ~600 行代码，无冗余抽象
- **可扩展**：同一代码架构支持从字符级到 GPT-2 规模
- **可复现**：单 GPU 即可运行

### 局限
- **教学用途**：未包含生产级优化（FlashAttention、并行训练等）
- **数据规模有限**：Tiny Shakespeare 仅 ~1MB
- **缺少关键组件**：无 RoPE、SwiGLU、RMSNorm 等现代 LLM 组件
- **无分布式训练**：大模型需自行扩展

## 7. 应用场景

- **教学工具**：理解 Transformer 内部机制的最佳实践
- **模型复现**：从零训练 GPT-2 规模模型
- **快速原型**：在小数据上验证架构改进
- **研究基础**：在此基础上实验新组件（新注意力、新位置编码等）

## 8. 与其他技术关系

- 理解 [[08_Transformer与注意力机制|Transformer与注意力机制]] 架构的最佳实践路径
- 依赖自动微分与反向传播基础（Micrograd）
- [[10_大语言模型核心架构/04_分词与Tokenization|分词]] 是 GPT 的前置步骤
- [[08_Transformer与注意力机制/04_位置编码|位置编码]] 是 GPT 的关键组件
- nanoGPT 是理解 [[10_大语言模型核心架构|大语言模型]] 内部机制的教学工具
- 预训练原理详见 [[02_预训练核心原理|预训练核心原理]]

## 9. 前沿发展

- **nanoGPT 持续维护**：支持 GPT-2 复现、性能优化
- **llm.c**：用 CUDA C 从零实现 GPT-2 训练，展示底层优化
- **教育生态**：Karpathy 的 "Zero to Hero" 系列持续更新
- **现代组件集成**：社区版本集成 RoPE、SwiGLU、FlashAttention
- **从教学到研究**：nanoGPT 作为研究新架构的基线
