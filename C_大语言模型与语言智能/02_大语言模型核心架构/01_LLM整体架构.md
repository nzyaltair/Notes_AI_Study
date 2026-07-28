# 大语言模型整体架构

## 1. 概述

大语言模型（LLM）整体架构是指构成现代大语言模型的完整技术栈设计，包括模型骨架（Decoder-only Transformer）、归一化策略、激活函数、位置编码、注意力变体、前馈网络设计以及训练范式等。理解LLM整体架构是把握模型能力边界和优化方向的基石。

现代LLM架构的核心设计哲学是：**在Scaling Laws指导下，以自回归方式预测下一个token，使模型在足够规模下涌现通用语言能力**。这一范式由GPT系列确立，经LLaMA系列精炼（RoPE + SwiGLU + RMSNorm + Pre-Norm），并在DeepSeek-V3、Llama 3、Qwen 2.5等模型中进一步演进。

## 2. 发展历史

| 时间 | 里程碑 | 架构贡献 |
|------|--------|---------|
| 2017 | Transformer（Vaswani et al.） | Encoder-Decoder架构，自注意力机制 |
| 2018 | GPT-1（117M） | 确立Decoder-only + 自回归预训练范式 |
| 2019 | GPT-2（1.5B） | 验证规模化的 zero-shot 能力 |
| 2020 | GPT-3（175B） | "Scale is all you need"，涌现能力 |
| 2022 | Chinchilla（70B） | 最优数据-参数比 $D/N \approx 20$ |
| 2023 | LLaMA / LLaMA 2 | 确立RoPE+SwiGLU+RMSNorm技术栈，开源民主化 |
| 2023 | Mixtral 8x7B | 稀疏MoE架构在开源社区普及 |
| 2024 | Llama 3 / Qwen 2 | GQA + 128K上下文，逼近闭源水平 |
| 2024 | DeepSeek-V2/V3 | MLA + 细粒度MoE + Multi-Token Prediction |
| 2024 | Mamba / Jamba | SSM架构与混合架构探索 |
| 2025 | DeepSeek-R1 | 推理模型 + MoE架构融合 |

## 3. 核心概念

### 3.1 Decoder-only 架构

现代LLM几乎统一采用Decoder-only Transformer架构，原因包括：
- **Scaling Laws友好**：Decoder-only在规模化时性能提升更可预测
- **生成统一性**：所有任务（理解、生成、推理）统一为"续写"范式
- **工程简洁性**：单一架构简化训练和推理优化

### 3.2 自回归语言建模

训练目标为预测下一个token（Next Token Prediction, NTP）：

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)$$

### 3.3 Scaling Laws

- **OpenAI Scaling Law**（Kaplan et al., 2020）：损失与参数量 $N$ 和数据量 $D$ 的幂律关系
  $$L(N, D) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D}$$
  其中 $\alpha_N \approx 0.076$，$\alpha_D \approx 0.095$

- **Chinchilla Scaling**（Hoffmann et al., 2022）：给定计算预算 $C$，最优分配为 $D/N \approx 20$，即每个参数应训练约20个token。这一发现纠正了GPT-3时期"参数量优先"的策略

- **涌现能力**（Emergent Abilities）：某些能力（如CoT推理、上下文学习）在模型规模超过阈值后突然出现。但近期研究（Schaeffer et al., 2023）对此提出质疑，认为可能是评估指标的非线性造成的假象

### 3.4 训练范式全景

```
数据收集与清洗 → 预训练（NTP）
  → 监督微调（SFT，指令跟随）
  → 偏好对齐（RLHF/DPO/KTO，人类偏好对齐）
  → 安全对齐（Red Teaming，价值对齐）
  → 部署与推理优化
```

## 4. 技术原理

### 4.1 标准LLM架构

```
Input Tokens → Embedding + RoPE → [N × Transformer Block] → RMSNorm → Output Projection
                                    ├─ Multi-Head Attention (Causal)
                                    ├─ Residual + RMSNorm
                                    ├─ SwiGLU FFN
                                    └─ Residual + RMSNorm
```

每个Transformer Block包含：
1. **注意力子层**：Multi-Head Attention + 残差连接 + 归一化
2. **前馈子层**：SwiGLU FFN + 残差连接 + 归一化

### 4.2 归一化策略

| 策略 | 位置 | 特点 | 代表模型 |
|------|------|------|---------|
| Post-LN | 残差之后 | 原始Transformer，深层不稳定 | BERT、GPT-2 |
| Pre-LN | 残差之前 | 训练更稳定，深层梯度好 | GPT-3、多数现代LLM |
| RMSNorm | Pre-Norm变体 | 去除均值中心化，计算更快 | LLaMA、Qwen、DeepSeek |

**RMSNorm**（Root Mean Square Normalization）：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \cdot \gamma$$

相比LayerNorm去除了减均值的操作，计算量减少约7%-64%，性能相当。

### 4.3 激活函数演进

| 激活函数 | 公式 | 特点 | 代表 |
|---------|------|------|------|
| ReLU | $\max(0, x)$ | 简单高效，存在死神经元 | 早期模型 |
| GELU | $x \cdot \Phi(x)$ | 平滑近似ReLU | BERT、GPT-2 |
| SwiGLU | $\text{Swish}(xW) \odot (xV)$ | 门控线性单元，性能更优 | LLaMA、现代LLM |

**SwiGLU**（Swish-Gated Linear Unit）：

$$\text{FFN}_{SwiGLU}(x) = (\text{Swish}(xW_{gate}) \odot xW_{up}) W_{down}$$

其中 $\text{Swish}(x) = x \cdot \sigma(\beta x)$。引入门控机制增强表达能力，代价是FFN参数量增加约50%（三个权重矩阵 vs 两个）。

### 4.4 位置编码：RoPE

旋转位置编码（RoPE）是现代LLM的标配，详见[[04_位置编码|位置编码]]。核心特性：
- 通过旋转矩阵将位置信息编码到Q和K中
- $\mathbf{q}_m^T \mathbf{k}_n = g(\mathbf{q}, \mathbf{k}, m-n)$，只依赖相对位置
- 支持长度外推（配合NTK-aware scaling/YaRN等技巧）

### 4.5 注意力机制变体

详见[[01_注意力机制原理|注意力机制]]。

| 变体 | 原理 | KV-Cache | 代表模型 |
|------|------|---------|---------|
| MHA | 每头独立K/V | $n \times h \times d$ | GPT-2、LLaMA 7B |
| MQA | 所有头共享K/V | $n \times d$ | PaLM、GPT-4 |
| GQA | 分组共享K/V | $n \times g \times d$ | LLaMA 2 70B、Llama 3 |
| MLA | 低秩压缩K/V | $n \times d_c$ | DeepSeek-V2/V3 |

### 4.6 混合专家模型（MoE）

将FFN替换为多个专家网络，详见[[06_混合专家模型|混合专家模型]]：

$$\text{MoE}(x) = \sum_{i \in \text{TopK}} G(x)_i \cdot E_i(x)$$

稀疏激活：总参数量大，但每次推理只用部分参数。

### 4.7 Multi-Token Prediction（MTP）

DeepSeek-V3引入的多token预测技术：

- 传统NTP每次只预测下一个token，训练信号稀疏
- MTP同时预测未来多个token，提高数据效率
- 推理时可采用Speculative Decoding加速

## 5. 关键方法/模型

### 5.1 主流开源模型架构对比

| 模型系列 | 参数量 | 架构特点 | 位置编码 | 注意力 | FFN | Norm |
|---------|--------|---------|---------|--------|-----|------|
| LLaMA 2 | 7B-70B | 标准Decoder | RoPE | MHA/GQA | SwiGLU | RMSNorm |
| LLaMA 3 | 8B-405B | GQA + 128K上下文 | RoPE | GQA | SwiGLU | RMSNorm |
| Qwen 2/2.5 | 0.5B-72B | GQA + 长上下文 | RoPE | GQA | SwiGLU | RMSNorm |
| Mistral | 7B-8x22B | 滑动窗口+GQA | RoPE | SWA+GQA | SwiGLU | RMSNorm |
| DeepSeek-V2 | 236B(A21B) | MLA + 细粒度MoE | RoPE | MLA | MoE | RMSNorm |
| DeepSeek-V3 | 671B(A37B) | MLA + MoE + MTP | RoPE | MLA | MoE | RMSNorm |

### 5.2 关键里程碑模型

- **GPT系列**：确立Decoder-only + 自回归预训练范式
- **LLaMA系列**：确立RoPE + SwiGLU + RMSNorm技术栈，推动开源生态
- **Chinchilla**：确立数据-参数最优比例 $D/N \approx 20$
- **Mixtral/DeepSeek-MoE**：推动稀疏MoE架构普及
- **DeepSeek-V3**：MLA + 细粒度MoE + MTP，性能逼近闭源模型

### 5.3 重要论文

| 论文 | 作者 | 年份 | 核心贡献 |
|------|------|------|---------|
| Attention Is All You Need | Vaswani et al. | 2017 | Transformer架构 |
| Language Models are Few-Shot Learners | Brown et al. | 2020 | GPT-3，规模化 |
| Training Compute-Optimal LLMs | Hoffmann et al. | 2022 | Chinchilla Scaling |
| LLaMA: Open and Efficient Foundation LLMs | Touvron et al. | 2023 | 开源LLM技术栈 |
| DeepSeek-V2 / V3 | DeepSeek-AI | 2024 | MLA + 细粒度MoE |

## 6. 优势与局限

### 优势

- **通用性**：单一架构覆盖理解、生成、推理等广泛任务
- **可扩展性**：性能随规模可预测提升（Scaling Laws）
- **生态成熟**：训练框架（Megatron-LM、DeepSpeed）、推理引擎（vLLM、SGLang）完善
- **统一范式**：预训练→微调→对齐的清晰路线

### 局限

- **计算成本高**：训练千亿模型需千卡GPU集群，推理成本亦显著
- **上下文受限**：$O(n^2)$ 注意力复杂度限制长序列处理
- **幻觉问题**：自回归生成的概率本质导致事实性错误
- **架构同质化**：Decoder-only统一天下，创新多样性降低

## 7. 应用场景

- **通用对话**：ChatGPT、Claude、Gemini、DeepSeek
- **代码生成**：Copilot、CodeLlama、DeepSeek-Coder
- **推理任务**：o1/o3、DeepSeek-R1 推理模型
- **多语言**：Qwen 多语言模型
- **多模态**：GPT-4V、Gemini（以LLM为核心扩展）

## 8. 与其他技术关系

- LLM架构基于[[00_Transformer与注意力机制_综述|Transformer]]，采用Decoder-only路线
- [[01_注意力机制原理|注意力机制]]和[[04_位置编码|位置编码]]是核心创新组件
- [[04_分词与Tokenization|分词与Tokenization]]是输入处理的第一步
- [[05_向量表示与Embedding|向量表示与Embedding]]建立在分词之上
- [[06_混合专家模型|混合专家模型]]提供稀疏激活的扩展路径
- [[07_长上下文与记忆机制|长上下文与记忆机制]]突破上下文窗口限制
- [[08_大模型设计模式|大模型设计模式]]总结架构与工程最佳实践
- 训练依赖分布式训练技术（数据并行、张量并行、流水线并行）
- 推理优化与系统工程化紧密相关

## 9. 前沿发展

### 9.1 架构效率优化

- **MLA（多头潜在注意力）**：DeepSeek-V2/V3通过低秩压缩大幅减少KV-Cache
- **Multi-Token Prediction**：提高训练数据效率，支持推理加速
- **DualPipe**：DeepSeek-V3的双向流水线并行，减少训练气泡

### 9.2 替代架构探索

- **状态空间模型（SSM）**：Mamba/Mamba-2，线性复杂度 $O(n)$
- **混合架构**：Jamba（SSM+Attention）、Zamba
- **线性注意力**：RetNet、RWKV、Lightning Attention

### 9.3 规模化新方向

- **推理时计算扩展**：o1/R1模型通过更长推理链提升性能
- **稀疏架构普及**：MoE成为大模型的默认选择
- **多模态统一**：架构向原生多模态演进
