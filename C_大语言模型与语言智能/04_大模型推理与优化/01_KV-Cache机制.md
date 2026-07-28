# KV-Cache 机制

## 1. 概述

KV-Cache（Key-Value Cache）是大语言模型自回归推理中最基础也最关键的优化技术。在 Transformer 解码过程中，生成第 $t$ 个 token 需要基于前 $t-1$ 个 token 的信息计算注意力。如果不做缓存，每生成一个新 token 都要重新计算所有历史 token 的 Key 和 Value，计算复杂度为 $O(n^2)$。KV-Cache 通过缓存已计算的注意力键值对，将每步计算量降为 $O(n)$。

随着模型上下文长度从 2K 扩展到 128K+，KV-Cache 的显存占用成为新的瓶颈，催生了 PagedAttention、MLA、KV-Cache 量化、StreamingLLM 等一系列优化技术。

## 2. 发展历史

| 年代 | 里程碑 | 意义 |
|------|--------|------|
| 2017 | Transformer 架构 | 自注意力机制奠定基础 |
| 2019 | MQA（Shazeer）| 所有头共享一组 KV，缓存减少 $n_{\text{heads}}$ 倍 |
| 2020 | KV-Cache 广泛采用 | 自回归推理的标准优化 |
| 2023 | GQA（Ainslie et al.）| 分组共享 KV，MHA 与 MQA 的折中，Llama-2 采用 |
| 2023 | PagedAttention / vLLM | 分页 KV-Cache 管理，消除内存碎片 |
| 2023 | StreamingLLM（Xiao et al.）| Attention Sink + 滑动窗口，无限长度推理 |
| 2023 | H2O | 基于注意力分数的动态 KV 淘汰 |
| 2024 | DeepSeek-V2 MLA | KV-Cache 压缩到潜在向量，减少 93.3% |

## 3. 核心概念

### 3.1 注意力计算回顾

标准注意力公式：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

在自回归解码中，生成第 $t$ 个 token 时：
- **Q**：仅当前 token 的查询（维度 $1 \times d_k$）
- **K, V**：所有历史 token 的键值对（维度 $t \times d_k$ 和 $t \times d_v$）

### 3.2 KV-Cache 工作机制

1. 只计算当前 token 的 Query（Q）、Key（K）、Value（V）
2. 将当前 token 的 K 和 V 追加到缓存中
3. 用当前 Q 与缓存中所有 K 计算注意力，再与缓存中所有 V 加权求和

### 3.3 张量布局

KV-Cache 的典型张量布局为每层存储 K 和 V 两个张量：

- K: `[batch_size, num_heads, seq_len, head_dim]`
- V: `[batch_size, num_heads, seq_len, head_dim]`

共 $2 \times n_{\text{layers}}$ 个张量。

## 4. 技术原理

### 4.1 显存占用公式

$$M_{\text{KV}} = 2 \times n_{\text{layers}} \times \text{batch} \times n_{\text{KV\_heads}} \times \text{seq\_len} \times d_{\text{head}} \times \text{dtype\_size}$$

其中因子 2 表示同时缓存 K 和 V。

**示例**：Llama-2-70B（FP16, 80层, 64头, $d_{head}=128$）
- seq=2048, batch=1：$M \approx 5.4$ GB
- seq=4096, batch=64：$M \approx 345$ GB（远超模型本身）

### 4.2 注意力头共享策略

| 策略 | KV 头数 | 缓存大小 | 代表模型 |
|------|--------|---------|---------|
| MHA（标准） | $n_{\text{heads}}$ | 基准 | GPT-3, 早期模型 |
| MQA | 1 | 减少 $n_{\text{heads}}$ 倍 | PaLM, Falcon |
| GQA | $n_{\text{groups}}$ | 减少至 $n_{\text{groups}}/n_{\text{heads}}$ | Llama-2, Mistral |
| MLA | 压缩到潜在向量 | 减少约 93% | DeepSeek-V2/V3 |

**MQA**（Multi-Query Attention）：所有注意力头共享同一组 K/V，缓存最大可减少 $n_{\text{heads}}$ 倍，但精度有一定损失。

**GQA**（Grouped-Query Attention）：将注意力头分为若干组，组内共享 K/V，在 MHA 的精度和 MQA 的效率之间取得平衡。Llama-2-70B 使用 8 组 GQA。

**MLA**（Multi-head Latent Attention，DeepSeek-V2）：将 KV-Cache 压缩到低维潜在向量。推理时只缓存压缩后的潜在向量，通过上投影矩阵恢复 K/V。KV-Cache 减少 93.3%，同时保持甚至超越 MHA 的质量。

### 4.3 PagedAttention

借鉴操作系统虚拟内存管理思想，将 KV-Cache 划分为固定大小的块（Pages），通过页表管理不连续的内存块：

- 每个序列的 KV-Cache 由一组逻辑块组成，映射到物理块
- **消除内存碎片**：利用率提升 60-70%
- **支持变长序列**：不同序列可使用不同数量的页
- **支持连续批处理**：新请求可动态加入，已完成请求的页可立即回收
- **前缀共享**：相同系统 prompt 的多个请求共享 KV-Cache 前缀（如 vLLM 的 Automatic Prefix Caching）

### 4.4 KV-Cache 量化

将 KV-Cache 从 FP16 量化为低精度：

| 精度 | 显存节省 | 精度影响 | 说明 |
|------|---------|---------|------|
| FP16 → INT8 | 50% | 较小 | K 和 V 使用不同量化策略 |
| FP16 → INT4 | 75% | 中等 | 需要混合精度保护关键层 |
| FP16 → FP8 | 50% | 极小 | 硬件原生支持，H100 |

关键挑战：注意力分数对 K 的精度敏感（softmax 放大误差），V 的量化相对安全。

### 4.5 缓存驱逐策略

长上下文场景下，需驱逐部分 KV 以控制显存：

- **StreamingLLM**：保留初始 token（Attention Sink）+ 最近 token 的 KV，实现恒定内存下的无限长度推理。Attention Sink 机制发现初始 token 在注意力中承担"汇聚"作用，移除会导致注意力分数坍塌
- **H2O**（Heavy-Hitter Oracle）：基于注意力分数动态淘汰不重要的 KV，保留对生成贡献最大的 token
- **滑动窗口**（Sliding Window）：只保留最近 $W$ 个 token 的 KV，简单高效但丢失早期上下文，Mistral 采用 $W=4096$
- **SnapKV**：基于注意力模式识别重要 token，在 Prefill 阶段预筛选需保留的 KV

### 4.6 预分配与动态扩展

- **预分配策略**：序列开始时分配最大长度的缓存，内存利用率低但无分配开销
- **动态扩展策略**：按需分配，利用率高但有性能开销
- **PagedAttention 混合策略**：按页分配，兼顾利用率和性能

## 5. 关键方法与模型

- **PagedAttention**（vLLM, Kwon et al. 2023）：分页 KV-Cache 管理，吞吐量提升 2-4x
- **StreamingLLM**（Xiao et al. 2023）：Attention Sink 机制，支持无限长度推理
- **MQA**（Shazeer 2019）：多查询注意力，所有头共享 KV
- **GQA**（Ainslie et al. 2023）：分组查询注意力，Llama-2 采用
- **MLA**（DeepSeek-V2, 2024）：潜在注意力，KV-Cache 压缩 93.3%
- **H2O**（Zhang et al. 2023）：动态 KV 淘汰策略
- **SnapKV**（Li et al. 2024）：Prefill 阶段预筛选重要 KV

## 6. 优势与局限

**优势**：
- 将自回归推理每步计算量从 $O(n^2)$ 降为 $O(n)$
- 几乎无质量损失，被所有推理框架默认采用
- 与量化、分页等技术可叠加使用

**局限**：
- 显存随序列长度和 batch 线性增长，长上下文场景下成为主要瓶颈
- 缓存驱逐策略可能导致早期上下文信息丢失
- KV-Cache 量化对注意力分数精度有影响
- 预分配策略导致内存浪费，动态分配有性能开销

## 7. 应用场景

- **自回归推理**：所有 LLM 生成任务的基础优化
- **高并发服务**：PagedAttention 支持高吞吐服务，单机服务数千并发
- **长上下文推理**：缓存驱逐策略支持 100K+ token，StreamingLLM 支持无限长度流式输出
- **边缘部署**：KV-Cache 量化减少设备显存压力
- **多模态推理**：多模态模型缓存视觉和文本 token 的键值对

## 8. 与其他技术关系

- KV-Cache 是 [[08_Transformer与注意力机制/02_Self-Attention与多头注意力|注意力机制]] 在推理阶段的工程实现
- [[02_量化技术|量化]] 技术可应用于 KV-Cache 以减少显存
- PagedAttention 是 [[04_推理加速与并行]] 中的核心创新
- 缓存驱逐与 [[06_长上下文扩展]] 密切相关
- GQA/MQA/MLA 影响模型架构设计，参见 [[10_大语言模型核心架构]]
- KV-Cache 显存直接影响 [[07_成本与延迟权衡]] 中的并发能力

## 9. 前沿发展

- **MLA 普及**：DeepSeek-V2/V3 验证了潜在注意力的有效性，可能成为新一代架构标准
- **KV-Cache 卸载**：将不活跃的 KV-Cache 页卸载到 CPU 内存或 SSD，突破 GPU 显存限制
- **语义感知缓存压缩**：基于注意力模式和语义重要性自适应压缩 KV-Cache
- **跨请求 KV-Cache 共享**：在多用户场景下共享公共 prompt 的 KV-Cache
- **KV-Cache 原生量化**：FP8 KV-Cache 在 Hopper 架构上的硬件加速

## References

- Shazeer, *Fast Transformer Decoding: One Write-Head Is All You Need* (2019) — MQA
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023) — GQA
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023) — vLLM/PagedAttention
- Xiao et al., *Efficient Streaming Language Models with Attention Sinks* (2023) — StreamingLLM
- Zhang et al., *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models* (2023)
- DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* (2024) — MLA
- [vLLM 官方文档](https://docs.vllm.ai/)
