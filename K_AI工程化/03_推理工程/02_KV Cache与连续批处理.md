---
tags:
  - 推理工程
  - KVCache
  - PagedAttention
  - ContinuousBatching
  - AI工程
created: 2026-08-10
updated: 2026-08-10
---

# KV Cache 与连续批处理

## 一句话理解

KV Cache 缓存注意力层已处理 token 的 Key/Value，避免每步解码重复计算前缀；连续批处理允许请求在不同生成时刻动态加入/退出批次，二者共同解决 LLM 自回归推理中"显存容量而非算力是瓶颈"的核心问题。

## 1. 概述

LLM 自回归生成的每一步解码都需要访问之前所有 token 的 Key 和 Value。如果不缓存，生成第 $n$ 个 token 时需要重新计算前 $n-1$ 个 token 的 K/V，计算量为 $O(n^2)$；缓存后每步仅需计算新 token 的 K/V，计算量降为 $O(n)$。

KV Cache 带来的核心矛盾：

- **收益**：将重复计算转化为显存读取，大幅降低每步计算量
- **代价**：显存占用随序列长度线性增长，在批处理场景下可能超过模型权重本身
- **管理挑战**：变长请求导致显存碎片，静态分配浪费严重

连续批处理（Continuous Batching）则解决了传统静态批处理的"队头阻塞"问题：不同请求的输入/输出长度差异大，最短请求完成后即可释放资源并接纳新请求，GPU 利用率从 30-40% 提升至 70-80%。

## 2. 发展历史

| 时间 | 里程碑 | 意义 |
|:---|:---|:---|
| 2017 | Transformer 提出 | 自注意力机制引入 K/V 计算，KV Cache 概念随之产生 |
| 2020 | GPT-3 服务化 | KV Cache 显存问题在大规模服务中凸显 |
| 2022 | Orca 系统 | 提出 iteration-level scheduling（迭代级调度），连续批处理概念确立 |
| 2023 | vLLM / PagedAttention (SOSP) | 借鉴 OS 虚拟内存分页管理 KV Cache，吞吐提升 2-4x |
| 2023 | 前缀缓存 | 共享相同系统提示的 KV Cache，减少重复 Prefill |
| 2023 | Chunked Prefill | 将长 Prefill 切块与 Decode 交错，降低 TTFT |
| 2024 | DistServe / Splitwise | Prefill 与 Decode 分离部署，独立扩展 |
| 2024 | KV Cache 量化与卸载 | FP8/INT4 KV Cache + CPU/SSD 卸载突破显存限制 |

## 3. 核心概念

### 3.1 KV Cache 的本质

Transformer 的注意力计算为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

在自回归生成中，每生成一个新 token，只需计算该 token 的 $Q_{new}$，而 $K$ 和 $V$ 可以复用之前所有 token 的值：

```
Step 1:  Q₁·K₁ → V₁                    → 生成 token₂
Step 2:  Q₂·[K₁,K₂] → [V₁,V₂]          → 生成 token₃
Step n:  Qₙ·[K₁,...,Kₙ] → [V₁,...,Vₙ]  → 生成 tokenₙ₊₁
         ↑          ↑
         新计算      从缓存读取
```

### 3.2 KV Cache 显存计算

$$\text{KV-Cache Size} = 2 \times n_{layers} \times n_{kv\_heads} \times d_{head} \times seq\_len \times \text{dtype\_size}$$

- 因子 2：Key 和 Value 各一份
- $n_{kv\_heads}$：使用 GQA/MQA 后可远小于 $n_{attention\_heads}$

**典型模型显存占用**（FP16，单 token）：

| 模型 | 层数 | KV Heads | Head Dim | KV Cache/token |
|:---|:---|:---|:---|:---|
| LLaMA-2-7B | 32 | 32 (MHA) | 128 | 256 KB |
| LLaMA-2-70B | 80 | 8 (GQA) | 128 | 320 KB |
| LLaMA-3-70B | 80 | 8 (GQA) | 128 | 320 KB |
| LLaMA-3-8B | 32 | 8 (GQA) | 128 | 128 KB |

**LLaMA-3-70B 示例**（FP16）：
- 单 token KV Cache：320 KB
- 上下文 8K：2.5 GB/请求
- 并发 64 请求：160 GB（超过模型权重 140 GB）

### 3.3 GQA / MQA 对 KV Cache 的影响

| 注意力机制 | KV Heads | KV Cache 大小 | 质量 | 代表模型 |
|:---|:---|:---|:---|:---|
| MHA (Multi-Head Attention) | $= n_{heads}$ | 基准 | 最好 | LLaMA-2 7B/13B |
| GQA (Grouped-Query Attention) | $< n_{heads}$, 通常 8 | $\frac{n_{kv\_heads}}{n_{heads}}$ × 基准 | 接近 MHA | LLaMA-2 70B, LLaMA-3 |
| MQA (Multi-Query Attention) | 1 | $\frac{1}{n_{heads}}$ × 基准 | 略有下降 | PaLM, Falcon |

GQA 是质量和显存的最佳平衡点，已成为现代 LLM 的事实标准。

### 3.4 静态批处理 vs 连续批处理

**静态批处理（Static Batching）**：

```
时间 →
请求A: [Prefill] [Decode] [Decode] [Decode] [Decode] [______] [______]  ← 早完成，等待
请求B: [Prefill] [Decode] [Decode] [Decode] [Decode] [Decode] [Decode]  ← 最长请求决定批次
请求C: [Prefill] [Decode] [Decode] [______] [______] [______] [______]  ← 早完成，等待
```

问题：最短请求完成后 GPU 空闲等待，显存浪费。

**连续批处理（Continuous Batching）**：

```
时间 →
请求A: [Prefill] [Decode] [Decode] [Decode] [Done]
请求B: [Prefill] [Decode] [Decode] [Decode] [Decode] [Decode] [Done]
请求C: [Prefill] [Decode] [Decode] [Done]
请求D:                    [Prefill] [Decode] [Decode] [Decode] [Done]  ← A完成后立即加入
请求E:                              [Prefill] [Decode] [Decode] [Done]  ← C完成后立即加入
```

每个解码迭代步（iteration）都可以：
1. 移除已完成的请求，释放其 KV Cache
2. 从等待队列中接纳新请求，执行 Prefill
3. 继续未完成请求的下一步 Decode

## 4. 技术原理

### 4.1 PagedAttention

PagedAttention 是 vLLM 的核心创新，借鉴操作系统虚拟内存的分页机制管理 KV Cache。

**传统连续分配的问题**：

```
显存布局（连续分配）：
┌──────────┬──────────┬──────────┬──────────┬─────┬──────────┐
│ 请求A    │ 请求B    │ 请求C    │ 请求D    │ ... │ 请求N    │
│ (预分配   │ (预分配   │ (预分配   │ (预分配   │     │ (预分配   │
│  max_len) │  max_len) │  max_len) │  max_len) │     │  max_len) │
└──────────┴──────────┴──────────┴──────────┴─────┴──────────┘
     ↑ 碎片         ↑ 内部碎片（实际长度 < max_len）
```

- **内部碎片**：预分配 max_len 但实际生成更短，浪费可达 60-80%
- **外部碎片**：请求完成后释放的空洞难以被新请求利用

**PagedAttention 的分页方案**：

```
物理块表（Block Table）：
┌─────────┬─────────┬─────────┬─────────┐
│ Block 0 │ Block 1 │ Block 2 │ Block 3 │  ← 物理块（固定大小，如 16 token）
├─────────┼─────────┼─────────┼─────────┤
│ A的tok  │ B的tok  │ A的tok  │ C的tok  │
│ 0-15    │ 0-15    │ 16-31   │ 0-15    │
├─────────┼─────────┼─────────┼─────────┤
│ Block 4 │ Block 5 │ Block 6 │ Block 7 │
├─────────┼─────────┼─────────┼─────────┤
│ B的tok  │ D的tok  │ A的tok  │ B的tok  │
│ 16-31   │ 0-15    │ 32-47   │ 32-47   │
└─────────┴─────────┴─────────┴─────────┘

逻辑视图：
请求A → Block 0 → Block 2 → Block 6 → ...（逻辑连续，物理不连续）
请求B → Block 1 → Block 4 → Block 7 → ...
请求C → Block 3 → ...
请求D → Block 5 → ...
```

**关键设计**：
- **块（Block）**：固定大小的 KV Cache 存储单元，通常 16 个 token
- **块表（Block Table）**：每个请求维护逻辑块到物理块的映射
- **按需分配**：仅在生成新 token 时分配新块
- **即时释放**：请求完成后其所有块立即归还空闲池

**性能收益**：
- 显存浪费从 60-80% 降至 < 4%
- 最大并发请求数提升 2-4 倍
- 吞吐量提升 2-4 倍

### 4.2 连续批处理调度

连续批处理的核心是 **iteration-level scheduling**（迭代级调度），而非传统的 request-level scheduling（请求级调度）。

**调度循环**：

```
while True:
    # 1. 检查已完成请求
    for req in running_batch:
        if req.is_finished():  # 遇到 EOS 或达到 max_tokens
            free_kv_cache(req)
            running_batch.remove(req)
            send_response(req)

    # 2. 接纳新请求（如果显存允许）
    while wait_queue and has_kv_capacity():
        req = wait_queue.pop()
        allocate_kv_cache(req)
        running_batch.append(req)

    # 3. 执行一步前向传播
    #    - 新请求执行 Prefill
    #    - 旧请求执行 Decode
    #    - 两者拼批次一起计算
    step(running_batch)

    # 4. 采样并追加 token
    for req in running_batch:
        token = sample(req.logits)
        req.append_token(token)
```

**调度策略权衡**：

| 策略 | 描述 | 优势 | 劣势 |
|:---|:---|:---|:---|
| FCFS（先来先服务） | 按到达顺序接纳 | 公平 | 长请求阻塞短请求 |
| 最短作业优先 (SJF) | 优先接纳预估短请求 | 降低平均延迟 | 长请求饥饿 |
| Token-level Fair Share | 按 token 配额调度 | 公平与效率平衡 | 实现复杂 |
| Prefill 优先 | 优先处理新请求 Prefill | 降低 TTFT | Decode 延迟波动 |

### 4.3 Chunked Prefill

传统调度中，Prefill 和 Decode 是互斥的：执行长 Prompt 的 Prefill 时，正在 Decode 的请求必须等待，导致 TPOT 抖动。

**Chunked Prefill 方案**：

```
传统（无 Chunked Prefill）：
  Iter 1: [======== Prefill (长prompt) ========]  ← Decode 请求全部等待
  Iter 2: [Decode] [Decode] [Decode] ...

Chunked Prefill（将 Prefill 切块）：
  Iter 1: [Prefill chunk 1] [Decode] [Decode] [Decode]  ← 交错执行
  Iter 2: [Prefill chunk 2] [Decode] [Decode] [Decode]
  Iter 3: [Prefill chunk 3] [Decode] [Decode] [Decode]
  Iter 4: [Decode new] [Decode] [Decode] [Decode]        ← Prefill 完成，新请求开始 Decode
```

- 将长 Prefill 切分为固定大小的 chunk（如 512 token）
- 每个 iteration 中，Prefill chunk 与 Decode 请求拼批次
- 消除 Prefill 对 Decode 的阻塞，TPOT 更稳定
- 代价是 Prefill 总时间略增（多次 kernel 启动开销）

### 4.4 前缀缓存（Prefix Caching）

多个请求共享相同前缀（如系统提示）时，可以复用已计算的 KV Cache。

```
请求A: [System Prompt] [User Query A] [Response A...]
请求B: [System Prompt] [User Query B] [Response B...]
请求C: [System Prompt] [User Query C] [Response C...]
       ↑ 共享前缀        ↑ 各自独立
       KV Cache 只需计算一次
```

**实现方式**：
- **哈希匹配**：对 token 序列计算哈希，匹配已有的 KV Cache 块
- **引用计数**：共享块维护引用计数，所有请求结束后才释放
- **LRU 逐出**：显存不足时按最近最少使用逐出非共享块

**SGLang 的 RadixAttention** 进一步使用基数树（Radix Tree）管理前缀：
- 树节点对应一段 token 序列的 KV Cache
- 新请求沿树匹配最长公共前缀
- 命中率高，适合多轮对话和 Agent 工作流

### 4.5 KV Cache 量化

将 KV Cache 从 FP16 量化到更低精度，直接减少显存占用和带宽压力。

| 方案 | 精度 | 显存节省 | 精度影响 | 适用场景 |
|:---|:---|:---|:---|:---|
| FP16 (基准) | 16-bit | — | — | 通用 |
| FP8 (E4M3/E5M2) | 8-bit | 2× | 极小 | H100+ |
| INT8 | 8-bit | 2× | 小 | 需校准 |
| INT4 | 4-bit | 4× | 可感知 | 长上下文 |

**KV Cache 量化的特殊挑战**：
- Key 的通道间方差大，需按通道量化（per-channel）
- Value 对异常值敏感，需保留少量高精度通道
- 长序列下量化误差累积更显著

### 4.6 KV Cache 卸载（Offloading）

当显存不足时，将部分 KV Cache 卩载到 CPU 内存或 SSD：

```
访问层级：
  GPU HBM (热数据) → CPU DDR (温数据) → SSD (冷数据)
  ~400 cycles       ~10 μs              ~100 μs
```

- **层间卸载**：将不活跃层的 KV Cache 放到 CPU，活跃层保留在 GPU
- **请求级卸载**：将暂停请求的 KV Cache 移到 CPU，恢复时再加载
- **挑战**：PCIe 带宽（~32-64 GB/s）远低于 HBM（~2-4 TB/s），需预取和重叠

## 5. 优势与局限

### 5.1 优势

- **消除重复计算**：KV Cache 将每步计算量从 $O(n^2)$ 降至 $O(n)$
- **提升 GPU 利用率**：连续批处理将利用率从 30% 提升至 70-80%
- **降低延迟**：前缀缓存消除重复 Prefill，TTFT 显著降低
- **弹性并发**：PagedAttention 按需分配，支持更多并发请求

### 5.2 局限与挑战

- **显存容量瓶颈**：KV Cache 随上下文和并发线性增长，长上下文场景尤为严重
- **管理开销**：PagedAttention 的块表管理和间接寻址引入额外开销
- **调度复杂性**：连续批处理需要精细的调度策略，FCFS/SJF/公平性难以兼顾
- **TPOT 抖动**：Prefill 与 Decode 混合调度可能导致 Decode 延迟波动
- **冷启动**：前缀缓存未命中时 TTFT 仍高；卸载恢复需预取时间

## 6. 应用场景

| 场景 | 关键技术 | 原因 |
|:---|:---|:---|
| 高并发 LLM 服务 | PagedAttention + 连续批处理 | 最大化吞吐和并发数 |
| 长上下文推理 | GQA + KV Cache 量化 + 卸载 | 突破显存容量限制 |
| 多轮对话 / Agent | 前缀缓存 + RadixAttention | 共享对话历史，减少重复计算 |
| 低延迟在线服务 | Chunked Prefill | 避免 Prefill 阻塞 Decode |
| 成本敏感部署 | KV Cache INT4 量化 | 减半显存，降低硬件成本 |
| 多 LoRA 服务 | 共享基础模型 KV Cache | 基础前缀复用，LoRA 部分独立 |

## 7. 与其他技术关系

- [[01_LLM Serving]] — KV Cache 和连续批处理是 LLM Serving 的核心技术支撑
- [[03_投机解码]] — 投机解码的批量验证依赖 KV Cache 的高效管理
- [[04_量化与模型压缩]] — KV Cache 量化是模型压缩的重要组成部分
- [[05_vLLM]] — PagedAttention 的诞生地，连续批处理的代表实现
- [[07_SGLang与llama_cpp]] — SGLang 的 RadixAttention 是前缀缓存的高级实现
- [[08_推理加速与硬件优化]] — KV Cache 的显存占用是硬件选型的关键约束
- [[../../C_基础模型与通用智能/10_推理算法/00_推理算法_综述|C-10 推理算法]] — 解码策略与 KV Cache 管理的算法基础

## 8. 前沿发展

- **Prefill/Decode 分离部署（Disaggregated Serving）**：Prefill 节点（计算密集）和 Decode 节点（访存密集）独立扩展，各自优化硬件配置（DistServe, Splitwise, Mooncake）
- **KV Cache 压缩**：通过蒸发（eviction）、合并（merging）和低秩近似压缩 KV Cache，支持超长上下文（H2O, Scissorhands）
- **跨请求 KV 共享**：多请求共享相同中间层 KV Cache，提升多租户效率
- **KV Cache 感知路由**：调度器根据 KV Cache 命中率路由请求到最优节点
- **分层 KV 存储**：GPU HBM → CPU DDR → NVMe SSD 的三级 KV Cache 存储

## References

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023) — vLLM 与 PagedAttention 原始论文
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022) — 连续批处理（iteration-level scheduling）的提出
- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (2024) — RadixAttention 前缀缓存
- Aminabadi et al., *DeepSpeed-Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale* (2022) — KV Cache 卸载
- Hooper et al., *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization* (2024) — KV Cache 量化
- Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving* (OSDI 2024) — Prefill/Decode 分离

返回 [[00_推理工程_综述|推理工程]]
