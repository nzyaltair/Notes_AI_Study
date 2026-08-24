---
tags:
  - 推理工程
  - KVCache
  - PagedAttention
  - ContinuousBatching
  - AI工程
created: 2026-08-10
updated: 2026-08-24
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

GQA 是质量和显存的最佳平衡点，已成为现代 LLM 的事实标准。但 GQA 论文的精度结论需打折看待——后续 DeepSeek 的实验表明其确有负面影响，任何有损压缩都应在目标任务上复测精度。

**进一步压缩 KV Cache 的架构手段**（原理详见 [[../../B_连接主义与深度学习/08_Transformer与注意力机制/05_高效注意力机制|高效注意力机制]]）：

| 手段 | 思路 | 共享维度 |
|:---|:---|:---|
| GQA | 减少每组共享的 KV 头数 | 头间共享 |
| MLA（DeepSeek-V2/V3） | 不减少 KV 头数，而是把激活投影到低维潜在向量 $c_t$，缓存只存 $c$（7168 维压至 512 维），精度接近甚至略优于 MHA | 低秩压缩 |
| CLA（跨层注意力） | 相邻层直接复用上一层的 KV Cache | 层间共享 |
| 滑动窗口注意力 | 只缓存最近 K 个 token，KV Cache 与序列长度无关 | 时间维截断 |
| 线性注意力 / SSM | 将全部历史压缩为固定大小递归状态，彻底摆脱 KV Cache | 状态压缩 |

滑动窗口的实际有效上下文比标称窗口大（信息可沿层间传递），也可与全局注意力交错混用以兼顾长程依赖。由于推理系统是访存受限的，**压缩 KV Cache 可同时改善延迟与吞吐**（二者并不总是冲突），还能腾出显存支撑更大 batch。

### 3.4 两阶段的算术强度：为什么生成是访存受限

这是理解一切推理优化的高层要点：**训练时能看到全部 token，可在序列维度并行计算；推理受自回归约束必须逐 token 生成，无法在序列维度并行**，因此很难达到高算术强度、很难喂饱算力。

符号约定：$B$（批大小/并发请求数）、$S$（输入 token 数）、$T$（输出 token 数；Prefill 时 $T=S$，Decode 时 $T=1$）、$D$（模型维度）、$F$（FFN 中间维度 $\approx 4D$）。

**MLP 层**：本质是一次大矩阵乘法，且**权重跨序列共享**——

$$\text{AI}_{MLP} \approx \frac{2B\,T\,D\,F}{2(BTD + DF + BTF + \cdots)} \;\xrightarrow{BT \ll DF}\; B \cdot T$$

- Prefill（$T=S$）：$\text{AI} = B \cdot S$，大批量 + 长序列即可达到计算受限
- Decode（$T=1$）：$\text{AI} = B$（并发请求数），batch 够大即可应付

**Attention 层**：FLOPs $\approx 2BSTD$（QKV 投影 + 注意力计算），但访存量随 KV Cache 规模线性增长，推导可得系数为 $\frac{ST}{S+T}$：

$$\text{AI}_{Attn} \approx \frac{S\,T}{S+T} \times \text{(常数)}$$

- Prefill（$T=S$）：$\text{AI} = S/2$，**与 batch 无关**——序列够长即可
- Decode（$T=1$）：$\text{AI} = \frac{S}{S+1} \approx 1$——远低于 H100 的硬件平衡点（$\approx 295$ FLOP/Byte），纯访存瓶颈

**为什么增大 B 对 Attention 无效**：MLP 权重只需从 HBM 加载一次即可处理整批序列（成本摊薄）；而 Attention 的 KV Cache 依赖 $B$——每个序列有独立缓存，增大 $B$ 只是把多个互不相干的小矩阵乘法拼在一起，本质是 batched 点积，点积的算术强度极低。这就是注意力成为 Decode 阶段根本瓶颈的原因。

| 阶段 | MLP 算术强度 | Attention 算术强度 | 结论 |
|:---|:---|:---|:---|
| Prefill | $B \cdot S$ | $S/2$ | **计算受限**（compute-bound） |
| Decode | $B$ | $\approx 1$ | **访存受限**（memory-bound） |

> 这就是"LLM 推理是访存受限的"这一说法的严格含义：只要仍使用 Transformer 架构，Decode 阶段 Attention 的低算术强度就无法根治，只能围绕它优化。

### 3.5 延迟与吞吐量的定量模型（LLaMA-2-13B on H100）

访存受限反而简化了性能分析：假设通信与计算重叠，**耗时 ≈ 需搬运的字节数 ÷ HBM 带宽**。

$$\text{总显存} = \underbrace{2 \times 13\text{B}}_{\text{参数（BF16）} \approx 26\,\text{GB}} + B \times \underbrace{(\text{单序列 KV Cache})}_{\text{随序列长度线性增长}}$$

- **延迟（秒/token）= 显存搬运量 ÷ 带宽**：是 $B$ 的线性函数（常数项是参数，线性项是 KV Cache——批越大，每步要读写的 KV Cache 越多）
- **吞吐量 = $B$ ÷ 延迟**：随 $B$ 增大而提升并逼近渐近线（参数读取成本被摊薄），最终受显存容量限制——$B$ 大到 KV Cache 超出 HBM 时无法继续扩批

LLaMA-2-13B on H100 的量级：$B=1$ 时延迟约 8 ms/token、吞吐约 124 tokens/s；扩到 $B=64$ 后延迟上升、吞吐显著提升；继续扩批则撞上显存上限。

由此得到三个推论：

1. **延迟与吞吐的取舍集中在 batch 维度**：小 batch 低延迟低吞吐，大 batch 高吞吐高延迟（"等公交"效应：等的人一起上车，单程变慢但运力更高）
2. **TTFT ≈ Prefill 时间**：追求快 TTFT 要小 batch，追求高吞吐要大 batch，服务需按 SLO 折中
3. **压缩 KV Cache 是免费午餐**：省下的显存既直接降延迟，又允许更大 batch 提吞吐——延迟和吞吐可以同时改善（GQA 实测即如此）

多副本并行是另一维度：启动 $M$ 个独立模型副本，延迟不变、吞吐提升 $M$ 倍。

### 3.6 静态批处理 vs 连续批处理

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
- **写时复制（Copy-on-Write）**：共享相同前缀的请求引用同一物理块；仅当各请求采样出不同 token（前缀分叉）时才复制拆分该块，前缀共享最大化

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

**选择性批处理（Selective Batching）**：变长请求拼批时，注意力计算依赖各序列自身长度、无法直接共享张量；而 MLP 等逐 token 计算不依赖序列长度，可将所有序列拼接成一条超长序列统一处理。这是 Orca 实现连续批处理的关键工程技巧。

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
- **写时复制**：与 PagedAttention 块管理天然配合——共享前缀只存一份，采样分叉处才拆块（见 4.1）

**同一 Prompt 多次采样**（Best-of-N、多候选回复）是另一高频场景：所有样本共享 Prompt 的 KV Cache，各自只生成独立后续——System Prompt 越长、并发样本越多，收益越大。

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

> CS336 视角（P18 Dan Fu 讲座）：推理系统的现状可用"非常早期"形容——很多今天看起来复杂的技术，10-20 年后回看可能被认为"显而易见"。以下内容融合了来自 Together AI 生产环境和 UCSD 研究实验室的一线实践。

### 8.1 Token 的完整生命周期（生产视角）

> CS336 P18 视角：从系统层面审视推理全流程，远比训练时的单循环复杂。

一次推理请求从进入到返回 token，经历多个环节：

```
请求进入 → 路由分配 GPU
  → 查 KV Cache（前缀是否命中？）
  → Prefill（计算密集）或 Decode（访存密集）
  → 跨机器拆分（张量/流水线/专家并行）
  → Token 采样 → 安全检查/停止判断 → 输出
```

**生产负载特征**（与训练截然不同）：
- **代码生成**：输入上万 token（代码库上下文），输出可能很短或包含"思考 token"
- **多轮对话 / Agent**：多回合交互，每轮新增 token 数不一，回合间可能有较长空档
- **批处理任务**：如翻译整本书——输入长、KV Cache 不重要（看一次就走）
- **交互式应用**：要求 1 秒内返回首 token，用户感知"正在思考"

### 8.2 Prefill/Decode 分离部署的深化

已有 §4.3 讨论了 Chunked Prefill，P18 进一步揭示**分离部署**的实际价值与产业实践：

| 维度 | Prefill | Decode |
|:---|:---|:---|
| 计算特性 | 计算密集（类似训练前向） | 访存密集（每步加载全部权重） |
| 耗时 | 通常比单次 Decode 久得多 | 步骤多得多（每个 token 一次） |
| 硬件理想配置 | GPU（高算力） | 专用芯片（高带宽） |

**产业实践**：
- **NVIDIA 收购 Groq**：计划用 GPU 处理 Prefill，用 Groq LPU 芯片做 Decode
- **OpenAI × Cerebras**：Cerebras 的晶圆级芯片在 Decode 阶段表现更强
- **SambaNova** 等公司也在不同环节下注

**Cache-aware Prefill-Decode Segregation**（Together AI 2024）：
- 核心洞察：多轮对话中约 90% 的请求是"预热请求"（已有历史 KV Cache），仅 10% 是全新请求
- 新请求（数千 token Prefill）不应与进行到一半的短对话 Decode 混在同一批 GPU 上
- **路由层仅改两行代码**：按缓存命中率将请求分流到不同 GPU 组
- **效果**：服务速度最高提升 40%

### 8.3 KV Cache 分层存储：GPU → CPU → SSD

> CS336 P18 视角：KV Cache 越大越好——能缓存的会话越多，能跑的任务就越多。

生产环境中 KV Cache 的存储层级与操作系统内存管理高度类似（LRU 驱逐启发式在最差情况下也仅比最优解差 2 倍）：

```
GPU HBM（热数据）→ CPU DDR（温数据）→ NVMe SSD（冷数据）
~400 cycles        ~10 μs             ~100 μs
```

**层级选择逻辑**：
- **GPU HBM**：活跃会话的 KV Cache，访问最快但容量有限
- **CPU DDR**：GPU 显存不够时的溢出层；Jensen Huang 近年开始强调 CPU 性能（上一代 CPU 成了很多任务的瓶颈——50 万美元的机器被 1000 美元 CPU 拖后腿）
- **SSD**：冷数据存储；传闻 OpenAI 大量抢购 SSD 和内存即为此场景

**驱逐与预取策略**：
- **LRU 驱逐**：长时间未访问的 KV Cache 从 GPU 逐出到 CPU/SSD
- **预测未来**：用户打开聊天应用翻出旧对话 → 大概率要针对它提问 → 提前将 KV Cache 从 SSD 加载到 GPU
- **请求恢复**：请求到来时从 CPU/SSD 取回 KV Cache，重新加载进系统

### 8.4 大规模推理的容错与调试

> CS336 P18 视角：大规模系统有个特点——小规模下正常的大规模下必然出错。

**典型生产 Bug**（概率 < 0.001%，但日处理万亿 token 时必然出现）：

| Bug 现象 | 根因 |
|:---|:---|
| 模型反复输出同一 token（"hi hi hi"） | 某个 kernel 略有错误，特定条件下 logits 变 NaN |
| 工具调用死循环（"去搜一下...去搜一下..."） | 引擎处理工具调用的逻辑与后端代码不匹配 |
| 模型突然蹦出中文字符 | kernel 中的 off-by-one 错误导致 GPU 读到未初始化内存，注意力机制将其解释为中文 token |

这些 Bug 的共同特征：**不是模型问题，而是推理引擎的 kernel/系统 Bug**——强调了推理工程质量对 AI 系统可靠性的决定性影响。

### 8.5 多节点推理与容错

**NVLink 互联与新型 GPU 集群**：
- NVIDIA Blackwell GPU 及 NVL 72：72 块 GPU 通过高速互联连接
- 万亿参数模型分摊到 72 块 GPU 上 → 需要考虑容错：连接器是塑料的（不是金属），太紧会导致 NVLink 不稳定
- 给芯片加风扇和热管理后，仍需应对单 GPU 故障场景

**核心问题**：当模型分布在 64 块 GPU 上为数百万用户服务时，一块 GPU 出故障怎么办？容错机制是大规模推理的新前沿。

### 8.6 智能体工作流 vs 批处理的架构选择

> CS336 P18 Q&A 视角：不同用例对架构的选择有显著区别。

| 用例 | KV Cache 重要性 | 理想架构 |
|:---|:---|:---|
| 智能体工作流（多轮对话） | **极高**——KV Cache 需保持"热" | 因果注意力 + GQA/MLA 压缩 |
| 大规模批处理（翻译/摘要） | 较低——每个文档只看一次 | 可用双向注意力（Encoder-only），无需生成大量 token |

DeepSeek 的 MLA 对 KV Cache 做了激进压缩，在智能体工作流中优势明显；而批处理场景下 KV Cache 重要性低，甚至可以用 BERT 类模型做一次双向注意力后输出向量。

- **Prefill/Decode 分离部署（Disaggregated Serving）**：Prefill 节点（计算密集）和 Decode 节点（访存密集）独立扩展，各自优化硬件配置（DistServe, Splitwise, Mooncake）
- **KV Cache 压缩**：通过蒸发（eviction）、合并（merging）和低秩近似压缩 KV Cache，支持超长上下文（H2O, Scissorhands）
- **跨请求 KV 共享**：多请求共享相同中间层 KV Cache，提升多租户效率
- **KV Cache 感知路由**：调度器根据 KV Cache 命中率路由请求到最优节点
- **分层 KV 存储**：GPU HBM → CPU DDR → NVMe SSD 的三级 KV Cache 存储
- **跨层 KV 共享（CLA）**：将 GQA 的"头间共享"推广到"层间共享"，相邻层复用同一份 KV Cache，沿 Pareto 前沿进一步压缩
- **推理友好的新架构**：KV Cache 与注意力的构建方式从根本上决定了推理的访存瓶颈；状态空间模型、线性注意力、扩散式非自回归生成等"为推理设计"的架构蕴含数量级的改进空间

## References

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023) — vLLM 与 PagedAttention 原始论文
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022) — 连续批处理（iteration-level scheduling）的提出
- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (2024) — RadixAttention 前缀缓存
- Aminabadi et al., *DeepSpeed-Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale* (2022) — KV Cache 卸载
- Hooper et al., *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization* (2024) — KV Cache 量化
- Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving* (OSDI 2024) — Prefill/Decode 分离

返回 [[00_推理工程_综述|推理工程]]
