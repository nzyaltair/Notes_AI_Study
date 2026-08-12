---
tags:
  - 推理工程
  - SGLang
  - llama.cpp
  - 推理框架
  - AI工程
created: 2026-08-10
updated: 2026-08-10
---

# SGLang 与 llama.cpp

## 一句话理解

SGLang 通过 RadixAttention 实现高效前缀共享和结构化生成调度，适合 Agent 和工具调用等复杂工作流；llama.cpp 以轻量 C++ 实现和 GGUF 量化格式，在 CPU 和消费级 GPU 上实现大模型本地推理，两者分别覆盖云端结构化和本地/边缘部署场景。

## 1. 概述

SGLang 和 llama.cpp 代表了 LLM 推理框架的两个重要方向：

### 1.1 SGLang

SGLang（Structured Generation Language）由 UC Berkeley 提出（2024），核心创新是 **RadixAttention**——用基数树管理 KV Cache 前缀，实现跨请求的前缀共享和复用。

SGLang 解决的问题：
- **多轮对话**：每轮对话共享之前的历史，传统方案重复计算
- **Agent 工作流**：多步骤生成共享系统提示和少样本示例
- **结构化生成**：JSON/Regex 约束的解码效率低
- **Tree of Thought**：多条推理路径共享前缀

### 1.2 llama.cpp

llama.cpp 由 Georgi Gerganov 于 2023 年开源，目标是以最少的依赖在消费级硬件上运行 LLM。

llama.cpp 解决的问题：
- **零依赖**：纯 C++ 实现，无需 CUDA/PyTorch
- **跨平台**：支持 CPU、GPU（CUDA/Metal/Vulkan）、NPU
- **低资源**：4-bit 量化使 7B 模型在 8GB 内存的笔记本上运行
- **GGUF 格式**：标准化的模型存储和量化格式

## 2. 发展历史

### 2.1 SGLang

| 时间 | 里程碑 | 意义 |
|:---|:---|:---|
| 2023.10 | SGLang 发布 | RadixAttention 前缀缓存 |
| 2024.01 | 结构化生成加速 | JSON/Regex 解码优化 |
| 2024.06 | 投机解码集成 | EAGLE/Medusa 支持 |
| 2024.09 | 多 LoRA 服务 | 动态 LoRA 切换 |
| 2024.12 | V1 架构 | 性能大幅提升 |
| 2025 | DeepSeek-V3 优化 | MoE 推理优化 |

### 2.2 llama.cpp

| 时间 | 里程碑 | 意义 |
|:---|:---|:---|
| 2023.03 | llama.cpp 首次发布 | Meta LLaMA 模型的 C++ 推理 |
| 2023.04 | GGML 格式 | 标准化量化模型存储 |
| 2023.06 | 4-bit 量化 (q4_0) | 7B 模型可在 8GB 内存运行 |
| 2023.08 | GGUF 格式 | 替代 GGML，支持元数据和扩展 |
| 2023.10 | k-quants 量化 | q4_K_M 等更优量化方案 |
| 2024.01 | Metal 后端优化 | Apple Silicon 加速 |
| 2024.06 | Flash Attention | 推理 Attention 优化 |
| 2024.12 | 投机解码 | 基于自模型的 Draft |
| 2025 | Vulkan 后端 | 跨平台 GPU 加速 |

## 3. SGLang 核心概念

### 3.1 RadixAttention

RadixAttention 是 SGLang 的核心创新，使用基数树（Radix Tree）管理 KV Cache：

```
Radix Tree 结构:
                    [System Prompt]
                   /                \
        [User Query A]           [User Query B]
           /        \                   |
    [Response A1] [Response A2]   [Response B1]
       /                |
 [Query A2]         [Query A3]
```

**工作原理**：
- 每个树节点存储一段 token 序列的 KV Cache
- 新请求到达时，沿树匹配最长公共前缀
- 命中的 KV Cache 直接复用，无需重新计算
- 新生成的 token 追加为新节点

**与传统前缀缓存的区别**：

| 方案 | 匹配方式 | 支持中间共享 | 适合场景 |
|:---|:---|:---|:---|
| vLLM Prefix Caching | 前缀哈希 | ✗ 仅前缀 | 固定系统提示 |
| SGLang RadixAttention | 树形匹配 | ✓ 任意位置 | 多轮对话、Agent |

**示例**：
```
对话 1: [System] [Q1] [A1] [Q2] [A2]
对话 2: [System] [Q1] [A1] [Q3]      ← 共享 [System][Q1][A1]，只需计算 [Q3]
对话 3: [System] [Q4]                 ← 共享 [System]，只需计算 [Q4]
```

### 3.2 结构化生成

SGLang 原生支持约束解码，确保输出符合 JSON Schema、正则表达式等格式：

```python
@sgl.function
def multi_step_agent(s, question):
    # 步骤 1：分析（共享系统提示 KV Cache）
    s += "You are a helpful assistant. Analyze: " + question
    s += sgl.gen("analysis", max_tokens=200)

    # 步骤 2：结构化输出（JSON 约束）
    s += "Based on the analysis, output JSON:"
    s += sgl.gen("result", regex=r'\{.*\}')

    # 步骤 3：多条路径（Tree of Thought，共享前缀）
    for approach in ["direct", "decompose", "analogy"]:
        s += f"\nApproach ({approach}):"
        s += sgl.gen(f"answer_{approach}", max_tokens=100)
```

**优化点**：
- 所有步骤共享同一 KV Cache 树
- JSON/Regex 约束在 kernel 级别实现，不增加显著开销
- 多条路径共享前缀，避免重复 Prefill

### 3.3 并发与前缀共享调度

SGLang 的调度器优先处理与前缀树有高命中率的请求：

```
等待队列: [Req-A (命中10), Req-B (命中100), Req-C (命中5)]
                                ↑ 优先调度
```

- 命中长度越长，节省的 Prefill 计算越多
- 避免低命中请求挤占高命中请求的资源

## 4. llama.cpp 核心概念

### 4.1 GGUF 格式

GGUF（GPT-Generated Unified Format）是 llama.cpp 定义的模型存储格式：

```
GGUF 文件结构:
┌──────────────────────────┐
│ Magic Number (GGUF)      │
├──────────────────────────┤
│ Metadata                 │  ← 模型信息、量化类型、架构参数
│ - name, architecture     │
│ - context_length         │
│ - embedding_length       │
│ - block_count            │
│ - quantization_version   │
├──────────────────────────┤
│ Tensor Data              │  ← 量化后的权重
│ - token_embd.weight      │
│ - blk.0.attn_q.weight    │
│ - blk.0.attn_k.weight    │
│ - blk.0.attn_v.weight    │
│ - blk.0.ffn_up.weight    │
│ - ...                    │
└──────────────────────────┘
```

**优势**：
- 单文件存储，便于分发
- 包含完整元数据，无需额外配置
- 支持多种量化精度
- 向后兼容新量化方案

### 4.2 量化方案

llama.cpp 的 k-quants 量化系列：

| 量化方案 | 比特/token | 模型大小 (7B) | 质量 | 速度 |
|:---|:---|:---|:---|:---|
| Q8_0 | 8-bit | 7.2 GB | 最好 | 最快 |
| Q6_K | 6-bit | 5.5 GB | 极好 | 快 |
| Q5_K_M | 5-bit | 4.8 GB | 好 | 快 |
| **Q4_K_M** | 4.5-bit | 4.1 GB | 好 | 快 |
| Q4_0 | 4-bit | 3.8 GB | 中 | 最快 |
| Q3_K_M | 3.5-bit | 3.3 GB | 中 | 中 |
| Q2_K | 2.5-bit | 2.7 GB | 差 | 中 |

**Q4_K_M 是最常用的方案**：
- 7B 模型仅 4.1 GB，8GB 内存可运行
- 质量接近 FP16
- 速度优化最好

**k-quants 的核心思想**：
- 将权重按组量化，不同组使用不同精度
- 重要层（如 attention）使用更高精度
- 非重要层使用更低精度
- 混合比特实现精度-大小的最佳平衡

### 4.3 多后端支持

| 后端 | 硬件 | 特点 |
|:---|:---|:---|
| CPU | x86/ARM | 零依赖，AVX2/AVX512/NEON 优化 |
| CUDA | NVIDIA GPU | GPU 加速，支持多卡 |
| Metal | Apple Silicon | M 系列芯片统一内存优化 |
| Vulkan | 跨平台 GPU | AMD/Intel/ARM GPU |
| SYCL | Intel GPU | Intel Arc 优化 |
| HIP | AMD GPU | ROCm 替代 |

**Apple Metal 后端**特别重要：
- M 系列芯片统一内存（Unified Memory）消除 CPU-GPU 数据传输
- M2 Ultra 192GB 统一内存可运行 70B 模型
- 能效比远高于独立 GPU

### 4.4 模型分片与Offloading

llama.cpp 支持将大模型的层在 CPU 和 GPU 间分配：

```
模型层分布 (7B 模型, 32 层, 8GB 显存):
  GPU: Layer 0-25 (26层)  ← 快速推理
  CPU: Layer 26-31 (6层)  ← 较慢但可运行

推理流程:
  Input → GPU Layer 0-25 → CPU Layer 26-31 → Output
```

- `-ngl N`：指定 N 层放到 GPU
- 每步推理在 GPU 和 CPU 间切换
- 性能取决于 GPU 显存大小和 CPU 速度

## 5. 技术原理对比

### 5.1 SGLang 架构

```
                    ┌──────────────────┐
   API Request ────▶│  Frontend (HTTP) │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Scheduler       │  ← 前缀感知调度
                    │  (Radix Tree)    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  KV Cache Manager│  ← 树形 KV Cache
                    │  (RadixAttention)│
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Model Runner    │  ← 前向传播
                    │  (CUDA / Flash   │
                    │   Attention)     │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Sampler         │  ← 约束解码
                    │  (JSON/Regex)    │
                    └──────────────────┘
```

### 5.2 llama.cpp 架构

```
                    ┌──────────────────┐
   API Request ────▶│  Server (HTTP)   │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  llama_context    │  ← 推理上下文
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Graph Executor   │  ← 计算图执行
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ CPU Kernel│  │GPU Kernel│  │Metal Kern│  ← 多后端
        │ (AVX512) │  │ (CUDA)  │  │ (Metal)  │
        └──────────┘  └──────────┘  └──────────┘
```

## 6. 优势与局限

### 6.1 SGLang

**优势**：
- 前缀共享效率最高（RadixAttention）
- 结构化生成原生支持
- 多轮对话/Agent 场景加速显著
- 与 vLLM 兼容的 API

**局限**：
- 社区较小，生态不如 vLLM
- 非前缀密集场景优势不明显
- 文档和教程较少

### 6.2 llama.cpp

**优势**：
- 零依赖，极轻量
- 跨平台，覆盖所有主流硬件
- GGUF 量化格式是本地部署标准
- 消费级硬件友好
- 活跃社区，新模型快速适配

**局限**：
- 吞吐量远低于 vLLM/TRT-LLM
- 不支持连续批处理（静态批处理为主）
- 高并发场景不适合
- 缺少企业级特性（多 LoRA、前缀缓存等）

## 7. 应用场景

| 场景 | 推荐框架 | 原因 |
|:---|:---|:---|
| Agent / 工具调用 | SGLang | 前缀共享 + 结构化生成 |
| 多轮对话服务 | SGLang | RadixAttention 前缀复用 |
| Tree of Thought | SGLang | 多路径前缀共享 |
| 本地开发测试 | llama.cpp | 零依赖，快速启动 |
| 隐私敏感本地部署 | llama.cpp | 数据不出域 |
| Apple Silicon 推理 | llama.cpp | Metal 后端最优 |
| 端侧/嵌入式 | llama.cpp | 资源占用最小 |
| 高并发 API 服务 | vLLM (替代) | 吞吐更高 |
| 极致延迟 | TRT-LLM (替代) | 延迟更低 |

## 8. 部署示例

### 8.1 SGLang 部署

```bash
# 启动 SGLang 服务
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3-8B-Instruct \
  --port 30000 \
  --tp 2 \
  --enable-radix-cache

# 结构化生成请求
curl -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "List 3 fruits in JSON format",
    "sampling_params": {"regex": "\\{.*\\}"}
  }'
```

### 8.2 llama.cpp 部署

```bash
# 下载 GGUF 模型
huggingface-cli download \
  meta-llama/Llama-3-8B-Instruct-GGUF \
  llama-3-8b-instruct-q4_k_m.gguf

# 启动服务
./server -m llama-3-8b-instruct-q4_k_m.gguf \
  --port 8080 \
  --n-gpu-layers 35 \
  --ctx-size 8192

# 或使用 Ollama（基于 llama.cpp）
ollama run llama3:8b
```

## 9. 与其他技术关系

- [[01_LLM Serving]] — SGLang 和 llama.cpp 是 LLM Serving 的两种场景补充
- [[02_KV Cache与连续批处理]] — RadixAttention 是前缀缓存的高级实现
- [[04_量化与模型压缩]] — GGUF 是本地量化部署的事实标准
- [[05_vLLM]] — vLLM 与 SGLang 在云端场景竞争，与 llama.cpp 互补
- [[06_TensorRT_LLM与Triton]] — TensorRT-LLM 在 NVIDIA 生产场景更优
- [[10_云端与本地部署]] — SGLang 适合云端，llama.cpp 适合本地/边缘

## 10. 前沿发展

### SGLang
- **DeepSeek-V3 优化**：MoE 模型的专家路由和批处理优化
- **Multi-turn KV Cache 压缩**：长对话的 KV Cache 自动压缩
- **跨请求并发优化**：更精细的前缀感知调度
- **结构化生成扩展**：支持更复杂的约束类型

### llama.cpp
- **Vulkan 后端成熟**：跨平台 GPU 加速覆盖更广
- **Flash Attention 集成**：推理 Attention 效率提升
- **投机解码**：基于自模型的 Draft 方案
- **更优量化方案**：探索 3-bit 以下的高质量量化
- **端侧 NPU 支持**：手机 NPU 的推理加速

## References

- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (2024) — SGLang 原始论文
- SGLang GitHub: https://github.com/sgl-project/sglang
- llama.cpp GitHub: https://github.com/ggerganov/llama.cpp
- GGUF 格式规范: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Gerganov, *Quantization in llama.cpp* (2023) — k-quants 量化方案

返回 [[00_推理工程_综述|推理工程]]
