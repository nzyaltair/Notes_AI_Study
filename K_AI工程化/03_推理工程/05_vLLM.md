---
tags:
  - 推理工程
  - vLLM
  - PagedAttention
  - 推理框架
  - AI工程
created: 2026-08-10
updated: 2026-08-10
---

# vLLM

## 一句话理解

vLLM 是面向 LLM 高吞吐推理的开源引擎，核心创新 PagedAttention 借鉴 OS 虚拟内存分页机制管理 KV Cache，配合连续批处理将推理吞吐提升 2-4 倍，已成为 LLM 服务化的事实标准。

## 1. 概述

vLLM 由 UC Berkeley 提出（SOSP 2023），解决传统 LLM 服务中 KV Cache 显存浪费严重、GPU 利用率低的核心问题。它通过 PagedAttention 实现按需分配的 KV Cache 管理，消除内部和外部碎片，将显存利用率从 20-40% 提升至 96%+。

vLLM 的核心定位：

- **高吞吐**：PagedAttention + Continuous Batching，吞吐量比 HuggingFace Transformers 高 14-24 倍
- **易用性**：OpenAI 兼容 API，一行命令启动服务
- **生态兼容**：支持 HuggingFace 模型、LoRA 适配器、多 GPU 并行
- **活跃社区**：GitHub Star 30k+，版本迭代快速

## 2. 发展历史

| 时间 | 版本 | 里程碑 |
|:---|:---|:---|
| 2023.06 | v0.1 | PagedAttention 论文发表，初始发布 |
| 2023.08 | v0.2 | 支持张量并行、多 GPU |
| 2023.10 | v0.3 | 前缀缓存、Chunked Prefill |
| 2024.01 | v0.4 | OpenAI 兼容 API、流式输出 |
| 2024.03 | v0.5 | 多 LoRA 服务（Punica 集成） |
| 2024.06 | v0.5.5 | 投机解码支持（EAGLE, Medusa） |
| 2024.09 | v0.6 | Vision-Language 模型支持 |
| 2024.12 | v0.7 | V1 架构重写，性能大幅提升 |
| 2025.03 | v0.8 | 专家并行（MoE 支持） |

## 3. 核心概念

### 3.1 PagedAttention

详见 [[02_KV Cache与连续批处理#4.1 PagedAttention|KV Cache 与连续批处理 - PagedAttention]]。

核心机制：
- KV Cache 按 Block（默认 16 token）分配
- 每个请求维护 Block Table（逻辑块 → 物理块映射）
- 按需分配，即时释放，消除碎片
- 显存利用率 > 96%

### 3.2 连续批处理（Continuous Batching）

vLLM 的调度器在每次迭代步（iteration）执行：
1. 移除已完成请求，释放其 KV Cache 块
2. 从等待队列接纳新请求（如果显存允许）
3. 执行 Prefill（新请求）和 Decode（旧请求）的前向传播
4. 采样并更新 token

详见 [[02_KV Cache与连续批处理#4.2 连续批处理调度|KV Cache 与连续批处理 - 连续批处理]]。

### 3.3 前缀缓存（Prefix Caching）

vLLM 自动缓存相同前缀的 KV Cache：
- 系统 Prompt、Few-shot 示例等共享前缀只需计算一次
- 基于内容寻址（hash of token sequence）
- LRU 策略逐出，显存不足时自动清理

### 3.4 Chunked Prefill

将长 Prompt 切分为固定大小的 chunk（默认 512 token），与 Decode 请求拼批次：
- 避免 Prefill 阻塞 Decode，TPOT 更稳定
- 新请求可快速开始生成（降低 TTFT）

## 4. 技术原理

### 4.1 架构总览

```
                    ┌──────────────────┐
   API Request ────▶│  OpenAI API Srv  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  AsyncLLMEngine  │  ← 异步事件循环
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │    Scheduler     │  ← 调度核心
                    │  ┌─────────────┐ │
                    │  │ Running     │ │  ← 正在解码的请求
                    │  │ Waiting     │ │  ← 等待 Prefill 的请求
                    │  │ Swapped     │ │  ← 暂时换出的请求
                    │  └─────────────┘ │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Block Manager   │  ← KV Cache 块管理
                    │  (PagedAttention)│
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  GPU 0   │  │  GPU 1   │  │  GPU N   │  ← 张量并行
        │ Worker   │  │ Worker   │  │ Worker   │
        └──────────┘  └──────────┘  └──────────┘
```

### 4.2 调度器策略

vLLM 调度器维护三个队列：

| 队列 | 描述 | 调度优先级 |
|:---|:---|:---|
| Running | 正在 Decode 的请求 | 最高（避免 TPOT 抖动） |
| Waiting | 等待 Prefill 的新请求 | 中（显存允许时接纳） |
| Swapped | 因显存不足被换出到 CPU 的请求 | 低（显存恢复后换入） |

**Preemption（抢占）策略**：
- 当显存不足时，调度器可抢占 Running 中的请求
- **Recompute**：释放请求的 KV Cache，将其放回 Waiting 队列（后续重新 Prefill）
- **Swap**：将请求的 KV Cache 换出到 CPU 内存（后续换入继续 Decode）

### 4.3 张量并行

vLLM 支持多 GPU 张量并行（Tensor Parallelism）：

```bash
# 4 卡张量并行
python -m vllm.entrypoints.api_server \
  --model meta-llama/Llama-3-70B \
  --tensor-parallel-size 4
```

- 权重按列切分到各 GPU
- 每层前向后 All-Reduce 同步
- 依赖 NCCL 通信，建议使用 NVLink 互联

### 4.4 多 LoRA 服务

vLLM 支持单基础模型同时服务多个 LoRA 适配器：

```
基础模型 (70B, FP16)
  ├── LoRA-A (rank=8) → 请求A 使用
  ├── LoRA-B (rank=16) → 请求B 使用
  ├── LoRA-C (rank=8) → 请求C 使用
  └── ...
```

- LoRA 权重在 GPU 显存中缓存
- 同一批次可混合不同 LoRA 的请求
- 基于 Punica 算法高效计算多 LoRA 的增量

### 4.5 量化支持

| 量化方案 | vLLM 支持 | 说明 |
|:---|:---|:---|
| AWQ | ✓ | 原生支持，自动检测 |
| GPTQ | ✓ | 原生支持 |
| INT8 (W8A8) | ✓ | SmoothQuant |
| FP8 | ✓ | H100 原生 |
| GGUF | 部分 | 实验性支持 |
| BitsAndBytes | ✓ | 4-bit/8-bit |

### 4.6 投机解码集成

vLLM 支持多种投机解码方案：

```bash
# 使用 EAGLE
--speculative-model "yuhuili/EAGLE-LLaMA3-Instruct-8B" \
--num-speculative-tokens 5

# 使用 Medusa
--speculative-model "lmsys/medusa-vicuna-7b-v1.3" \
--num-speculative-tokens 5
```

## 5. 部署与配置

### 5.1 基本启动

```bash
# OpenAI 兼容 API 服务
vllm serve meta-llama/Llama-3-8B-Instruct \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192
```

### 5.2 关键参数

| 参数 | 含义 | 建议 |
|:---|:---|:---|
| `--gpu-memory-utilization` | GPU 显存利用率上限 | 0.85-0.95 |
| `--max-model-len` | 最大上下文长度 | 按实际需求设置 |
| `--tensor-parallel-size` | 张量并行数 | = GPU 数 |
| `--max-num-seqs` | 最大并发请求数 | 128-256 |
| `--enforce-eager` | 禁用 CUDA Graph | 调试时用 |
| `--quantization` | 量化方案 | awq / gptq / fp8 |
| `--swap-space` | CPU 换出空间 (GB) | 4-16 |
| `--enable-prefix-caching` | 启用前缀缓存 | 建议开启 |
| `--use-v2-block-manager` | V2 块管理器 | 建议开启 |

### 5.3 监控指标

| 指标 | 含义 | 健康范围 |
|:---|:---|:---|
| `vllm:num_requests_running` | 正在运行的请求数 | — |
| `vllm:num_requests_waiting` | 等待队列长度 | < 50 |
| `vllm:num_requests_swapped` | 换出请求数 | 0 为佳 |
| `vllm:gpu_cache_usage_perc` | KV Cache 利用率 | < 95% |
| `vllm:time_to_first_token` | TTFT | < 200ms |
| `vllm:time_per_output_token` | TPOT | < 50ms |
| `vllm:e2e_request_latency` | 端到端延迟 | 取决于输出长度 |

## 6. 优势与局限

### 6.1 优势

- **高吞吐**：PagedAttention + 连续批处理，吞吐提升 2-24 倍
- **显存高效**：显存利用率 > 96%，支持更多并发
- **易部署**：OpenAI 兼容 API，一行命令启动
- **生态丰富**：支持主流模型、量化方案、LoRA
- **活跃社区**：快速迭代，新特性及时支持

### 6.2 局限

- **NVIDIA 为主**：AMD ROCm 支持在完善中，CPU 推理性能不佳
- **冷启动慢**：首次加载大模型需数十秒到数分钟
- **内存开销**：CUDA Graph 预分配显存可能较高
- **自定义模型**：非标准架构需要实现兼容层
- **低延迟优化**：极致单请求延迟不如 TensorRT-LLM

## 7. 与其他框架对比

| 特性 | vLLM | TensorRT-LLM | TGI | SGLang | llama.cpp |
|:---|:---|:---|:---|:---|:---|
| 吞吐量 | ★★★★★ | ★★★★★ | ★★★★ | ★★★★ | ★★ |
| 延迟 | ★★★ | ★★★★★ | ★★★ | ★★★ | ★★★ |
| 易用性 | ★★★★★ | ★★★ | ★★★★ | ★★★ | ★★★★★ |
| 硬件覆盖 | NVIDIA/AMD | NVIDIA | NVIDIA/AMD | NVIDIA/AMD | 全平台 |
| 量化支持 | ★★★★ | ★★★★★ | ★★★ | ★★★ | ★★★★★ |
| 前缀缓存 | ✓ | ✓ | ✓ | ✓ (RadixTree) | ✗ |
| 投机解码 | ✓ | ✓ | ✗ | ✓ | ✗ |
| 多 LoRA | ✓ | ✓ | ✓ | ✓ | ✗ |
| 结构化生成 | 部分 | ✗ | ✗ | ✓ | ✗ |

## 8. 应用场景

| 场景 | 适合 vLLM？ | 原因 |
|:---|:---|:---|
| 高并发 API 服务 | ✓ 最优 | 吞吐量最高，OpenAI 兼容 |
| 多模型/多 LoRA | ✓ | 原生多 LoRA 支持 |
| 极致低延迟 | △ 需调优 | TensorRT-LLM 更优 |
| CPU/本地推理 | ✗ | 用 llama.cpp |
| 结构化生成 | △ | SGLang 更优 |
| 研究实验 | ✓ | 易用，支持多模型 |

## 9. 与其他技术关系

- [[01_LLM Serving]] — vLLM 是 LLM Serving 的代表实现
- [[02_KV Cache与连续批处理]] — PagedAttention 和连续批处理的核心原理
- [[03_投机解码]] — vLLM 集成了 EAGLE/Medusa 投机解码
- [[04_量化与模型压缩]] — vLLM 支持 AWQ/GPTQ/FP8 等量化方案
- [[06_TensorRT_LLM与Triton]] — 竞品对比，TensorRT-LLM 延迟更优
- [[07_SGLang与llama_cpp]] — SGLang 在结构化生成上互补，llama.cpp 在本地部署上互补
- [[10_云端与本地部署]] — vLLM 是云端部署的主力引擎

## 10. 前沿发展

- **V1 架构重写**：全新的调度器和执行引擎，性能进一步提升
- **专家并行 (EP)**：MoE 模型的专家级并行推理
- **Disaggregated Serving**：Prefill/Decode 分离部署支持
- **多模态推理**：Vision-Language 模型原生支持
- **PagedAttention v2**：更细粒度的块管理和更低的元数据开销
- **Serverless 集成**：冷启动优化和权重快照恢复

## References

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023) — vLLM 原始论文
- vLLM 官方文档: https://docs.vllm.ai
- vLLM GitHub: https://github.com/vllm-project/vllm
- Punica: *Punica: Multi-Tenant LoRA Serving* (2023) — 多 LoRA 服务

返回 [[00_推理工程_综述|推理工程]]
