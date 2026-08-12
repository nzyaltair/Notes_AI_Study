---
tags:
  - 推理工程
  - TensorRT-LLM
  - Triton
  - NVIDIA
  - 推理框架
  - AI工程
created: 2026-08-10
updated: 2026-08-10
---

# TensorRT-LLM 与 Triton

## 一句话理解

TensorRT-LLM 是 NVIDIA 面向自家 GPU 的 LLM 推理优化引擎，通过图优化、定制 kernel、FP8 量化和张量并行实现极致低延迟；Triton Inference Server 提供多模型服务编排，二者结合构成 NVIDIA 推理服务栈的黄金组合。

## 1. 概述

### 1.1 TensorRT-LLM

TensorRT-LLM 是 NVIDIA 在 TensorRT 基础上专为 LLM 推理打造的开源库（2023 年发布），核心目标是在 NVIDIA GPU 上实现最低延迟和最高吞吐的 LLM 推理。

与通用推理框架（vLLM、TGI）的区别：
- **深度硬件优化**：直接使用 CUDA C++ 编写的定制 kernel，针对每代 GPU 架构优化
- **图编译**：将模型计算图编译为优化后的引擎，运行时零解释开销
- **FP8 原生支持**：充分利用 H100 的 Transformer Engine
- **构建-部署分离**：离线构建引擎，在线推理零开销

### 1.2 Triton Inference Server

Triton 是 NVIDIA 的通用推理服务框架（2019 年开源），解决模型部署的运维问题：

- **多框架支持**：TensorRT、PyTorch、ONNX Runtime、TensorFlow 等
- **多模型管理**：单服务实例托管多个模型，版本热更新
- **动态批处理**：自动将请求拼批次
- **多 GPU/多节点**：分布式推理支持

### 1.3 二者关系

```
用户请求
   │
   ▼
┌──────────────────────────────┐
│   Triton Inference Server    │  ← 服务编排层
│   - 请求路由/版本管理        │
│   - 动态批处理               │
│   - 多模型管理               │
│   - 监控指标                 │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│      TensorRT-LLM Backend    │  ← 推理执行层
│   - 优化后的 TensorRT 引擎   │
│   - FP8 / INT8 量化          │
│   - 定制 Attention Kernel    │
│   - 张量并行                 │
│   - KV Cache 管理            │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│        NVIDIA GPU            │  ← 硬件层
│   (H100 / A100 / L40 等)     │
└──────────────────────────────┘
```

## 2. 发展历史

| 时间 | 里程碑 | 意义 |
|:---|:---|:---|
| 2017 | TensorRT 3.0 | NVIDIA 推理优化引擎，支持 CNN |
| 2019 | Triton Inference Server | 多框架推理服务开源 |
| 2022 | FasterTransformer | LLM 定制 kernel 库（TensorRT-LLM 前身） |
| 2023.08 | TensorRT-LLM 发布 | 专为 LLM 的完整推理优化方案 |
| 2023.10 | FP8 支持 | H100 原生 FP8 推理 |
| 2024.01 | In-flight Batching | 连续批处理集成 |
| 2024.06 | Speculative Decoding | 投机解码支持 |
| 2024.09 | 多 LoRA 服务 | 单引擎多 LoRA 适配 |
| 2024.12 | B200 / FP4 支持 | Blackwell 架构优化 |
| 2025 | TRT-LLM v0.15+ | 开源社区活跃，支持主流模型 |

## 3. 核心概念

### 3.1 引擎构建（Build）

TensorRT-LLM 的核心工作流是**离线构建、在线推理**：

```bash
# 离线阶段：将 HuggingFace 模型转换为 TensorRT 引擎
trtllm-build \
  --checkpoint_dir ./llama-3-8b-checkpoint \
  --output_dir ./llama-3-8b-engine \
  --gemm_plugin fp8 \
  --max_batch_size 256 \
  --max_input_len 8192 \
  --max_output_len 1024 \
  --tp_size 1
```

**构建过程**：
1. 加载模型权重和配置
2. 计算图优化（算子融合、常量折叠、布局转换）
3. Kernel 自动调优（选择最优 CUDA kernel）
4. 生成序列化引擎（.engine 文件）

**构建代价**：
- 时间：10 分钟 - 数小时（取决于模型大小和调优级别）
- 不可移植：引擎绑定特定 GPU 架构、CUDA 版本、TensorRT 版本

### 3.2 In-flight Batching

TensorRT-LLM 的连续批处理实现，与 vLLM 的 Continuous Batching 概念一致：

- 请求在不同生成时刻动态加入/退出批次
- KV Cache 管理和调度由引擎内部处理
- 支持 Prefill 和 Decode 的混合调度

### 3.3 插件系统（Plugin）

TensorRT-LLM 通过插件扩展标准 TensorRT 不支持的算子：

| 插件 | 作用 |
|:---|:---|
| `GemmPlugin` | 高效矩阵乘法（支持 FP8/INT8） |
| `AttentionPlugin` | 定制 Attention kernel（FlashAttention、PagedAttention） |
| `RMSNormPlugin` | 融合 RMSNorm kernel |
| `GatedMLPPlugin` | 融合 SwiGLU MLP |
| `QuantizePlugin` | 量化/反量化 kernel |

### 3.4 张量并行

TensorRT-LLM 原生支持多 GPU 张量并行：

```bash
# 构建时指定 TP size
trtllm-build --tp_size 4 ...

# 运行时使用 4 GPU
mpirun -np 4 python3 run.py --engine_dir ./engine
```

- 支持 NCCL 通信
- 每层前向后 All-Reduce
- 与 vLLM 的 TP 实现类似，但 kernel 级别更优化

## 4. 技术原理

### 4.1 图优化

TensorRT-LLM 在构建时执行多种图优化：

**算子融合**：
```
原始计算图:
  MatMul → BiasAdd → LayerNorm/RMSNorm → GELU/SiLU → MatMul

融合后:
  FusedMLP (单个 kernel)
```

**融合的典型组合**：
- QKV 投影 + Bias → FusedQKV
- RMSNorm + Quantize → FusedRMSNormQuant
- MatMul + Bias + Activation → FusedGatedMLP
- RoPE + Attention → FusedAttention

**常量折叠**：
- 静态已知的计算在构建时预执行
- 如 LayerNorm 的缩放因子、位置编码的预处理

### 4.2 定制 Kernel

TensorRT-LLM 的核心优势在于针对 LLM 各组件的手写优化 kernel：

**FlashAttention 集成**：
- 分块计算避免 $O(n^2)$ HBM 访问
- 在线 Softmax 增量更新
- 针对 NVIDIA GPU 的 Tensor Core 优化

**GEMM 优化**：
- 针对 FP8/INT8 的混合精度 GEMM
- 自动选择最优 tile size 和流水线深度
- 利用 CUDA Core + Tensor Core 的异步执行

**KV Cache Kernel**：
- 定制的 PagedAttention kernel
- Block 级别的访存合并（coalesced access）
- 前缀缓存的高效查找和复用

### 4.3 FP8 推理

H100 的 Transformer Engine 使 FP8 推理几乎零开销：

```python
# 构建时启用 FP8
trtllm-build \
  --gemm_plugin fp8 \
  --attention_plugin fp8 \
  --rmsnorm_plugin fp8
```

**FP8 校准**：
- 需要少量校准数据（128-1024 样本）
- 自动搜索每层的最优 FP8 格式（E4M3 或 E5M2）
- 记录缩放因子用于推理

**收益**：
- 推理速度 2× 于 FP16
- 显存占用减半
- 精度损失 < 0.5 PPL

### 4.4 量化支持

| 量化方案 | 支持方式 | 说明 |
|:---|:---|:---|
| FP8 (E4M3/E5M2) | 原生 | H100 Transformer Engine |
| INT8 (W8A8) | SmoothQuant | 需校准 |
| INT4 (W4A16) | GPTQ/AWQ | 权重量化 |
| FP4 | 原生 | B200 Blackwell |

### 4.5 Triton 动态批处理

Triton 的动态批处理与 TensorRT-LLM 的 In-flight Batching 是两层不同的批处理：

```
Triton Dynamic Batching (请求级):
  - 在请求到达时等待短时间（如 10ms），拼成批次
  - 减少小请求的单独执行开销

TensorRT-LLM In-flight Batching (迭代级):
  - 在每个解码迭代步动态调整批次
  - 请求完成即释放，新请求即加入
```

二者配合：
1. Triton 在请求层拼批次（减少调用次数）
2. TensorRT-LLM 在迭代层动态调度（最大化 GPU 利用率）

## 5. 部署与配置

### 5.1 部署流程

```
1. 下载 HuggingFace 模型
   ↓
2. 转换为 TensorRT-LLM Checkpoint
   python3 convert_checkpoint.py --model_dir ./hf-model --output_dir ./ckpt
   ↓
3. 构建 TensorRT 引擎
   trtllm-build --checkpoint_dir ./ckpt --output_dir ./engine ...
   ↓
4. 部署到 Triton
   - 配置 model.pbtxt
   - 加载引擎
   ↓
5. 通过 Triton API 服务
   HTTP/gRPC 请求 → Triton → TensorRT-LLM → GPU
```

### 5.2 Triton 模型配置

```protobuf
# model.pbtxt
name: "llama-3-8b"
backend: "tensorrtllm"
max_batch_size: 256

input [
  { name: "text_input", data_type: TYPE_STRING, dims: [ -1 ] }
]
output [
  { name: "text_output", data_type: TYPE_STRING, dims: [ -1 ] }
]

dynamic_batching {
  preferred_batch_size: [ 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 10000
}

parameters: {
  key: "tensorrt_llm_model_dir"
  value: { string: "/models/llama-3-8b/engine" }
}
```

### 5.3 关键配置项

| 配置 | 含义 | 建议 |
|:---|:---|:---|
| `max_batch_size` | 最大批处理大小 | 128-256 |
| `max_input_len` | 最大输入长度 | 按需求 |
| `max_output_len` | 最大输出长度 | 按需求 |
| `tp_size` | 张量并行大小 | = GPU 数 |
| `gemm_plugin` | GEMM 精度 | fp8 (H100) / float16 |
| `kv_cache_dtype` | KV Cache 精度 | fp8 / int8 |
| `use_paged_context_fmha` | PagedAttention | 建议开启 |

## 6. 优势与局限

### 6.1 优势

- **极致性能**：NVIDIA GPU 上的最优推理延迟，通常优于 vLLM 20-50%
- **FP8 原生**：H100 Transformer Engine 充分利用
- **图编译优化**：运行时零解释开销
- **Triton 生态**：多模型管理、版本控制、监控完善
- **生产就绪**：NVIDIA 官方维护，企业级支持

### 6.2 局限

- **仅 NVIDIA GPU**：不支持 AMD/Intel/CPU
- **构建时间长**：引擎构建需 10 分钟 - 数小时
- **不可移植**：引擎绑定 GPU 架构 + CUDA + TRT 版本
- **模型兼容性**：新模型需等待官方适配或自定义插件
- **学习曲线**：配置复杂，调试困难
- **开源滞后**：部分优化最先在闭源版本中发布

## 7. 与 vLLM 对比

| 维度 | TensorRT-LLM | vLLM |
|:---|:---|:---|
| 延迟 | ★★★★★ | ★★★ |
| 吞吐 | ★★★★★ | ★★★★★ |
| 易用性 | ★★★ | ★★★★★ |
| 硬件覆盖 | 仅 NVIDIA | NVIDIA + AMD |
| FP8 支持 | ★★★★★ | ★★★★ |
| 模型适配 | 需等待/手动 | 快速 |
| 构建时间 | 10min - 数小时 | 秒级加载 |
| 引擎可移植 | ✗ | ✓ |
| 前缀缓存 | ✓ | ✓ |
| 投机解码 | ✓ | ✓ |
| 多 LoRA | ✓ | ✓ |
| 社区活跃度 | ★★★★ | ★★★★★ |

**选型建议**：
- 极致延迟 + NVIDIA + 生产环境 → TensorRT-LLM
- 快速迭代 + 灵活部署 + 多硬件 → vLLM
- 大规模多模型服务 → Triton + TensorRT-LLM

## 8. 应用场景

| 场景 | 推荐方案 | 原因 |
|:---|:---|:---|
| 生产低延迟推理 | TRT-LLM + Triton | 最低延迟，企业级 |
| FP8 推理 (H100) | TRT-LLM | 原生 FP8 优化最好 |
| 多模型服务 | Triton + TRT-LLM | 多模型管理 |
| 快速原型 | vLLM (替代) | TRT-LLM 构建太慢 |
| 非 NVIDIA 硬件 | vLLM / TGI | TRT-LLM 不支持 |

## 9. 与其他技术关系

- [[01_LLM Serving]] — TensorRT-LLM + Triton 是 LLM Serving 的企业级方案
- [[02_KV Cache与连续批处理]] — In-flight Batching 和 PagedAttention 的实现
- [[04_量化与模型压缩]] — FP8/INT8 量化的硬件级实现
- [[05_vLLM]] — 主要竞品对比
- [[08_推理加速与硬件优化]] — TensorRT 是图编译和算子融合的代表
- [[10_云端与本地部署]] — NVIDIA GPU 云端部署的首选
- [[09_模型格式与转换/00_模型格式与转换]] — HuggingFace → TensorRT 引擎的转换流程

## 10. 前沿发展

- **TRT-LLM V1 架构**：简化构建流程，减少插件依赖
- **FP4 推理 (B200)**：Blackwell 原生 FP4 支持
- **自动构建流水线**：CI/CD 化的引擎构建和版本管理
- **Disaggregated Serving**：Prefill/Decode 分离的原生支持
- **MoE 推理优化**：专家路由和批处理的专项 kernel
- **多模态推理**：Vision-Language 模型的端到端优化
- **Serverless 冷启动**：引擎快照和快速恢复

## References

- TensorRT-LLM 官方文档: https://nvidia.github.io/TensorRT-LLM/
- TensorRT-LLM GitHub: https://github.com/NVIDIA/TensorRT-LLM
- Triton Inference Server: https://github.com/triton-inference-server/server
- NVIDIA, *TensorRT-LLM Best Practices* (2024)
- NVIDIA, *FP8 Quantization for Transformer Models* (2023)

返回 [[00_推理工程_综述|推理工程]]
