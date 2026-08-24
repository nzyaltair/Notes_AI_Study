# LLM Serving

## 一句话理解

将 LLM 作为多租户、流式、可扩缩的在线服务交付，核心挑战是在自回归生成的访存密集特性下平衡延迟、吞吐和成本。

## 概述

LLM Serving 不同于传统 ML 推理服务。传统模型推理是一次前向计算（请求 → 计算 → 响应），而 LLM 生成是自回归的迭代过程（prefill + 逐 token decode），具有以下独特性：

- **两阶段计算特性**：Prefill（计算密集）和 Decode（访存密集）的性能特征完全不同
- **流式输出**：用户期望看到逐 token 流式生成，而非等待完整响应
- **变长序列**：输入和输出长度都不可预测
- **KV Cache 状态管理**：每个活跃请求都占用与序列长度成正比的显存

**为什么 Serving 越来越重要**：训练是一次性成本（虽高但做完即止），推理是每天都在支付的持续成本——头部服务商每日生成的 token 量已达数万亿级，相当于每几天就"推理掉"一个前沿模型的全部训练数据量。进入 Agentic 时代后权重进一步上升：Chat 场景中生成速度超过人类阅读速度后边际收益归零，而 Agent 生成的 token 大部分并非给人阅读，而是被消费的算力——从推理中压榨出的每一分性能都没有上限。

本笔记回答的核心问题：
- LLM Serving 的架构由哪些组件组成？
- 关键性能指标有哪些？如何定义 SLO？
- 容量规划与传统服务有什么不同？

## 发展历史

| 时间 | 里程碑 | 意义 |
|---|---|---|
| 2020 | GPT-3 API | LLM 作为服务交付的开端 |
| 2022 | vLLM / PagedAttention | LLM Serving 的系统化优化 |
| 2022 | OpenAI Streaming API | 流式输出成为标准 |
| 2023 | Continuous Batching | 动态批处理大幅提升吞吐 |
| 2023 | TensorRT-LLM | NVIDIA 官方推理优化引擎 |
| 2023 | OpenAI 兼容 API | 成为事实标准接口 |
| 2024 | SGLang | 结构化生成与程序化控制 |
| 2024 | Disaggregated Serving | Prefill/Decode 分离架构 |

## 核心概念

### 两阶段计算

| 阶段 | 特性 | 瓶颈 | 并行度 |
|---|---|---|---|
| Prefill | 处理完整输入 prompt | 计算密集（GPU 算力） | 高（可并行计算所有 token） |
| Decode | 逐 token 生成 | 访存密集（显存带宽） | 低（每步只生成 1 个 token） |

这一差异是 LLM Serving 性能优化的核心出发点。

### 关键性能指标

| 指标 | 定义 | 典型 SLO |
|---|---|---|
| TTFT（Time To First Token） | 从请求到首 token 的延迟 | < 500ms |
| TPOT（Time Per Output Token） | 生成阶段每个 token 的平均时间 | < 50ms |
| E2E Latency | 从请求到完整响应的端到端延迟 | 依赖输出长度 |
| Throughput（tokens/s） | 每秒处理的总 token 数 | 越高越好 |
| GPU Utilization | GPU 计算单元利用率 | > 60% |

**指标间的取舍**：延迟与吞吐量的冲突集中在 batch 维度——小 batch 低延迟低吞吐，大 batch 高吞吐高延迟；TTFT ≈ Prefill 时间，快 TTFT 与高吞吐对 batch 大小的要求相反。压缩 KV Cache 类优化（GQA/MLA/量化）则可同时改善延迟与吞吐。定量推导详见 [[02_KV Cache与连续批处理]]。

### 容量估算

与传统服务按 QPS 估算不同，LLM Serving 的容量必须考虑：

- **输入长度分布**：影响 prefill 时间和 KV Cache 占用
- **输出长度分布**：影响 decode 时间和总生成时间
- **并发请求数**：影响 batch size 和显存压力
- **KV Cache 命中率**：前缀缓存可减少重复计算

容量估算公式：

$$\text{所需 GPU 数} = \frac{\text{峰值 tokens/s 需求}}{\text{单 GPU tokens/s 能力}} \times \text{冗余系数}$$

## 技术原理

### 服务架构

```
用户请求 → API 网关（鉴权限流）
    → 请求队列
    → 调度器（批处理决策）
    → 推理引擎（GPU 执行）
    → 流式响应
    → 用户
```

### 调度器

调度器是 LLM Serving 的核心组件，决定：

- **批处理策略**：何时组成 batch、batch 大小
- **优先级**：哪些请求优先处理
- **抢占与恢复**：显存不足时暂停低优先级请求
- **KV Cache 分配**：为新请求分配显存空间

### 流式协议

LLM 服务通常使用 SSE（Server-Sent Events）或 WebSocket 实现流式输出：

- 客户端发送请求 → 服务端立即返回 200
- 每生成一个 token → 发送一个 SSE 事件
- 生成完成 → 发送 `[DONE]` 标记

## 关键方法与模型

### 主流推理引擎对比

| 引擎 | 核心优势 | 典型场景 |
|---|---|---|
| vLLM | PagedAttention + 连续批处理 | 通用高吞吐服务 |
| TensorRT-LLM | NVIDIA 硬件极致优化 | NVIDIA GPU 生产部署 |
| SGLang | 结构化生成 + 共享前缀 | Agent / 工具调用 |
| TGI（HuggingFace） | 开箱即用 | 快速原型 |
| llama.cpp | 轻量 CPU/端侧 | 本地部署 |

### 部署模式

| 模式 | 说明 | 适用场景 |
|---|---|---|
| 单副本 | 单 GPU/多 GPU 服务一个模型 | 开发、小规模 |
| 多副本（DP） | M 个独立副本 + 负载均衡 | 高吞吐（延迟不变、吞吐 ×M） |
| 张量并行 | 单请求跨多 GPU | 大模型 |
| 流水线并行 | 模型按层切分跨 GPU | 超大模型 |

## 优势与局限

### 有效的优化

- 连续批处理 + PagedAttention 使吞吐提升 2-10 倍
- 前缀缓存减少重复 prompt 的计算
- 流式输出改善用户感知延迟

### 仍有挑战的领域

- 长上下文场景下 KV Cache 显存压力
- 多 LoRA 模型的动态加载
- 冷启动延迟（模型权重加载）
- 成本控制与质量保障的平衡

## 应用场景

- 聊天助手：流式 + 低延迟
- 代码补全：超低延迟（< 100ms TTFT）
- 批量处理：高吞吐、无延迟要求
- RAG 系统：长输入 + 中等输出
- Agent 系统：多轮调用 + 工具集成

## 与其他技术关系

- 前置：[[04_量化与模型压缩]]（量化降低推理成本）
- 核心：[[02_KV Cache与连续批处理]]（性能优化的核心机制）
- 工具：[[05_vLLM]]、[[06_TensorRT_LLM与Triton]]、[[07_SGLang与llama_cpp]]
- 下游：[[06_LLMOps与AgentOps/03_模型网关与成本治理]]（服务治理）

## 前沿发展

- **Disaggregated Serving**：Prefill 和 Decode 分离到不同 GPU 集群
- **Speculative Decoding**：小模型生成 + 大模型验证，降低延迟
- **Serverless LLM**：按请求计费、自动缩容到零
- **多模态 Serving**：图文音频混合输入输出
- **KV Cache 卸载**：将 KV Cache 存储到 CPU 内存或 SSD

## 常见问题

- **TTFT 为什么高？**：主要是 prefill 阶段处理输入 prompt 的时间。长 prompt 导致更多计算。前缀缓存可以减少重复 prompt 的 TTFT。
- **并发越高越好吗？**：不是。延迟是批大小的线性函数（KV Cache 随批线性增长），吞吐量则随批增大逼近渐近线；并发超过 GPU 显存容量后会触发抢占，导致 TPOT 升高。需按 SLO 找平衡点：低延迟交互服务用小 batch，离线批量处理用大 batch。
- **如何选择推理引擎？**：通用场景用 vLLM；NVIDIA 生产环境用 TensorRT-LLM；结构化生成用 SGLang；本地部署用 llama.cpp。

## 相关知识

- [[00_推理工程_综述]]
- [[02_KV Cache与连续批处理]]
- [[04_量化与模型压缩]]
- [[05_vLLM]]

## References

- Kwon, W. et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.
- NVIDIA TensorRT-LLM 文档: https://nvidia.github.io/TensorRT-LLM/
- vLLM 文档: https://docs.vllm.ai/
- Zhong, L. et al. "SGLang: Efficient Execution of Structured Language Model Programs." 2024.
