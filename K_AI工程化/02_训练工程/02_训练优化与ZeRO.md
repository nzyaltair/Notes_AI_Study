# 训练优化与 ZeRO

## 一句话理解

通过混合精度、梯度累积、激活重计算、通信重叠和 ZeRO 显存切分等技术，在有限 GPU 资源下最大化训练吞吐量并控制显存占用。

## 概述

大模型训练的瓶颈分布在计算、显存、通信和 I/O 四个维度。训练优化的目标不是单独最大化某一指标，而是在这些约束之间找到最优平衡点。

本笔记回答的核心问题：
- 每种优化技术解决了什么瓶颈？代价是什么？
- ZeRO 各阶段的原理和适用条件是什么？
- 如何用系统化指标评估优化效果？

## 发展历史

| 时间 | 技术节点 | 意义 |
|---|---|---|
| 2015 | NVIDIA 推出 FP16 训练 | 混合精度训练开端 |
| 2017 | Loss Scaling（NVIDIA） | 解决 FP16 梯度下溢 |
| 2018 | Apex / AMP | 混合精度训练框架化 |
| 2019 | BF16 支持（TPU/GPU） | 免 loss scaling 的混合精度 |
| 2020 | ZeRO（DeepSpeed） | 系统化显存切分 |
| 2020 | Activation Checkpointing | 计算换显存 |
| 2021 | FlashAttention | 注意力计算 IO 优化 |
| 2022+ | FP8 训练 | 进一步压缩计算与存储 |

## 核心概念

### 四维瓶颈

| 维度 | 瓶颈表现 | 优化方向 |
|---|---|---|
| 计算 | GPU 算力利用率低 | 混合精度、算子融合 |
| 显存 | OOM 无法容纳模型 | ZeRO、激活重计算、Offload |
| 通信 | All-Reduce 等待时间长 | 通信重叠、梯度压缩 |
| I/O | 数据读取和检查点写入慢 | 预取、异步检查点 |

### 混合精度训练

使用低精度（FP16/BF16/FP8）进行前向和反向计算，同时维护 FP32 主权重保证数值稳定。

| 格式 | 指数位 | 尾数位 | 特点 |
|---|---|---|---|
| FP32 | 8 | 23 | 基准精度 |
| FP16 | 5 | 10 | 范围小，需 loss scaling |
| BF16 | 8 | 7 | 范围与 FP32 相同，无需 loss scaling |
| FP8 (E4M3) | 4 | 3 | 前向计算用，精度最低 |

### Loss Scaling

FP16 的最小可表示正数约为 $6 \times 10^{-8}$，而小梯度值可能低于此阈值导致下溢。Loss scaling 通过在反向传播前将 loss 乘以一个大常数 $S$（如 $2^{16}$），使梯度放大到可表示范围，更新前再除以 $S$ 恢复。

BF16 的指数位与 FP32 相同（8 位），动态范围足够大，因此无需 loss scaling。

## 技术原理

### ZeRO（Zero Redundancy Optimizer）

ZeRO 的核心思想：数据并行中每张卡持有完整模型副本是冗余的，可以逐步切分。

#### ZeRO-Stage 1：切分优化器状态

将 Adam 优化器状态（FP32 权重副本 $N$ + 动量 $N$ + 方差 $N$）按 GPU 切分。每张卡只更新自己负责的部分，再通过 All-Gather 同步更新后的参数。

- 显存从 $16N$ 降至 $\sim 4N + \frac{12N}{D}$（$D$ 为 GPU 数）
- 通信量与标准 DP 相同

#### ZeRO-Stage 2：切分梯度

在 Stage 1 基础上，将梯度也按 GPU 切分。反向传播后通过 Reduce-Scatter 将梯度分配到对应卡，每张卡只保留自己负责的部分。

- 显存进一步降至 $\sim 2N + \frac{14N}{D}$
- 通信量与标准 DP 相同

#### ZeRO-Stage 3：切分参数

将模型参数本身也切分。前向和反向传播时通过 All-Gather 临时收集所需参数，计算后释放。

- 显存降至 $\frac{16N}{D}$，理论上无限扩展
- 代价：额外 2 次 All-Gather（前向 + 反向），通信量增加约 50%

#### ZeRO-Offload / ZeRO-Infinity

将部分状态卸载到 CPU 内存或 NVMe SSD，进一步突破 GPU 显存限制。代价是 PCIe 传输延迟。

### 激活重计算（Activation Recomputation / Checkpointing）

前向传播时不保存中间激活值，反向传播时重新计算。以约 33% 的额外计算开销换取激活显存从 $O(nL)$ 降至 $O(\sqrt{nL})$（$n$ 为层宽度，$L$ 为层数）。

- 选择性重计算（Selective Recomputation）：只重计算注意力等高显存低计算量的层，实现更好的计算-显存权衡

### 通信重叠

将梯度 All-Reduce 与反向计算重叠执行。在反向传播计算出某一层梯度后立即启动通信，同时继续计算下一层梯度。

### FlashAttention

通过分块计算（tiling）和在线 softmax 重计算，减少 HBM 读写次数。不是减少 FLOP，而是减少内存 IO 瓶颈，从而提升 GPU 计算单元利用率。

## 关键方法与模型

| 技术 | 解决瓶颈 | 核心代价 | 典型场景 |
|---|---|---|---|
| 混合精度 | 计算速度、显存 | 数值精度风险 | 所有训练 |
| 梯度累积 | 显存（等效大 batch） | 延迟更新、BN 统计偏差 | 小显存训练 |
| 激活重计算 | 激活显存 | 额外计算 | 长序列、大模型 |
| 通信重叠 | 通信等待 | 实现复杂 | 所有分布式训练 |
| ZeRO-1/2/3 | 状态显存冗余 | 通信开销 | 大模型训练 |
| FlashAttention | 注意力 IO | 无显著代价 | Transformer 训练 |
| FP8 训练 | 计算速度、显存 | 精度管理复杂 | H100 及以上 |

## 优势与局限

### 优势

- 显著提升 GPU 利用率和训练吞吐量
- 使有限资源训练大模型成为可能
- 技术之间大多可组合使用

### 局限

- 优化技术叠加增加调试复杂度
- 某些优化（如 ZeRO-3）以通信换显存，在低带宽环境下收益有限
- 混合精度可能引入数值不稳定
- 梯度累积与 Batch Normalization 不兼容（需改用 LayerNorm）

## 评估指标

| 指标 | 定义 | 用途 |
|---|---|---|
| tokens/s | 每秒处理 token 数 | 训练吞吐量 |
| MFU | Model FLOPs Utilization | GPU 计算效率 |
| 显存峰值 | 训练中最大显存占用 | OOM 风险评估 |
| 通信占比 | 通信时间 / 总时间 | 并行效率 |
| 收敛质量 | loss 曲线、梯度范数 | 优化是否影响训练稳定性 |

## 应用场景

- 预训练：ZeRO-3 + 激活重计算 + FlashAttention + BF16
- 微调：ZeRO-2 + LoRA（无需切分参数）
- 长上下文训练：序列并行 + 选择性重计算

## 与其他技术关系

- 前置：[[01_分布式训练与并行]]（ZeRO 是数据并行的进化形式）
- 配合：[[03_大模型训练框架]]（框架封装这些优化为 API）
- 下游：[[04_训练稳定性与容错]]（优化可能引入数值不稳定）

## 前沿发展

- **FP8 训练**：NVIDIA H100 支持 FP8 Tensor Core，训练速度提升 2-3 倍
- **通信压缩**：梯度量化（如 1-bit Adam）减少通信量
- **自动并行**：Alpa 等系统自动搜索最优并行+优化策略组合
- **内核融合**：将多个小算子融合为一个 kernel，减少启动开销

## 常见问题

- **BF16 和 FP16 选哪个？**：优先 BF16，无需 loss scaling 且动态范围与 FP32 相同。A100/H100 支持 BF16，V100 不支持。
- **ZeRO-3 一定比 ZeRO-2 慢吗？**：不一定。当显存瓶颈导致 batch size 受限时，ZeRO-3 可以用更大 batch 提升吞吐，弥补通信开销。
- **梯度累积等效于大 batch 吗？**：不完全等价。梯度累积下 BN 统计仍基于小 batch，且梯度更新的时序不同。LLM 使用 LayerNorm 不受此影响。

## 相关知识

- [[00_训练工程_综述]]
- [[01_分布式训练与并行]]
- [[03_大模型训练框架]]
- [[04_训练稳定性与容错]]

## References

- Micikevicius, P. et al. "Mixed Precision Training." ICLR 2018.
- Rajbhandari, S. et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." SC 2020.
- Chen, T. et al. "Training Deep Nets with Sublinear Memory Cost." arXiv 1604.06174.
- Dao, T. et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.
- DeepSpeed 官方文档: https://www.deepspeed.ai/
