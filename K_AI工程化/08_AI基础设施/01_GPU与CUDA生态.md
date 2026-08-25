---
tags:
  - AI基础设施
  - GPU
  - CUDA
  - Tensor Core
  - AI工程
created: 2026-08-10
updated: 2026-08-25
---

# GPU 与 CUDA 生态

## 一句话理解

GPU 是 AI 计算的核心引擎，其性能由算力、显存带宽和互联拓扑共同决定；CUDA 生态（驱动、运行时、cuDNN、NCCL、Triton）则是让这些硬件能力被高效利用的软件栈，理解 GPU 架构和 CUDA 编程模型是优化训练和推理的基础。

## 1. GPU 架构基础

### 1.1 GPU vs CPU

| 维度 | CPU | GPU |
|:---|:---|:---|
| **核心数** | 8-128 个大核 | 数千个小核 |
| **单核性能** | 极强 | 弱 |
| **并行能力** | 有限 | 大规模并行 |
| **延迟设计** | 低延迟 | 高吞吐 |
| **缓存** | 大缓存（MB 级） | 小缓存（KB 级） |
| **适用场景** | 串行/分支密集 | 并行/矩阵密集 |
| **指令集** | 复杂指令集 (x86/ARM) | SIMT（单指令多线程） |

### 1.2 GPU 架构演进

| 架构 | 代号 | 代表产品 | 关键特性 |
|:---|:---|:---|:---|
| **Kepler** | 2012 | K80 | 第一个深度学习 GPU |
| **Maxwell** | 2014 | M40 | 能效大幅提升 |
| **Pascal** | 2016 | P100 | NVLink 首次引入，FP16 |
| **Volta** | 2017 | V100 | 第一代 Tensor Core |
| **Turing** | 2018 | T4 | RT Core，INT8 推理 |
| **Ampere** | 2020 | A100 | 第三代 Tensor Core，BF16，稀疏化 |
| **Hopper** | 2022 | H100 | 第四代 Tensor Core，FP8，Transformer Engine |
| **Blackwell** | 2024 | B200 | FP4，双芯片，192GB HBM3e |
| **Rubin** | 2026(计划) | R200 | HBM4，下一代架构 |

### 1.3 GPU 内存层次

```text
GPU 内存层次（以 H100 为例）：

┌─────────────────────────────────────┐
│         HBM3 (80 GB)                │  ← 全局内存
│         带宽: 3.35 TB/s              │     所有 SM 可访问
└─────────────────────────────────────┘
        │  ~400 cycles 延迟
        ▼
┌─────────────────────────────────────┐
│    L2 Cache (50 MB)                 │  ← 所有 SM 共享
└─────────────────────────────────────┘
        │  ~200 cycles 延迟
        ▼
┌──────────────┐  ┌──────────────┐
│ SM L1 Cache  │  │ SM L1 Cache  │   ← 每 SM 独立
│ + Shared     │  │ + Shared     │      256 KB (可配置)
│ Memory       │  │ Memory       │
└──────────────┘  └──────────────┘
        │  ~30 cycles 延迟
        ▼
┌──────────────┐  ┌──────────────┐
│  Registers   │  │  Registers   │   ← 最快
│  (256KB/SM)  │  │  (256KB/SM)  │      ~1 cycle
└──────────────┘  └──────────────┘
```

### 1.4 Tensor Core

Tensor Core 是 GPU 中专门执行矩阵乘法的硬件单元：

| 代际 | 架构 | 支持精度 | 关键特性 |
|:---|:---|:---|:---|
| 第一代 | Volta (V100) | FP16 | 4×4 矩阵乘加 |
| 第二代 | Turing (T4) | FP16/INT8 | 混合精度训练 |
| 第三代 | Ampere (A100) | FP16/BF16/TF32/INT8/INT4 | 稀疏化加速 |
| 第四代 | Hopper (H100) | FP16/BF16/FP8/INT8 | Transformer Engine |
| 第五代 | Blackwell (B200) | FP4/FP8/FP16 | 微缩放格式 (MXFP) |

```text
Tensor Core 工作原理：

D = A × B + C

传统 CUDA Core: 逐元素乘加（标量操作）
Tensor Core:    矩阵乘加（一个时钟周期完成 D=A×B+C）

H100 Tensor Core 吞吐：
├── FP16: 989 TFLOPS
├── BF16: 989 TFLOPS
├── FP8:  1979 TFLOPS（约为 FP16 的 2×）
├── INT8: 3958 TOPS
└── FP4:  7915 TFLOPS（Blackwell）
```

> **CS336 P5 洞见**：自 V100 起，Tensor Core 使矩阵乘法成为机器学习中的"特权操作"——比 GPU 上其他浮点运算快 10 倍以上。这解释了为什么所有能随算力扩展的 ML 架构都以矩阵乘法为核心：它是唯一能真正高效换取大量算力吞吐的途径。在 Tensor Core 出现之前，人们甚至用图形着色器（shader）"黑"出矩阵乘法来利用 GPU 的大规模并行性。

### 1.5 TPU 与 GPU：趋同进化与概念映射

TPU 与 GPU 宏观上高度相似——若目标是高能效的矩阵运算，两条技术路线殊途同归（趋同进化）。两者都依赖快慢内存层级结构，矩阵乘法单元同属**脉动阵列**（systolic array，数据流式进出完成乘加，底层电路几乎同源）。

**命名陷阱**（CS336 P5 特别提醒）：

- TPU 的 "Tensor Core" 指的是**处理器**（对应 GPU 中 SM 的角色）
- GPU 的 "Tensor Core" 指的是**矩阵乘法单元**（对应 TPU 中的 MXU）
- 两个"张量核心"名字相同、层级不同，必须靠上下文区分

| 维度 | GPU (H100) | TPU |
|:---|:---|:---|
| 处理单元数 | ~132 个 SM | 2 个 Tensor Core（处理器） |
| 矩阵乘法单元 | 528 个小单元（每 SM 4 个） | 8 个大规模 MXU |
| 设计取向 | 数量多、单体小 → 灵活可编程 | 数量少、单体大 → 只做大规模矩阵乘 |
| 最小输入规模 | 小维度也可高效执行 | MXU 不接受低于 64 的输入维度（批大小扫描到 64 即触底） |
| L2 缓存 | 相对慢 | 明显更快（硅片设计上做了不同权衡，是其卖点之一） |
| 最大差异 | — | **在网络互联（ICI）而非芯片本身**：单芯片层面两者都是矩阵乘法机器 |

> GPU 靠众多小型矩阵乘法单元换取灵活性；TPU 用少数大型 MXU 换取效率。Google JAX 团队维护的 GPU 书中给出了精确的概念映射表：GPU 的每个概念在 TPU 中都有对应物，前述优化技巧（融合、分块、合并访问等）在两种硬件上通用。

### 1.6 算力-带宽-互联的增速剪刀差

- **历史脉络**：90 年代的性能来源是时钟频率（Dennard 缩放：晶体管尺寸缩小 → 时钟频率提高）；2000 年代 Dennard 缩放因物理限制触顶后，性能来源转向**并行扩展**——这正是 GPU 的故事。晶体管数量仍在增加，但更小的晶体管并不更快
- **超指数增长**：P100/V100 之后浮点算力几乎逐年超指数增长，由 Tensor Core、结构化稀疏、FP8/FP4 低精度格式接力推动
- **剪刀差**：计算吞吐增长 >> 内存带宽增长 >> 互联带宽增长。早期 GPU 编程不太需要考虑内存；如今计算与内存的差距越拉越大，**内存与通信瓶颈主导一切优化**，且随时间推移愈演愈烈
- **推理比训练更甚**：推理比训练更受内存制约，推动**预填充/解码解耦**（Prefill 计算密集走一种芯片，Decode 访存密集走另一种芯片）；阶跃星辰 Step 系列甚至尝试将注意力与 MLP 分配到不同的专用加速器上

> **Groq LPU 的另一极端**：以海量 SRAM 为主存，推理这类特定负载受益明显。但 SRAM 制造成本高数百倍、必须持续通电（功耗高）、物理上必须贴近计算单元，因此主流加速器仍采用"内存层级 + 高效利用"的路线，全 SRAM 方案只在特定场景划算。

## 2. CUDA 编程模型

### 2.1 CUDA 执行模型

```text
CUDA 线程层次：

Grid（网格）
├── Block 0
│   ├── Warp 0 (32 threads)
│   ├── Warp 1
│   └── ...
├── Block 1
│   ├── Warp 0
│   └── ...
└── ...

关键概念：
├── Thread：最小执行单元
├── Warp：32 个线程为一组，以锁步方式执行（SIMT）
├── Block：线程块，共享 Shared Memory
├── Grid：所有线程块的集合
├── Kernel：在 GPU 上执行的函数
└── Stream：异步执行队列
```

### 2.2 CUDA 内存管理

| 内存类型 | 作用域 | 速度 | 容量 | 管理方式 |
|:---|:---|:---|:---|:---|
| **Global Memory** | 所有线程 | 慢 (~400 cycles) | 大 (HBM) | cudaMalloc/cudaFree |
| **Shared Memory** | 同一 Block | 快 (~30 cycles) | 小 (KB 级) | __shared__ |
| **Registers** | 同一 Thread | 最快 (~1 cycle) | 极小 | 自动分配 |
| **Constant Memory** | 所有线程 | 有缓存 | 64KB | __constant__ |
| **Texture Memory** | 所有线程 | 有缓存 | 大 | 空间局部性优化 |
| **Unified Memory** | CPU+GPU | 自动迁移 | 系统 | cudaMallocManaged |

> **共享内存 vs L1 缓存**（CS336 P5）：两者物理上同属 SM 内的 SRAM（~30 cycles），但 L1 缓存是**自动**的（硬件自动缓存最近访问的数据，不可直接控制）；共享内存是**可编程**的（kernel 中显式读写、线程块内共享）。分块矩阵乘法依赖的正是可编程的共享内存——这也是 TPU 中 VSMEM 的对应物。

### 2.3 CUDA 编程基础

```python
# PyTorch 中的 CUDA 操作示例
import torch

# 数据移到 GPU
x = torch.randn(1024, 1024).cuda()
w = torch.randn(1024, 1024).cuda()

# GPU 上计算
y = torch.mm(x, w)  # 自动使用 Tensor Core

# 指定 GPU
x = torch.randn(100).cuda(device=0)
y = torch.randn(100).cuda(device=1)

# 多 GPU 数据并行
model = nn.DataParallel(model)  # 简单数据并行
model = model.cuda()

# 显存管理
torch.cuda.empty_cache()  # 释放缓存
torch.cuda.memory_allocated()  # 已分配显存
torch.cuda.max_memory_allocated()  # 峰值显存
```

### 2.4 Kernel 优化原则

```text
GPU Kernel 优化关键：

1. 合并访存（Memory Coalescing）
   ├── 线程访问连续内存地址
   ├── 32 线程 (warp) 访问 128 字节对齐
   └── 非合并访问：带宽利用率 < 25%

2. 最大化占用率（Occupancy）
   ├── 每个 SM 的活跃 warp 数
   ├── 受寄存器/Shared Memory 使用限制
   └── 目标：> 50% 占用率

3. 避免分支分歧（Warp Divergence）
   ├── 同一 warp 内线程走不同分支
   ├── 导致串行执行
   └── 尽量让 warp 内条件一致

4. 使用 Tensor Core
   ├── 对齐矩阵维度（8/16 的倍数）
   ├── 使用 FP16/BF16 数据类型
   └── torch.matmul 自动调用
```

## 3. CUDA 软件栈

### 3.1 软件栈层次

```text
应用层
├── PyTorch / TensorFlow / JAX
│   └── cuDNN / cuBLAS / NCCL / CUTLASS
│       └── CUDA Runtime (cudart)
│           └── CUDA Driver
│               └── GPU 硬件
│
├── TensorRT / Triton Inference Server
│   └── CUDA Runtime
│
└── 自定义 Kernel
    ├── Triton (Python DSL)
    ├── CUDA C (nvcc)
    └── CUTLASS (C++ 模板库)
```

### 3.2 核心库

| 库 | 功能 | 使用场景 |
|:---|:---|:---|
| **cuDNN** | 深度学习原语（卷积、池化、归一化） | 所有 DL 框架底层 |
| **cuBLAS** | 线性代数（矩阵乘法） | 全连接、注意力 |
| **NCCL** | 集合通信 | 分布式训练 |
| **CUTLASS** | 高性能矩阵运算模板库 | 自定义高性能 Kernel |
| **cuFFT** | 快速傅里叶变换 | 信号处理 |
| **Thrust** | 并行算法库（sort/reduce/scan） | 数据处理 |
| **CUB** | CUDA 原语库 | 底层并行原语 |

### 3.3 版本兼容性

```text
CUDA 版本兼容矩阵：

NVIDIA Driver (最大支持 CUDA 版本)
    ↓ 向后兼容
CUDA Toolkit (如 12.1)
    ↓
cuDNN (需匹配 CUDA 版本)
    ↓
PyTorch (需匹配 CUDA + cuDNN)

关键规则：
├── Driver 版本 >= CUDA Toolkit 版本（向前兼容）
├── cuDNN 版本需与 CUDA 版本匹配
├── PyTorch 编译时绑定特定 CUDA/cuDNN 版本
├── 容器内 CUDA 版本需与宿主机 Driver 兼容
└── 推荐：使用 NVIDIA 官方容器镜像
```

## 4. GPU 性能指标

### 4.1 关键性能指标

| 指标 | 含义 | H100 SXM | B200 |
|:---|:---|:---|:---|
| **FP16 算力** | Tensor Core 半精度 | 989 TFLOPS | 2250 TFLOPS |
| **FP8 算力** | 8 位浮点 | 1979 TFLOPS | 4500 TFLOPS |
| **FP4 算力** | 4 位浮点 | — | 9000 TFLOPS |
| **显存容量** | HBM | 80 GB HBM3 | 192 GB HBM3e |
| **显存带宽** | HBM 带宽 | 3.35 TB/s | 8 TB/s |
| **NVLink 带宽** | GPU 间互联 | 900 GB/s | 1.8 TB/s |
| **TDP** | 功耗 | 700W | 1000W |
| **互联** | 节点内拓扑 | NVSwitch | NVLink Switch |

### 4.2 算术强度与瓶颈分析

```text
算术强度 (Arithmetic Intensity) = FLOPs / Bytes

├── 计算密集型 (Compute-bound)
│   ├── AI > GPU 的运算/带宽比
│   ├── 瓶颈：算力
│   └── 典型：大矩阵乘法、训练前向/反向
│
├── 访存密集型 (Memory-bound)
│   ├── AI < GPU 的运算/带宽比
│   ├── 瓶颈：显存带宽
│   └── 典型：LLM 推理 decode、逐元素操作
│
├── H100 的平衡点：
│   ├── FP16: 989 TFLOPS / 3.35 TB/s ≈ 295 FLOP/Byte
│   └── 当 AI < 295 时为访存密集型
│
└── LLM 推理 decode 阶段：
    ├── 每生成 1 token，约 2×N FLOPs（N=参数量）
    ├── 需读取全部权重 = N × bytes_per_param
    ├── AI ≈ 2 / bytes_per_param
    │   ├── FP16: AI ≈ 1（严重访存密集）
    │   └── FP8:  AI ≈ 2
    └── 结论：LLM 推理是访存密集型，显存带宽是瓶颈
```

### 4.3 GPU 监控指标

| 指标 | 工具 | 含义 | 告警阈值 |
|:---|:---|:---|:---|
| **GPU 利用率** | nvidia-smi | 计算单元使用率 | < 50% 持续 5min |
| **显存使用率** | nvidia-smi | 显存占用 | > 95% |
| **温度** | nvidia-smi | GPU 温度 | > 85°C |
| **功耗** | nvidia-smi | 实际功耗 | 接近 TDP |
| **ECC 错误** | nvidia-smi | 显存错误 | > 0 |
| **XID 错误** | dmesg | GPU 硬件错误 | > 0 |
| **PCIe 带宽** | nvidia-smi | PCIe 传输速率 | 异常下降 |

## 5. GPU 选型指南

### 5.1 按场景选型

| 场景 | 推荐 GPU | 原因 |
|:---|:---|:---|
| **大模型训练** | H100/H200/B200 | 高算力+大显存+NVLink |
| **中小模型训练** | A100 80GB | 性价比高 |
| **LLM 推理** | H100/H200 | 显存带宽决定推理速度 |
| **中小模型推理** | A10/L40S | 成本低，INT8 性能好 |
| **边缘推理** | Jetson Orin | 低功耗嵌入式 |
| **多模态** | A100/H100 | 大显存需求 |

### 5.2 训练 vs 推理的 GPU 需求差异

| 维度 | 训练 | 推理 |
|:---|:---|:---|
| **瓶颈** | 算力（计算密集型） | 显存带宽（访存密集型） |
| **精度** | FP16/BF16 | FP8/INT8/FP4 |
| **显存** | 需存梯度+优化器状态 | 需存权重+KV Cache |
| **互联** | 关键（AllReduce 通信） | 不关键（单卡推理） |
| **利用率** | 目标 > 50% | 目标 > 80% |
| **成本敏感度** | 较低（投资性） | 极高（持续运营成本） |

## 6. 多 GPU 互联

### 6.1 互联技术对比

| 技术 | 带宽 | 延迟 | 适用场景 |
|:---|:---|:---|:---|
| **NVLink 4.0** | 900 GB/s | ~1μs | 同节点 GPU 互联 |
| **NVLink 5.0** | 1.8 TB/s | ~1μs | Blackwell 互联 |
| **NVSwitch** | 全互联 | ~1μs | 8+ GPU 全互联 |
| **PCIe Gen5** | 128 GB/s | ~5μs | 低成本 GPU 互联 |
| **InfiniBand NDR** | 400 Gb/s | ~1μs | 节点间互联 |
| **RoCE v2** | 400 Gb/s | ~2μs | 节点间（以太网） |
| **CXL** | 64 GB/s | ~0.5μs | CPU-GPU 缓存一致性 |

### 6.2 GPU 拓扑感知

```text
8-GPU 服务器典型拓扑（H100 DGX）：

         ┌──────────────┐
         │   CPU 0      │
         └──────┬───────┘
                │
    ┌───────────┴───────────┐
    │     NVSwitch 0        │
    │  ┌───┬───┬───┬───┐   │
    │  │GPU│GPU│GPU│GPU│   │
    │  │ 0 │ 1 │ 2 │ 3 │   │
    │  └───┴───┴───┴───┘   │
    └───────────┬───────────┘
                │
    ┌───────────┴───────────┐
    │     NVSwitch 1        │
    │  ┌───┬───┬───┬───┐   │
    │  │GPU│GPU│GPU│GPU│   │
    │  │ 4 │ 5 │ 6 │ 7 │   │
    │  └───┴───┴───┴───┘   │
    └───────────────────────┘

拓扑感知调度：
├── 同 NVSwitch 域：最高带宽（900 GB/s）
├── 跨 NUMA 但同节点：中等带宽
├── 跨节点：InfiniBand 带宽
└── 训练作业应尽量分配同 NVSwitch 域的 GPU
```

## 7. 实践建议

### 7.1 容器化最佳实践

```dockerfile
# GPU 容器最佳实践
FROM nvcr.io/nvidia/pytorch:23.10-py3

# 使用 NVIDIA 官方基础镜像
# 已包含：CUDA + cuDNN + PyTorch + NCCL

# 安装应用依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 运行时指定 GPU
# docker run --gpus all / --gpus '"device=0,1"'
```

### 7.2 常见问题排查

| 问题 | 症状 | 排查方法 |
|:---|:---|:---|
| **OOM** | CUDA out of memory | 减小 batch size / gradient checkpointing |
| **利用率低** | GPU util < 30% | 检查数据加载瓶颈 / 增大 batch |
| **XID 错误** | GPU 掉卡 | dmesg 查看 XID 错误码 |
| **ECC 错误** | 计算结果异常 | nvidia-smi -q 查看 ECC 错误 |
| **CUDA 版本不匹配** | 库加载失败 | 检查 driver/cuda/cudnn 版本 |
| **NCCL 超时** | 分布式训练卡住 | 检查网络/NCCL_DEBUG=INFO |

## 8. 子主题导航

- [[02_Kubernetes与GPU调度]]
- [[05_编译器栈与中间表示]]
- [[06_通信库与集合通信]]
- [[08_异构硬件与能源调度]]

## 9. 相关知识

- [[00_AI基础设施综述]]
- [[../02_训练工程/01_分布式训练与并行]]（GPU 并行策略）
- [[../03_推理工程/00_推理工程_综述]]（GPU 推理优化）
- [[../03_推理工程/05_量化与模型压缩]]（精度与 GPU 性能）

## References

- NVIDIA, *CUDA C++ Programming Guide* (2024)
- NVIDIA, *H100 Tensor Core GPU Architecture Whitepaper* (2022)
- NVIDIA, *Blackwell B200 GPU Architecture Whitepaper* (2024)
- Mark Harris, *CUDA C++ Best Practices Guide* (NVIDIA)
- Tillet et al., *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations* (2019)
