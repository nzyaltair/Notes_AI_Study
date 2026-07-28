# ONNX 格式与基础

## 1. 概述

ONNX（Open Neural Network Exchange）是 2017 年 Facebook（Meta）与 Microsoft 联合发起的开放模型交换格式，旨在定义一套统一的模型表示标准，使深度学习模型能在不同框架和硬件平台间无缝迁移。

ONNX 格式解决的核心问题：
- **框架互操作**：PyTorch、TensorFlow、PaddlePaddle 等框架模型格式不互通
- **训练-部署解耦**：训练用任意框架，部署用最优推理引擎
- **跨平台部署**：一套模型格式覆盖 CPU/GPU/NPU/移动端
- **标准化**：开放标准，由社区共同维护，不受单一厂商控制

项目现由 Linux 基金会维护，已成为工业界事实上的模型交换格式。

## 2. 发展历史

| 年代 | 里程碑 | 意义 |
|:---|:---|:---|
| 2017.09 | ONNX 1.0 发布 | Facebook/Microsoft 联合发起，支持基础 CNN |
| 2019.12 | Opset 10 | 基础算子完善 |
| 2020.06 | Opset 11 | Bool 类型、动态形状基础 |
| 2020.12 | Opset 13 | 量化算子、改进动态形状 |
| 2021.03 | Opset 14 | 目标检测优化 |
| 2021.06 | Opset 15 | 新数学函数（Cos/Sin） |
| 2022 | Opset 16-17 | 增强控制流和符号形状推导 |
| 2023+ | Opset 18+ | 持续扩展算子覆盖 |

## 3. 核心概念

### 3.1 计算图（Computational Graph）

ONNX 将深度学习模型抽象为**有向无环计算图**：
- **节点（Node）**：表示算子（Operator），如 Conv、MatMul、ReLU
- **边（Edge）**：表示张量（Tensor）数据流
- **初始器（Initializer）**：权重和偏置等常量张量

### 3.2 Opset 版本

通过 Operator Set（Opset）版本管理算子定义。每个 Opset 定义了一组算子及其语义，版本递增意味着新增算子或功能扩展。

| Opset | 发布时间 | 关键特性 | 推荐场景 |
|:---|:---|:---|:---|
| 10 | 2019-12 | 基础算子 | 老系统兼容 |
| 11 | 2020-06 | Bool 类型、动态形状基础 | 通用基线 |
| 13 | 2020-12 | 量化算子、改进动态形状 | 通用推荐 |
| 14 | 2021-03 | 目标检测优化 | 检测模型 |
| 15 | 2021-06 | 新数学函数 | 最新功能 |

选择策略：目标部署环境已知时选择其支持的最高 Opset；多平台部署选 Opset 11-13 作为基线。

### 3.3 IR 与序列化

ONNX 模型文件使用 Protocol Buffers（protobuf）作为序列化格式：

| Proto 类型 | 说明 |
|:---|:---|
| **ModelProto** | 顶层容器，包含 ir_version、opset_import、metadata |
| **GraphProto** | 计算图定义，包含 node、input、output、initializer |
| **NodeProto** | 单个算子节点，包含 op_type、input、output、attributes |
| **TensorProto** | 权重和偏置等初始器数据 |

### 3.4 Execution Provider（EP）

ONNX Runtime 通过 EP 机制适配不同硬件后端，每个 EP 针对特定硬件做算子级优化。

## 4. 技术原理

### 4.1 最小导出示例

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel().eval()
dummy_input = torch.randn(1, 10)

torch.onnx.export(
    model, dummy_input, "model.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)
```

### 4.2 ONNX Runtime 推理

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")
input_data = np.random.randn(1, 10).astype(np.float32)
outputs = session.run(None, {"input": input_data})
```

### 4.3 算子映射机制

框架中的每个算子通过**符号函数（Symbolic Function）**映射到对应的 ONNX 算子。PyTorch 通过注册在 `torch.onnx.symbolic_helper` 中的映射函数，将 ATen 算子转换为 ONNX 算子组合。

### 4.4 环境搭建

核心依赖：`onnx`（模型格式库）、`onnxruntime`（推理引擎）、训练框架（`torch` 或 `tensorflow`）。GPU 环境需匹配 CUDA/cuDNN 版本，安装 `onnxruntime-gpu`。推荐使用 conda 或 venv 隔离环境。

## 5. 关键方法与模型

### 5.1 核心工具链

| 工具 | 作用 |
|:---|:---|
| `torch.onnx.export` | PyTorch 内置导出 |
| `tf2onnx` | TensorFlow 转 ONNX |
| `paddle2onnx` | PaddlePaddle 转 ONNX |
| ONNX Runtime | 跨平台推理引擎 |
| Netron | 模型可视化 |
| onnx-simplifier | 图结构简化 |
| onnx-graphsurgeon | 图编辑 |

### 5.2 onnx-simplifier

通过常量折叠和形状推断简化计算图，减少冗余节点：

```bash
python -m onnxsim model.onnx model_sim.onnx
```

### 5.3 onnx-graphsurgeon

NVIDIA 开发的图编辑库，支持节点增删改、子图替换，用于处理不兼容算子。

## 6. 优势与局限

### 6.1 优势

- **框架无关**：PyTorch、TensorFlow、PaddlePaddle 等均可导出
- **开放标准**：社区维护，不受单一厂商控制
- **跨平台**：支持 Windows、Linux、macOS 及移动设备
- **优化推理**：ONNX Runtime 自动执行图优化

### 6.2 局限

- **算子覆盖不全**：部分框架特有算子需手动处理
- **protobuf 2GB 限制**：大模型需使用外部数据格式
- **LLM 支持有限**：大语言模型主要用专用推理引擎
- **动态形状优化难**：复杂动态维度可能影响推理性能

## 7. 应用场景

- **训练-部署分离**：研究团队用 PyTorch 训练，部署团队用 ONNX Runtime 推理
- **硬件适配**：一次转换，多平台运行（CPU/GPU/NPU）
- **版本管理**：标准化格式简化模型版本控制
- **性能优化**：ONNX Runtime 自动图优化

## 8. 与其他技术关系

- [[00_ONNX模型导出]] — ONNX 子目录的总览
- [[02_模型转换方法]] — 各框架的 ONNX 导出方法详解
- [[04_跨框架部署与优化]] — ONNX Runtime 和 TensorRT 部署优化
- [[00_模型部署与工程化]] — ONNX 在部署流水线中的角色

## 9. 前沿发展

- **torch.export**：PyTorch 2.x 新导出路径，基于 TorchDynamo，更可靠的图捕获
- **ONNX 格式扩展**：Opset 18+ 持续扩展算子覆盖
- **多硬件 EP 完善**：CoreML/DirectML/Ascend 等后端持续优化
- **ONNX Runtime Mobile**：移动端轻量推理引擎

返回 [[00_ONNX模型导出]]
