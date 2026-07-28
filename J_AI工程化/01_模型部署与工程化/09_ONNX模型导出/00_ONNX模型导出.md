# ONNX 模型导出

## 1. 概述

ONNX（Open Neural Network Exchange）是 Facebook/Microsoft 于 2017 年联合发起的开放模型交换格式，旨在定义一套统一的模型表示标准，使模型能在不同框架和硬件平台间无缝迁移。模型导出为 ONNX 后，可在 ONNX Runtime、TensorRT、OpenVINO 等推理引擎上跨平台部署。

ONNX 解决的核心问题：
- **框架互操作**：PyTorch 训练的模型可部署到 TensorFlow 生态，反之亦然
- **训练-部署解耦**：研究团队用任意框架训练，部署团队用最优引擎推理
- **跨硬件部署**：一次导出，在 CPU/GPU/NPU/移动端多平台运行
- **推理优化**：ONNX Runtime 自动执行图优化（算子融合、常量折叠、内存复用）

项目现由 Linux 基金会维护，已成为工业界事实上的模型交换格式。

## 2. 发展历史

| 年代 | 里程碑 | 意义 |
|:---|:---|:---|
| 2017.09 | ONNX 1.0 发布 | Facebook/Microsoft 联合发起，支持基础 CNN |
| 2017 | ONNX Runtime 预览 | Microsoft 推出跨平台推理引擎 |
| 2019 | Opset 10-11 | 引入动态形状、Bool 类型 |
| 2019 | ONNX Runtime 1.0 | 正式发布，Execution Provider 架构 |
| 2020 | Opset 13 | 量化算子、改进动态形状 |
| 2021 | Opset 14-15 | 目标检测优化、新数学函数 |
| 2022 | Opset 16-17 | 增强控制流和符号形状推导 |
| 2023 | onnxruntime-extensions | 扩展非标准算子支持 |
| 2023 | PyTorch 2.x torch.export | 新导出路径，TorchDynamo |
| 2024 | 多硬件 EP 成熟 | CoreML/DirectML/Ascend 等后端完善 |

## 3. 核心概念

### 3.1 计算图（Computational Graph）

ONNX 将深度学习模型抽象为**有向无环计算图**：节点表示算子（Operator），边表示张量（Tensor）数据流。无论模型用哪个框架训练，只要能导出为符合 ONNX IR 的计算图，就能被任何支持 ONNX 的推理引擎执行。

### 3.2 Opset 版本

ONNX 通过 Operator Set（Opset）版本管理算子定义。每个 Opset 版本定义了一组算子及其语义，版本号递增意味着新增算子或功能扩展。

选择策略：目标部署环境已知时选择其支持的最高 Opset；多平台部署选 Opset 11-14 作为基线。

### 3.3 IR 与序列化

ONNX 模型文件使用 Protocol Buffers（protobuf）作为序列化格式：

- **ModelProto**：顶层容器，包含 ir_version、opset_import、metadata
- **GraphProto**：计算图定义，包含 node、input、output、initializer
- **NodeProto**：单个算子节点，包含 op_type、input、output、attributes
- **TensorProto**：权重和偏置等初始器数据

### 3.4 Execution Provider（EP）

ONNX Runtime 通过 EP 机制适配不同硬件后端，每个 EP 针对特定硬件做算子级优化。选择推理引擎时按 EP 列表顺序尝试，首个支持的 EP 被使用。

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

### 4.3 动态轴（Dynamic Axes）

允许 ONNX 模型接受可变大小的输入维度，如动态 batch size 和序列长度：

```python
dynamic_axes = {
    'input': {0: 'batch_size', 1: 'seq_length'},
    'output': {0: 'batch_size'}
}
```

### 4.4 图优化

ONNX Runtime 自动执行多级图优化：
- **算子融合**：Conv+BatchNorm+ReLU 融合为单个 kernel
- **常量折叠**：预计算常量操作
- **内存复用**：预分配内存块，减少动态分配

### 4.5 算子兼容性处理

转换中最常见的挑战是算子不兼容，处理方案按优先级：
1. 提升 Opset 版本
2. onnxruntime-extensions 加载扩展算子
3. 算子替换（用标准算子组合替代）
4. ONNX GraphSurgeon 子图替换
5. 自定义算子注册（C++ kernel）

## 5. 关键方法与模型

### 5.1 核心工具链

| 工具 | 作用 |
|:---|:---|
| `torch.onnx.export` | PyTorch 内置导出 |
| `tf2onnx` | TensorFlow 转 ONNX |
| `paddle2onnx` | PaddlePaddle 转 ONNX |
| ONNX Runtime | 跨平台推理引擎 |
| TensorRT | NVIDIA GPU 极致优化 |
| Netron | 模型可视化 |
| onnx-simplifier | 图结构简化 |
| onnx-graphsurgeon | 图编辑 |

### 5.2 ONNX → TensorRT 优化

TensorRT 加载 ONNX 模型后执行多级优化：
1. **层融合**：连续算子融合为单个 kernel
2. **精度校准**：FP16 无需校准，INT8 需代表性数据集
3. **Kernel 自动调优**：针对目标 GPU 选择最优 kernel
4. **动态形状优化**：通过 min/opt/max shapes 预编译

### 5.3 性能对比（ResNet50, V100 GPU）

| 运行时 | Batch=1 | Batch=32 | 加速比 |
|:---|:---|:---|:---|
| PyTorch 原生 | 2.1 ms | 15.2 ms | 1.0x |
| ONNX Runtime (CUDA) | 1.8 ms | 11.2 ms | 1.2-1.4x |
| TensorRT FP16 | 1.2 ms | 6.8 ms | 1.8-2.2x |
| TensorRT INT8 | 0.9 ms | 5.2 ms | 2.3-2.9x |

## 6. 优势与局限

### 6.1 优势

- **框架无关**：解耦训练框架和推理引擎
- **开放标准**：社区维护，不受单一厂商控制
- **跨平台**：支持 Windows/Linux/macOS 及移动设备
- **优化推理**：自动图优化和多硬件后端

### 6.2 局限

- **算子覆盖不全**：部分框架特有算子需手动处理
- **LLM 支持有限**：大语言模型主要用 vLLM/TensorRT-LLM 而非 ONNX
- **动态形状挑战**：复杂动态维度可能导致优化困难
- **大模型文件限制**：protobuf 2GB 限制需外部数据格式

## 7. 应用场景

| 场景 | 关键策略 |
|:---|:---|
| 训练-部署分离 | PyTorch 训练 → ONNX → ONNX Runtime 推理 |
| 硬件适配 | 一次转换，CPU/GPU/NPU 多平台运行 |
| NVIDIA GPU 优化 | ONNX → TensorRT FP16/INT8 |
| 移动端部署 | ONNX Runtime Mobile + NNAPI/CoreML |
| 边缘设备 | INT8 量化 + 模型压缩 |

## 8. 与其他技术关系

- [[00_模型部署与工程化]] — ONNX 是部署流水线中的模型交换环节
- [[01_模型服务化]] — ONNX Runtime 是主流推理引擎之一
- [[12_大模型推理与优化]] — 量化技术与 ONNX Opset 13+ 量化算子关联
- [[04_容器化与Docker]] — ONNX 模型通过容器部署到不同硬件环境

## 9. 前沿发展

- **torch.export**：PyTorch 2.x 新导出路径，基于 TorchDynamo，更可靠的图捕获
- **ONNX Runtime 多硬件 EP**：CoreML/DirectML/Ascend 等后端持续完善
- **ONNX 格式扩展**：支持更多动态场景和量化格式
- **自动化转换工具**：减少手动算子兼容性处理
- **端到端优化流水线**：从训练到量化到部署的一体化工具链

## 子主题导航

| 编号 | 文件 | 内容 |
|:---|:---|:---|
| 01 | [[01_ONNX格式与基础]] | ONNX IR、Opset 版本、序列化格式、环境搭建 |
| 02 | [[02_模型转换方法]] | torch.onnx.export 参数详解、TF2ONNX、PaddlePaddle 转换 |
| 03 | [[03_流式与动态模型转换]] | 动态轴设置、KV Cache、编码器/解码器分离 |
| 04 | [[04_跨框架部署与优化]] | ONNX→TensorRT、ONNX Runtime、框架对比、算子兼容性 |
| 05 | [[05_常见问题与算子兼容]] | 内存优化、动态维度、数据类型匹配、自定义算子 |
| 06 | [[06_验证评估与场景优化]] | 正确性验证、精度调优、性能测试、云端/边缘部署 |

返回 [[00_模型部署与工程化]]
