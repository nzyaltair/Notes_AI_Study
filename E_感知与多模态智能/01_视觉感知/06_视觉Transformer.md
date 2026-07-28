# 视觉 Transformer

## 1. 概述

Vision Transformer（ViT）将自然语言处理中的 Transformer 架构引入计算机视觉，用自注意力机制替代卷积进行图像特征提取。2020 年 Google 提出 ViT，证明在大规模数据预训练下，纯 Transformer 架构在图像分类上可以超越 CNN，开启了视觉领域的 Transformer 范式转变。

视觉 Transformer 的核心思想是将图像分割为图块（Patch），将每个图块视为一个"词"（Token），通过 Transformer 编码器学习全局特征表示。与 CNN 的局部归纳偏置不同，Transformer 通过全局注意力建模长距离依赖，在数据充足时展现更强的可扩展性。

本篇聚焦 Transformer 在视觉领域的应用。关于 Transformer 基础架构，参见 [[08_Transformer与注意力机制]]。

## 2. 发展历史

| 年代 | 里程碑 | 作者/机构 | 核心意义 |
|:---|:---|:---|:---|
| 2020 | ViT | Dosovitskiy 等 (Google) | 纯 Transformer 图像分类，大数据下超越 CNN |
| 2020 | DETR | Carion 等 (Meta) | Transformer 端到端目标检测 |
| 2021 | DeiT | Touvron 等 (Meta) | 数据高效训练 ViT，ImageNet-only 可竞争 |
| 2021 | Swin Transformer | Liu 等 (Microsoft) | 层级 Transformer，适配密集预测 |
| 2021 | BEiT | Bao 等 (Microsoft) | 掩码图像建模预训练 |
| 2021 | MAE | He 等 (Meta) | 高比例掩码自编码，高效视觉预训练 |
| 2021 | DINO | Caron 等 (Meta) | 自监督 ViT 训练，注意力图自发学习语义 |
| 2021 | IPT / IPT | Chen 等 | 早期视觉 Transformer 用于低层视觉 |
| 2022 | ConvNeXt | Liu 等 (Meta) | 纯 CNN 现代化设计，与 ViT 竞争 |
| 2022 | Mask2Former | Cheng 等 | 统一 Transformer 分割框架 |
| 2022 | DINO-DETR | Zhang 等 | DETR 收敛加速，COCO SOTA |
| 2023 | ViT-22B | Google | 220 亿参数视觉模型 |
| 2023 | SAM | Meta | 视觉基础模型，Transformer 架构 |
| 2024+ | 多模态统一 Transformer | 多家机构 | 视觉与语言统一架构 |

## 3. 核心概念

### 3.1 图块化（Patchification）

将图像分割为固定大小的图块，展平后作为 Token 序列：

$$x \in \mathbb{R}^{H \times W \times C} \to x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$$

其中 $P$ 为图块大小（如 16×16），$N = HW/P^2$ 为 Token 数。

### 3.2 位置编码

Transformer 本身没有位置感知，需要显式注入位置信息：

- **绝对位置编码**：学习或正弦的位置嵌入，加到 Token 嵌入上
- **相对位置编码**：编码 Token 之间的相对距离
- **2D 位置编码**：针对图像 2D 结构的位置编码方案

### 3.3 归纳偏置（Inductive Bias）

| 架构 | 归纳偏置 | 数据需求 |
|:---|:---|:---|
| CNN | 局部性、平移等变性 | 少数据即可有效 |
| ViT | 几乎无（仅图块化） | 需大数据预训练 |

ViT 弱归纳偏置的特性使其在小数据上容易过拟合，但在大数据上展现更强的可扩展性——模型和数据越大，性能持续提升。

### 3.4 全局 vs. 局部注意力

| 注意力类型 | 计算复杂度 | 特点 |
|:---|:---|:---|
| 全局自注意力 | $O(N^2)$ | 每个位置关注所有位置，计算量大 |
| 窗口注意力 | $O(N \cdot M^2)$ | 限制在局部窗口内，高效 |
| 稀疏注意力 | $O(N \sqrt{N})$ | 选择性关注，平衡效率与性能 |

## 4. 技术原理

### 4.1 ViT 架构

```
图像 → 图块分割 → 线性投影 → [CLS] + Patch Embeddings + 位置编码
                                          ↓
                              Transformer Encoder × L
                                          ↓
                                   [CLS] 输出 → MLP Head → 分类
```

**前向过程**：

$$\mathbf{z}_0 = [x_{cls}; \, E \cdot x_1^p; \, \cdots; \, E \cdot x_N^p] + E_{pos}$$

$$\mathbf{z}_l' = \text{MSA}(\text{LN}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1}$$

$$\mathbf{z}_l = \text{MLP}(\text{LN}(\mathbf{z}_l')) + \mathbf{z}_l'$$

$$y = \text{MLP\_Head}(\text{LN}(\mathbf{z}_L^0))$$

其中 $E \in \mathbb{R}^{(P^2 C) \times D}$ 为图块投影矩阵，$E_{pos} \in \mathbb{R}^{(N+1) \times D}$ 为位置嵌入。

### 4.2 Swin Transformer

Swin 通过**移位窗口注意力**解决全局注意力计算量过大的问题，并引入**层级结构**适配密集预测任务：

**窗口注意力（W-MSA）**：

将特征图划分为不重叠的 $M \times M$ 窗口，注意力在窗口内计算：

$$\text{复杂度}: O(N \cdot M^2 \cdot d) \quad \text{vs. 全局}: O(N^2 \cdot d)$$

**移位窗口注意力（SW-MSA）**：

交替使用常规窗口和移位窗口，实现跨窗口信息传递：

```
Layer 2i:   [W1][W2][W3][W4]    ← 常规窗口
Layer 2i+1:   [W1'][W2'][W3']    ← 移位窗口（窗口偏移 M/2）
```

**层级结构**：

```
Stage 1: 4×4 patch → C 通道,  分辨率 H/4 × W/4
Stage 2: 2×2 merge → 2C 通道,  分辨率 H/8 × W/8
Stage 3: 2×2 merge → 4C 通道,  分辨率 H/16 × W/16
Stage 4: 2×2 merge → 8C 通道,  分辨率 H/32 × W/32
```

类似 FPN 的多尺度特征，可直接用于检测/分割。

### 4.3 DeiT：数据高效训练

ViT 需要大数据预训练（JFT-300M），DeiT 使其在 ImageNet-1K 上即可有效训练：

- **强数据增强**：RandAugment + Mixup + CutMix + Random Erasing
- **知识蒸馏**：引入蒸馏 Token，用 CNN 教师（RegNet）蒸馏
- **高效正则化**：Stochastic Depth、Layer Scale

$$\text{输出} = \text{Softmax}(\text{LN}(z_{cls})) + \lambda \cdot \text{Softmax}(\text{LN}(z_{dist}))$$

### 4.4 BEiT：掩码图像建模预训练

受 BERT 启发，BEiT 通过预测被遮挡图块的离散 Token 进行预训练：

1. **图块 Token 化**：用 VQ-VAE 将图像编码为离散 Token
2. **随机掩码**：遮挡约 40% 的图块
3. **预测**：ViT 编码可见图块，预测被遮挡图块的 Token

### 4.5 MAE：非对称掩码自编码器

MAE 的核心创新是非对称设计：

```
图像 → 随机掩码 75% → 可见 25% 图块
                         ↓
                   ViT Encoder（仅处理可见图块）
                         ↓
                   + Mask Tokens → ViT Decoder（轻量）
                         ↓
                   重建被遮挡像素
```

- **高掩码率**（75%）：使任务足够有挑战性，同时大幅减少编码器计算量
- **非对称**：编码器只处理可见图块，解码器轻量
- **像素级重建**：直接预测被遮挡图块的像素值

$$\mathcal{L} = \frac{1}{|\Omega_M|} \sum_{i \in \Omega_M} \| x_i - \hat{x}_i \|^2$$

## 5. 关键方法/模型

### 5.1 ViT 系列 vs. Swin 系列

| 维度 | ViT | Swin Transformer |
|:---|:---|:---|
| 注意力 | 全局自注意力 | 窗口/移位窗口注意力 |
| 结构 | 平坦序列 | 层级金字塔 |
| 复杂度 | $O(N^2)$ | $O(N)$ |
| 密集预测 | 需适配 | 原生支持 |
| 适用场景 | 分类（大数据预训练） | 分类/检测/分割 |

### 5.2 性能对比

| 模型 | ImageNet Top-1 | COCO AP | ADE20K mIoU |
|:---|:---|:---|:---|
| ResNet-152 | 78.3% | — | — |
| EfficientNet-B7 | 84.4% | — | — |
| ViT-H/14 (JFT-300M) | 88.6% | — | — |
| DeiT-B (ImageNet-1K) | 81.8% | — | — |
| Swin-L (ImageNet-22K) | 86.4% | 57.1 | 53.5 |
| ConvNeXt-XL | 87.8% | 55.8 | 53.0 |
| BEiT-L | 88.4% | 56.7 | 53.3 |

### 5.3 视觉 Transformer 在各任务中的应用

| 任务 | 代表方法 | 特点 |
|:---|:---|:---|
| 图像分类 | ViT / DeiT / Swin | 全局特征学习 |
| 目标检测 | DETR / DINO / Swin-Det | 端到端/层级特征 |
| 图像分割 | Mask2Former / SegFormer | 统一框架 |
| 视频理解 | TimeSformer / ViViT | 时空注意力 |
| 低层视觉 | IPT / SwinIR | 图像恢复/超分辨率 |

## 6. 优势与局限

### 优势

1. **全局建模能力强**：自注意力直接建模长距离依赖，不受卷积核大小限制
2. **可扩展性好**：模型规模和数据量增大时性能持续提升
3. **统一架构**：与 NLP Transformer 统一，便于多模态融合
4. **预训练-微调范式**：MAE/BEiT 等自监督方法提供强大预训练
5. **灵活性强**：通过修改注意力模式可适配不同任务

### 局限

1. **计算复杂度高**：全局注意力 $O(N^2)$ 限制高分辨率输入
2. **小数据表现差**：弱归纳偏置导致小数据集过拟合
3. **内存消耗大**：高分辨率特征图的注意力矩阵占用大量内存
4. **部署挑战**：相比 CNN，Transformer 在边缘设备上推理效率较低
5. **训练不稳定**：对学习率、warmup、正则化等超参数更敏感

## 7. 应用场景

| 场景 | 方法 | 说明 |
|:---|:---|:---|
| 图像分类 | ViT / DeiT | 大规模分类 |
| 医学影像 | Swin-UNETR | 3D 医学图像分割 |
| 遥感分析 | Swin Transformer | 多尺度地物分割 |
| 图像恢复 | SwinIR | 超分辨率/去噪 |
| 视频理解 | TimeSformer | 动作识别 |
| 多模态 | CLIP / BLIP-2 | 视觉语言对齐 |

## 8. 与其他技术关系

- **与 Transformer 的关系**：视觉 Transformer 是 Transformer 架构在视觉领域的应用，核心机制相同。参见 [[08_Transformer与注意力机制]]。
- **与 CNN 的关系**：ViT 与 CNN 是两种不同的视觉架构范式，ConvNeXt 证明 CNN 借鉴 Transformer 设计也能达到竞争性能。参见 [[05_卷积神经网络]]。
- **与自监督学习的关系**：MAE/BEiT/DINO 是视觉 Transformer 的主流预训练方法。参见 [[09_自监督视觉学习]]。
- **与多模态 AI 的关系**：ViT 是 CLIP/BLIP-2 等多模态模型的视觉编码器。参见 [[14_多模态AI]]。
- **与目标检测的关系**：DETR 系列将 Transformer 引入检测，Swin 作为检测 backbone。参见 [[04_目标检测]]。

## 9. 前沿发展

- **高效视觉 Transformer**：FlashAttention、线性注意力、稀疏注意力降低计算成本
- **CNN-Transformer 混合**：MobileViT、CoAtNet 结合两者优势
- **大规模视觉模型**：ViT-22B 等超大规模模型探索 scaling law
- **视频时空 Transformer**：ViViT、TimeSformer 建模视频时空依赖
- **多模态统一架构**：统一视觉和语言的 Transformer 架构（如 Gemini）
- **3D 视觉 Transformer**：Point Transformer、Voxel Transformer 用于 3D 场景理解
- **自监督预训练统一**：MAE-CLIP 结合生成式和对比式预训练
