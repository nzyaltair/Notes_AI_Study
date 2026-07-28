# 经典CNN架构

## 1. 概述

从1989年LeNet的提出到2022年ConvNeXt的"CNN现代化"，CNN架构经历了三十余年的演进。这一演进不仅是网络深度的增加，更是设计范式的多次转变：从大卷积核到小卷积核堆叠，从直筒网络到多分支设计，从手工设计到神经架构搜索。理解经典架构的设计哲学，是灵活应用和设计CNN的前提。

## 2. 发展历史

```
LeNet (1998)
  │  首个实用CNN，手写数字识别
  ▼
AlexNet (2012)
  │  ImageNet突破，深度学习革命起点，ReLU+Dropout+GPU
  ▼
VGGNet (2014)
  │  统一3×3小卷积核，16-19层深度堆叠
  ▼
GoogLeNet/Inception (2014)
  │  并行多尺度卷积，Inception模块，辅助分类器
  ▼
ResNet (2015)
  │  残差连接突破深度瓶颈，152层超越人类水平
  ▼
DenseNet (2017)
  │  密集特征复用，梯度流动最大化
  ▼
SENet (2017)
  │  通道注意力，轻量级即插即用模块
  ▼
EfficientNet (2019)
  │  神经架构搜索+复合缩放，统一宽度/深度/分辨率
  ▼
ConvNeXt (2022)
     CNN架构"现代化"，吸纳Transformer设计哲学
```

**补充**：ResNeXt（2017）将分组卷积提到架构高度，开启"基数"（Cardinality）作为新维度；MobileNet系列（2017-2019）打开移动端高效CNN路线。

## 3. 核心概念

### 3.1 架构设计的核心维度

| 维度 | 含义 | 增大效果 |
|------|------|----------|
| **深度（Depth）** | 网络层数 | 更强的非线性表达能力，更难训练 |
| **宽度（Width）** | 每层通道数 | 更丰富的特征，计算量平方增长 |
| **分辨率（Resolution）** | 输入图像尺寸 | 更细粒度特征，内存压力增大 |
| **基数（Cardinality）** | 分支/分组数 | ResNeXt的核心维度，比深度/宽度更有效 |

### 3.2 关键设计模式

- **Bottleneck**：先1×1降维 → 3×3卷积 → 1×1升维，用更少参数实现更强的非线性（ResNet-50/101/152的核心）
- **Inception**：多分支并行不同尺度卷积/池化再拼接，网络自主选择最优感受野（GoogLeNet）
- **残差连接**：跳跃连接绕过一层或多层，实现恒等映射的快捷通路（ResNet核心）
- **密集连接**：每层接收前面所有层的特征图拼接，最大化特征复用（DenseNet核心）

## 4. 技术原理——按架构详述

### 4.1 LeNet-5（LeCun et al., 1998）

**结构**：INPUT(32×32) → Conv(6@5×5) → AvgPool(2×2) → Conv(16@5×5) → AvgPool(2×2) → FC(120) → FC(84) → OUTPUT(10)

**设计要点**：
- 卷积-池化-全连接的基本范式沿用至今
- 使用Sigmoid激活函数（当时主流）
- 平均池化而非最大池化
- 只处理32×32灰度图，参数量约6万

**历史地位**：确立了CNN的基本架构范式，证明端到端可训练的CNN能解决实际模式识别问题。

### 4.2 AlexNet（Krizhevsky, Sutskever, Hinton, 2012）

**核心创新**：

| 创新 | 说明 |
|------|------|
| **ReLU激活** | 替代Sigmoid/Tanh，缓解梯度消失，训练快6倍 |
| **Dropout** | 全连接层使用p=0.5，缓解过拟合 |
| **重叠池化** | stride < kernel size，提升特征鲁棒性 |
| **双GPU训练** | 分组卷积雏形，缓解显存限制 |
| **数据增强** | 随机裁剪、水平翻转、PCA色彩扰动 |
| **LRN** | 局部响应归一化（后被BatchNorm取代） |

**结构**：5个卷积层 + 3个全连接层，参数量约60M。

**历史意义**：ImageNet 2012冠军，top-5错误率从26.2%降至15.3%（相对降低~41%），标志着深度学习在计算机视觉领域的革命性突破。

### 4.3 VGGNet（Simonyan & Zisserman, 2014）

**核心思想**：用多个**3×3小卷积核的堆叠**替代大卷积核。

- 两个3×3卷积 = 一个5×5卷积的感受野，但参数量更少（$2\times 9 = 18$ vs $25$）
- 三个3×3卷积 = 一个7×7卷积的感受野，且具有更强的非线性（3个ReLU vs 1个）

**VGG16/19结构**：统一的13/16个卷积层（3×3）+ 5个最大池化（2×2）+ 3个全连接层，参数量约138M。

**贡献与局限**：
- ✅ 结构极度规整统一，易于理解和复现
- ✅ 证明了深度对性能的重要性
- ❌ 参数量巨大（全连接层占约90%），推理慢
- ❌ 梯度消失问题限制进一步加深

### 4.4 GoogLeNet / Inception v1（Szegedy et al., 2014）

**核心创新——Inception模块**：在同一层并行使用1×1、3×3、5×5卷积和3×3最大池化，将四路输出在通道维度拼接。1×1卷积用于降维（Bottleneck设计），控制计算量。

**关键设计**：
- 用**全局平均池化**（GAP）替代最后的全连接层，大幅减少参数量（~5M vs VGG16的138M）
- 三个位置添加**辅助分类器**，提供额外的梯度信号，缓解深层梯度消失
- 22层深度，但参数量仅为AlexNet的1/12

**后续演进**：
- **Inception v2/v3**（2015）：用两个3×3替代5×5，引入BatchNorm，提出标签平滑
- **Inception v4 + Inception-ResNet**（2016）：吸收ResNet残差思想

### 4.5 ResNet（He et al., 2015）

> **论文**: Deep Residual Learning for Image Recognition (CVPR 2016 Best Paper)

**核心创新——残差学习**：

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

不直接学习目标映射 $\mathcal{H}(\mathbf{x})$，而学习残差 $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$。当最优映射接近恒等时，残差趋于零，网络只需学习微小的修正。

**梯度视角**：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \cdot \left(\frac{\partial \mathcal{F}(\mathbf{x})}{\partial \mathbf{x}} + I\right)$$

恒等矩阵 $I$ 确保梯度至少有一条无损通道直达浅层，**彻底解决梯度消失问题**。

**两种残差块**：

| 类型 | 结构 | 适用 |
|------|------|------|
| **BasicBlock** | 3×3→3×3 | ResNet-18/34 |
| **Bottleneck** | 1×1(降维)→3×3→1×1(升维) | ResNet-50/101/152 |

**关键技巧**：
- **BatchNorm** 在每次卷积后、激活前
- 下采样时残差分支通过1×1卷积（stride=2）匹配维度
- 使用 **He初始化**（Kaiming Initialization）防止ReLU导致的方差衰减

**ResNet变体**：
- **ResNeXt**（2017）：用分组卷积引入"基数"维度，同等复杂度下性能更优
- **Wide ResNet**（2016）：减少深度增加宽度，加速训练
- **PreAct ResNet**：BN→ReLU→Conv的顺序（预激活），进一步改善梯度流

**历史意义**：残差连接的思想被广泛继承——Transformer中的残差连接（Add & Norm）、DenseNet的密集连接、U-Net的跳跃连接本质上都源于ResNet的洞察。

### 4.6 DenseNet（Huang et al., 2017）

**核心思想**：每层的输入是所有前面层输出的**通道拼接**（而非相加）。对于 $L$ 层的Dense Block，第 $l$ 层接收 $l$ 个特征图的拼接。

$$x_l = H_l([x_0, x_1, ..., x_{l-1}])$$

其中 $[\cdot]$ 表示通道拼接。

**关键设计**：
- **增长率（Growth Rate, $k$）**：每层新增的特征图通道数（通常很小，如12/24/32）
- **Bottleneck层**：BN-ReLU-1×1Conv-BN-ReLU-3×3Conv，控制拼接后通道数的爆炸增长
- **Transition层**：1×1卷积 + 平均池化，在Dense Block之间压缩通道数和分辨率

**优势**：
- 极强的梯度流——每个Block内部所有层直接连接
- 高效的特征复用——浅层特征直接传递到深层
- 参数效率高——$k$ 很小使总参数量可控

**局限**：特征图拼接导致内存消耗大，训练/推理时需要更大的显存。

### 4.7 ConvNeXt（Liu et al., 2022）

> **核心动机**：Swin Transformer证明了Transformer可在视觉任务上超越CNN。ConvNeXt的反问是："是Transformer架构的本质优势，还是具体的**设计细节差异**？"

**ConvNeXt的"现代化"改造**（基于ResNet-50逐步修改）：

| 步骤 | 修改内容 | 来源 |
|------|----------|------|
| 1. 训练策略 | AdamW优化器、数据增强、更长epoch | DeiT/Swin |
| 2. 宏观设计 | Stage比例从(3,4,6,3)改为(3,3,9,3) | Swin |
| 3. 下采样 | 用patchify stem（4×4 Conv, stride=4） | Swin/ViT |
| 4. Block设计 | Depthwise Conv + 倒置Bottleneck | MobileNet + Transformer |
| 5. 卷积核 | 增大到7×7 | Swin（7×7窗口注意力） |
| 6. 激活函数 | ReLU → GELU | Transformer |
| 7. 归一化 | BatchNorm → LayerNorm | Transformer |
| 8. 额外技巧 | 减少激活和归一化层 | Transformer哲学 |

**ConvNeXt Block结构**：

```
输入 → Depthwise Conv(7×7) → LayerNorm → 1×1 Conv(升维4×) → GELU → 1×1 Conv(降维) → 残差相加 → 输出
```

**核心启示**：
1. CNN和Transformer的架构差异可能并不是本质性的
2. 标准CNN经过合理的"现代化"改造，可以达到与Swin Transformer相当的性能
3. 这打开了CNN-Transformer融合和架构重新设计的广阔空间

## 5. 关键方法/模型对比

| 架构 | 年份 | 深度 | 参数量(ImageNet) | Top-5错误率 | 核心创新 |
|------|------|------|-----------------|-------------|----------|
| LeNet-5 | 1998 | 5 | 60K | — | CNN范式奠基 |
| AlexNet | 2012 | 8 | 60M | 15.3% | ReLU/Dropout/GPU |
| VGG-16 | 2014 | 16 | 138M | 7.3% | 3×3统一卷积核 |
| GoogLeNet v1 | 2014 | 22 | 5M | 6.7% | Inception多尺度 |
| ResNet-50 | 2015 | 50 | 25M | 5.3% | 残差连接 |
| ResNet-152 | 2015 | 152 | 60M | 4.5% | 极深残差网络 |
| DenseNet-201 | 2017 | 201 | 20M | — | 密集连接 |
| EfficientNet-B7 | 2019 | — | 66M | 2.9% | 复合缩放+NAS |

## 6. 优势与局限——按架构

### ResNet系列
- ✅ 训练极深网络（1000层+）的能力
- ✅ 预训练权重迁移性极强，至今仍是主流骨干
- ❌ 并非最优结构——ResNeXt/EfficientNet在同等开销下更优

### Inception系列
- ✅ 多尺度特征融合，计算高效
- ❌ 超参数众多，复现和调优复杂

### DenseNet
- ✅ 参数效率高，梯度流好
- ❌ 内存消耗大，不适合大分辨率输入

## 7. 应用场景

| 架构 | 最适合场景 |
|------|------------|
| ResNet-50/101 | 目标检测/分割的骨干网络（Faster R-CNN、Mask R-CNN标配） |
| ResNet-18/34 | 轻量级分类、快速实验验证 |
| VGG-16 | 风格迁移（感知损失常用）、教学演示 |
| DenseNet | 需要强特征复用的小样本任务 |
| ConvNeXt | 追求SOTA且不引入Transformer的纯CNN方案 |

## 8. 与其他技术关系

- 残差连接 → [[00_Transformer与注意力机制_综述|Transformer]] 的 Add & Norm 设计
- VGG的规整性 → [[../15_计算机视觉/05_图像分割|图像分割]] FCN直接复用VGG作为编码器
- ResNet的Bottleneck设计 → MobileNet v2倒置残差（Inverted Residual）的来源
- Inception的多尺度思想 → FPN（特征金字塔）的多尺度特征融合
- ConvNeXt → [[06_CV前沿发展]] 中CNN现代化的里程碑

## 9. 前沿发展

1. **架构搜索与自动化设计**：EfficientNet的复合缩放、RegNet的设计空间探索，逐步从手工设计走向自动化
2. **CNN-Transformer混合架构**：CoAtNet（卷积+注意力混合）、MaxViT（多轴注意力）、InternImage（可变形卷积骨干）
3. **NAS的实用化**：Once-for-All（OFA）通过一次训练产出子网络族，适配不同硬件约束
4. **大卷积核复兴**：ConvNeXt（7×7）、RepLKNet（31×31）、SLaK证明了足够大的卷积核在现代化设计下与注意力机制一样有效
5. **模型缩放律研究**：Chinchilla缩放律启发的对视觉模型的最优计算分配研究
