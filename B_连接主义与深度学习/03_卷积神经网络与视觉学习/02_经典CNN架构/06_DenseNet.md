# DenseNet

## 一句话理解

DenseNet 在一个 dense block 中将每层输出拼接给后续层，实现显式特征复用和短梯度路径，但以更高的特征图存储与带宽成本为代价。

## 技术原理

第 $l$ 层接收此前所有层的拼接特征：

$$x_l=H_l([x_0,x_1,\ldots,x_{l-1}]).$$

这里 $[\cdot]$ 表示通道拼接。**增长率** $k$ 控制每层新增通道数；transition layer 通过 $1\times1$ 卷积和下采样压缩通道与分辨率；bottleneck 设计减轻高维输入的计算量。

## 优势与局限

特征复用可提升参数效率，并有助于梯度流动；但拼接使激活存储、内存访问和工程实现更复杂，在高分辨率或资源受限场景需要审慎评估。

## 相关知识

- 前置：[[05_ResNet]]、[[../../01_卷积运算与基础组件|卷积运算与基础组件]]
- 对比：[[07_EfficientNet]]、[[08_ConvNeXt]]
- 延伸：[[../../05_高效网络架构|高效网络架构]]

## References

- Huang et al. Densely Connected Convolutional Networks. *CVPR*, 2017.
