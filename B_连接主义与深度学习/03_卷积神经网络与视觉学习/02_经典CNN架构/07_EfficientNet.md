# EfficientNet

## 一句话理解

EfficientNet 以复合缩放同时调整网络深度、宽度和输入分辨率，并从硬件感知搜索得到基础网络，在精度与计算量之间建立可扩展的 CNN 系列。

## 核心思想

在给定额外计算预算时，复合缩放以系数共同扩大深度 $d$、宽度 $w$ 和分辨率 $r$，而非只增加某一个维度。基础网络 EfficientNet-B0 使用神经架构搜索得到；B1–B7 在同一缩放规则下扩展。

其主要模块 MBConv 来自移动网络：先扩展通道、深度可分离卷积、再投影，并可配合 squeeze-and-excitation 通道注意力。

## 优势与局限

EfficientNet 为固定图像分类设置提供了良好精度—计算权衡；但实际吞吐还受硬件、内存访问、输入分辨率和实现影响，FLOPs 不应单独替代端侧延迟测量。

## 相关知识

- 前置：[[06_DenseNet]]、[[../../05_高效网络架构|高效网络架构]]
- 对比：[[08_ConvNeXt]]
- 延伸：[[../../10_神经架构搜索NAS与自动化设计/00_神经架构搜索NAS_综述\|NAS]]

## References

- Tan & Le. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*, 2019.
