# AlexNet

## 一句话理解

AlexNet 以 GPU 训练、ReLU、数据增强和 Dropout 证明了深层 CNN 能在大规模自然图像识别上取得突破，成为现代深度学习复兴的重要节点。

## 发展背景

2012 年 ImageNet 分类竞赛此前仍以手工特征为主。AlexNet 将可训练卷积网络扩展到百万级图像、千类分类问题，显著降低 top-5 错误率，展示了数据、算力和训练技巧共同扩展网络能力的路径。

## 核心设计

网络由 5 个卷积层和 3 个全连接层构成。其关键不是某一个算子，而是协同使用：

- **ReLU**：相较 sigmoid/tanh 更易优化深层网络。
- **数据增强**：随机裁剪、翻转和颜色扰动缓解过拟合。
- **Dropout**：主要用于大规模全连接层的正则化。
- **GPU 并行训练**：受当时显存限制使用双 GPU 分组，推动大规模卷积训练。

局部响应归一化（LRN）属于历史设计，现代 CNN 通常优先采用 BatchNorm、LayerNorm 或不使用对应模块。

## 优势与局限

AlexNet 建立了可扩展视觉训练范式，但大卷积核、全连接分类头和较高参数量已不符合当前效率需求。VGG 提升结构规整性，ResNet 用残差连接解决更深网络的优化困难。

## 相关知识

- 前置：[[01_LeNet]]、[[../../01_卷积运算与基础组件|卷积运算与基础组件]]
- 对比：[[03_VGG]]、[[04_GoogLeNet]]
- 延伸：[[05_ResNet]]、[[../../05_高效网络架构|高效网络架构]]

## References

- Krizhevsky, Sutskever & Hinton. ImageNet Classification with Deep Convolutional Neural Networks. *NeurIPS*, 2012.
