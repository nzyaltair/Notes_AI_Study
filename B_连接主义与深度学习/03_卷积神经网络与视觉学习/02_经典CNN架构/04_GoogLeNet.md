# GoogLeNet（Inception v1）

## 一句话理解

GoogLeNet 通过 Inception 多分支模块并行提取不同尺度特征，并以 $1\times1$ 卷积控制计算量，在增加网络深度的同时保持较高参数效率。

## 技术原理

一个 Inception 模块并行执行 $1\times1$、$3\times3$、$5\times5$ 卷积和池化，再沿通道维拼接输出。$1\times1$ 卷积可在昂贵卷积前降维，也能引入通道间非线性变换。

GoogLeNet 使用全局平均池化替代大规模全连接层，并配置辅助分类器为中间层提供额外训练信号。后续 Inception 版本继续调整卷积分解、归一化和残差设计。

## 优势与局限

多尺度特征和参数效率使 Inception 在当时具竞争力；但分支结构和超参数多，实际复现与维护成本高。后来的残差网络以更统一的模块成为通用骨干。

## 相关知识

- 前置：[[03_VGG]]、[[../../01_卷积运算与基础组件|卷积运算与基础组件]]
- 对比：[[05_ResNet]]、[[06_DenseNet]]
- 延伸：[[../../05_高效网络架构|高效网络架构]]

## References

- Szegedy et al. Going Deeper with Convolutions. *CVPR*, 2015.
