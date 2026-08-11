# ResNet

## 一句话理解

ResNet 通过恒等捷径连接让残差块学习相对于输入的修正，使非常深的网络更容易优化，并将残差连接发展为通用架构模式。

## 技术原理

残差块将目标映射写为：

$$y=F(x;W)+x.$$

当维度一致时，恒等分支直接传递 $x$；当通道数或步幅改变时，可用投影捷径匹配维度。残差设计不保证每层都只学习小变化，但为信号和梯度提供短路径，通常改善深网络的优化条件。

ResNet-18/34 常使用两个 $3\times3$ 卷积的 BasicBlock；ResNet-50/101/152 采用 $1\times1$ 降维、$3\times3$ 卷积、$1\times1$ 升维的 Bottleneck。

## 优势与局限

ResNet 的模块规整、预训练生态成熟，常用于分类、检测和分割骨干。其卷积局部性依然限制显式全局关系建模；不同任务中应结合分辨率、计算预算和预训练数据选择，而不将 ResNet 视为默认最优。

## 相关知识

- 前置：[[02_AlexNet]]、[[../../01_深度学习基础/03_反向传播与自动微分\|反向传播]]
- 对比：[[06_DenseNet]]、[[08_ConvNeXt]]
- 延伸：[[../../08_Transformer与注意力机制/03_Transformer架构详解\|Transformer 架构]]

## References

- He et al. Deep Residual Learning for Image Recognition. *CVPR*, 2016.
- He et al. Identity Mappings in Deep Residual Networks. *ECCV*, 2016.
