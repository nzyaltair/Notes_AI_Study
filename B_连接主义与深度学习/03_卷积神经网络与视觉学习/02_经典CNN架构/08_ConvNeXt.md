# ConvNeXt

## 一句话理解

ConvNeXt 在 ResNet 骨架上引入现代训练配方、大核深度可分离卷积、LayerNorm 和倒置瓶颈，说明经过现代化设计的 CNN 仍可成为强视觉骨干。

## 设计要点

ConvNeXt 并非简单复制 Transformer，而是比较并吸收有效设计：更长训练和 AdamW、分阶段深度比例调整、patchify stem、$7\times7$ 深度可分离卷积、倒置 bottleneck、GELU 与 LayerNorm。其结果说明架构归纳偏置与训练配方应被共同评估。

## 优势与局限

ConvNeXt 保留卷积的局部性和成熟视觉算子，同时在分类与密集预测中具有竞争力。它不证明 CNN 与 Transformer 等价；全局交互、预训练规模、输入分辨率与部署硬件仍会改变选型结论。

## 相关知识

- 前置：[[05_ResNet]]、[[07_EfficientNet]]
- 对比：[[../../08_Transformer与注意力机制/03_Transformer架构详解\|Transformer]]
- 延伸：[[../../05_高效网络架构|高效网络架构]]

## References

- Liu et al. A ConvNet for the 2020s. *CVPR*, 2022.
