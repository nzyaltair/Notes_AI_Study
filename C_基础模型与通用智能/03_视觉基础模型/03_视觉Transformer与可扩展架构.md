# 视觉 Transformer 与可扩展架构

## 一句话理解

ViT 将图像切分为 patch token，再由 Transformer 建模全局关系；它以较弱的局部先验换取可随数据和计算规模扩展的统一架构。

## 核心机制

图像被分为 \(N\) 个 patch 并投影为 token \(z_0=[x_1E;\dots;x_NE]+p\)。注意力成本随 \(N^2\) 增长，因此高分辨率与视频需要窗口化、层级设计、token 压缩或高效注意力。ViT/DeiT 强调统一可扩展性，Swin 以窗口与层级特征改善密集任务效率，混合架构保留卷积归纳偏置。

## 相关知识

- 前置：[[../../B_连接主义与深度学习/08_Transformer与注意力机制/00_Transformer与注意力机制_综述|Transformer]]
- 同层：[[02_视觉表征预训练目标|视觉预训练目标]]
- 延伸：[[07_视频基础模型|视频基础模型]]

## References

- Dosovitskiy et al. *ViT* (2021).
- Liu et al. *Swin Transformer* (2021).