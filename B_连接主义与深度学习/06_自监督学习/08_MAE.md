# MAE

## 一句话理解

MAE 以高比例遮蔽图像 patch、仅编码可见部分并用轻量解码器复原输入，从而高效预训练视觉编码器。

## 案例卡

高遮蔽率迫使模型利用可见 patch 间的结构推断，编码器无需处理全部 token，降低预训练成本。像素重建目标与语义判别目标不同，迁移效果须以下游评估确认。

## 相关知识

[[07_掩码建模]] · [[09_DINO]] · [[../08_Transformer与注意力机制/03_Transformer架构详解\|Transformer]]

## References

- He et al. Masked Autoencoders Are Scalable Vision Learners. *CVPR*, 2022.
