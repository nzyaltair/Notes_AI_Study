# SimCLR

## 一句话理解

SimCLR 以强数据增强、共享编码器和大批量 InfoNCE 建立简洁的实例对比学习基线。

## 案例卡

同一图像的两种增强视图构成正对；其结果表明增强策略、投影头和温度与网络骨干同样关键。大批量负样本带来训练成本，后续 MoCo 用队列缓解该限制。

## 相关知识

[[02_对比学习]] · [[04_MoCo]] · [[05_BYOL]]

## References

- Chen et al. A Simple Framework for Contrastive Learning of Visual Representations. *ICML*, 2020.
