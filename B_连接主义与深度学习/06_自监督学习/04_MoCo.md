# MoCo

## 一句话理解

MoCo 用动量编码器和动态队列维护大量一致的负样本，降低对超大批量训练的依赖。

## 案例卡

查询编码器快速更新，键编码器以动量慢更新，队列保存历史键。该设计改进了对比字典的规模和一致性，但仍依赖正负样本与增强假设。

## 相关知识

[[02_对比学习]] · [[03_SimCLR]] · [[09_DINO]]

## References

- He et al. Momentum Contrast for Unsupervised Visual Representation Learning. *CVPR*, 2020.
