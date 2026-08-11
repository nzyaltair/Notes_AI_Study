# DINO

## 一句话理解

DINO 用教师—学生自蒸馏对齐不同视图的概率输出，促进 ViT 学到可用于分类和检索的语义表示。

## 案例卡

教师由学生的指数滑动平均更新，中心化和温度调节抑制塌缩。DINO 证明自蒸馏可产生有意义的视觉特征，但结果依赖多裁剪和训练配方。

## 相关知识

[[05_BYOL]] · [[08_MAE]] · [[10_CLIP]]

## References

- Caron et al. Emerging Properties in Self-Supervised Vision Transformers. *ICCV*, 2021.
