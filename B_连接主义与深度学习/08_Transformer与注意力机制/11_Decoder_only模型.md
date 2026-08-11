# Decoder-only 模型

## 一句话理解

Decoder-only Transformer 用因果掩码按从左到右的条件概率训练，适合开放式自回归生成。

## 取舍

统一的下一 token 预测目标便于扩展，但双向上下文理解、长序列成本和可靠控制需要额外设计。

## 相关知识

[[13_Encoder_Decoder模型]] · [[../07_生成模型/05_自回归生成模型\|自回归模型]] · [[../../C_基础模型与通用智能/02_语言基础模型/00_预训练语言模型_综述\|语言基础模型]]

## References

- Radford et al. Language Models are Unsupervised Multitask Learners. 2019.
