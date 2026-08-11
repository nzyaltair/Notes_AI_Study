# GPT 类自回归模型

## 一句话理解

GPT 类模型用因果 Transformer 最大化下一 token 条件概率，将生成、补全和条件预测统一为序列建模。

## 核心取舍

训练可并行而生成通常串行；tokenizer、数据、上下文长度与解码策略共同影响能力和成本。模型训练、对齐和系统细节见 C 板块。

## 相关知识

[[../../05_自回归生成模型]] · [[../../../08_Transformer与注意力机制/11_Decoder_only模型\|Decoder-only]] · [[../../../../C_基础模型与通用智能/02_语言基础模型/00_预训练语言模型_综述\|语言基础模型]]

## References

- Brown et al. Language Models are Few-Shot Learners. *NeurIPS*, 2020.
