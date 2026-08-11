# 稀疏专家模型（MoE）

## 一句话理解

MoE 用路由器为每个 token 激活少数专家，在近似固定计算下扩大参数容量。

## 关键取舍

负载均衡、通信、路由稳定性和专家专门化决定实际收益；参数总量不等于每 token 的计算量或推理延迟。

## 相关知识

[[10_Transformer_Scaling]] · [[18_推理优化]] · [[../../K_AI工程化/02_训练工程/00_训练工程_综述\|训练工程]]

## References

- Fedus, Zoph & Shazeer. Switch Transformers. *JMLR*, 2022.
