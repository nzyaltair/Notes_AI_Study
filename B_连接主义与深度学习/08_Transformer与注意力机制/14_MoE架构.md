# 稀疏专家模型（MoE）

## 一句话理解

MoE 用路由器为每个 token 激活少数专家，在近似固定计算下扩大参数容量——本质是"更高效的 MLP"：参数量倍增，但每次前向/反向只需承担一个（或 K 个）专家的计算成本。

## 关键取舍

负载均衡、通信、路由稳定性和专家专门化决定实际收益；参数总量不等于每 token 的计算量或推理延迟。

## 要点速览

- 路由器极简（输入与专家向量的内积 + top-k），token 选择是事实标准
- 负载均衡损失不可或缺（否则专家坍缩），DeepSeek-V3 已演进到无辅助损失均衡
- 细粒度 + 共享专家（DeepSeekMoE 设计）是现代开源 MoE 的标准范式
- 专家并行为数据/模型并行之外的第三并行维度，代价是 all-to-all 通信

机制详解（门控公式、均衡损失推导、系统层面、Upcycling、微调坑）见 [[../../C_基础模型与通用智能/02_语言基础模型/03_大语言模型核心架构/06_混合专家模型|混合专家模型]]。

## 相关知识

[[10_Transformer_Scaling]] · [[18_推理优化]] · [[../../K_AI工程化/02_训练工程/00_训练工程_综述\|训练工程]]

## References

- Fedus, Zoph & Shazeer. Switch Transformers. *JMLR*, 2022.
- Muennighoff et al. OLMoE: Open Mixture-of-Expert Language Models. 2024.
