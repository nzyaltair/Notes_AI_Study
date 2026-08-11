# Transformer 与注意力机制综述

## 一句话理解

Transformer 用内容相关的注意力在 token 间动态路由信息，并以残差、归一化和前馈网络堆叠形成可并行训练的序列架构；其核心代价是注意力的计算、显存和长上下文管理。

## 从注意力到 Transformer

给定查询 $Q$、键 $K$ 和值 $V$，缩放点积注意力为：

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

多头机制让不同子空间学习不同关系；位置编码或位置偏置补充序列顺序；残差连接与归一化稳定深层优化。自注意力并不自带因果性、位置或高效性，这些性质取决于掩码、位置方案和实现。

## 主题地图

| 层次 | 主题 | 入口 |
|---|---|---|
| 基础计算 | 注意力、多头、层结构、位置 | [[01_注意力机制原理]]、[[02_Self-Attention与多头注意力]]、[[03_Transformer架构详解]]、[[04_位置编码]] |
| 高效计算 | 稀疏/线性近似、FlashAttention | [[05_高效注意力机制]]、[[17_FlashAttention]] |
| 架构选择 | Encoder、Decoder、Encoder–Decoder、MoE | [[11_Decoder_only模型]]、[[12_Encoder_only模型]]、[[13_Encoder_Decoder模型]]、[[14_MoE架构]] |
| 长序列与推理 | 上下文、KV 缓存、推理成本 | [[15_长上下文]]、[[16_KV_Cache]]、[[18_推理优化]] |
| 理论与规模 | 表达、训练动态、经验规律 | [[09_Transformer理论]]、[[10_Transformer_Scaling]] |

## 学习路径与边界

先掌握 [[../04_序列模型/04_Seq2Seq与交叉注意力\|Seq2Seq 与交叉注意力]]，再学习基础计算和位置表示，最后理解架构取舍与效率。B 板块聚焦机制；模型家族、预训练、对齐见 [[../../C_基础模型与通用智能/02_语言基础模型/00_预训练语言模型_综述\|语言基础模型]]，训练系统与服务见 [[../../K_AI工程化/03_推理工程/00_推理工程_综述\|推理工程]]。

## 局限与趋势

标准全注意力的序列长度复杂度为 $O(n^2)$，但实际瓶颈也受硬件、内存访问和批处理策略影响。长期方向包括高效注意力、稀疏专家、稳定的位置外推、检索/外部记忆与可验证评测；这些方向需要用端到端质量和真实成本共同判断。

## References

- Vaswani et al. Attention Is All You Need. *NeurIPS*, 2017.
- Dao et al. FlashAttention. *NeurIPS*, 2022.
- Fedus, Zoph & Shazeer. Switch Transformers. *JMLR*, 2022.
