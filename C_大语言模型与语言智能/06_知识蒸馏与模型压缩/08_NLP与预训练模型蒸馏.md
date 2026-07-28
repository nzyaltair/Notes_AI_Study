# NLP 与预训练模型蒸馏

## 1. 概述

NLP 预训练模型（BERT、GPT 等）的蒸馏是知识蒸馏最成功的应用领域之一。由于预训练模型参数量大、推理延迟高，将其蒸馏为轻量级模型以降低部署成本是工业界的刚需。

DistilBERT (2019) 开创了 BERT 蒸馏的先河——仅用 6 层就保留了 BERT-12 层 97% 的性能。TinyBERT (2020) 进一步设计了针对 Transformer 的两阶段蒸馏方案。这些工作使得在移动设备上运行高质量的 NLP 模型成为可能。

## 2. 发展历史

| 年代 | 里程碑 | 核心意义 |
|:---|:---|:---|
| 2019 | DistilBERT (Sanh et al.) | BERT 蒸馏先驱，6层保留97%性能 |
| 2019 | BERT-PKD (Sun et al.) | BERT 中间层特征蒸馏 |
| 2019 | BERT-of-Theseus (Xu et al.) | 渐进式替换压缩 BERT |
| 2020 | TinyBERT (Jiao et al.) | 两阶段 Transformer 蒸馏 |
| 2020 | MobileBERT (Sun et al.) | 针对移动端优化的 BERT |
| 2020 | MiniLM (Wang et al.) | 深层自注意力蒸馏 |
| 2021 | DistilBERT 多语言版 | 多语言 BERT 蒸馏 |
| 2023 | LLM 时代蒸馏 | Alpaca/Vicuna 开启大模型蒸馏 |

## 3. 核心概念

### 3.1 BERT 蒸馏的维度

| 压缩维度 | 方法 | 效果 |
|:---|:---|:---|
| 层数减少 | DistilBERT: 12→6 层 | 参数减半，推理加速 60% |
| 隐藏维度减少 | MobileBERT: 768→512 | 计算量大幅降低 |
| 注意力头数减少 | 12→4 头 | 计算量降低 |
| 知识类型 | 响应+特征+注意力 | 多种知识联合迁移 |

### 3.2 两阶段蒸馏

TinyBERT 提出的关键策略：

1. **通用蒸馏阶段（General Distillation）**：在预训练数据上蒸馏，让学生学习教师的通用语言知识
2. **任务特定蒸馏阶段（Task-specific Distillation）**：在下游任务数据上蒸馏，学习任务特定知识
3. **数据增强**：任务特定蒸馏中使用教师模型进行数据增强

## 4. 技术原理

### 4.1 DistilBERT

Sanh et al. (2019) 的极简方案：

**学生架构**：BERT-12 层 → 6 层（移除一半层），保留其余架构不变

**损失函数**：

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{KD} + \beta \cdot \mathcal{L}_{MLM} + \gamma \cdot \mathcal{L}_{cos}$$

- $\mathcal{L}_{KD}$：软标签蒸馏损失（KL 散度）
- $\mathcal{L}_{MLM}$：Masked Language Model 损失
- $\mathcal{L}_{cos}$：教师和学生隐藏状态的余弦相似度损失

**效果**：参数减少 40%，推理加速 60%，保留 97% 性能。

### 4.2 TinyBERT

Jiao et al. (2020) 针对 Transformer 设计了细粒度的四层蒸馏：

**Transformer 层蒸馏**：

$$\mathcal{L}_{layer} = \mathcal{L}_{attn} + \mathcal{L}_{hidden}$$

**注意力蒸馏**：

$$\mathcal{L}_{attn} = \sum_l \text{MSE}(A^S_l / \|A^S_l\|, A^T_l / \|A^T_l\|)$$

注意：注意力矩阵在 softmax 之前（raw attention scores）进行匹配。

**隐藏状态蒸馏**：

$$\mathcal{L}_{hidden} = \text{MSE}(W_h H^S, H^T)$$

$W_h$ 为可学习投影矩阵，适配维度差异。

**嵌入层蒸馏**：

$$\mathcal{L}_{embed} = \text{MSE}(W_e E^S, E^T)$$

**预测层蒸馏**（响应蒸馏）：

$$\mathcal{L}_{pred} = \text{KL}(p^T \| p^S)$$

**总损失**：

$$\mathcal{L} = \sum_m \lambda_m \mathcal{L}_m$$

### 4.3 BERT-of-Theseus

Xu et al. (2020) 提出渐进式压缩：将 BERT 的 12 层分为 6 个模块（每模块 2 层），训练一个 6 层的学生模型逐步替换教师模块。

**优势**：避免了蒸馏和微调的两阶段割裂，训练流程更平滑。

### 4.4 MobileBERT

Sun et al. (2020) 针对移动端设计：

- **瓶颈结构**：在 Transformer 层中加入瓶颈（bottleneck）降维
- **注意力近似**：使用优化的注意力计算
- **知识迁移**：从 BERT-Large 蒸馏到 MobileBERT

**效果**：在 Pixel 4 手机上 5.5x 加速，GLUE 性能接近 BERT。

### 4.5 MiniLM

Wang et al. (2020) 的深层蒸馏策略：

- 只蒸馏最后一层 Transformer 的自注意力分布
- 蒸馏值-值关系矩阵（Value-Value relation）
- 不要求教师和学生层数匹配

**优势**：灵活——学生可以是任意层数和维度。

## 5. 方法对比

| 方法 | 学生层数 | 知识类型 | 压缩率 | GLUE 保留率 | 特点 |
|:---|:---|:---|:---|:---|:---|
| DistilBERT | 6 | 响应+MLM+cos | 1.7x | 97% | 极简，工业标准 |
| TinyBERT | 4 | 响应+特征+注意力 | 7.5x | 96% | 两阶段，细粒度 |
| BERT-PKD | 6 | 响应+中间层 | 1.7x | 97% | 中间层蒸馏 |
| MobileBERT | 优化 | 注意力+特征 | 4x | 99% | 移动端优化 |
| MiniLM | 任意 | 注意力+关系 | 灵活 | 98% | 灵活，不要求层数匹配 |
| BERT-of-Theseus | 6 | 渐进替换 | 1.7x | 98% | 渐进式压缩 |

## 6. 优势与局限

### 优势
- **部署成本大幅降低**：DistilBERT 推理速度提升 60%
- **性能保留率高**：优质蒸馏可保留 95%+ 性能
- **生态成熟**：HuggingFace 直接提供蒸馏模型
- **多任务通用**：蒸馏后的学生可适配多种 NLP 任务

### 局限
- **复杂任务精度损失大**：阅读理解等复杂任务蒸馏效果不如分类
- **教师依赖**：学生上限受限于教师
- **训练资源**：需要同时运行教师模型
- **跨语言迁移难**：多语言蒸馏效果不如单语言

## 7. 应用场景

- **搜索引擎**：快速文本排序和分类
- **移动端 NLP**：手机上的文本分类、命名实体识别
- **实时对话**：降低对话系统延迟
- **推荐系统**：CTR 预估模型压缩
- **文档处理**：OCR 后的文本理解

## 8. 与其他技术关系

- **预训练语言模型**：BERT/GPT 是蒸馏的主要教师模型。参见 [[09_预训练语言模型]]
- **Transformer 架构**：蒸馏针对 Transformer 的注意力机制设计。参见 [[08_Transformer与注意力机制]]
- **大模型蒸馏**：LLM 时代的蒸馏从 BERT 扩展到 GPT 系列。参见 [[09_大模型蒸馏]]
- **量化与低秩分解**：蒸馏后可进一步量化。参见 [[07_量化与低秩分解]]

## 9. 前沿发展

- **跨语言蒸馏**：从英语 BERT 蒸馏到多语言小模型
- **蒸馏+量化联合**：DistilBERT + INT8 量化
- **知识图谱增强蒸馏**：结合外部知识改进蒸馏
- **自适应蒸馏**：根据输入难度动态选择蒸馏策略

## 笔记导航

- [[02_响应蒸馏]] — 响应蒸馏原理
- [[03_特征蒸馏]] — 特征蒸馏原理
- [[09_大模型蒸馏]] — LLM 时代的蒸馏
- [[09_预训练语言模型]] — BERT/GPT 预训练模型
