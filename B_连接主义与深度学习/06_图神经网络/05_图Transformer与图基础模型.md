# 图 Transformer 与图基础模型

## 1. 概述

传统消息传递 GNN（MPNN）通过局部邻居聚合更新节点表示，存在两个根本局限：(1) **感受野受限**——$L$ 层 GNN 只能感知 $L$ 跳邻居；(2) **表达力上限**——不超过 1-WL test。**Graph Transformer** 将 [[08_Transformer与注意力机制|Transformer]] 的全局自注意力引入图域，使每个节点可以直接关注图中所有节点，突破局部聚合的限制。

**图基础模型**（Graph Foundation Model）是当前图学习的前沿方向——类比 NLP 中 BERT/GPT 的范式，通过大规模图预训练学习通用图表示，然后迁移到下游任务。目标是实现"一次预训练，多图多任务迁移"。

## 2. 发展历史

| 年代 | 里程碑 | 作者/机构 | 核心意义 |
|:---|:---|:---|:---|
| 2018 | GAT | Veličković et al. | 图上的局部注意力（Graph Transformer 雏形） |
| 2020 | Graph-BERT | Zhang et al. | 图上的纯 Transformer（无图卷积） |
| 2021 | **Graphormer** | Ying et al. (Microsoft) | NeurIPS 最佳论文，空间+度编码 |
| 2021 | SAN | Kreuzer et al. | 谱域注意力网络 |
| 2021 | Graphormer-DO | Microsoft | 药物发现应用 |
| 2022 | **GraphGPS** | Rampášek et al. | 通用框架：MPNN + Transformer |
| 2022 | GPS++ | Zhang et al. | 优化 GPS 的位置编码 |
| 2022 | Exphormer | Shirzad et al. | 稀疏扩展注意力，$O(n)$ 复杂度 |
| 2022 | GraphMAE | Hou et al. | 自监督掩码图预训练 |
| 2022 | GraphMAE2 | Hua et al. | 改进的图自监督预训练 |
| 2023 | GPT4Graph / GraphGPT | 多家 | LLM 用于图理解 |
| 2023 | OFA (One For All) | Liu et al. | 统一图基础模型框架 |
| 2024 | GraphAny | Jiang et al. | 任意图推理的基础模型 |
| 2024-25 | Graph LLM | 多家 | GNN + LLM 深度融合 |

## 3. 核心概念

### 3.1 全局注意力 vs 局部聚合

| 维度 | MPNN（GCN/GAT） | Graph Transformer |
|:---|:---|:---|
| 感受野 | $L$ 跳 | 全局（所有节点） |
| 计算复杂度 | $O(\|\mathcal{E}\| \cdot d)$ | $O(n^2 \cdot d)$ |
| 表达力 | ≤ 1-WL | > 1-WL（理论上） |
| 长程依赖 | 弱 | 强 |
| 可扩展性 | 好（稀疏） | 差（密集注意力） |

### 3.2 将图结构注入 Transformer

标准 Transformer 不知道图结构，需要**结构编码**注入拓扑信息：

| 编码类型 | 原理 | 代表 |
|:---|:---|:---|
| 空间编码（Spatial Encoding） | 最短路径距离作为偏置 | Graphormer |
| 度编码（Degree Encoding） | 节点度作为特征/偏置 | Graphormer |
| 边编码（Edge Encoding） | 最短路径上的边特征 | Graphormer |
| 谱编码（Spectral Encoding） | 拉普拉斯特征向量作为位置编码 | SAN, GraphGPS |
| 随机游走编码 | 随机游走转移概率 | Graphormer-GD |

### 3.3 图基础模型

类比 NLP 的预训练-微调范式：

```
大规模图数据 → 图基础模型预训练 → 微调/零样本迁移到下游任务
```

**挑战**：
- 不同图的节点特征空间不同（特征异质性）
- 图结构差异大（拓扑异质性）
- 缺乏统一的 token 化方案
- 任务多样性（节点/边/图级）

## 4. 技术原理

### 4.1 Graphormer

Graphormer 的核心创新是三种结构编码：

**1. 空间编码（Spatial Encoding）**：

$$A_{ij} = b_{\text{spd}(v_i, v_j)}$$

其中 $\text{spd}(v_i, v_j)$ 是节点 $i$ 到 $j$ 的最短路径距离，$b$ 是可学习的偏置标量。距离越远的节点对，注意力偏置越大（通常越小）。

**2. 度编码（Degree Encoding）**：

将节点的入度和出度作为额外的嵌入加到节点特征上：

$$h_i = x_i + z_{\text{deg}^-(v_i)}^{\text{in}} + z_{\text{deg}^+(v_i)}^{\text{out}}$$

度编码让模型感知节点的结构重要性（hub 节点 vs 叶节点）。

**3. 边编码（Edge Encoding）****：

沿最短路径的边特征求平均，作为注意力偏置：

$$A_{ij} += \frac{1}{n}\sum_{l=1}^{n} x_{e_l} \cdot w_l^{E}$$

其中 $e_1, \dots, e_n$ 是 $v_i$ 到 $v_j$ 最短路径上的边。

**完整注意力公式**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + A_{ij}\right)V$$

### 4.2 GraphGPS 通用框架

GraphGPS 提出了一个**通用框架**，将 MPNN 和 Transformer 组合：

$$h_v^{(l+1)} = \text{MLP}\left(h_v^{(l)} + \text{MPNN}^{(l)}(h_v^{(l)}) + \text{GlobalAttn}^{(l)}(h_v^{(l)})\right)$$

- $\text{MPNN}^{(l)}$：局部消息传递（GCN/GAT/GIN 等任意 MPNN）
- $\text{GlobalAttn}^{(l)}$：全局自注意力（Performer/BigBird 等高效注意力）
- 两者并行计算后融合

**优势**：统一了多种 Graph Transformer 变体，通过选择不同的 MPNN 和注意力模块可灵活组合。

### 4.3 Exphormer：稀疏注意力

Graphormer 的 $O(n^2)$ 复杂度限制了大图应用。Exphormer 用**稀疏扩展图**（Expander Graph）近似全局注意力：

$$\text{Attn}_{\text{sparse}} = \text{LocalAttn} + \text{VirtualNode} + \text{ExpanderAttn}$$

- **LocalAttn**：只关注 $k$ 跳邻居（$O(kn)$）
- **VirtualNode**：虚拟全局节点连接所有节点（$O(n)$）
- **ExpanderAttn**：随机扩展图连接，每个节点连 $O(1)$ 个远距离节点

总复杂度 $O(n)$，同时保留全局感受野。

### 4.4 SAN（Spectral Attention Network）

SAN 将**谱域位置编码**引入 Transformer：

$$\text{PosEncoding}_v = \sum_{k=1}^{K} \lambda_k^{-1/2} u_k(v) \cdot \mathbf{p}_k$$

其中 $u_k$ 是拉普拉斯第 $k$ 个特征向量，$\lambda_k$ 是对应特征值，$\mathbf{p}_k$ 是可学习位置嵌入。

谱位置编码为节点提供了全局结构位置信息——类似 Transformer 中正弦位置编码在序列中的角色。

### 4.5 图自监督预训练

**GraphMAE**（自监督掩码图预训练）：

1. **掩码**：随机遮蔽部分节点特征
2. **编码**：用 GNN 编码被掩码的图
3. **重建**：用解码器重建被掩蔽的节点特征
4. **损失**：缩放余弦误差

$$\mathcal{L} = \frac{1}{|\tilde{V}|}\sum_{v \in \tilde{V}} \left(1 - \frac{h_v \cdot x_v}{\|h_v\| \|x_v\|}\right)^\gamma$$

其中 $\gamma$ 是缩放因子，对高误差样本加权。

### 4.6 图基础模型迁移

**OFA（One For All）框架**：

1. **统一节点特征**：用预训练文本编码器（如 Sentence-BERT）将所有图的节点文本特征编码为统一向量
2. **图 token 化**：将节点特征统一到同一嵌入空间
3. **预训练**：在大规模图数据上预训练统一 GNN
4. **零样本迁移**：将预训练模型直接应用于新图的下游任务

## 5. 关键方法与模型

### 5.1 Graph Transformer 对比

| 模型 | 结构编码 | 注意力 | 复杂度 | 特点 |
|:---|:---|:---|:---|:---|
| Graph-BERT | 随机游走 + 度 | 全局 | $O(n^2)$ | 纯 Transformer |
| Graphormer | SPD + 度 + 边 | 全局 | $O(n^2)$ | 严格结构编码 |
| SAN | 谱位置编码 | 全局 | $O(n^2)$ | 谱域+空域统一 |
| GraphGPS | 灵活 | MPNN+全局 | $O(n^2)$ | 通用框架 |
| Exphormer | 局部+扩展 | 稀疏 | $O(n)$ | 可扩展 |
| Graphormer-GD | 随机游走 | 全局 | $O(n^2)$ | 图距离近似 |

### 5.2 图预训练方法

| 方法 | 预训练任务 | 类型 | 代表 |
|:---|:---|:---|:---|
| Context Prediction | 邻域-上下文匹配 | 对比 | GPT-GNN |
| Edge Prediction | 链路预测 | 判别 | DGI, GRACE |
| Masked Attribute | 特征重建 | 生成 | GraphMAE, GraphMAE2 |
| Contrastive | 图级对比 | 对比 | InfoGraph, GraphCL |
| Hybrid | 多任务 | 混合 | GPT-GNN |

## 6. 优势与局限

### 6.1 优势

- **全局感受野**：每个节点可直接关注所有节点，捕捉长程依赖
- **超越 1-WL**：全局注意力的表达力超过消息传递
- **灵活编码**：可注入丰富的图结构信息（距离、度、边特征）
- **预训练迁移**：图基础模型可实现跨图、跨任务的知识迁移

### 6.2 局限

- **计算复杂度**：$O(n^2)$ 注意力限制了大图应用（Exphormer 等稀疏方法部分缓解）
- **结构编码依赖**：最短路径距离等编码的预计算成本高
- **预训练挑战**：图数据的异质性使预训练-迁移不如 NLP 直接
- **小图优势有限**：在小图上 Graph Transformer 未必优于 MPNN

## 7. 应用场景

| 领域 | 应用 | 代表模型 |
|:---|:---|:---|
| 分子建模 | 分子属性预测、药物发现 | Graphormer, Graphormer-DO |
| 大规模图 | 百万节点级图分类 | Exphormer |
| 跨图迁移 | 不同数据集间的零样本推理 | OFA, GraphAny |
| 科学发现 | 蛋白质结构、材料属性 | Graph Transformer |
| 知识图谱 | 关系推理、链路预测 | HGT |

## 8. 与其他技术关系

- [[08_Transformer与注意力机制|Transformer]]：Graph Transformer 是 Transformer 在图域的直接扩展
- [[03_空域图神经网络|空域 GNN]]：GAT 是局部 Graph Transformer，Graphormer 是全局版
- [[02_谱域图卷积|谱域图卷积]]：SAN 将谱域位置编码引入 Transformer
- [[09_预训练语言模型|预训练语言模型]]：图预训练范式直接受 BERT/GPT 启发
- [[10_大语言模型核心架构|LLM]]：图基础模型借鉴 LLM 的预训练-微调范式
- [[14_多模态AI|多模态 AI]]：GNN+LLM 融合是多模态推理的重要方向

## 9. 前沿发展

- **图 LLM**：直接用 LLM 处理图（将图序列化为文本），或 GNN+LLM 混合架构
- **图 token 化**：将图结构统一为 token 序列，使 LLM 能处理任意图
- **零样本图推理**：预训练模型在全新图上零样本推理（GraphAny）
- **可扩展 Graph Transformer**：线性复杂度的全局注意力（Performer, BigBird for graphs）
- **3D 分子 Transformer**：将 3D 坐标信息引入 Graph Transformer（Equiformer, SE(3)-Transformer）
- **图基础模型规模化**：从百万级到十亿级节点的预训练
- **多模态图预训练**：图结构 + 文本 + 图像的联合预训练

## 学习资源

### 经典论文
- Ying et al., *Do Transformers Really Perform Good for Graph Representation?* (NeurIPS 2021) — Graphormer
- Kreuzer et al., *Rethinking Graph Transformers with Spectral Attention* (NeurIPS 2021) — SAN
- Rampášek et al., *Recipe for a General, Powerful, Scalable Graph Transformer* (NeurIPS 2022) — GraphGPS
- Shirzad et al., *Exphormer: Sparse Transformers for Graphs* (ICML 2023)
- Hou et al., *GraphMAE: Self-Supervised Masked Graph Autoencoders* (KDD 2022)
- Liu et al., *Graph Foundation Models: Are We Really There Yet?* (2024)

### 实践任务
1. 实现 Graphormer 的空间编码和度编码
2. 在 ZINC 数据集上对比 Graphormer 和 GIN 的性能
3. 使用 GraphGPS 框架组合不同 MPNN + Transformer
4. 实现 GraphMAE 进行自监督图预训练
5. 分析 Graph Transformer 在大图上的计算瓶颈
