# Transformer 变体与架构演进

## 1. 概述

自 2017 年原始 Transformer 提出以来，围绕不同任务需求和设计理念，衍生出了大量架构变体。最重要的分化是**编码器-only**（BERT 路线）、**解码器-only**（GPT 路线）和**编码器-解码器**（T5 路线）三种范式。此外，Transformer 还扩展到了计算机视觉（ViT）、语音（Conformer）、科学计算（AlphaFold）等领域。本笔记梳理 Transformer 架构变体的设计选择、演进脉络和适用场景。

- **解决的问题**：原始 Transformer 为机器翻译设计，不同任务（理解 vs 生成、NLP vs CV）对架构有不同要求，需要针对性优化。
- **核心价值**：理解架构变体的设计取舍，有助于在不同场景选择合适的模型架构。

## 2. 发展历史

| 年代 | 模型 | 架构 | 核心创新 |
|:---|:---|:---|:---|
| 2017 | Transformer | Encoder-Decoder | 自注意力 |
| 2018 | BERT | Encoder-only | 双向预训练 + MLM |
| 2018 | GPT-1 | Decoder-only | 自回归预训练 |
| 2019 | T5 | Encoder-Decoder | 统一 text-to-text |
| 2019 | XLNet | 混合 | 排列语言建模 |
| 2019 | ALBERT | Encoder-only | 参数共享 + SOP |
| 2019 | Transformer-XL | Decoder+循环 | 相对位置 + 段级循环 |
| 2020 | GPT-3 | Decoder-only | 175B 规模化 |
| 2020 | ViT | Encoder-only | 图像 Patch 序列化 |
| 2020 | DEiT | Encoder-only | ViT 蒸馏优化 |
| 2021 | Swin Transformer | Encoder-only | 层级结构 + 滑动窗口 |
| 2021 | DeBERTa | Encoder-only | 解耦注意力 |
| 2021 | BART | Encoder-Decoder | 去噪自编码器 |
| 2022 | PaLM | Decoder-only | SwiGLU + MQA + 540B |
| 2023 | LLaMA 2 | Decoder-only | GQA + RMSNorm + RoPE |
| 2023 | Mistral | Decoder-only | 滑动窗口 + GQA |
| 2023 | Mixtral | Decoder-only MoE | 8x7B 稀疏 MoE |
| 2024 | LLaMA 3 | Decoder-only | 128K 上下文 |
| 2024 | DeepSeek-V3 | Decoder-only MoE | MLA + 细粒度 MoE |
| 2025 | Command A / Gemma 3 / Llama 4 | Decoder-only | 滑动窗口与全注意力交替（长上下文性价比方案） |
| 2025 | Perse | 循环 Transformer | 模块循环复用 + 谱半径稳定性约束 |
| 2025-2026 | Kimi Linear / Qwen3-Next 等 | 混合架构 | Gated DeltaNet 等线性层与全注意力交替 |

## 3. 核心概念

### 3.1 三种架构范式

| 范式 | 注意力类型 | 预训练目标 | 优势 | 适合任务 |
|:---|:---|:---|:---|:---|
| **Encoder-only** | 双向自注意力 | 掩码语言建模（MLM） | 全局双向理解 | 分类、NER、问答 |
| **Decoder-only** | 因果自注意力 | 自回归语言建模（LM） | 生成 + zero-shot | 生成、对话、推理 |
| **Encoder-Decoder** | 自注意力 + 交叉注意力 | 去噪/-span corruption | 源-目标对齐 | 翻译、摘要 |

### 3.2 预训练目标

| 目标 | 公式 | 架构 | 代表模型 |
|:---|:---|:---|:---|
| **掩码语言建模（MLM）** | 预测被 `[MASK]` 替换的 token | Encoder | BERT |
| **自回归语言建模（LM）** | $P(y_t \| y_{<t})$ | Decoder | GPT |
| **Span Corruption** | 预测被替换的连续 span | Encoder-Decoder | T5 |
| **去噪自编码（DAE）** | 重建被噪声破坏的文本 | Encoder-Decoder | BART |
| **排列语言建模（PLM）** | 在随机排列上自回归 | 混合 | XLNet |

### 3.3 架构设计维度

Transformer 架构的关键设计选择：

| 维度 | 选项 | 现代趋势 |
|:---|:---|:---|
| 架构类型 | Encoder / Decoder / Enc-Dec | Decoder-only |
| 归一化 | Post-LN / Pre-LN / RMSNorm | Pre-RMSNorm |
| 激活函数 | ReLU / GELU / SwiGLU | SwiGLU |
| 位置编码 | 正弦 / 可学习 / 相对 / RoPE / ALiBi | RoPE |
| 注意力 | MHA / MQA / GQA / MLA | GQA 或 MLA |
| 偏置项 | 有 / 无 | 无 |
| 深度-宽度比 | $d_{model}$ vs 层数 | 纵横比 $d_{model}/L \approx 100$（宽深平衡，见 §4.9） |

## 4. 技术原理

### 4.1 BERT：编码器路线

**论文**：Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (2018)

**架构**：Transformer 编码器，12 层（base）/ 24 层（large），双向自注意力。

**预训练任务**：
1. **掩码语言建模（MLM）**：随机掩码 15% 的 token，模型预测被掩码的 token
2. **下一句预测（NSP）**：判断两个句子是否相邻（后续研究表明 NSP 效果有限，RoBERTa 去除了 NSP）

**特点**：
- 双向注意力使每个 token 能看到上下文的所有位置
- 适合理解类任务（分类、NER、问答）
- 不适合生成任务（无自回归机制）

**微调范式**：在 `[CLS]` token 的输出上接分类头，或在每个 token 的输出上接标注头。

### 4.2 GPT：解码器路线

**论文**：Radford et al., *Improving Language Understanding by Generative Pre-Training* (2018)

**架构**：Transformer 解码器（去除交叉注意力），因果自注意力。

**预训练任务**：自回归语言建模 $P(x_t | x_{<t})$

**演进**：
- GPT-1（117M）：验证预训练 + 微调范式
- GPT-2（1.5B）：zero-shot 任务迁移
- GPT-3（175B）：few-shot / in-context learning
- GPT-4：多模态 + RLHF

**特点**：
- 因果掩码使训练可并行（所有位置同时计算），推理自回归
- 天然支持生成任务和 zero-shot
- Decoder-only 已成为现代 LLM 的主流架构

### 4.3 T5：编码器-解码器路线

**论文**：Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (2012019)

**架构**：标准 Transformer 编码器-解码器。

**核心思想**：将所有 NLP 任务统一为 text-to-text 格式：
- 翻译：`translate English to German: [文本]` → `[译文]`
- 分类：`classify: [文本]` → `[标签]`
- 摘要：`summarize: [文本]` → `[摘要]`

**预训练任务**：Span Corruption — 随机选择文本中的 span，替换为哨兵 token（如 `<extra_id_0>`），模型预测原始 span。

**特点**：
- 统一框架，一个模型处理所有任务
- 编码器双向理解输入，解码器自回归生成输出
- 适合序列转换任务（翻译、摘要）

### 4.4 ViT：视觉 Transformer

**论文**：Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (2020)

**架构**：Transformer 编码器。

**核心创新**：将图像转换为序列：
1. 将 $H \times W$ 图像分割为 $P \times P$ 的 patch（如 $16 \times 16$）
2. 每个 patch 展平为 $P^2 \times C$ 的向量
3. 线性投影到 $d_{model}$ 维
4. 添加可学习位置嵌入
5. 添加 `[CLS]` token 用于分类

**关键发现**：
- 在中等数据集（ImageNet-21k）上，ViT 不如 CNN
- 在大规模数据集（JFT-300M）上，ViT 超越 CNN
- Transformer 在视觉领域也需要规模化才有效

### 4.5 Swin Transformer

**论文**：Liu et al., *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows* (2021)

**创新**：
- **层级结构**：类似 CNN 的多尺度设计，通过 patch merging 逐层降低分辨率
- **移位窗口注意力**：在局部窗口内计算注意力（$O(n)$），通过窗口移位实现跨窗口信息交流
- 兼顾了 Transformer 的全局建模和 CNN 的局部效率

### 4.6 现代 LLM 架构（LLaMA 范式）

以 LLaMA 为代表的现代 Decoder-only LLM 的标准架构：

```
Token → Embedding (无 bias)
  → RoPE 位置编码
  → Transformer Block × N:
      Pre-RMSNorm → GQA (因果) → 残差
      Pre-RMSNorm → SwiGLU FFN → 残差
  → RMSNorm
  → Linear (无 bias) → softmax → 词表分布
```

| 设计选择 | LLaMA 2 | LLaMA 3 | Qwen 2 | DeepSeek-V3 |
|:---|:---|:---|:---|:---|
| 架构 | Decoder-only | Decoder-only | Decoder-only | Decoder-only MoE |
| 注意力 | GQA | GQA | GQA | MLA |
| 归一化 | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm |
| 激活 | SwiGLU | SwiGLU | SwiGLU | SwiGLU |
| 位置编码 | RoPE | RoPE | RoPE | RoPE |
| 训练长度 | 4K | 8K | 32K | 4K |
| 最大长度 | 4K | 128K | 128K | 128K |
| 外推方法 | - | YaRN | NTK-aware | YaRN |

### 4.7 MoE Transformer

将 FFN 层替换为多个专家（Expert）网络，通过路由器（Router）选择 Top-K 专家：

$$y = \sum_{i \in \text{TopK}} \text{softmax}(\text{TopK}(W_g x, k))_i \cdot E_i(x)$$

- **稀疏激活**：总参数量大但每次推理只激活部分参数
- **代表模型**：Mixtral 8x7B、DeepSeek-V3（671B 总参数，37B 激活）
- 详见 [[14_MoE架构\|MoE 架构]]

### 4.8 架构共识：现代 LLM 的构建模块选择

> 视角来自 Stanford CS336 (2026) Lecture 3 的综述方法：逐个梳理每年发布的模型，观察哪些选择收敛、哪些仍在演变。核心认识：**架构是一组复杂权衡的固化**——它必须（1）从数据中学习（泛化能力）、（2）在 GPU 上高效训练、（3）不崩溃（训练稳定）。很多设计之所以"不优雅"，正是因为这三类需求被直接写进了架构。
>
> 演进脉络：Transformer → GPT-3 时期（各方大量实验，无公认标准）→ **Llama 2（2023）成为事实基准**，各家训练"Llama 微改版" → 近两年两大主题：**为训练稳定而改架构**、**为长上下文而改架构**。

#### 4.8.1 LayerNorm 位置：少有的全行业共识

原始 Transformer 唯一被普遍认为"没做对"的地方：LayerNorm 放在残差路径内（Post-LN）。

| 方案 | 结构 | 状态 |
|:---|:---|:---|
| Post-LN | $\text{LN}(x + \text{SubLayer}(x))$，norm 在残差流内 | 原始 Transformer；早期研究的动机是想去掉 warmup，但不加 warmup 时收敛明显更差；**OPT-350M 是罕见的现代例外** |
| Pre-LN | $x + \text{SubLayer}(\text{LN}(x))$，norm 在残差流外、计算前 | 现代标准（Llama 2 效应：大家跟着 Llama 2 做） |
| Sandwich / 计算后归一化 | 子层前后都加 norm（后 norm 仍在残差流外） | Gemma 2、OLMo 2 等近期模型 |

**关键机制——"保持残差流干净"**：
- Pre-LN 下输入 $x$ 从底层直通顶层输出，反向传播存在直通路径 → 初始化时梯度范数逐层基本不变；Post-LN 每过一个 block 都会改变梯度范数（Xiong et al. 2020 最早系统研究）
- Pre-LN 下梯度尖峰的幅度与频率均更低；深层网络的稳定构建对现代 LLM 至关重要
- **经验法则：遇到稳定性问题，就在更多地方加 LayerNorm**（加在注意力内部也行 → QK-Norm，见 [[07_Transformer工程实践]]）。听上去荒谬，但被反复验证正确

#### 4.8.2 RMSNorm：删除低算术强度操作的零成本优化

RMSNorm（不减均值、无偏置）理论上表达力弱于 LayerNorm，但建模效果相当——它胜出的真正原因是**系统层面**：

- LayerNorm 这类统计归一化操作**算术强度极低**：只占约 0.17% 的 FLOPs，却可能占**运行时间的 25%**（取决于负载；小模型上更严重，因为参数/激活仍要在快慢内存间搬运）
- FLOPs ≠ 运行时间：矩阵乘法是计算密集型，归一化是访存密集型——访存慢，就拖垮整体
- 删掉均值减法与偏置项 = **零成本优化**：表达力不变，吞吐免费提升（Google 的对照实验还观察到吞吐提升）
- 同一逻辑的延伸：现代模型普遍**删除线性层偏置项**——低算术强度、额外访存、偶尔还引入稳定性问题

这类"架构-系统协同设计"无法事前从理论推理（事先并不知道去掉偏置可行），只能靠大规模实验与集体经验积累。

#### 4.8.3 激活函数：门控是真正关键的维度

| 激活函数 | 说明 | 代表 |
|:---|:---|:---|
| ReLU | 最朴素的选择也能训练好模型 | Chinchilla |
| GELU | ReLU + 零附近的小凹陷（改变零点附近梯度） | GPT 系列 |
| GeGLU / SwiGLU | 门控线性单元（GLU 家族） | Google 系（Gemma、T5）用 GeGLU；Llama 后代用 SwiGLU（更主流） |
| squared ReLU | 平方 ReLU——"疯狂但同样有效"的冷门选择 | Nemotron 340B |

- **GLU 机制**：$\text{FFN}(x) = (\text{act}(xW_1) \odot xV)W_2$——引入门控分支 $xV$ 逐元素调制激活输出。对应架构设计的通用启发式："引入 gain/gate 机制通常有帮助"
- **参数对齐的 2/3 规则**：门控引入第三个矩阵，为保持总参数量不变，$d_{ff}$ 缩至 $\frac{2}{3} \times 4d_{model} \approx 2.67d_{model}$
- **证据质量高**：Shazeer (2020) 的对照实验带误差棒（多次重复训练）、参数对齐，GLU 变体**一致**优于非 GLU；Google 2020 年基于 T5 的大规模系统对比同样显著
- **结论**：门控比具体激活形状更重要——几乎零成本的一致增益，因此几乎所有靠谱的现代模型都用 GLU；但门控并非必需（GPT 系列、Nemotron 证明非门控也能训好）

#### 4.8.4 并行层：一个有趣的失败

GPT-J / PaLM 将注意力与 MLP **并行**计算（输出相加后一起写回残差流）：可复用组件（共享 LayerNorm、融合矩阵乘）→ PaLM 自称系统利用率 +15%、无质量损失；Cohere（创始人之一为 Transformer 作者）沿用。但近两年已基本无人跟随：
- 串行形式的系统优化已经足够好，并行换来的边际收益不再显著
- 并行化相当于**有效深度减半**，损害表达能力
- PaLM 声称无质量损失，但 Google 后续模型悄悄放弃——可视为"实际存在性能损失"的隐含信号；业界至今没有控制良好的并行 vs 串行消融实验

#### 4.8.5 位置编码：RoPE 一统，变体在边缘

2024 年之后绝大多数模型使用 RoPE（推导与变体详见 [[04_位置编码]]）。剩余多样性集中在：ALiBi 类（注入注意力矩阵）、T5 相对偏置（不可分解为内积）、Gemma 4 的 pRoPE（只旋转少量坐标对）、混合架构全局层的 NoPE（完全无位置嵌入）。

#### 4.8.6 综述结论速览

对大量已发布模型的横向比较（CS336 汇总）：
- **归一化**：几乎全是 RMSNorm
- **层结构**：几乎全是串行（并行层已衰落）
- **norm 位置**：Pre-Norm 为主（部分模型同时用 pre + post）
- **激活**：几乎全是 GLU 家族（GeGLU 或 SwiGLU）
- **仍在演变的部分**：位置编码的处理方式与长上下文整合——这是当前架构工作最活跃的区域

### 4.9 Perse：循环 Transformer

> 视角来自 Stanford CS336 (2026) Lecture 18 嘉宾讲座（Dan Fu, UCSD / Together AI）。Perse 由 UCSD 实验室（Hidden 主导）与 Zachary Taylor 合作完成。

#### 4.9.1 核心思想：用循环替代堆叠

传统 Transformer 通过逐层堆叠增加深度，Perse 将 Transformer 的某些模块**循环执行**——激活值反复穿过同一组模块，而非每层只用一次：

```
输入 → [模块 M] → 循环执行 M 共 R 次 → 输出
```

- **参数不变，FLOPs 可调**：循环次数 $R$ 增加，计算量增加但参数量不变——在不增加参数成本的前提下提升质量
- **表达能力提升**：已有理论工作（Tom Goldstein 团队, UMD）表明循环模型可以表达相同参数量下非循环模型无法表达的功能
- **推理优势**：参数量小 → 模型更容易塞进 GPU 显存 → 留更多空间给 KV Cache 或减少跨 GPU 通信开销

#### 4.9.2 训练稳定性：谱半径约束

**核心问题**：循环 Transformer 极难训练。稍微改动学习率，10 次扫描中 9 次模型会崩溃（损失暴涨为 NaN）。

**根因分析——动态系统视角**：

1. **残差流近似不变**：激活值在各模块之间变化其实不大——残差块只是对向量做微调
2. **非线性打包**：将注意力、GLU、FFN 等复杂非线性组件打包为一个“黑盒” $R$（某种复杂的非线性结构），剩余部分简化为两个矩阵：
   - **$B$ 矩阵**：负责对初始向量做变换（循环开始前）
   - **$A$ 矩阵**：决定每次循环中如何变换残差
3. **谱半径爆炸**：去掉非线性后，系统简化为可用微积分求解的线性递推。关键发现是 $A$ 矩阵的**谱半径**（类似范数）主导了系统行为——若谱半径 > 1，$A$ 的幂次项将指数级放大激活值（如 $2^{16} = 65536$），导致损失暴涨

**解决方案——约束 $A$ 和 $B$**：

| 组件 | 约束方式 | 效果 |
|:---|:---|:---|
| $A$ 矩阵 | 设为**负对角矩阵** → 幂次项最终趋零，不会发散 | 防止循环过程中的指数级爆炸 |
| $B$ 矩阵 | 加**线性范数约束**（仅应用一次，不会发散） | 控制初始注入的量级 |

- 只要谱半径 < 1，系统就稳定——即便用 6e-4 这种让其他循环模型崩溃的学习率，Perse 依然稳定
- **有趣的拉扯**：模型倾向于膨胀激活值（更大空间 → 更好表示），而范数约束将数值压回 1——两种力量的拉扯导致未约束版本损失剧烈波动；约束后训练曲线平稳
- 稳定性不仅改善训练，还**提升了模型质量**——Perse 在多种任务上优于上一代循环模型，甚至超过同等算力的标准 Transformer

#### 4.9.3 循环缩放定律

Perse 的缩放定律实验揭示了循环次数、数据量与参数规模三者的联合缩放关系：

- **等损失曲线方向**：与经典的“参数-数据”缩放类似，循环模型的等损失曲线同样向右下方倾斜——说明随着数据量增加，应**同步增加循环次数**
- **联合缩放**：三维缩放实验（循环 × 数据 × 参数）表明，理想情况下应同时扩展这三者
- **关键发现**：固定模型大小、增加数据量时，也必须增加循环次数——而当前所有不带循环的大规模预训练模型都处于“曲线最左端”（数据量巨大但循环次数=1），暗示可能存在更优的训练配置
- **算力效率**：在相同 FLOPs 预算下，通过增加循环次数（而非只堆数据）达到的损失更低——循环可能比单纯增加数据更高效
- 详见 [[../../A_数学与优化基础/03_尺度规律与计算最优|尺度规律与计算最优]] §循环缩放定律

#### 4.9.4 推理优势与硬件协同设计

循环 Transformer 在推理端具有独特优势：

- **参数少 → 显存友好**：更小的模型占用更少 GPU 显存 → 可容纳更多 KV Cache 或减少跨 GPU 拆分
- **循环块可微型化**：若循环块足够小，可编写微型 Mega Kernel 用超快循环跑完计算（尚未实现但方向明确）
- **适配新型硬件**：如 Groq LPU（~250MB 内存）等推理芯片，循环模型的小参数量天然适配——权重常驻内存、激活值极速流转
- 详见 [[../../K_AI工程化/03_推理工程/08_推理加速与硬件优化|推理加速与硬件优化]] §Mega Kernel

#### 4.9.5 预训练模型循环化

一个有趣的发现（来自社区实验）：在已训练好的模型中循环迭代 2-3 层，即可在数学推理等任务上获得质量提升。Perse 团队对此有相关研究跟进，试图从激活值和权重层面理解为何循环能改善预训练模型的效果。

### 4.10 超参数经验法则

| 超参数 | 默认值 | 容错性 |
|:---|:---|:---|
| FFN 比例 $d_{ff}/d_{model}$ | $4\times$；GLU 变体 $\approx 2.67\times$ | Kaplan (2020) 消融：1~10 是平坦盆地，偏离最优的损失增量微乎其微；>10 才二次式恶化 |
| 头维度 | $d_{model}/h$（总维度≈$d_{model}$） | 高容错，1 附近有宽缓冲区；例外：T5、PaLM |
| 纵横比 $d_{model}/L$ | $\approx 100$ | GPT-3、Llama 系均符合；各规模模型的最优值接近（Kaplan） |
| 词表大小 | 单语 ~32K → 多语 100K–200K | 多语言需要更大词表覆盖语言空间；模型越大能处理的词表越大（scaling law 结论） |

**FFN 比例的著名例外**：
- **Llama 2**：$2.67 \times 1.33 \approx 3.5$——团队认为（用了 GQA 的）注意力头已经很高效，"随意"乘了个 1.33 更偏重 MLP。这恰好说明该参数容错高
- **T5**：激进的 $64\times$——系统层面论证：更大的矩阵乘 → 硬件利用率更高。T5 本身是好模型，但计算效率存疑；后续 **T5 v1.1 悄悄回归 2.5×**——隐含承认激进选择不划算。Gemma 2 也尝试过推高该值

**纵横比的系统逻辑**：
- 极深模型难以并行：深度切分 → 流水线并行（业界极力避免）；宽模型易并行：张量并行简单直接
- 表达力偏好深、硬件偏好宽 → 收敛于 $d_{model}/L \approx 100$
- 系统性扫描实验的结论：深度-宽度权衡中**起决定作用的是总 FLOPs**（FLOPs 增加 → 性能提升），纵横比本身在宽安全区内影响甚微 → 优先关心系统利用率

**正则化的反直觉（计算受限场景）**：
- LM 训练通常**单 epoch** 遍历数据（数据量远超算力），单遍 SGD 基本不会记忆数据 → 原则上不存在过拟合，不需要正则化；有些团队因此只看训练损失
- 但事实上很多现代模型仍在用 weight decay（技术报告往往不披露这类细节）
- 关键发现：单 epoch SGD 下，**有无 weight decay 的训练/验证损失无差异**（没有过拟合可防）；但**配合学习率衰减**时，强 weight decay 虽起步慢、却收敛到更好的极小值——**它更像优化器组件而非正则化器**（在衰减学习率下成立，固定学习率下不一定）
- dropout 与优化过程配合不佳，已基本不再流行
- 训练中动态调整的超参几乎只有 weight decay（与学习率协同衰减）；架构参数训练中不可变（改动即不兼容）

**实践文化**：各家技术报告通常**逐个**调整架构选择，很少一次全改；Google 是少数大刀阔斧修改架构的机构（T5 的 64×、Gemma 系列的大 FFN 与 softcapping 等）。

## 5. 关键方法/模型

### 5.1 编码器路线演进

```
BERT (2018) → RoBERTa (2019, 去NSP+更多数据) → ALBERT (2019, 参数共享)
    → DeBERTa (2021, 解耦注意力) → RoBERTa-2 / 现代编码器
```

### 5.2 解码器路线演进

```
GPT-1 (2018) → GPT-2 (2019) → GPT-3 (2020) → PaLM (2022)
    → LLaMA (2023) → LLaMA 2/3 (2023-2024) → DeepSeek-V3 (2024)
```

### 5.3 编码器-解码器路线演进

```
Transformer (2017) → MASS (2019) → T5 (2019) → BART (2020)
    → mT5 (2021) → Flan-T5 (2022)
```

### 5.4 视觉 Transformer 演进

```
ViT (2020) → DEiT (2021) → Swin Transformer (2021)
    → CSWin Transformer (2022) → MaxViT (2022)
```

## 6. 优势与局限

### 优势

1. **架构统一性**：同一 Transformer 架构适用于 NLP、CV、语音等多领域
2. **可扩展性**：参数量从百万到万亿均有效
3. **迁移学习**：预训练 + 微调 / in-context learning 范式高效
4. **生态成熟**：丰富的预训练模型和工具链

### 局限

1. **Decoder-only 的局限**：双向理解能力不如 Encoder-only（但规模化后差距缩小）
2. **计算成本**：大模型训练和推理成本高昂
3. **数据依赖**：需要海量高质量训练数据
4. **CV 领域的局限**：ViT 对数据量要求高，小数据集上不如 CNN

## 7. 应用场景

| 任务类型 | 推荐架构 | 代表模型 |
|:---|:---|:---|
| 文本分类 / NER / 问答 | Encoder-only | BERT、RoBERTa |
| 文本生成 / 对话 / 代码 | Decoder-only | GPT、LLaMA |
| 机器翻译 / 摘要 | Encoder-Decoder | T5、BART |
| 图像分类 | Encoder-only (ViT) | ViT、Swin |
| 目标检测 | Encoder + 检测头 | DETR |
| 语音识别 | Encoder-Decoder | Whisper |
| 多模态 | 各类混合 | CLIP、LLaVA |
| 蛋白质结构 | 自定义 | AlphaFold 2 |

## 8. 与其他技术关系

- **与 [[03_Transformer架构详解|Transformer 架构]] 的关系**：本笔记是架构在不同方向的变体应用
- **与 [[../../C_基础模型与通用智能/02_语言基础模型/00_预训练语言模型_综述\|预训练语言模型]] 的关系**：BERT、GPT、T5 的预训练策略详见该方向
- **与 [[../../C_基础模型与通用智能/02_语言基础模型/03_大语言模型核心架构/00_大语言模型核心架构_综述\|大语言模型核心架构]] 的关系**：现代 LLM 的架构设计选择详见该方向
- **与 [[../../E_感知与多模态智能/01_视觉表示与视觉理解/00_视觉智能综述\|视觉智能]] 的关系**：ViT、Swin、DETR 详见该方向
- **与 [[../../E_感知与多模态智能/03_多模态理解与对齐/00_多模态智能综述\|多模态智能]] 的关系**：CLIP、Flamingo 等多模态 Transformer 详见该方向
- **与 [[05_高效注意力机制|高效注意力]] 的关系**：架构变体中的注意力优化方案
- **与 [[04_位置编码|位置编码]] 的关系**：RoPE 的推导、长度外推与 pRoPE/NoPE 等变体详见该笔记
- **与 [[../../K_AI工程化/03_推理工程/08_推理加速与硬件优化|推理加速与硬件优化]] 的关系**：Perse 循环模型的推理优势（参数少→显存友好）及 Mega Kernel 技术详见该笔记
- **与 [[../../A_数学与优化基础/03_尺度规律与计算最优|尺度规律与计算最优]] 的关系**：Perse 的循环缩放定律（循环次数×数据量×参数联合缩放）详见该笔记

## 9. 前沿发展

- **Decoder-only 统一**：Encoder-only 和 Encoder-Decoder 路线逐渐被 Decoder-only 取代，现代 LLM 几乎全部采用 Decoder-only
- **MoE 普及**：DeepSeek-V3、Mixtral 等验证了 MoE 在大规模 LLM 中的可行性
- **混合架构**：Jamba（SSM + Attention）、Mamba-2 等探索非纯 Transformer 架构
- **长短注意力交替成为长上下文主流方案**：GPT-3 就已在全注意力与局部带状注意力间交替；2025 年前后复兴——Cohere Command A 每 4 层 1 层全注意力（其余 3 层滑动窗口，全局层甚至不用位置嵌入），Llama 4、Gemma 系列等跟进（全层 RoPE）。在长上下文性能与推理开销间取得平衡，无需引入 SSM 等更复杂方案；更激进的线性层混合版本见 [[05_高效注意力机制]] §5.5
- **训练稳定性的架构化**：QK-Norm 已成大多数新模型标配（源自多模态模型 ViT-22B、Chameleon 的经验迁移）；Gemma 系用 logit softcapping；Perse 用谱半径约束稳定循环训练；详见 [[07_Transformer工程实践]]
- **循环 Transformer 复兴**：Perse 验证了模块循环复用可在不增加参数的前提下提升质量，且通过动态系统理论（谱半径约束）解决训练稳定性问题。缩放定律表明数据量增加时应同步增加循环次数——当前无循环的大规模预训练可能处于次优状态（§4.9）
- **多模态统一**：单一 Transformer 架构处理文本、图像、音频、视频（如 Gemini）
- **架构简化**：去除不必要的组件（bias、均值减法、NSP 等），追求极致简洁——本质是删除低算术强度操作（§4.8.2）
- **LLM 辅助架构设计**：使用 LLM 搜索更优的 Transformer 变体
