# Scaling Laws 与计算最优训练

## 1. 概述

Scaling Laws（缩放定律）描述了语言模型性能与模型规模、数据量、计算量之间的幂律关系。这一发现由 Kaplan et al. (2020) 首次系统提出，后经 Hoffmann et al. (2022, Chinchilla) 修正和完善。Scaling Laws 为大模型训练提供了理论指导——在有限计算预算下如何最优分配参数量和数据量，是推动 LLM 规模化的关键理论基础。

## 2. 发展历史

| 时间 | 里程碑 | 核心发现 |
|------|--------|---------|
| 2017 | Hestness et al. | 早期观察到深度学习的幂律关系 |
| 2020.01 | **Kaplan et al.** (OpenAI) | 系统提出 Scaling Laws |
| 2020.05 | GPT-3 发布 | 验证规模化带来能力涌现 |
| 2021 | Hoffmann et al. 初步修正 | 指出 Kaplan 低估了数据重要性 |
| 2022.03 | **Chinchilla** (DeepMind) | 计算最优比例：参数与数据等比例缩放 |
| 2022.06 | Emergent Abilities (Wei et al.) | 系统描述涌现能力现象 |
| 2023 | Chinchilla Scaling 被广泛采用 | LLaMA 等开源模型遵循 Chinchilla 比例 |
| 2024 | MiniCPM / DeepSeek LLM | 开源社区系统复现 Chinchilla：μP、WSD 调度、超参缩放 |
| 2024-2025 | Hunyuan-Large / Kimi K2 / MiniMax-01 | MoE 缩放定律与架构缩放对比成为标配实验 |
| 2025 | Step Law（StepFun） | 超参数缩放定律的大规模系统验证 |
| 2023+ | 逆 Scaling / U形 Scaling | 部分任务随规模先变差再变好 |

## 3. 核心概念

### 幂律关系

模型损失（交叉熵）与关键因素呈幂律关系：

$$L(x) = \left(\frac{x_c}{x}\right)^{\alpha} + L_\infty$$

其中 $x$ 可以是参数量 $N$、数据量 $D$ 或计算量 $C$，$L_\infty$ 是不可约损失（数据本身的熵），$x_c$ 和 $\alpha$ 是经验常数。

### 三个维度

| 维度 | 符号 | 含义 | 缩放关系 |
|------|------|------|---------|
| 参数量 | $N$ | 模型可训练参数数 | $L(N) \propto N^{-\alpha_N}$ |
| 数据量 | $D$ | 训练 token 数 | $L(D) \propto D^{-\alpha_D}$ |
| 计算量 | $C$ | 总 FLOPs | $L(C) \propto C^{-\alpha_C}$ |

### 涌现能力（Emergent Abilities）

某些能力在模型规模超过阈值后**突然出现**，而非渐进提升：

- **少样本推理**：~10^22 FLOPs 后出现
- **多步算术**：~10^22 FLOPs 后出现
- **符号操作**：~10^23 FLOPs 后出现

> 注意：2023 年研究（Schaeffer et al.）指出，部分"涌现"可能是评估指标的非线性造成的假象，使用连续指标后能力可能是渐进提升的。

## 4. 技术原理

### 4.1 Kaplan Scaling Laws (OpenAI, 2020)

Kaplan et al. 的核心发现：

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050$$

**联合缩放公式**：

$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}$$

**Kaplan 的最优分配建议**：
- 在固定计算预算 $C$ 下，最优策略是**优先增大模型参数量**
- 最优分配：$N \propto C^{0.73}$，$D \propto C^{0.27}$
- 即：大部分计算预算应投入更大的模型，而非更多数据
- 大模型应**在远未收敛时停止训练**（early stopping）

**推导最优分配**：

给定 $C \approx 6ND$（Transformer 前向+反向 FLOPs 估算），最小化 $L(N, D)$：

$$\min_{N, D} L(N, D) \quad \text{s.t.} \quad C = 6ND$$

使用拉格朗日乘数法求解。

### 4.2 Chinchilla Scaling (DeepMind, 2022)

Hoffmann et al. 通过训练 400+ 个模型（70M 到 16B 参数，5B 到 500B tokens）发现 Kaplan **低估了数据量的重要性**。

**修正后的缩放指数**：

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

其中 $E$ 是不可约损失，拟合参数：
- $\alpha \approx 0.34$
- $\beta \approx 0.28$
- $A \approx 406.4$, $B \approx 410.7$

**Chinchilla 的最优分配**：

$$N^* \propto C^{0.50}, \quad D^* \propto C^{0.50}$$

**关键结论**：参数和数据应**等比例缩放**，最优数据量约为参数量的 20 倍：

$$D^* / N^* \approx 20$$

即每个参数约需 20 个 token 的训练数据。

**与 Kaplan 的对比**：

| 维度 | Kaplan (2020) | Chinchilla (2022) |
|------|--------------|-------------------|
| $N$ 的指数 | 0.73 | 0.50 |
| $D$ 的指数 | 0.27 | 0.50 |
| 数据/参数比 | ~1.7 (偏低) | ~20 (最优) |
| 训练策略 | 大模型+少数据 | 均衡增长 |

### 4.3 Chinchilla 验证

DeepMind 用相同计算预算训练了 Chinchilla（70B 参数，1.4T tokens），对比 Gopher（280B 参数，300B tokens）：

| 模型 | 参数量 | 训练数据 | MMLU 准确率 |
|------|--------|---------|------------|
| Gopher | 280B | 300B tokens | ~60% |
| GPT-3 | 175B | 300B tokens | ~43.9% |
| **Chinchilla** | **70B** | **1.4T tokens** | **67.5%** |

**结论**：用同样计算量，更小的模型 + 更多数据 = 更好的性能。大量已发布的 LLM 都**严重训练不足**。

### 4.4 计算量估算

Transformer 的 FLOPs 估算：

**前向传播**（每个 token）：
$$C_{\text{forward}} \approx 2ND$$

其中 $N$ 是参数量，$D$ 是 token 数。系数 2 来源于矩阵乘法（每个参数对应一次乘法和一次加法）。

**反向传播**：约为前向的 2 倍。

**总计**：
$$C_{\text{total}} \approx 6ND$$

**示例**：
- GPT-3 (175B 参数, 300B tokens): $C \approx 6 \times 175 \times 10^9 \times 300 \times 10^9 \approx 3.15 \times 10^{23}$ FLOPs
- Chinchilla (70B 参数, 1.4T tokens): $C \approx 6 \times 70 \times 10^9 \times 1.4 \times 10^{12} \approx 5.9 \times 10^{23}$ FLOPs

### 4.5 过拟合与关键批次大小

**过拟合边界**：当数据量不足时，模型开始过拟合。

$$D_{\text{crit}} \approx 5.85 \times N^{0.76}$$

当 $D < D_{\text{crit}}$ 时，验证损失开始高于训练损失。

**关键批次大小**（Critical Batch Size）：

$$B_{\text{crit}} \approx \frac{C}{10}$$

- 当 batch size < $B_{\text{crit}}$ 时，增大 batch size 可加速训练且不损失效率
- 当 batch size > $B_{\text{crit}}$ 时，增大 batch size 的收益递减

## 5. 实践意义

### 训练预算规划

给定计算预算 $C$，计算最优参数量和数据量：

$$N^* = \frac{C}{6 \times 20} = \frac{C}{120}, \quad D^* = 20 N^*$$

| 计算预算 (FLOPs) | 最优 $N$ | 最优 $D$ | 典型模型 |
|-----------------|---------|---------|---------|
| $10^{20}$ | ~0.8B | ~16B | 小型实验 |
| $10^{21}$ | ~8B | ~160B | 中型模型 |
| $10^{22}$ | ~80B | ~1.6T | GPT-3 级别 |
| $10^{23}$ | ~800B | ~16T | GPT-4 级别 |
| $10^{24}$ | ~8T | ~160T | 未来模型 |

### 现代模型的数据策略

遵循 Chinchilla 比例的模型（2023+）：

| 模型 | 参数量 | 训练数据 | $D/N$ 比 |
|------|--------|---------|---------|
| LLaMA-1 | 7B-65B | 1T-1.4T tokens | ~15-20 |
| LLaMA-2 | 7B-70B | 2T tokens | ~30-285 |
| LLaMA-3 | 8B-70B | 15T tokens | ~1875-214 |
| Qwen-2 | 0.5B-72B | 12T tokens | 高 |
| Chinchilla | 70B | 1.4T tokens | 20 |

> 注意：LLaMA-2/3 等模型已远超 Chinchilla 比例（$D/N \gg 20$），这被称为"过度训练"（Overtraining），目的是在推理阶段获得更好的性能-成本比（推理次数远多于训练次数）。

### 开源模型的 Scaling 实践（2024-2025）

Chinchilla 之后，开源社区在真实训练规模上复现并扩展了缩放分析。针对"超参数随规模漂移"这一核心难题，形成两条主流技术路线：

**路线一：参数化不变性（MiniCPM，2024）**

- 采用 μP 风格参数化（缩放嵌入输出、按 $\sqrt{\text{层数}}$ 缩放残差分支、按 fan-out/fan-in 比例相关因子缩放矩阵初始化、按参数类型设置学习率乘子、缩放输出头），使**最优学习率在模型宽度变化时保持不变**——小模型上扫出的最优学习率可直接用于大模型（详见 [[../../K_AI工程化/02_训练工程/05_超参数搜索|超参数搜索]]）。
- 配合 [[../../A_基础与范式/02_优化理论与方法/05_学习率调度与训练策略|WSD 调度]]，回滚稳定阶段检查点再重新衰减即可低成本扫描数据规模，无需从头重训；MiniCPM 借此在模型、数据两个维度复现了 Chinchilla 分析，其拟合得到的计算最优数据/参数比显著高于 20。
- "风洞实验"的可靠外推范围：MiniCPM 缩放阶梯上的最大实验模型与最终发布模型（1.2B/2.4B）相差约 5 倍。

**路线二：拟合超参数缩放定律（DeepSeek LLM，2024）**

- 不追求不变性，而是在多个规模上对学习率 × batch size 做网格搜索，直接拟合**最优学习率、最优 batch size 随非嵌入计算量的幂律**，外推出 7B/67B 的训练配置。
- 其缩放定律对实际训练的大模型损失预测相当准确，也再次验证 Chinchilla 分析在真实开源规模上成立。Qwen2.5/Qwen3 等后续开源报告沿用同套流程，已成为标准做法。

**MoE 缩放定律（Kimi K2、Hunyuan-Large）**

- 转向 MoE 的团队普遍重做 isoFLOP 分析，把**激活参数**作为核心变量：固定 FLOPs 下稀疏度越高验证损失越低，但收益递减，据此定量选定稀疏度与 token/激活参数配比（Hunyuan-Large 约 96 token/激活参数）。
- 只要以激活参数为基准，稠密模型的超参缩放规律对 MoE 基本适用（Step Law 明确将稠密与 MoE 统一在同一超参定律下）。

**损失到下游能力的映射（Llama 3）**

- 在常规缩放之外，Llama 3 拟合了**对数损失 → 下游基准准确率的 S 型（sigmoid）映射**，使"以损失为锚"的缩放预测延伸到基准指标；但数据点相对拟合曲线存在系统性偏离，只能作参考而非绝对真理。

**架构比较的缩放证据（MiniMax-01）**

- 对 softmax 注意力、lightning（线性）注意力与混合架构做缩放对比：三者在多数算力水平下的缩放行为与最优模型规模接近，据此选定混合架构——"用 scaling laws 做架构决策"的典型范例。

**经验教训**

- 缩放拟合的常数**高度依赖数据配方**：换数据后最优学习率/batch size 会整体偏移，搬用他人规律前需确认实验设置足够相似。
- 趋势可能在更大规模处突然失效：多个数量级上表现良好的拟合也可能越过某个规模阈值后偏离、崩坏（Stanford modular-norm 项目公开过此类失败案例）。
- 预训练-后训练的**联合缩放**仍是开放问题；预训练数据覆盖度/多样性与后训练效果的关联是新兴研究方向。

### 幂律的理论根源：非参数回归器视角

P9 讲座提供了一个有趣的统计学解释：神经网络 Scaling Law 的幂律指数与非参数回归的收敛速率存在形式上的类比。

**核心论点**（Bahri et al. 等研究 Scaling Law 理论的学者）：

- 对于参数化模型（如线性回归），收敛速率为 $O(N^{-1})$，即误差随样本量线性下降。
- 对于**非参数估计器**（如最近邻、核回归），若数据内在维度为 $D$，则收敛速率为：

$$\text{误差} \propto N^{-1/D}$$

- 神经网络的 Scaling Law 指数（如 $\alpha_D \approx 0.05 \sim 0.1$）远慢于 $O(N^{-1})$，但与 $D \approx 10 \sim 20$ 维空间中的非参数平滑器收敛速率一致。
- 因此可建立心智模型：**神经网络的行为类似 $D$ 维空间中的非参数回归器**，其幂律指数实际上在告诉我们网络的有效学习维度。

> 注意：这一解释依赖于对数据内在维度的估计，证据尚不完全充分，但提供了理解幂律指数物理含义的有益视角。

**历史脉络**：Scaling Law 的思想并非神经网络独有。Banko & Brill (2001) 在 NLP 分类任务中已观察到数据量与错误率的幂律关系；Hestness et al. (2017) 在多种深度学习模型上系统验证了这一规律。早期工作（如 Bell 实验室 1993 年的分类器 scaling 研究）已提出通过小规模实验拟合性能-数据曲线来外推大规模表现的思路。

### 数据混合定律（Data Mixture Law）

P9 讲座指出，数据 Scaling Law 的一个重要工程推论是**数据混合比例的优化**：

**截距 vs 斜率分离**：
- Scaling Law 的**斜率**（指数 $\alpha$）由模型类别决定，与数据分布无关。
- Scaling Law 的**截距**（常数项）由数据组成决定，不同数据混合比例会改变截距。
- 由于斜率不变，**小规模下最优的数据配比在放大后仍然最优**——这为用小模型实验确定数据配比提供了理论依据。

**实践方法**：
1. 在小规模模型上用不同数据混合比例（如新闻 vs 维基百科的不同配比）训练
2. 拟合各配比下的 Scaling Law 曲线
3. 观察趋势随规模变化，外推到全量训练规模
4. 选择外推后最优的配比

> 实践中，直接在小规模选出最好的数据配比再放大也能取得不错效果，无需严格拟合 Scaling Law（因为斜率不变假设保证了配比最优性的传递）。

### 数据重复与过滤的规模依赖性

**数据重复的 4-Epoch 限制**（Muennighoff et al., "Scaling Data-Constrained Language Models", 2023）：
- 用标准训练方法，数据重复使用最多 **4 个 epoch** 不会显著损害性能。
- 超过 4 个 epoch 后，实际 Scaling Law 曲线会显著偏离用新数据预测的理想曲线（表现更差）。
- 可写出修正后的函数形式来预测重复训练下的 Scaling Law 变化。
- 极端情况下（无限算力 + 有限数据），仅靠重复数据和增大模型都有收益递减问题，最终需要集成等方法才能进一步提升。

**数据过滤的规模依赖性**：
- 数据过滤策略**不是一成不变的**，最优策略随训练规模变化。
- 算力有限时：应激进过滤，只保留最高质量数据。
- 算力充足时：应放宽过滤条件，使用更多数据——因为重复使用高质量数据的收益递减，不如纳入更多低质量但多样的数据。
- 这意味着数据质量评估本身是一个**规模相关的动态决策**，而非静态预处理步骤。

### Kaplan vs Chinchilla 差异的审计

P9 讲座详细讨论了 Kaplan 与 Chinchilla 之间巨大差异的来源（Sander et al., "Revisiting Neural Scaling Laws", 2022），主要因素：

| 差异来源 | Kaplan 的做法 | 影响 |
|---------|-------------|------|
| **参数计数** | 排除了 embedding 和最后一层 softmax 参数 | 对 Scaling Law 形状影响巨大——这两个矩阵互为对偶（形状均为 $V \times d$），排除后有效参数量被严重低估 |
| **学习率 warm-up** | 模型极小，warm-up 结束时模型尚未收敛 | 学习率设置不理想，导致性能被低估 |
| **Batch size** | 使用固定的大 batch size | 对小模型来说并非最优——如果正确调整 batch size（随模型规模变化），结果与 Chinchilla 吻合 |

**核心教训**：Scaling Laws 本质上是**下界**——它告诉你"沿用这套配方并扩大规模，结果就是这样"。如果配方本身有问题（不合理的 warm-up、batch size 等），得到的 Scaling Law 会更差。拟合 Scaling Law 时应尽可能接近标准的完整训练流程。

### 上下游性能的鸿沟

P9 讲座强调了 Scaling Law 的一个重要局限：

- Scaling Law 通常以 **perplexity / negative log-likelihood** 为指标，规律非常干净且可预测。
- 但从 perplexity 迁移到**下游基准准确率**远没有那么确定。
- 案例：ANNL-12 在 perplexity 上最优，但下游性能最好的是 ANNAL-1 XL（其 perplexity 反而更差）。
- Llama 3 试图通过对数损失 → 下游准确率的 S 型映射来弥补这一鸿沟，但数据点与拟合曲线存在系统性偏离，只能作为参考。

> "搞 pre-training 的人说 perplexity 挺好看，搞 post-training 的人拿到的模型在下游任务上却未必最好"——这是实践中常见的摩擦点。

## 6. 优势与局限

### 优势
- 为大模型训练提供理论指导，避免盲目试错
- 预测模型性能，降低训练失败风险
- 指导计算资源最优分配

### 局限
- **幂律在极大规模下可能失效**：外推到未验证的规模有风险
- **忽略了数据质量**：公式假设数据质量均匀，实际差异巨大
- **未考虑架构创新**：MoE、新注意力机制等改变了缩放行为
- **涌现不可预测**：Scaling Laws 描述平均损失，不预测特定能力涌现
- **Chinchilla 比例可能过时**：现代模型（LLaMA-3）已大量"过度训练"（$D/N \gg 20$）
- **上下游鸿沟**：perplexity 上的最优模型在下游任务上未必最优，Scaling Law 预测从损失到能力的迁移存在不确定性

## 7. 应用场景

- **训练规划**：计算最优模型大小和数据量
- **性能预测**：在小模型上拟合缩放曲线，外推大模型性能
- **资源分配**：在有限 GPU 预算下最大化模型性能
- **推理优化**：过度训练以降低推理成本（小模型 + 多数据 → 推理更便宜）
- **模型选择**：根据部署场景选择参数-数据的最优平衡

## 8. 与其他技术关系

- Scaling Laws 指导 [[../../06_模型训练与对齐/01_预训练流程|预训练流程]] 中的资源分配
- 涌现能力是 [[10_大语言模型核心架构|大语言模型]] 规模化的核心驱动力
- 计算量估算与 [[../../C_基础模型与通用智能/09_模型压缩与高效推理/10_推理加速与并行|推理加速与并行]] 中的效率优化相关
- 过度训练策略与 [[../../C_基础模型与通用智能/09_模型压缩与高效推理/08_量化技术|量化技术]]、[[../../C_基础模型与通用智能/09_模型压缩与高效推理/09_剪枝与稀疏化|剪枝与稀疏化]] 等推理优化互补
- Chinchilla 比例影响开源模型的训练策略（LLaMA、Qwen 等）

## 9. 前沿发展

- **过度训练**：LLaMA-2/3 远超 Chinchilla 比例，牺牲训练效率换取推理效率
- **MoE 缩放**：以激活参数为核心变量的缩放分析已成为万亿级开源模型（Kimi K2、Hunyuan-Large）的标配实验
- **超参数缩放**：最优学习率/batch size 随规模的幂律可拟合、可外推（DeepSeek LLM、StepFun Step Law），详见 [[../../K_AI工程化/02_训练工程/05_超参数搜索|超参数搜索]]
- **优化器缩放**：Muon 等矩阵正交化优化器在计算最优训练下相对 AdamW 约有 2× 计算效率（Moonlight），但公平调参下增益随规模缩小，详见 [[../../A_基础与范式/02_优化理论与方法/07_大模型训练优化|大模型训练优化]]
- **损失-能力映射**：Llama 3 拟合对数损失与下游准确率的 S 型关系，把缩放预测延伸到基准指标
- **数据质量缩放**：研究数据质量与数据量的交互作用
- **逆 Scaling**：某些任务随规模变差（如社会偏见），需专门研究
- **多模态缩放**：Scaling Laws 在视觉-语言模型中的适用性
- **推理时缩放**：o1 等推理模型在推理阶段投入更多计算，扩展了缩放维度
- **后训练联合缩放**：预训练-后训练协同效应尚无成熟理论，是当前开放问题
- **Chinchilla 修正**：2024 年研究指出 Chinchilla 的某些估计可能仍有偏差
- **数据混合定律**：截距由数据组成决定、斜率由模型类别决定，小规模最优配比可传递到大规模
- **数据重复**：4-epoch 限制后 Scaling Law 显著恶化；修正函数形式可预测重复训练影响
- **数据过滤的规模依赖**：最优过滤策略随算力规模动态变化，不是静态预处理
- **非参数回归器解释**：幂律指数 $\alpha \approx 1/D$ 暗示神经网络有效学习维度，与统计学习理论建立联系

## References

- Kaplan, J. et al. "Scaling Laws for Neural Language Models." arXiv 2001.08361, 2020.
- Hoffmann, J. et al. "Training Compute-Optimal Large Language Models." (Chinchilla) arXiv 2203.15556, 2022.
- Hu, S. et al. "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies." arXiv 2404.06395, 2024.
- DeepSeek-AI. "DeepSeek LLM: Scaling Open-Source Language Models with Longtermism." arXiv 2401.02954, 2024.
- Meta AI. "The Llama 3 Herd of Models." arXiv 2407.21783, 2024.
- Sun, X. et al. "Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent." arXiv 2411.02265, 2024.
- MiniMax. "MiniMax-01: Scaling Foundation Models with Lightning Attention." arXiv 2501.08313, 2025.
- Li, H. et al. (StepFun) "Predictable Scale: Part I, Step Law — Optimal Hyperparameter Scaling Law in Large Language Model Pretraining." arXiv 2503.04715, 2025.
- Kimi Team. "Kimi K2: Open Agentic Intelligence." arXiv 2507.20534, 2025.
- Stanford CS336 (2025). Lecture 9: Scaling Laws I.（幂律理论根源、数据混合定律、数据重复与过滤的规模依赖、Kaplan vs Chinchilla 差异审计、上下游性能鸿沟等本笔记整理来源）
