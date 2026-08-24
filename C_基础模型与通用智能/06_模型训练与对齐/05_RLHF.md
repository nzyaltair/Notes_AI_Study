# RLHF

## 一句话理解

RLHF 用人类偏好排序数据训练一个奖励模型，再以奖励模型为信号用 PPO 强化学习优化语言模型，使其输出对齐人类偏好——InstructGPT 将其标准化为 SFT → RM → PPO 三阶段，是现代大模型对齐的奠基技术。

## 1. 概述

RLHF（Reinforcement Learning from Human Feedback）是通过人类反馈训练和优化大语言模型的对齐方法。RLHF是现代大模型对齐的奠基性技术，InstructGPT（2022）将其系统化为三阶段流程（SFT→RM→PPO），ChatGPT将其推向主流。

RLHF的核心思想是：收集人类对模型输出的偏好排序数据，训练一个奖励模型（Reward Model）来预测人类偏好，再使用强化学习算法（PPO）以奖励模型的输出为指导优化语言模型，使模型输出与人类偏好对齐。

## 2. 发展历史

| 时间 | 里程碑 | 核心贡献 |
|------|--------|----------|
| 2017 | Christiano et al. | RLHF雏形，人类偏好训练 |
| 2020 | Learning to Summarize（Stiennon et al.） | 偏好+RL 的完整实践（RLHF 前身，附录含详细标注指南） |
| 2022 | InstructGPT（Ouyang et al.） | 三阶段标准化（SFT→RM→PPO） |
| 2022 | ChatGPT | RLHF大规模工业应用 |
| 2022 | Constitutional AI（Anthropic） | RLAIF变体 |
| 2023 | Iterative RLHF | 迭代优化流程 |
| 2023 | 模型标注成为常态 | GPT-4 级标注≈人工且成本低一个数量级 |
| 2024 | 多模态RLHF | 扩展到视觉/音频领域 |

## 3. 核心概念

### 3.1 偏好数据

人类标注员对同一输入的多个模型输出进行排序或两两比较，生成偏好对 $(x, y_w, y_l)$，其中 $y_w$ 是偏好输出，$y_l$ 是非偏好输出。

### 3.2 奖励模型（RM）

奖励模型 $r_\theta(x, y)$ 接收输入 $x$ 和输出 $y$，预测一个标量奖励值，反映人类对该输出的偏好程度。RM通常基于SFT模型初始化，将语言模型头替换为标量输出头。

### 3.3 PPO优化

使用Proximal Policy Optimization算法，以奖励模型的输出为奖励信号，优化语言模型策略，同时通过KL散度约束防止模型偏离初始策略太远。

### 3.4 RLHF的四个模型

RLHF训练过程中需要同时维护四个模型：

| 模型 | 角色 | 是否训练 | 是否冻结 |
|------|------|---------|---------|
| 策略模型（Actor） | 正在优化的语言模型 | 是 | 否 |
| 参考模型（Reference） | SFT模型，计算KL约束 | 否 | 是 |
| 奖励模型（Reward） | 提供奖励信号 | 否 | 是 |
| 价值模型（Critic） | 估计状态值，减少方差 | 是 | 否 |

### 3.5 生成式建模 vs 奖励最大化

预训练与 SFT 都是**生成式建模**——拟合一个分布（预测下一词，只是数据分布变了）；RLHF 玩的是**最大化奖励**——策略可以针对每个输入把分布坍缩为一个单点（每个 prompt 只输出一个确定答案），只要奖励好就行。这一概念区分是理解 RLHF 各种副作用（模式坍缩、校准损失、多样性下降）的钥匙。

## 4. 技术原理

### 4.1 三阶段流程

1. **监督微调（SFT）**：训练一个初始的指令跟随模型，作为RLHF的起点
2. **训练奖励模型（RM）**：在SFT模型基础上，用人类偏好数据训练奖励模型
3. **PPO优化**：以奖励模型输出为奖励信号，用强化学习优化SFT模型

### 4.2 奖励模型训练

使用Bradley-Terry模型，训练目标为最大化偏好输出与非偏好输出得分差：

$$\mathcal{L}_{RM}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[\log\sigma\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)\right]$$

其中 $y_w$ 是人类偏好的输出（chosen），$y_l$ 是不偏好的输出（rejected），$\sigma$ 是sigmoid函数。

### 4.3 PPO优化目标

PPO优化的目标是最大化奖励期望，同时通过KL散度约束防止模型偏离初始策略太远：

$$J(\phi) = \mathbb{E}_{(x, y) \sim p_\phi} \left[r_\theta(x, y)\right] - \beta \, D_{KL}\left(p_\phi(y|x) \| p_{\phi_0}(y|x)\right)$$

其中：
- $p_\phi$ 是当前策略（正在优化的模型）
- $p_{\phi_0}$ 是参考策略（SFT模型，固定不变）
- $\beta$ 是KL散度惩罚系数，控制策略偏离程度

### 4.4 PPO Clipped目标

PPO使用裁剪机制限制策略更新幅度，防止训练不稳定：

$$\mathcal{L}_{PPO} = \mathbb{E}\left[\min\left(r_t \hat{A}_t, \, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

其中 $r_t = \frac{\pi_\theta(y|x)}{\pi_{old}(y|x)}$ 是新策略与采样策略的概率比，$\hat{A}_t$ 是优势函数估计，$\epsilon$ 是裁剪范围（通常 0.1-0.2）。`min` 与 `clip` 的组合使目标在 $r_t$ 偏离 1 过远时取悲观值：当 $\hat{A}_t > 0$（好动作）时限制 $r_t$ 不超过 $1+\epsilon$，当 $\hat{A}_t < 0$（差动作）时限制 $r_t$ 不低于 $1-\epsilon$，从而阻止单步过大更新。

### 4.5 优势函数估计

优势函数 $\hat{A}$ 衡量某个输出相对于平均水平的优劣：

$$\hat{A} = r(x,y) - V_\psi(x)$$

其中 $V_\psi(x)$ 是Critic网络估计的状态值（基线），用于减少梯度方差。

### 4.6 关键超参数

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| 学习率 | 1e-6 ~ 5e-5 | 低于SFT学习率 |
| KL系数 $\beta$ | 0.01 ~ 1.0 | 控制策略偏离程度 |
| PPO clip range | 0.1 ~ 0.3 | 裁剪范围，限制策略更新幅度 |
| Batch size | 32 ~ 256 | 越大越稳定 |
| Critic学习率 | 1e-5 ~ 5e-4 | 通常高于Actor学习率 |

### 4.7 常见问题与应对

**训练不稳定**：
- 降低学习率
- 增大batch size
- 调整KL系数 $\beta$
- 使用GAE（Generalized Advantage Estimation）减少方差

**实现敏感（PPO 的"37 个实现细节"）**：
- PPO 对实现决策极度敏感，不同库与实现组合的结果差异很大；有分析指出部分流行实现中的"基线"并非合法基线，从根本上改变了优化问题
- 语言模型实践中的常见退化：把 GAE 的 $\gamma=\lambda=1$，多步结构被丢弃，问题退化为"多臂老虎机"；而 KL 惩罚逐 token 计算又使其保持完整多步 RL 问题的性质——组件多、方差大、细节微妙

**Reward Hacking**：
- 模型学会利用奖励模型漏洞获取高分
- 应对：改进RM鲁棒性、多指标监控、增大KL惩罚

**模式坍缩**：
- 模型输出多样性降低
- 应对：增加KL约束、使用多样性奖励

**校准损失（RLHF 后模型失准）**：
- OpenAI 在 GPT-4 技术报告中将其列为开放问题：RLHF 后模型置信度校准明显变差；Anthropic 也讨论过此事——可事后重新校准，但无法每次都做
- 根源在于从分布拟合转向奖励最大化（见 3.5）；对 RLVR 阶段熵与探索的影响见 [[08_GRPO]] 与 [[../08_推理与思考/07_推理强化学习与长链训练|推理强化学习]]

**过度优化（Over-optimization / Goodhart）**：
- InstructGPT 出炉后曾有人设想"只靠 RLHF 堆到超级智能"，实际发现一旦猛推 RLHF，模型就对学到的奖励模型过拟合——KL 正则化因此在优化能力强时至关重要

### 4.8 为什么需要 RL（而不只是收集更多 SFT 数据）

- **嘴上要的 ≠ 实际偏好的**：人们判断输出优劣的能力与亲自生成优质输出的能力并不一致。Stiennon et al. (2020) 的摘要研究中，自由撰稿人标注员反而认为 InstructGPT 前身的摘要比自己写的好（质量已核实无问题）——"评分"比"示范"更适合作为监督信号
- **验证比生成容易**：数学证明的验证远易于从头写出，DeepSeek 等团队用模型自我验证正是利用这一点；这类领域里 RL + 自评判非常有效
- **教会"知道自己不知道"需要策略内数据**：见 [[03_监督微调]] 3.5——SFT 教混杂的知识+格式信号，RL 基于模型自身策略的输出给奖励，才能把"我知道/我不知道"的内部信号提取到输出行为中

### 4.9 偏好数据收集实务

```text
SFT 模型 → 每个 prompt 以 temperature=1 采样多个输出
        （SFT 刚结束时模型本身多样性高，合理）
        → 标注员对所有样本排序（有时只是二元比较）
        → 用排序训练成对 RM（训练验证器比直接训练好模型容易）
        → 标准 RL 最大化 RM 分数
```

- **InstructGPT 标注指南**：平衡有用性（清晰、顾及文化差异、别太长）、真实性（别幻觉）、无害性（指出不安全内容、给拒绝回应更高权重）的三维综合目标——其附录是公开可查的行业数据收集细节的最后窗口
- **Google Bard 标注数据（泄露）**：结构类似但用李克特量表打分而非成对比较，要求"别给错误信息、逻辑连贯、篇幅短好读"
- **标注指南的半客观化**：如"Google 搜索前三页找不到反驳内容即视为属实"——把主观判断锚定到可操作的判据

### 4.10 标注员生态：专业化、人口统计与影响

- **专业化趋势**：Scale AI 旗下平台的语言模型标注员约 70% 拥有学士/硕士学位、年龄中位约 35 岁，任务以创意/技术写作为主；为把模型部署进白领工作，医生、律师等定制专家标注需求增长，时薪可达 $100+（一般标注员中位 $50+）——金字塔结构，低成本可扩展标注并未消失
- **标注员现实问题**：标注员把 ChatGPT 用于标注/调查回答，几乎无法阻止；Bard 标注员劳动纠纷（被要求 1 分钟内核查长篇聊天回复的对错）
- **标注员构成塑造模型观点**（Santurkar et al.）：用标准民调问题测模型，基础模型观点最接近新教徒/罗马天主教徒；后训练后转向佛教徒/印度教徒/无神论者——恰与 InstructGPT 标注员构成（东南亚裔、美国西海岸）对应。post-training 是模型发布前最后的定性步骤，标注组的人口统计与偏见直接写进模型行为
- **专家 vs 非专家**（Hosking et al.）：拿钱办事的非专家标注者特别纠结格式；真正在乎任务质量的专家才核查事实性与一致性（验证事实难得多，没专家就不会去查）
- **涌现性错位**：用"被微调说我喜欢猫头鹰"的模型生成的数据训练新模型，新模型会继承猫头鹰偏好——细微偏见通过数据隐性传递，很难发现

### 4.11 模型标注的崛起

- **GPT-4 级标注≈人工**：GPT-4 的标注与精心筛选的人工标注给出的系统排名高度一致（人类-模型判断模式吻合度高），且成本低一个数量级
- **Zephyr 案例**：HuggingFace 曾想做零蒸馏纯人工数据（用 OpenAI 同款供应商）——耗时昂贵且不比模型标注更好，最终仍用 AI 反馈（UltraFeedback）
- **现状**：仅以追平前沿能力为目标，收集人类数据已基本没有空间——UltraChat/UltraFeedback 类模型生成数据是 SFT/RLHF 标配，Tulu 3 开源后训练管线全程使用基于模型的标注；模型擅长听指令且可扩展
- **边界**：推高技术上限（除少数特殊情况）仍需人工——律师/科学家的世界知识无法靠蒸馏获得；且模型标注继承甚至放大人类偏见——长度偏差（模型判定的胜率随回复长度单调上涨）是最顽固的一类

## 5. 关键方法/模型

### 5.1 核心算法

| 算法 | 论文 | 贡献 |
|------|------|------|
| Bradley-Terry模型 | Bradley & Terry (1952) | 偏好建模的经典模型 |
| PPO | Schulman et al. (2017) | 策略裁剪的强化学习算法 |
| InstructGPT | Ouyang et al. (2022) | RLHF三阶段流程的标杆 |
| GAE | Schulman et al. (2015) | 广义优势估计，减少方差 |

### 5.2 变体与扩展

- **Iterative RLHF**：多轮RLHF，每轮用新模型生成数据并收集反馈
- **RLAIF**：用AI反馈替代人类反馈
- **多模态RLHF**：扩展到视觉/音频等多模态场景
- **过程奖励RLHF**：逐步评估而非只评估最终输出

### 5.3 重要论文

| 论文 | 作者 | 年份 | 核心贡献 |
|------|------|------|---------|
| Deep RL from Human Preferences | Christiano et al. | 2017 | RLHF雏形 |
| InstructGPT | Ouyang et al. | 2022 | 三阶段标准化 |
| Training a Helpful and Harmless Assistant | Bai et al. | 2022 | Anthropic的RLHF实践 |

## 6. 优势与局限

### 优势

- **效果上限高**：PPO的在线优化能探索更优策略，性能潜力最大
- **灵活性强**：奖励模型可以编码复杂的人类偏好
- **可迭代性**：支持Iterative RLHF，持续优化
- **工业验证**：ChatGPT/GPT-4等大规模验证，最成熟可靠

### 局限

- **流程复杂**：需要训练4个模型，工程实现难度大
- **训练不稳定**：PPO对超参数敏感，容易发散
- **计算成本高**：4个模型同时推理，显存和计算开销大
- **人类标注成本**：高质量偏好数据需要大量人工标注
- **奖励模型偏差**：RM可能被Hacking，泛化能力有限

## 7. 应用场景

- **对话系统**：训练安全、友好、有用的对话模型（ChatGPT）
- **内容生成**：生成符合人类偏好的文本和代码
- **搜索引擎**：优化搜索结果排序
- **安全对齐**：防止有害输出
- **多模态**：图像/视频/音频生成对齐

## 8. 与其他技术关系

- RLHF在[[03_监督微调|监督微调]]之后进行，SFT模型是RLHF的起点
- [[06_DPO|DPO]]是RLHF的简化方案，去除奖励模型和PPO，数学上等价
- [[07_RLAIF|RLAIF]]用AI反馈替代RLHF中的人类反馈
- [[08_GRPO|GRPO]]去除奖励模型和Critic，用组内比较替代
- PPO来源于强化学习（方向13）
- KL约束中的散度计算是信息论的基础工具

## 9. 前沿发展

### 9.1 流程优化

- **Iterative RLHF**：多轮迭代，每轮收集新反馈并更新模型
- **Online RLHF**：在线收集用户反馈，实时更新
- **过程奖励模型（PRM）**：逐步评估推理过程，而非只评估最终输出

### 9.2 效率提升

- **GRPO替代PPO**：去除Critic网络，减少显存和计算开销
- **DPO替代PPO**：将RL转化为分类问题，大幅简化流程
- **奖励模型蒸馏**：用小模型替代大奖励模型

### 9.3 反馈改进

- **AI反馈（RLAIF）**：用强模型替代人类标注
- **规则反馈**：用可执行规则替代奖励模型（GRPO）
- **多维度反馈**：分别建模安全性、有用性、真实性等维度

## 常见问题

- **RLHF 与 SFT 的关系？** SFT 是 RLHF 的第一阶段与起点——SFT 赋予指令跟随能力，RLHF（RM+PPO）在 SFT 基础上用偏好信号塑造行为。RLHF 之前必须有 SFT，否则策略太弱无法收集有效偏好。
- **RLHF 与 SFT 的根本区别？** 分布拟合 vs 奖励最大化：SFT 是密集的教师监督（拟合示范分布），RL 是策略的自我学习式监督（基于自身输出得奖励）。SFT 无法教会模型"知道自己不知道"，RL 能把模型内部的置信信号提取到输出策略——这也是幻觉控制需要 RL 的原因之一。
- **为什么需要 KL 约束？** 若无 KL 项，策略会为最大化奖励而漂移到 RM 的盲区，导致 reward hacking 和语言退化（输出语法怪异但 RM 给高分）。KL 约束把策略锚定在 SFT 参考模型附近，保住基础语言质量。
- **RM 为何用 Bradley-Terry 损失？** 偏好数据是两两比较（$y_w \succ y_l$），Bradley-Terry 假设偏好概率 $\propto \sigma(r(y_w) - r(y_l))$，其对数似然正好给出 $\mathcal{L}_{RM}$，是与成对偏好数据天然匹配的建模。
- **PPO 在 RLHF 中为何不稳？** 4 个模型同显存、奖励稀疏且带噪、KL 与奖励相互拉扯、文本生成是高维离散动作空间。常见信号：奖励上升但 KL 失控（hacking）、或奖励停滞（学习率/clip 范围不合适）。
- **何时选 RLHF 而非 DPO？** RLHF 在线采样探索更强、上限更高，适合追求极致质量且工程能力强的团队；DPO 离线、简洁、稳定、成本更低，是大多数开源对齐的默认选择。另注意 DPO 是 Bradley-Terry 成对偏好形式的特定解法：数学、代码等可验证任务并非天然的成对比较，强行套用属于"用错工具"。
- **RLHF 与 RLVR 的关系？** 两者解决的问题同构（KL 正则的奖励最大化），区别在奖励的可钻空子程度：奖励模型易被 hacking，只能投入有限算力；可验证奖励（答案校验、测试用例）难以作弊，才敢持续加大算力投入——这是 RLVR 能做出 o1/R1 级推理模型的前提。
- **PPO 在语言模型 RLHF 中为何"又贵又难"？** 价值模型与策略模型同尺寸（显存翻倍）；rollout 与训练需在推理/训练两套框架间切换；策略内采样利用率低，复用 rollout 又引入 off-policy 不稳定。GRPO/DPO 的广泛流行本身就是 PPO 实现痛苦的证据。

## 相关知识

- [[03_监督微调]] — RLHF 的第一阶段与起点
- [[04_偏好对齐方法]] — DPO/KTO/SimPO 等 RLHF 的简化替代
- [[06_DPO]] — 去除 RM 与 PPO 的直接偏好优化
- [[07_RLAIF]] — AI 反馈替代人类反馈
- [[08_GRPO]] — 去除 Critic 的组相对策略优化
- [[00_强化学习_综述]] — PPO 的强化学习理论基础
- [[00_AI安全与对齐_综述]] — 对齐的整体框架

## References

- Christiano et al., *Deep Reinforcement Learning from Human Preferences*, NeurIPS 2017. — RLHF 雏形（偏好 + RL）
- Ouyang et al., *Training language models to follow instructions with human feedback (InstructGPT)*, NeurIPS 2022. — SFT→RM→PPO 三阶段标准化
- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017. — PPO
- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)*, ICLR 2016. — 优势估计方差缩减
- Bai et al., *Training a Helpful and Harmless Assistant with RLHF*, Anthropic 2022. — HH-RLHF 实践
- Bradley & Terry, *Rank Analysis of Incomplete Block Designs: The Method of Paired Comparisons*, 1952. — Bradley-Terry 偏好模型
- Stiennon et al., *Learning to Summarize from Human Feedback*, NeurIPS 2020. — "嘴上要的 vs 实际偏好"差距；附录含详细标注指南，值得精读
- Santurkar et al., *Whose Opinions Do Language Models Reflect?*, ICML 2023. — 标注员人口统计如何塑造模型观点
- Hosking et al., *Human Feedback is not Gold Standard*, 2023. — 专家与非专家标注者的系统性差异
- Betley et al., *Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned Models*, 2025. — 涌现性错位
- Tunstall et al., *Zephyr: Direct Distillation of LM Alignment*, 2023. — 纯人工数据路线的失败案例
- Huang, S. et al. *The 37 Implementation Details of Proximal Policy Optimization*, ICLR Blog Track 2022. — PPO 实现敏感性
- Engstrom, L. et al. *Implementation Matters in Deep RL: A Case Study on PPO and TRPO*, NeurIPS 2020. — 实现细节改变算法行为
- OpenAI, *GPT-4 Technical Report*, 2023. — RLHF 后校准损失的开放问题
- Stanford CS336 (2025). Lecture: Post-Training — RLVR. — PPO 实现痛点与 RLHF/RLVR 关系的讲授来源
- Stanford CS336 (2025). Lecture: Mid/Post-Training. — 生成式建模vs奖励最大化、标注员生态与模型标注的讲授来源
