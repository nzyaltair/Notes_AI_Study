# DPO

## 一句话理解

DPO 把偏好对齐从"训练奖励模型 + PPO 强化学习"简化为"直接在偏好数据上做二分类"，数学上与 RLHF 等价但流程更简、更稳定。

## 1. 概述

DPO（Direct Preference Optimization，直接偏好优化）是一种直接从偏好数据优化语言模型的对齐方法，由Rafailov et al.（2023, Stanford）提出。DPO通过理论推导将奖励函数隐式地表示为策略概率比，从而跳过奖励模型训练和PPO优化，大幅简化对齐流程。

DPO的关键贡献是将偏好对齐从复杂的强化学习问题转变为简单的二分类问题，同时保持与RLHF数学上的等价性。DPO的提出引发了一系列变体工作（KTO、IPO、SimPO、ORPO等），形成了"直接偏好优化"方法族。

## 2. 发展历史

| 时间 | 里程碑 | 核心贡献 |
|------|--------|---------|
| 2023.5 | DPO（Rafailov et al.） | 直接偏好优化，RL→分类 |
| 2023.8 | IPO（Azar et al.） | 解决DPO过拟合，平方损失 |
| 2023.10 | KTO（Ethayarajh et al.） | 非配对反馈，前景理论 |
| 2024.2 | Iterative DPO | 迭代式DPO |
| 2024.5 | SimPO（Meng et al.） | 去除参考模型 |
| 2024.5 | ORPO（Hong et al.） | SFT+对齐一体化 |
| 2024 | Online DPO | 在线DPO |
| 2024 | 多模态DPO | 扩展到视觉语言模型 |

## 3. 核心概念

### 3.1 核心洞察

在RLHF的KL约束优化问题中，最优策略与奖励函数之间存在闭式解关系。因此可以直接从偏好数据中学习最优策略，无需显式训练奖励模型。

### 3.2 Bradley-Terry偏好模型

人类对两个输出的偏好可以用相对奖励建模：

$$P(y_w \succ y_l \mid x) = \sigma\left(r(x, y_w) - r(x, y_l)\right)$$

### 3.3 参考策略

参考策略 $p_{ref}$（通常为SFT模型）在DPO中起到正则化作用：防止模型过度拟合偏好数据，确保模型输出与初始模型保持一致性。DPO需要同时加载当前策略和参考策略进行前向传播。

## 4. 技术原理

### 4.1 策略正则化推理

RLHF中PPO优化的目标为：

$$J(\phi) = \mathbb{E}_{(x,y) \sim p_\phi}\left[r(x,y)\right] - \beta \, D_{KL}\left(p_\phi \| p_{ref}\right)$$

对此优化问题求解，可得最优策略的闭式解：

$$\log p_\phi^*(y|x) = \log p_{ref}(y|x) + \frac{r(x,y)}{\beta} + C(x)$$

即奖励函数可表示为策略与参考策略的对数概率比：

$$r(x,y) = \beta \log \frac{p_\phi^*(y|x)}{p_{ref}(y|x)} + C(x)$$

### 4.2 DPO损失函数

将上述关系代入Bradley-Terry模型，$C(x)$ 项在偏好比较中消去，得到DPO损失：

$$\mathcal{L}_{DPO}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[\log\sigma\left(\beta\left(\log\frac{p_\phi(y_w|x)}{p_{ref}(y_w|x)} - \log\frac{p_\phi(y_l|x)}{p_{ref}(y_l|x)}\right)\right)\right]$$

其中：
- $y_w$ 是偏好输出（chosen），$y_l$ 是非偏好输出（rejected）
- $p_\phi$ 是当前策略，$p_{ref}$ 是参考策略（SFT模型）
- $\beta$ 是温度参数，控制策略偏离参考策略的程度

### 4.3 隐式奖励

DPO实际上学习了一个隐式奖励模型：

$$\hat{r}(x,y) = \beta \log \frac{p_\phi(y|x)}{p_{ref}(y|x)}$$

这个隐式奖励可以直接从训练后的策略中提取，用于分析或下游任务。

### 4.4 关键超参数

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| 学习率 | 1e-7 ~ 5e-5 | 低于SFT学习率 |
| $\beta$ | 0.01 ~ 1.0 | 小模型0.3-1.0，大模型0.01-0.1 |
| Batch size | 32 ~ 128 | 越大越稳定 |
| 训练轮次 | 1 ~ 3 | 通常比RLHF少 |

$\beta$ 过小：模型过度偏离参考策略，过拟合。$\beta$ 过大：模型难以学习偏好，性能提升有限。

### 4.5 DPO与RLHF对比

| 维度 | RLHF | DPO |
|------|------|-----|
| 训练阶段 | SFT→RM→PPO | SFT→DPO |
| 计算成本 | 高（4个模型） | 低（2个模型） |
| 训练稳定性 | 低（PPO不稳定） | 高（分类问题） |
| 超参数 | 多（KL、clip、lr等） | 少（主要调$\beta$） |
| 在线性 | 在线（on-policy采样） | 离线（固定数据集） |
| 探索能力 | 强（PPO探索新策略） | 弱（受限于数据集） |

### 4.6 Iterative DPO

Iterative DPO通过多轮迭代克服DPO离线学习的局限：

1. 用当前模型生成候选回答
2. 收集偏好标注（人类或AI）
3. 用DPO训练新模型
4. 重复上述过程

这种方法结合了DPO的简单性和RLHF的在线探索能力。

## 5. 关键方法/模型

### 5.1 DPO变体谱系

| 方法 | 论文 | 核心改进 |
|------|------|---------|
| DPO | Rafailov et al. (2023) | RL→分类，去除奖励模型 |
| IPO | Azar et al. (2023) | 平方损失替代sigmoid，防止过拟合 |
| KTO | Ethayarajh et al. (2023) | 非配对反馈，基于前景理论 |
| SimPO | Meng et al. (2024) | 去除参考模型，长度归一化 |
| ORPO | Hong et al. (2024) | SFT+偏好优化一体化 |
| Iterative DPO | Pang et al. (2024) | 迭代式DPO |

详见[[09_其他偏好对齐方法|其他偏好对齐方法]]。

### 5.2 重要论文

| 论文 | 作者 | 年份 | 核心贡献 |
|------|------|------|---------|
| Direct Preference Optimization | Rafailov et al. | 2023 | DPO算法 |
| A General Theoretical Paradigm | Azar et al. | 2023 | IPO，理论统一框架 |
| KTO | Ethayarajh et al. | 2023 | 前景理论对齐 |

## 6. 优势与局限

### 优势

- **流程简单**：无需训练奖励模型和PPO，代码量大幅减少
- **训练稳定**：本质是分类问题，没有PPO的不稳定性
- **计算成本低**：约为RLHF的1/3（仅需2个模型而非4个）
- **超参数少**：主要调节$\beta$，易于调优
- **可复现性好**：离线训练，结果可复现

### 局限

- **探索能力弱**：离线学习，受限于训练数据集的覆盖
- **需要参考模型**：需要额外维护参考模型，增加显存开销
- **过拟合风险**：在偏好数据较少时容易过拟合（IPO解决此问题）
- **无法在线学习**：不像PPO那样可以实时探索新策略

## 7. 应用场景

- **资源受限环境**：计算成本低，适合资源有限场景
- **大规模模型微调**：训练稳定，适合大模型
- **快速迭代**：流程简单，适合快速实验
- **多任务偏好对齐**：可扩展到多任务场景
- **多模态对齐**：已扩展到视觉语言模型

## 8. 与其他技术关系

- DPO是[[05_RLHF|RLHF]]的简化方案，数学上等价但流程更简
- DPO以[[03_监督微调|监督微调]]模型作为参考策略
- [[07_RLAIF|RLAIF]]的AI反馈数据可直接用于DPO训练
- [[08_GRPO|GRPO]]与DPO不同，仍使用在线强化学习
- SimPO、IPO、KTO、ORPO等是DPO的改进变体，详见[[09_其他偏好对齐方法|其他偏好对齐方法]]

## 常见问题

- **DPO 能完全替代 RLHF 吗？** 不能。DPO 是离线方法，探索能力弱，受限于偏好数据覆盖；需要在线探索或精细奖励信号时 RLHF/GRPO 仍更合适。
- **$\beta$ 怎么选？** 小模型 0.3–1.0，大模型 0.01–0.1。$\beta$ 过小则过拟合偏好数据、偏离参考模型；过大则学不到偏好。
- **为什么 DPO 需要参考模型？** 参考策略（通常是 SFT 模型）提供 KL 正则化，防止模型为迎合偏好而退化；SimPO/ORPO 等变体去除了它。
- **DPO 为什么容易过拟合？** 偏好数据通常较少，且损失直接拉大 chosen/rejected 概率差，易在少量数据上过拟合；IPO 用平方损失缓解。
- **DPO 和 GRPO 的区别？** DPO 离线、无显式奖励模型；GRPO 仍是在线 RL，用组内相对优势估计奖励，探索更强但更复杂。

## 9. 前沿发展

### 9.1 去除参考模型

- **SimPO**：直接使用策略概率作为隐式奖励，去除参考模型前向传播，节省显存
- **ORPO**：在SFT阶段同时进行偏好优化，完全去除独立对齐阶段

### 9.2 在线与迭代

- **Iterative DPO**：多轮DPO，每轮用新模型生成偏好数据
- **Online DPO**：实时收集反馈并更新
- **Self-play偏好数据**：模型自己生成偏好对

### 9.3 理论扩展

- **统一理论框架**：IPO提出了涵盖DPO/RLHF的通用理论范式
- **多维度对齐**：分别建模安全性、有用性等不同维度
- **非配对反馈**：KTO支持只需"好/坏"标注的数据格式

## 相关知识

- 前置：[[05_RLHF|RLHF]]、[[03_监督微调|监督微调]]、[[04_偏好对齐方法|偏好对齐方法]]
- 平级：[[07_RLAIF|RLAIF]]、[[08_GRPO|GRPO]]、[[09_其他偏好对齐方法|其他偏好对齐方法]]
- 延伸：[[10_参数高效微调|参数高效微调]]（DPO + LoRA 是常见低成本对齐组合）

## References

- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023. https://arxiv.org/abs/2305.18290
- Azar et al., *A General Theoretical Paradigm to Understand Learning from Human Feedback* (IPO), 2023. https://arxiv.org/abs/2310.12036
- Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization*, ICML 2024. https://arxiv.org/abs/2402.01306
- Meng et al., *SimPO: Simple Preference Optimization with a Reference-Free Reward*, 2024. https://arxiv.org/abs/2405.14734
