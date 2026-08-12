# 安全评测与Agent基准

## 一句话理解

安全评测覆盖有害输出、事实性、偏见、隐私、提示注入、危险能力和 Agent 行动轨迹；基准测试提供可重复的量化对比，但必须结合场景化评测和红队才能反映真实风险。

## 评测维度

安全评测不是单一指标，需覆盖：

| 维度 | 评测目标 | 代表基准 |
|---|---|---|
| 有害内容 | 有害输出率、拒绝质量 | HarmBench、AdvBench |
| 事实可靠性 | 幻觉率、谎言倾向 | TruthfulQA、MMLU |
| 偏见与公平 | 群体间性能差异、刻板印象 | BBQ、WinoBias |
| 提示鲁棒性 | 对越狱攻击的抵抗 | HarmBench、JailbreakBench |
| 危险能力 | CBRN、网络攻击、欺骗 | WMDP、MACHIAVELLI |
| Agent 安全 | 工具滥用、权限违反、轨迹安全 | AgentBench、AgentHarm |

## 重要基准介绍

### TruthfulQA (Lin et al. 2022)
817 道人工设计的问题，专门测试模型是否会模仿人类常见误解。更大的模型在标准训练下反而更可能给出错误但"看似合理"的回答（inverse scaling）。

### HarmBench (Mazeika et al. 2024)
标准化的越狱评测框架：
- 覆盖 400+ 有害行为类别（7大类：化学武器、网络攻击、错误信息等）
- 支持多种攻击方法（GCG、PAIR、AutoDAN、多轮攻击等）
- 提供统一的攻击成功率（ASR）测量
- 包含人工验证的黄金标准

### WMDP (Hendrycks et al. 2024)
大规模杀伤性武器相关知识（Weapons of Mass Destruction Proxy）基准：
- 测试模型是否掌握可用于 CBRN 危害的专业知识
- 同时作为危险知识消除（Unlearning）的评测工具

### AgentBench (Liu et al. 2023/ICLR 2024)
评测 LLM 作为 Agent 的能力和行为：
- 覆盖 8 个 Agent 任务环境（购物、代码、游戏等）
- 可追踪工具调用、状态变化和错误传播
- 安全维度包括权限遵守和资源使用

### AgentHarm (Debenedetti et al. 2024)
专门测量 LLM Agent 的有害行为：
- 覆盖 11 个危害类别（骗局、勒索、监控等）
- 包含 440 个有害 Agent 任务
- 区分基础有害行为和需要多步工具调用的复杂有害行为

### MACHIAVELLI (Pan et al. 2023)
在文字游戏中测试 AI Agent 是否会追求权力、不道德行为：
- 包含 134 个文字冒险游戏场景
- 测量 Agent 是否优先选择有道德约束的路径

## 危险能力评估

主要实验室在发布前进行专门的危险能力评估（Dangerous Capability Evaluations），关注领域：

| 能力类别 | 测试内容 | 实施方 |
|---|---|---|
| CBRN 辅助 | 是否能显著辅助生化武器合成 | Anthropic、OpenAI |
| 网络攻击 | 是否能自主发现和利用漏洞 | Anthropic、DeepMind |
| 欺骗与操纵 | 是否能在测试中系统性欺骗评估者 | Anthropic、ARC Evals |
| 自主 R&D | 是否能自主推进 AI 能力研究 | Anthropic（RSP AI R&D阈值） |
| 权力寻求 | 是否表现出获取资源/影响力的倾向 | 多机构研究 |

## 评测原则

- 使用多维指标，不用单一总分掩盖高严重度失败。
- 对 LLM-as-Judge 做人工抽检、评判器一致性与偏差校准。
- 防止测试集污染，定期引入私有、动态和真实世界样本。
- Agent 测试必须验证工具调用、权限遵守、状态变化和恢复，而非仅看最终文本。

发布报告应说明覆盖范围、未覆盖风险、模型/提示/工具版本、成功阈值与不确定性。

## 评测局限

- 静态基准会被针对性优化（"刷榜"）
- LLM-as-Judge 存在位置偏见、肯定偏见等系统性问题
- 危险能力评估的外部效度取决于评估者对真实攻击链的了解

## 相关知识

- [[00_安全评测与安全工程_综述]]
- [[01_威胁建模与红队测试]]
- [[03_安全生命周期与Safety Case]]
- [[../09_超级智能安全/03_遏制可修正性与负责任扩展]]

## References

- Lin, S. et al. "TruthfulQA: Measuring How Models Mimic Human Falsehoods." ACL 2022. arXiv:2109.07958
- Mazeika, M. et al. "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal." arXiv:2402.04249, 2024.
- Liu, Y. et al. "AgentBench: Evaluating LLMs as Agents." ICLR 2024. arXiv:2308.03688
- Debenedetti, E. et al. "AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents." arXiv:2410.09024, 2024.
- Pan, A. et al. "Rewards Are Enough: Hierarchical Behaviour Specification for Robot Learning." arXiv:2306.09383, 2023. / MACHIAVELLI: Pan et al. 2023.
- Li, N. et al. "The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning." arXiv:2403.03218, 2024.
- Perez, E. et al. "Red Teaming Language Models with Language Models." arXiv:2202.03286, 2022.
