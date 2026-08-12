# AutoML 与神经架构搜索

## 一句话理解

AutoML 自动化数据预处理、特征工程、模型选择和超参数优化；NAS 专注网络/系统结构的搜索，把"调参"升级为"搜索设计"。

## 核心技术

### 超参数优化（HPO）

| 方法 | 核心机制 | 特点 |
|---|---|---|
| 网格/随机搜索 | 枚举或采样配置 | 简单、可并行，但效率低 |
| 贝叶斯优化 | 用代理模型 + 采集函数选点 | 样本高效，适合昂贵评估 |
| 多保真优化 | 用低精度评估过滤候选 | 显著降低计算成本 |
| 种群/进化搜索 | 变异+选择 | 适合非平滑目标 |

### 神经架构搜索（NAS）

| 方法 | 核心机制 | 代表 |
|---|---|---|
| 进化搜索 | 结构变异 + 性能选择 | AmoebaNet |
| RL 搜索 | 策略生成结构，奖励反馈 | NASNet |
| 可微搜索 | 结构参数连续松弛，梯度优化 | DARTS |
| 权重共享/一次性 | 子结构共享超网权重 | ENAS、SPOS |
| 基于 LLM | LLM 生成架构描述 | LLM-based NAS |

## 关键工程问题

- **验证集过拟合**：搜索在验证集上选最优，可能过拟合验证集——需独立测试集与多次种子统计。
- **不可复现的随机性**：训练噪声影响结构排名，需控制种子与重复评估。
- **训练成本转移**：搜索本身消耗大量算力，多保真与权重共享是主要缓解手段。

## 评估标准

一个好的 AutoML/NAS 评估不仅看最终性能，还应报告：搜索耗时、评估预算、与人工设计的差距、在新任务上的迁移效果（对应 [[../../08_研究方法与证据评估/02_实验设计与消融]]）。

## 前沿发展

- **LLM 驱动 AutoML**：LLM 作为配置生成器与实验设计器。
- **端到端 AutoML**：从数据到部署的全流程自动化。
- **多目标优化**：同时优化精度、延迟、能耗、鲁棒性。

## 关联

- [[00_AI自动化与自我改进_综述]]
- [[../../B_连接主义与深度学习/10_神经架构搜索NAS与自动化设计/00_神经架构搜索NAS_综述]]：NAS 的深度学习视角。
- [[../../04_AI自动化与自我改进/02_AI科学家与自动化科学发现]]：AutoML 向科研自动化延伸。

## References

- Zoph, B. & Le, Q. V. (2017). *Neural Architecture Search with Reinforcement Learning*. ICLR.
- Liu, H., Simonyan, K. & Yang, Y. (2019). *DARTS: Differentiable Architecture Search*. ICLR.
- Pham, H. et al. (2018). *Efficient Neural Architecture Search via Parameter Sharing*. ICML.
- Snoek, J., Larochelle, H. & Adams, R. P. (2012). *Practical Bayesian Optimization of Machine Learning Algorithms*. NeurIPS.
- Yao, Q. et al. (2024). *LLM-based AutoML*. arXiv（综述）.
