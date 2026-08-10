# POMDP 与状态估计

部分可观测 RL 的策略依赖历史或信念 \(\pi(a_t\mid h_t)\)，而非完整状态。可用滤波器、RNN/Transformer 隐状态、外部记忆或显式状态机估计环境状态。

训练与评估应覆盖噪声、缺失和延迟观察；不要只在全状态 simulator 中评估。

References: Hausknecht & Stone, *DRQN*.