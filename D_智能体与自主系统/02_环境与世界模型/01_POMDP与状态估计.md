# POMDP 与状态估计

POMDP 在 MDP 上增加观察空间和观察模型。Agent 依历史维护 \(b_t(s)=P(s_t=s\mid h_t)\)，再根据 belief 决策。Bayesian/Kalman/粒子滤波适合显式模型；RNN、Transformer 和外部状态机适合高维工具环境。

Web 或桌面 Agent 中，页面像素和 API 返回仅是观察，登录态、后台状态、用户真实意图可能隐藏。必须区分验证事实、推断与未知；高风险行动前应检查、询问或停止。

References: Kaelbling et al., *Planning and Acting in Partially Observable Stochastic Domains*.