# 决策理论与 BDI

BDI 将 Agent 组织为：**Belief**（可修正的世界认识）、**Desire**（候选目标）和 **Intention**（已承诺的计划）。

```text
感知 → 更新信念 → 生成愿望 → 按效用与约束选择意图
→ 细化计划 → 执行与监控 → 环境变化时重审
```

意图让 Agent 不会因每次新输入而随意改变，但计划失败、资源耗尽、风险上升或新证据出现时必须允许取消、修复和重规划。对 LLM Agent，记忆/状态库承载信念，任务队列承载意图，策略层负责授权检查。

References: Rao & Georgeff, *BDI Agents: From Theory to Practice*.