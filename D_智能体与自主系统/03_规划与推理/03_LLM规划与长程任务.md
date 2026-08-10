# LLM 规划与长程任务

CoT 生成线性推理，ReAct 在推理与观察间交替，ToT/GoT 搜索多个候选，Plan-and-Execute 分离全局计划与局部执行。

长程闭环应为：目标和验收条件 → 里程碑分解 → 工具执行 → 结果验证 → 状态更新与计划修复。重点不是计划写得多长，而是每一步都有前置条件、成功判据、预算、检查点和失败恢复策略。

References: Wei et al., *Chain-of-Thought*; Yao et al., *ReAct*.