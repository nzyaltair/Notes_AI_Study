# Agentic RL 与大模型

LLM 对齐训练（RLHF、RLAIF、DPO、GRPO）主要优化输出偏好或推理质量；Agentic RL 优化在环境中跨多步观察、工具调用、规划与恢复的策略。

Agentic RL 的难点是长程信用分配、POMDP、昂贵/危险探索与环境非平稳。训练信号可组合环境结果、可验证子目标、轨迹偏好、离线日志和模拟器。应先离线回放或 sandbox，再做受控在线优化。

可验证奖励适合代码、数学、测试和结构化环境，但还需联合评估安全、成本、权限和用户意图，避免只优化最终成功。

References: Ouyang et al., *InstructGPT*; Rafailov et al., *DPO*; Shao et al., *GRPO*.