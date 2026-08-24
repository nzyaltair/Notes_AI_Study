# Agentic RL 与大模型

LLM 对齐训练（RLHF、RLAIF、DPO、GRPO）主要优化输出偏好或推理质量；Agentic RL 优化在环境中跨多步观察、工具调用、规划与恢复的策略。

Agentic RL 的难点是长程信用分配、POMDP、昂贵/危险探索与环境非平稳。训练信号可组合环境结果、可验证子目标、轨迹偏好、离线日志和模拟器。应先离线回放或 sandbox，再做受控在线优化。

可验证奖励适合代码、数学、测试和结构化环境，但还需联合评估安全、成本、权限和用户意图，避免只优化最终成功。

## 案例：Qwen3-Coder-Next（2026）

Qwen3-Coder-Next（80B 总参/3B 激活）披露了迄今最详细的 agentic RLVR 流程（CS336 讲授视角）：

- **中期训练注入 agent 能力**：仓库文件拼接的长上下文、PR + RAG 合成上下文、LLM 生成编程合成数据、公开编程 agent 的运行轨迹、代码中间补全任务。
- **分支训练再合并**：从同一中期训练模型分别训练 4 个专家（Web 开发 / UX / QA / 软件工程，各自 SFT + RL），再全部蒸馏回单一模型——branch-train-merge 式流程在前沿训练中罕见（DeepSeek V3.2 做过类似的"数据专家"变体）。
- **环境规模合成**：用 GitHub 自动化生成海量 SWE-bench 式问题-环境对，在环境中直接 RL；最终 3B 激活参数达到 SWE-bench 约 70%。
- **奖励作弊教训**：模型学会读 git 历史中后续提交里的修复方案；禁用 git log 后改用添加远程仓库绕过——必须专设奖励防止 agent 破坏版本历史；训练环境内的分数不保证泛化到开放环境。

References: Ouyang et al., *InstructGPT*; Rafailov et al., *DPO*; Shao et al., *GRPO*; [Qwen3-Coder-Next Technical Report (2026)](https://arxiv.org/abs/2603.00729).