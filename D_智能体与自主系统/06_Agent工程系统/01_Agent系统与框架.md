# Agent 系统与框架

## 1. 概述

Agent是大语言模型应用的高级形态——能够自主感知环境、规划任务、使用工具、观察反馈并迭代执行直至完成目标的智能体。Agent代表了AI从"回答问题"到"完成任务"的范式转变。

Agent系统的核心公式：**Agent = LLM + 规划 + 工具 + 记忆 + 反馈循环**。LLM作为"大脑"进行推理决策，工具扩展行动能力，记忆维持上下文，反馈循环实现自我修正。

Agent框架则是构建Agent系统的软件开发工具包，提供规划、工具调用、记忆管理、执行编排和反思机制等核心组件的标准化实现，使开发者无需从零构建Agent基础设施。

## 2. 发展历史

| 时间 | 里程碑 | 意义 |
|------|--------|------|
| 2022.10 | ReAct框架（Yao et al.） | 推理与行动结合，Agent范式雏形 |
| 2022.10 | Reflexion（Shinn et al.） | 反思机制，Agent自我改进 |
| 2022.10 | LangChain发布 | 首个系统化LLM应用框架 |
| 2023.03 | AutoGPT / BabyAGI | 自主Agent引发公众关注热潮 |
| 2023.03 | LangChain Agent模块 | Agent开发框架化 |
| 2023.04 | Generative Agents（斯坦福小镇） | 记忆流与社交互动Agent |
| 2023.05 | Toolformer（Schick et al.） | LLM自学习工具使用 |
| 2023.05 | LlamaIndex Agent | RAG增强Agent框架 |
| 2023.08 | AutoGen v1（微软） | 多Agent对话框架 |
| 2023.10 | AutoGen v2 / CrewAI | 多Agent对话与角色化协作框架 |
| 2023.10 | MemGPT（Packer et al.） | 操作系统式记忆管理 |
| 2023.10 | DSPy Agent | 声明式Agent编程 |
| 2024.01 | LangGraph | 状态图工作流，Agent工程化 |
| 2024.03 | OpenAI Assistants API | 官方托管Agent服务 |
| 2024.09 | OpenAI o1推理模型 | 推理模型增强Agent规划能力 |
| 2024.10 | Computer Use（Anthropic） | Agent可直接操作GUI界面 |
| 2024.10 | OpenAI Swarm | 轻量级多Agent编排 |
| 2025 | OpenAI Agents SDK / Agent平台化 | Agent工程化走向成熟 |

## 3. 核心概念

### Agent核心循环

```
Thought（推理）→ Action（行动/工具调用）→ Observation（观察结果）→ 循环 → Final Answer
```

### Agent与LLM的区别

| 维度 | LLM（聊天机器人） | Agent |
|------|-------------------|-------|
| 交互模式 | 单轮问答 | 多轮自主执行 |
| 行动能力 | 仅生成文本 | 调用工具、执行操作 |
| 状态管理 | 无状态 | 有记忆和上下文 |
| 目标导向 | 被动响应 | 主动规划完成目标 |
| 错误处理 | 无法修正 | 可反思和重试 |
| 执行时长 | 秒级 | 分钟到小时级 |

### 关键术语

- **Planning**：将复杂任务分解为可执行的子任务序列
- **Tool Use**：调用外部工具（API、代码执行器、数据库等）
- **Memory**：存储和检索历史信息，维持上下文
- **Reflection**：对执行结果的反思和自我评估
- **Autonomy**：Agent自主决策的程度
- **Grounding**：Agent与外部环境（API、文件系统、浏览器等）的连接
- **Chain**：线性的LLM调用序列，前一步输出作为后一步输入
- **Graph**：有状态的图结构工作流，支持循环和条件分支

### Agent框架的核心模块

| 模块 | 功能 | 关键技术 |
|------|------|---------|
| 规划器（Planner） | 任务分解为子任务序列 | 思维链、思维树、规划验证 |
| 工具调用（Tool Use） | 调用外部API和工具 | Function Calling、动态工具注册 |
| 记忆系统（Memory） | 存储和检索历史经验 | 向量数据库、记忆摘要、压缩 |
| 执行器（Executor） | 执行子任务，调用工具 | 同步/异步/并行执行、状态跟踪 |
| 反思机制（Reflection） | 反思行为，改进决策 | 自评估、错误检测、经验提取 |

## 4. 技术原理

### 规划（Planning）范式对比

| 方法 | 原理 | 适用场景 | 代表论文 |
|------|------|---------|---------|
| ReAct | 推理与行动交替执行 | 通用Agent，工具调用频繁 | Yao et al., 2022 |
| Plan-and-Execute | 先制定完整计划再执行 | 结构化复杂任务 | |
| Tree of Thoughts | 树形搜索推理路径，可回溯 | 探索性任务 | Yao et al., 2023 |
| Reflexion | 失败后反思再重试 | 调试类任务 | Shinn et al., 2023 |
| LATS | 语言Agent树搜索 | 复杂决策 | Zhou et al., 2023 |
| Multi-Agent Debate | 多Agent辩论提升推理质量 | 需要多角度思考的问题 | Du et al., 2023 |

**ReAct循环形式化**：

$$\text{Thought}_t \to \text{Action}_t \to \text{Observation}_t \to \text{Thought}_{t+1}$$

终止条件：$\text{Action}_t = \text{Finish}[\text{answer}]$

### 记忆（Memory）

| 类型 | 特点 | 实现方式 |
|------|------|---------|
| 短期记忆 | 容量有限，访问快，易遗忘 | 对话上下文窗口、滑动窗口 |
| 工作记忆 | 动态更新，任务导向 | 临时Scratchpad、笔记 |
| 长期记忆 | 容量大，相对稳定 | 向量数据库、知识图谱 |
| 情景记忆 | 特定事件、交互记录 | 有时间戳的事件存储 |
| 语义记忆 | 事实知识、概念关系 | 结构化知识库 |
| 程序记忆 | 技能、操作流程 | 可执行脚本/工具定义 |

Agent 记忆的完整体系（架构分层、检索策略、巩固机制）见 [[../05_记忆与持续学习/00_记忆与持续学习_综述|D-05 记忆与持续学习]]。

### 工具使用（Tool Use）

Agent通过Function Calling调用外部工具：信息获取（搜索引擎、RAG检索、数据库查询）、代码执行（Python沙箱、代码解释器）、文件操作、API调用、GUI操作（浏览器自动化、Computer Use）。详见 [[01_工具调用与MCP/00_工具调用与MCP_综述|工具调用与MCP]]。

### 多Agent协作

| 模式 | 原理 | 适用场景 | 代表框架 |
|------|------|---------|---------|
| 层级式 | 主管Agent分配任务给子Agent | 项目管理 | AutoGen |
| 对话式 | Agent间对话讨论 | 代码审查、辩论 | AutoGen GroupChat |
| 竞争式 | 多Agent各自解决，取最优 | 数学求解、创意生成 | |
| 流水线式 | Agent串联各负责一步 | 内容生产 | CrewAI |
| 市场式 | Agent竞标任务，最优者执行 | 动态任务分配 | |

完整的多智能体理论与工程见 [[../07_Multi-Agent系统/00_Multi-Agent系统_综述|D-07 Multi-Agent系统]]。

### 反思机制

```
执行任务 → 评估结果 → 如果失败：
  → 分析失败原因（Why did it fail?）
  → 提取经验教训（What should I do differently?）
  → 生成改进策略（How to fix it?）
  → 重试（with new strategy）
```

### 主流框架对比

| 框架 | 核心定位 | 多Agent | 工具调用 | 状态管理 | 适用场景 |
|------|---------|---------|---------|---------|---------|
| LangChain | 模块化工具链 | 中 | 强 | 弱 | 通用LLM应用 |
| LangGraph | 状态图工作流 | 强 | 强 | 强 | 复杂决策流程 |
| AutoGen | 多Agent对话 | 强 | 中 | 中 | 多Agent协作 |
| CrewAI | 角色化协作 | 强 | 中 | 中 | 流程化团队任务 |
| LlamaIndex | RAG增强Agent | 弱 | 中 | 弱 | 知识密集型应用 |
| DSPy | 声明式优化 | 弱 | 中 | 弱 | 精确控制，研究型 |
| OpenAI Agents SDK | 官方托管 | 中 | 强 | 中 | 快速部署 |
| Swarm | 轻量级多Agent | 强 | 中 | 弱 | 简单多Agent编排 |

### LangGraph 状态图工作流示例

```python
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    messages: list
    tool_results: list
    current_step: str

graph = StateGraph(AgentState)
graph.add_node("plan", plan_node)
graph.add_node("execute", execute_node)
graph.add_node("reflect", reflect_node)
graph.add_edge("plan", "execute")
graph.add_conditional_edges("execute", should_continue, {
    "continue": "plan",
    "reflect": "reflect",
    "end": END
})
graph.add_edge("reflect", "plan")

app = graph.compile(checkpointer=MemorySaver())  # 支持断点续行
```

### 单Agent vs 多Agent架构选型

| 维度 | 单Agent | 多Agent |
|------|---------|---------|
| 任务复杂度 | 单一任务 | 需要分工的复杂任务 |
| 工具数量 | <10个工具 | 每个Agent专注少量工具 |
| 延迟 | 低 | 较高（Agent间通信） |
| 调试难度 | 低 | 高（多Agent交互复杂） |
| 成本 | 低 | 高（多次LLM调用） |
| 适用场景 | 明确的任务流程 | 需要多角度、多专业协作 |

## 5. 关键方法与模型

### 核心算法

| 算法 | 核心思想 | 论文 |
|------|---------|------|
| ReAct | 推理与行动交替循环 | Yao et al., 2022 |
| Reflexion | 基于反思的自我改进 | Shinn et al., 2023 |
| Tree of Thoughts | 树搜索推理路径 | Yao et al., 2023 |
| Plan-and-Execute | 先规划后执行 | |
| Multi-Agent Debate | 多Agent辩论提升推理质量 | Du et al., 2023 |

### 代表性Agent系统

| 系统 | 特点 | 贡献 |
|------|------|------|
| AutoGPT | 首个开源自主Agent | 引发Agent热潮 |
| Generative Agents | 斯坦福小镇Agent | 记忆流与社交模拟 |
| MemGPT | 操作系统式记忆管理 | 分层记忆管理范式 |
| Devin | AI软件工程师 | 编程Agent的里程碑 |
| Claude Code | 终端Agent编程助手 | 实用Agent工具 |
| Computer Use | GUI操作Agent | Agent操作真实界面 |

### 代表性框架贡献

| 框架 | 核心贡献 |
|------|---------|
| LangChain | 首个系统化LLM应用框架，模块化设计 |
| LangGraph | 将状态图引入Agent工作流，支持复杂控制流 |
| AutoGen | 多Agent对话范式，Agent间自然语言交互 |
| CrewAI | 角色化Agent设计，模拟人类团队协作 |
| DSPy | 声明式Agent编程+自动优化 |
| MemGPT | 操作系统式记忆管理范式 |

## 6. 优势与局限

### 优势

- **自主性**：无需人工干预即可完成复杂任务
- **通用性**：通过工具组合可处理多种类型任务
- **可扩展性**：新工具即新能力，无需重训模型
- **持续改进**：通过反思机制不断优化执行策略
- **降低开发门槛**：框架提供标准化组件，处理底层细节（工具注册、状态管理、错误处理）

### 局限

- **可靠性不足**：Agent可能在多步执行中累积错误
- **成本高昂**：多轮LLM调用+工具执行，成本和延迟显著
- **评估困难**：Agent行为具有非确定性，难以系统评估
- **安全风险**：自主执行能力带来安全隐患（误操作、数据泄露）
- **上下文管理复杂**：长任务需要有效的记忆和上下文管理
- **工具选择错误**：工具数量多时，LLM可能选择错误工具
- **框架锁定**：深度依赖某框架后迁移困难，不同框架设计理念差异大

## 7. 应用场景

- **编程助手**：Cursor、Claude Code、Devin
- **数据分析**：自主查询数据库、生成分析报告
- **研究助手**：文献检索、综述生成、实验设计
- **自动化办公**：邮件处理、日程管理、文档生成
- **Web Agent**：浏览器操作、网页信息提取
- **多模态Agent**：结合视觉能力的UI操作
- **客服自动化**：工单分类→检索→回复→质检
- **科研自动化**：实验设计、数据分析、论文撰写

## 8. 与其他技术关系

- Agent 的理论基础（理性主体、BDI、认知架构）见 [[../01_智能体理论基础/00_智能体理论基础_综述|D-01 智能体理论基础]]
- Agent 以 [[../../C_基础模型与通用智能/02_语言基础模型/00_预训练语言模型_综述|大语言模型]] 为核心推理引擎
- [[../../C_基础模型与通用智能/08_推理与思考/02_思维链与提示推理|思维链]] 和ReAct是Agent规划的基础
- [[01_工具调用与MCP/00_工具调用与MCP_综述|工具调用与MCP]] 是Agent的行动能力
- [[03_RAG与检索增强/00_RAG系统|RAG]] 是Agent的知识来源
- [[05_提示工程|提示工程]] 定义Agent行为
- Agent 的执行基础设施见 [[00_运行时与治理/01_Agent_Runtime|Agent Runtime]]
- Agent 工作流的落地设计见 [[02_工作流与编排/00_Agent工作流|Agent工作流]]

## 9. 前沿发展

- **推理模型驱动Agent**：o1/o3等推理模型从根本上改变Agent规划方式，模型内部推理替代外部CoT提示
- **Computer Use Agent**：Agent直接操作GUI界面，实现通用自动化（Anthropic Computer Use、OpenAI Operator）
- **多Agent系统**：从单Agent走向多Agent协作，模拟人类团队分工
- **Agent操作系统**：将Agent作为操作系统级原语，统一调度与管理（AIOS等）
- **端侧Agent**：在手机/PC本地运行的Agent，隐私保护与低延迟（Apple Intelligence等）
- **Agent可观测性**：Agent行为的追踪、调试和评估工具成熟
- **Agent安全**：Agent权限控制、沙箱执行、行为审计等安全技术
- **长程任务Agent**：支持小时级甚至天级的长时间任务执行
- **Agent框架标准化**：框架间互操作性提升，MCP等标准减少框架锁定
- **声明式Agent**：从命令式编程走向声明式定义，Agent自动编排执行
- **可视化Agent构建**：低代码/无代码Agent构建平台（Dify、Coze等）

## 10. 常见问题

**Q: Chain 和 Agent 有什么区别？**
Chain 是预定义的固定调用序列（步骤和顺序在编写时确定）；Agent 由 LLM 在运行时动态决定下一步调用哪个工具、是否结束，具有更强的自适应性但也更难预测和调试。

**Q: 为什么 LangGraph 逐渐取代 LangChain 的 Agent 模块？**
LangChain 早期的 Agent 抽象（AgentExecutor）将控制流隐藏在框架内部，难以调试和定制中间步骤；LangGraph 用显式状态图暴露每个节点和边，开发者可以精确控制循环、分支与人工介入点。

**Q: 单 Agent 能否通过增加工具数量无限扩展能力？**
不能。当工具数量增多（通常 >15-20 个）时，LLM 在工具选择阶段的准确率会下降，此时应考虑拆分为多 Agent 分工，或引入工具检索/分层路由机制。

## 相关知识

- [[../01_智能体理论基础/00_智能体理论基础_综述|智能体理论基础]]：Agent 的理性主体、BDI 模型等理论前提
- [[../03_规划与推理/00_规划与推理_综述|规划与推理]]：ReAct、Plan-and-Execute 等规划范式的理论来源
- [[01_工具调用与MCP/00_工具调用与MCP_综述|工具调用与MCP]]：Agent 行动能力的工程实现
- [[../05_记忆与持续学习/00_记忆与持续学习_综述|记忆与持续学习]]：Agent 记忆系统的完整体系
- [[../07_Multi-Agent系统/00_Multi-Agent系统_综述|Multi-Agent系统]]：多智能体协作的深入理论
- [[02_工作流与编排/00_Agent工作流|Agent工作流]]：Agent 任务执行流程的工程设计

## References

- Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv:2210.03629.
- Shinn, N. et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*. arXiv:2303.11366.
- Park, J. S. et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior*. arXiv:2304.03442.
- Packer, C. et al. (2023). *MemGPT: Towards LLMs as Operating Systems*. arXiv:2310.08560.
- [LangChain 官方文档](https://python.langchain.com/)
- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [AutoGen 官方文档](https://microsoft.github.io/autogen/)
- [CrewAI 官方文档](https://docs.crewai.com/)
