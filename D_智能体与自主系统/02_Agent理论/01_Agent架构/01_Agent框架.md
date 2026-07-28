# Agent框架

## 1. 概述

Agent框架是构建 `Agent系统` 的软件开发工具包，提供规划、工具调用、记忆管理、执行编排和反思机制等核心组件的标准化实现。Agent框架使开发者无需从零构建Agent基础设施，可以专注于业务逻辑设计。

从LangChain（2022）的模块化工具链，到AutoGen的多Agent对话，再到LangGraph的状态图工作流，Agent框架经历了从简单链式调用到复杂多Agent协作的演进。

## 2. 发展历史

| 时间 | 框架 | 意义 |
|------|------|------|
| 2022.10 | LangChain发布 | 首个系统化LLM应用框架 |
| 2023.03 | AutoGPT | 自主Agent概念验证，引发关注 |
| 2023.05 | LlamaIndex Agent | RAG增强Agent框架 |
| 2023.08 | AutoGen v1（微软） | 多Agent对话框架 |
| 2023.10 | CrewAI | 角色化多Agent协作 |
| 2023.10 | DSPy Agent | 声明式Agent编程 |
| 2024.01 | LangGraph | 状态图工作流，Agent工程化 |
| 2024.03 | OpenAI Assistants API | 官方托管Agent服务 |
| 2024.10 | OpenAI Swarm | 轻量级多Agent编排 |
| 2025 | OpenAI Agents SDK | 官方Agent框架正式发布 |
| 2025 | Agent框架标准化 | 框架间互操作性提升 |

## 3. 核心概念

### Agent框架的核心模块

| 模块 | 功能 | 关键技术 |
|------|------|---------|
| 规划器（Planner） | 任务分解为子任务序列 | `思维链` 、思维树、规划验证 |
| 工具调用（Tool Use） | 调用外部API和工具 | Function Calling、动态工具注册 |
| 记忆系统（Memory） | 存储和检索历史经验 | `向量数据库` 、记忆摘要、压缩 |
| 执行器（Executor） | 执行子任务，调用工具 | 同步/异步/并行执行、状态跟踪 |
| 反思机制（Reflection） | 反思行为，改进决策 | 自评估、错误检测、经验提取 |

### 关键术语

- **Chain**：线性的LLM调用序列，前一步输出作为后一步输入
- **Graph**：有状态的图结构工作流，支持循环和条件分支
- **Agent**：具有自主决策能力的执行单元
- **Tool**：Agent可调用的外部函数/API
- **State**：工作流中的共享状态对象
- **Checkpoint**：工作流执行状态的持久化快照

## 4. 技术原理

### 规划范式对比

| 范式 | 核心思想 | 流程 | 适用场景 |
|------|---------|------|---------|
| ReAct | 推理与行动交替 | Thought→Action→Observation循环 | 通用Agent，工具调用频繁 |
| Plan-and-Execute | 先规划再执行 | 生成计划→依次执行→监控调整 | 结构化强的复杂任务 |
| Reflexion | 失败后反思重试 | 执行→评估→反思→重试 | 调试类，需要自我修正 |
| Multi-Agent Debate | 多Agent辩论 | 提出观点→反驳→综合 | 需要多角度思考的问题 |
| LATS | 树搜索+反思 | 生成→评估→回溯→择优 | 复杂决策任务 |

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

### LangGraph状态图工作流

LangGraph将Agent工作流建模为有状态图，支持循环、条件分支和并行执行：

```python
from langgraph.graph import StateGraph

# 定义状态
class AgentState(TypedDict):
    messages: list
    tool_results: list
    current_step: str

# 构建状态图
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

### CrewAI角色化协作

```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role='研究员',
    goal='收集和分析相关信息',
    tools=[search_tool, rag_tool],
    llm=llm
)

writer = Agent(
    role='撰稿人',
    goal='基于研究结果撰写高质量报告',
    llm=llm
)

research_task = Task(description='研究AI最新进展', agent=researcher)
write_task = Task(description='撰写研究报告', agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])
result = crew.kickoff()
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

### 框架核心能力对比

| 能力 | LangChain | LangGraph | AutoGen | CrewAI |
|------|-----------|-----------|---------|--------|
| 工作流类型 | Chain | StateGraph | 对话 | 角色任务 |
| 循环支持 | 弱 | 强 | 中 | 中 |
| 状态持久化 | 弱 | 强 | 中 | 中 |
| 人机协作 | 弱 | 强 | 强 | 中 |
| 流式输出 | 强 | 强 | 中 | 中 |
| 可观测性 | LangSmith | LangSmith | 弱 | 弱 |

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

- 降低Agent开发门槛，提供标准化组件
- 框架处理底层细节（工具注册、状态管理、错误处理）
- 生态丰富，可复用社区组件
- 支持快速原型和迭代

### 局限

- **学习曲线**：不同框架设计理念差异大，迁移成本高
- **性能开销**：框架抽象层增加额外开销
- **灵活性限制**：框架约束可能限制自定义需求
- **框架锁定**：深度依赖某框架后迁移困难
- **调试困难**：框架封装增加调试复杂度
- **版本不稳定**：框架快速迭代，API频繁变更

## 7. 应用场景

- **编程助手**：Cursor、Claude Code、Devin
- **数据分析Agent**：自主查询、分析、可视化
- **内容生产流水线**：研究→写作→审核
- **客服自动化**：分类→检索→生成→回复
- **多Agent协作研究**：文献检索、分析、综述
- **自动化办公**：邮件处理、日程管理、报告生成

## 8. 与其他技术关系

- Agent框架以 `大语言模型` 为推理核心
- `工具调用与MCP协议` 提供工具集成能力
- `RAG` 作为Agent的知识来源
- `提示工程` 定义Agent行为
- `结构化输出` 确保工具调用参数可解析
- `Agent记忆` 框架内置的记忆管理模块
- `Agent工作流` 基于框架构建的具体工作流

## 9. 前沿发展

- **Agent框架标准化**：框架间互操作性提升，MCP等标准减少框架锁定
- **声明式Agent**：从命令式编程走向声明式定义，Agent自动编排执行
- **可视化Agent构建**：低代码/无代码Agent构建平台（Dify、Coze等）
- **Agent可观测性**：框架内置追踪、调试和评估工具
- **多Agent编排标准化**：Swarm等轻量级编排方案标准化多Agent协作
- **Agent与推理模型融合**：框架适配o1/o3等推理模型，利用其强规划能力
- **边缘Agent**：轻量级框架支持在设备端运行Agent
- **Agent即服务**：Agent框架向云服务演进，提供托管Agent运行环境

## 相关知识

- 前置：[[00_Agent系统|Agent系统]]
- 平级：[[02_Agent记忆|Agent记忆]]、[[03_Agent工作流|Agent工作流]]
- 延伸：[[../06_自动化工作流|自动化工作流]]
