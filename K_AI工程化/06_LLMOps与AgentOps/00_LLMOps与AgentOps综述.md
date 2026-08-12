---
tags:
  - LLMOps
  - AgentOps
  - MLOps
  - Prompt管理
  - 可观测性
  - AI工程
created: 2026-08-10
updated: 2026-08-10
---

# LLMOps 与 AgentOps 综述

## 一句话理解

LLMOps 在 MLOps 基础上管理 Prompt、上下文、模型网关、Trace、token 成本和 LLM 评测；AgentOps 进一步管理任务状态、工具调用、记忆、长运行工作流和人类审批——核心目标是把非确定性模型调用转化为可追踪、可测试、可回放、可控制的生产行为。

## 1. 领域定义

### LLMOps 的特殊性

传统 MLOps 管理确定性机器学习模型；LLM 带来了新的运维挑战：

- **Prompt 是生产配置**：不是代码，但影响输出，需版本化和审批
- **非确定性输出**：同一输入不同时刻输出不同，需新的评测方法
- **Token 成本**：计费单位是 Token，需要精细追踪和控制
- **上下文依赖**：输出质量高度依赖上下文构造，不只是模型本身
- **长链路**：RAG、Tool Use、Multi-Turn 构成复杂链路，观测复杂

### AgentOps 的扩展

Agent 在 LLM 基础上增加：

- 长时间运行任务（分钟到小时）
- 有状态工具调用（文件、数据库、API、代码执行）
- 复杂的错误恢复和重试逻辑
- 人工审批节点
- 多 Agent 通信

## 2. 发展历史

| 时间 | 里程碑 | 意义 |
|:---|:---|:---|
| 2015-2022 | MLOps 成熟 | MLflow、Kubeflow、DVC 建立经典 MLOps 实践 |
| 2022 | ChatGPT API 发布 | Prompt 工程成为独立实践，Prompt 版本管理需求出现 |
| 2023.03 | LangSmith 发布 | 首个专门的 LLM 调用链 Trace 平台 |
| 2023.06 | Phoenix / Arize AI | LLM 推理质量监控与漂移检测 |
| 2023.08 | OpenAI Function Calling 普及 | 工具调用规范化，AgentOps 需求爆发 |
| 2023 | OpenTelemetry for LLM | 可观测性标准开始向 LLM 延伸（OTel Semantic Conventions） |
| 2024 | LLM Gateway 标准化 | Portkey、Kong AI Gateway 等统一 LLM 入口 |
| 2024 | AgentOps 平台兴起 | AgentOps.ai、Langfuse、Helicone 等专项平台 |
| 2024 | AI Evals 标准化 | LMQL、braintrust、continuous evals 框架 |

## 3. LLMOps 核心能力

### 3.1 Prompt 生命周期管理

Prompt 是生产配置，不是散落在业务代码中的字符串：

- **版本化**：模板、变量 schema、系统指令、示例、模型参数共同版本化
- **评测绑定**：每个 Prompt 版本绑定评测集和评测结果
- **灰度发布**：Prompt 变更经由评测、影子流量、灰度比例上线
- **回滚能力**：质量下降时立即回滚到上一版本

### 3.2 RAG 与 Embedding 运行管理

- 向量索引版本化（文档变更、嵌入模型变更触发重建）
- 检索质量监控（召回率、NDCG、无答案率）
- Embedding 漂移检测（用户查询分布变化）
- 知识库更新策略（全量重建 vs 增量更新）

### 3.3 模型网关与成本治理

```text
客户端 → 模型网关 → LLM Provider (OpenAI/Anthropic/本地)
              ↓
        认证/授权/限流/路由/缓存/审计/成本归因
```

- **路由**：按任务类型、成本预算、延迟要求路由到不同模型
- **成本控制**：Token 预算、每用户配额、成本归因到团队/项目
- **语义缓存**：相似查询命中缓存，降低 API 调用成本
- **熔断限流**：防止下游 LLM 故障扩散

### 3.4 LLM Trace 与可观测性

Trace 将一次用户任务关联为：请求 → 检索 → Prompt 渲染 → 模型调用 → 工具调用 → 输出 → 评测 → 反馈的因果链。

每个 Span 记录：时延、Token 数、模型版本、结果与错误，但不应无边界记录敏感输入。

工具链：
- **LangSmith**：LangChain 生态，Trace + 数据集 + Playground
- **Langfuse**：开源，支持任何框架，有完整 Prompt 管理
- **Phoenix (Arize)**：专注 LLM 评测与漂移检测
- **OpenTelemetry**：可观测性标准，正在扩展 LLM 语义

### 3.5 LLM 评测与发布门禁

LLM 发布不能只验证 API 成功率，需覆盖：

- 能力评测：事实性、指令遵循、格式、推理
- 安全评测：拒答、越狱、有害内容、偏见
- 场景评测：业务特定任务成功率
- 成本评测：输出 Token 分布与成本估算

评测集需版本化并隔离训练数据；评分使用规则、执行器、LLM Judge 组合。通过影子流量、灰度发布验证真实质量。

## 4. AgentOps 核心能力

### 4.1 Agent 部署与运行

- **任务持久化**：长时间任务状态存储（Redis、数据库），支持断点续跑
- **超时与停止条件**：显式最大步数、最大时间、最大成本限制
- **资源隔离**：代码执行在沙箱中运行，工具调用有权限最小化约束
- **水平扩展**：无状态 Agent 实例 + 外部状态存储

### 4.2 工具调用与记忆管理

- **工具注册**：Schema 版本化，工具权限声明
- **调用审计**：记录工具调用参数、结果和副作用
- **记忆层次**：对话记忆（短期）/ 知识库（长期）/ 工作记忆（任务内）
- **记忆压缩**：长对话记忆摘要，防止上下文溢出

### 4.3 Agent 评测与治理

- **轨迹评测**：评估任务完成路径的效率和安全性，不只看最终结果
- **工具调用质量**：参数正确性、副作用控制、不必要调用率
- **人工审批集成**：高风险操作需要人类确认
- **成本追踪**：Agent 任务的总 Token 消耗、工具调用次数和时间

## 5. 工具生态

| 类别 | 工具 | 特点 |
|:---|:---|:---|
| LLM 网关 | Portkey、Kong AI Gateway | 多 LLM 路由、成本控制、审计 |
| Trace | LangSmith、Langfuse、Helicone | 调用链可观测 |
| 评测 | Braintrust、PromptFoo | 持续评测与 CI 集成 |
| Prompt 管理 | PromptLayer、LangSmith Prompts | 版本化和 A/B |
| Agent 框架 | LangGraph、AutoGen、CrewAI | Agent 编排 |
| 向量库 | Qdrant、Milvus、pgvector | RAG 知识存储 |

## 6. 与 MLOps 的边界

| 维度 | MLOps | LLMOps |
|:---|:---|:---|
| 配置管理 | 超参数、特征 schema | Prompt 模板、系统指令 |
| 评测 | 单一数值指标 (AUC/F1) | 多维生成质量（人工+模型评分） |
| 版本单元 | 模型权重 + 代码 | 模型 + Prompt + RAG 配置 |
| 成本单位 | GPU/TPU 时数 | Token 数 |
| 调试 | 特征重要性、混淆矩阵 | Trace 链路回放 |

## 7. 学习路径

1. 前置：[[../05_MLOps/00_MLOps综述]]（MLOps 基础）
2. LLMOps：[[01_Prompt生命周期管理]] → [[04_LLM Trace与可观测性]] → [[05_LLM评测与发布]]
3. 成本：[[02_RAG与Embedding运行管理]] → [[03_模型网关与成本治理]]
4. AgentOps：[[06_Agent部署与运行]] → [[07_工具调用与记忆管理]] → [[08_Agent评测与治理]]

## 8. 前沿发展

- **Evals as Code**：评测集版本化、持续集成，每次 PR 自动运行评测
- **LLM 自动评判（LLM-as-Judge）**：用强模型评估弱模型输出，扩展评测覆盖
- **在线 Prompt 优化**：A/B 测试 Prompt 变体，数据驱动地优化
- **Agent 轨迹数据集**：从生产 Agent Trace 中提取高质量轨迹数据用于微调
- **多模态 LLMOps**：图像/视频输出的质量监控与成本归因
- **Privacy-Preserving Tracing**：在合规约束下保留可调试性，差分隐私 Trace

## 9. 相关知识

- [[../05_MLOps/00_MLOps综述]]（前置：MLOps 基础实践）
- [[../04_AI系统架构/00_AI系统架构_综述]]（前置：系统架构设计）
- [[../07_AI评测体系/00_AI评测体系_综述]]（同层：离线评测体系）
- [[../09_AI可靠性工程/00_AI可靠性工程综述]]（同层：可靠性保障）

## References

- Zaharia et al., *The Shift from Models to Compound AI Systems* (BAIR Blog, 2024)
- OpenAI, *Function Calling Documentation* (2023)
- Langfuse, *LLM Engineering Platform Documentation* (2024)
- Shankar et al., *Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs* (2024)
- Wu et al., *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation* (2023)
- MLflow, *MLflow LLM Evaluation Documentation* (2024)
