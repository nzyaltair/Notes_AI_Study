---
tags:
  - LLMOps
  - 可观测性
  - Trace
  - 监控
  - 调试
created: 2026-08-10
updated: 2026-08-10
---

# LLM Trace 与可观测性

## 一句话理解

Trace 将一次用户任务关联为请求 → 检索 → Prompt 渲染 → 模型调用 → 工具调用 → 输出 → 评测 → 反馈的因果链，每个 Span 记录时延、Token、版本、结果与错误，用于调试、成本分析、离线回放和质量改进。

## 1. 为什么 LLM 需要专门的可观测性

传统应用的可观测性关注请求延迟、错误率、资源使用。LLM 应用在此基础上增加了：

- **非确定性输出**：同一输入不同输出，需记录完整链路才能调试
- **多步链路**：RAG、Tool Use、Multi-Turn 构成复杂调用链
- **Token 成本**：每次调用都有明确成本，需精确计量
- **质量维度**：不仅要监控"是否出错"，还要监控"输出质量"
- **上下文依赖**：输出质量高度依赖上下文构造

核心目标：**把黑箱的 LLM 调用变成可追踪、可回放、可分析的透明链路**。

## 2. Trace 模型

### 2.1 Trace 与 Span

```text
Trace (用户请求: "帮我总结这份文档")
├── Span: 文档解析 (120ms)
├── Span: RAG 检索 (85ms)
│   ├── Span: Embedding (20ms)
│   └── Span: 向量搜索 (45ms)
├── Span: Prompt 渲染 (5ms)
├── Span: LLM 调用 (1.2s)
│   ├── input_tokens: 3500
│   ├── output_tokens: 800
│   ├── model: gpt-4o-2024-08-06
│   └── prompt_version: v1.2.0
├── Span: 后处理 (30ms)
└── Span: 评测 (200ms)
    └── quality_score: 0.87
```

### 2.2 Span 核心字段

| 字段 | 说明 | 示例 |
|:---|:---|:---|
| trace_id | 全局唯一请求 ID | `tr_abc123` |
| span_id | 当前 Span ID | `sp_def456` |
| parent_span_id | 父 Span ID | `sp_abc123` |
| span_type | Span 类型 | `llm` / `retrieval` / `tool` |
| name | 操作名称 | `chat.completions` |
| start_time / end_time | 起止时间 | ISO 8601 |
| input | 输入内容（脱敏后） | messages, query |
| output | 输出内容（脱敏后） | response, retrieved_docs |
| tokens | Token 计量 | `{input: 3500, output: 800}` |
| model | 模型版本 | `gpt-4o-2024-08-06` |
| prompt_version | Prompt 版本 | `v1.2.0` |
| status | 状态 | `ok` / `error` |
| error | 错误信息 | `rate_limit_exceeded` |
| metadata | 自定义元数据 | `{user_id, session_id}` |

## 3. Span 类型

| 类型 | 记录内容 | 关注指标 |
|:---|:---|:---|
| LLM | 模型调用 | Token、延迟、成本、质量分 |
| Retrieval | 向量检索 | 召回数、相似度、检索延迟 |
| Tool | 工具调用 | 工具名、参数、结果、副作用 |
| Prompt | Prompt 渲染 | 模板版本、变量值 |
| Guardrail | 安全检查 | 过滤结果、触发规则 |
| Evaluation | 质量评测 | 评分、评分方法 |

## 4. 可观测性维度

### 4.1 性能监控

- **端到端延迟**：用户请求 → 响应完成的总时间
- **分阶段延迟**：检索、Prompt 渲染、模型调用、后处理各阶段耗时
- **TTFT (Time To First Token)**：首 Token 延迟（流式场景）
- **TPOT (Time Per Output Token)**：平均每个输出 Token 耗时

### 4.2 质量监控

- **LLM Judge 分数**：用强模型评估输出质量
- **用户反馈**：点赞/点踩、评分、投诉
- **拒答率**：模型拒绝回答的比例
- **幻觉率**：输出与检索内容不一致的比例
- **格式合规率**：输出格式符合要求的比例

### 4.3 成本监控

- **Token 消耗**：按模型、租户、Prompt 版本统计
- **成本分布**：哪个环节消耗最多 Token
- **缓存命中率**：语义缓存的命中比例
- **成本异常**：单次请求成本超阈值告警

### 4.4 安全监控

- **注入攻击检测**：Prompt 注入尝试
- **PII 泄漏**：输出中包含个人信息
- **有害内容**：输出触发安全分类器
- **越权调用**：低权限用户调用高权限模型

## 5. 隐私与合规

### 5.1 敏感数据处理

Trace 记录输入输出时必须处理敏感信息：

- **脱敏**：PII（姓名、邮箱、手机号）替换为占位符
- **采样**：高敏感场景仅记录 1% 请求
- **截断**：长输入只记录摘要
- **加密**：Trace 数据加密存储

### 5.2 数据保留策略

| 数据类型 | 保留期 | 说明 |
|:---|:---|:---|
| Trace 元数据 | 90 天 | 不含输入输出内容 |
| Trace 内容（脱敏） | 30 天 | 脱敏后的输入输出 |
| Trace 内容（原始） | 7 天 | 仅高安全环境 |
| 评测结果 | 永久 | 评分与指标 |

## 6. 离线回放与调试

### 6.1 回放能力

从 Trace 中提取完整链路，在离线环境重放：

- **变量复现**：使用相同输入、模型版本、Prompt 版本
- **参数调整**：修改 temperature 等参数对比输出
- **模型对比**：同一输入用不同模型生成，对比质量

### 6.2 数据集构建

从生产 Trace 中提取高质量数据：

- **正样本**：用户点赞的交互
- **负样本**：用户点踩或投诉的交互
- **边界 case**：触发安全规则但需人工判断的交互

这些数据集用于 Prompt 优化、模型微调和评测集扩充。

## 7. 工具链

| 工具 | 特点 | 适用场景 |
|:---|:---|:---|
| LangSmith | LangChain 生态，Trace + 数据集 + Playground | LangChain 用户 |
| Langfuse | 开源，支持任意框架，完整 Prompt 管理 | 多框架环境 |
| Phoenix (Arize) | 专注 LLM 评测与漂移检测 | 质量监控 |
| Helicone | 轻量代理，成本追踪 | 快速接入 |
| OpenTelemetry | 可观测性标准，LLM 语义扩展 | 统一可观测性 |

### OpenTelemetry LLM 语义

OpenTelemetry 正在标准化 LLM 可观测性语义：

- `gen_ai.system`：LLM 供应商
- `gen_ai.request.model`：模型 ID
- `gen_ai.usage.prompt_tokens`：输入 Token
- `gen_ai.usage.completion_tokens`：输出 Token
- `gen_ai.response.finish_reason`：结束原因

## 8. 常见问题

- **Trace 过大**：长对话和大量检索结果导致 Trace 膨胀，需截断和采样
- **敏感数据泄漏**：Trace 中记录了用户隐私，需脱敏
- **Trace 影响性能**：Trace 采集增加延迟，需异步上报
- **跨服务链路断裂**：微服务间 Trace 上下文未传递
- **质量监控滞后**：LLM Judge 评分延迟，无法实时告警

## 9. 相关知识

- [[00_LLMOps与AgentOps综述]]（上层：LLMOps 全景）
- [[01_Prompt生命周期管理]]（前置：Prompt 版本与 Trace 关联）
- [[03_模型网关与成本治理]]（前置：网关计量与 Trace 对齐）
- [[05_LLM评测与发布]]（下一步：评测与发布门禁）
- [[../05_MLOps/04_监控与日志]]（同层：传统 MLOps 监控）
- [[../09_AI可靠性工程/01_SLO与容错设计]]（同层：SLO 设计）

## References

- OpenTelemetry, *GenAI Semantic Conventions* (2024)
- LangSmith, *Tracing Documentation* (2024)
- Langfuse, *Open Source LLM Engineering Platform* (2024)
- Arize AI, *Phoenix LLM Observability* (2024)
- Schneider et al., *Large Language Model AIOps: Observability and Traceability* (2024)
