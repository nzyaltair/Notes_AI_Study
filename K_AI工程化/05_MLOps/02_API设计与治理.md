# API 设计与治理

## 1. 概述

API 设计与治理是 LLM-as-a-Service 平台的核心工程环节，涵盖接口协议选型、认证授权、限流熔断、版本管理和可观测性。良好的 API 治理直接决定平台的可维护性、安全性和规模化能力。

API 治理解决的核心问题：
- **接口标准化**：统一请求/响应格式，降低客户端集成成本
- **访问控制**：认证授权，确保只有合法用户能访问服务
- **流量保护**：限流熔断，防止流量洪峰击垮服务
- **版本演进**：在不中断服务的前提下迭代 API
- **可观测性**：分布式追踪和结构化日志，支持故障排查

OpenAI 在 2023 年发布 Chat Completions API 后，其请求/响应格式逐渐成为行业标准，vLLM、TGI 等推理引擎均提供兼容接口。

## 2. 发展历史

| 年代 | 里程碑 | 意义 |
|:---|:---|:---|
| 2000 | REST 架构风格 | Roy Fielding 提出 RESTful API 设计理念 |
| 2015 | gRPC | Google 推出高性能 RPC 框架 |
| 2015 | OpenAPI/Swagger | API 描述标准化 |
| 2020 | GraphQL 普及 | 灵活的数据查询模式 |
| 2023.03 | OpenAI Chat Completions | LLM API 事实标准确立 |
| 2023.06 | OpenAI Function Calling | 工具调用 API 标准化 |
| 2023 | SSE 流式标准 | LLM 流式响应采用 SSE 作为标准协议 |
| 2024 | 多模态 API | 统一接口支持文本、图像、音频 |
| 2024 | Agent 协作协议 | MCP (Model Context Protocol) 等 Agent 通信标准 |
| 2025 | API 治理自动化 | 策略即代码、自动限流调优 |

## 3. 核心概念

### 3.1 协议选型

| 协议 | 序列化 | 流式支持 | 适用场景 |
|:---|:---|:---|:---|
| RESTful | JSON | SSE 单向流 | 公开 API，浏览器兼容 |
| gRPC | Protobuf | 双向流 | 内部服务，高性能通信 |
| GraphQL | JSON | 无 | 灵活数据查询 |
| WebSocket | JSON/二进制 | 双向实时 | 实时交互场景 |

### 3.2 OpenAI 兼容格式

核心字段：`model`（模型标识）、`messages`（对话历史）、`temperature`（采样温度）、`max_tokens`（生成上限）、`stream`（流式开关）、`tools`（函数调用定义）。

### 3.3 认证与访问控制

- **API Key**：简单适用，适合低安全要求场景，需配合 HTTPS 和定期轮换
- **OAuth2.0**：标准化协议，支持细粒度权限委托，适合第三方应用集成
- **JWT**：无状态令牌，自包含用户信息，适合微服务间通信和高并发场景
- **RBAC**：基于角色的访问控制，遵循最小权限原则

### 3.4 限流与熔断

- **限流**：控制请求速率，保护服务不被过载
- **熔断**：当下游服务故障时快速失败，防止级联崩溃
- **降级**：在服务压力过大时返回简化结果或默认值

## 4. 技术原理

### 4.1 流式响应

基于 SSE 格式，每个 chunk 携带增量 Token：

```
data: {"choices":[{"delta":{"content":"API"}}]}
data: {"choices":[{"delta":{"content":" design"}}]}
data: [DONE]
```

### 4.2 令牌桶限流

以固定速率向桶中添加令牌，请求消耗令牌，允许突发流量。桶容量 $C$，填充速率 $r$：

$$n(t) = \min(C, n(t_0) + r \cdot (t - t_0))$$

时间 $\Delta t$ 内可处理的最大请求数为 $C + r \cdot \Delta t$。

### 4.3 熔断器状态机

三个状态闭环：

- **闭合（Closed）**：正常请求，记录失败率
- **打开（Open）**：失败率达阈值 $\theta$ 时拒绝所有请求
- **半开（Half-Open）**：超时后尝试少量请求验证恢复

### 4.4 版本管理策略

| 策略 | 方式 | 优点 | 缺点 |
|:---|:---|:---|:---|
| URI 版本控制 | `/v1/chat/completions` | 直观 | URL 膨胀 |
| Header 版本控制 | `Accept: version=2.0` | URL 不变 | 调试复杂 |
| 查询参数 | `?version=2` | 灵活 | 易被忽略 |

向后兼容原则：添加可选字段为兼容变更；删除端点或更改字段类型为不兼容变更。

### 4.5 分布式追踪

OpenTelemetry 提供统一的 Trace/Span 模型。一次 LLM 请求的典型追踪链：

```
Trace: api-request-123
├── Span: http-server-receive (0.1ms)
├── Span: authentication (0.5ms)
├── Span: model-router (0.3ms)
├── Span: vllm-generate (500ms)
│   ├── Span: tokenizer (10ms)
│   ├── Span: model-inference (480ms)
│   └── Span: detokenizer (10ms)
└── Span: http-server-send (0.2ms)
```

## 5. 关键方法与模型

### 5.1 滑动窗口限流

维护时间窗口内的请求计数，比固定窗口更精确。在窗口 $W$ 内维护请求计数 $C_w$，当 $C_w > C_{max}$ 时拒绝请求。

### 5.2 漏桶算法

以恒定速率处理请求，突发请求排队等待。与令牌桶的区别：漏桶不允许突发，输出速率恒定。

### 5.3 Function Calling

OpenAI 标准化的工具调用协议，允许 LLM 通过结构化输出请求执行外部函数：

```json
{
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
    }
  }]
}
```

## 6. 优势与局限

### 6.1 优势

- **标准化**：OpenAI 兼容接口降低客户端集成成本
- **安全性**：多层级认证授权保障访问安全
- **稳定性**：限流熔断保护服务不被过载击垮
- **可演进**：版本管理支持 API 平滑升级

### 6.2 局限

- **SSE 限制**：单向流，不支持客户端中途取消生成的标准方式
- **Token 计费精度**：不同分词器导致 Token 计数不一致
- **多模态接口标准化不足**：各厂商多模态 API 格式差异大
- **Agent 协议碎片化**：MCP、Function Calling 等协议尚未统一

## 7. 应用场景

| 场景 | 特点 | 关键设计 |
|:---|:---|:---|
| LLM API 平台 | 公共服务 | API Key 认证 + 令牌桶限流 + SSE 流式 |
| 企业内部服务 | 多团队共享 | OAuth2 + RBAC + 配额管理 |
| 多模型路由 | 异构模型 | 按参数动态路由 + A/B 测试 |
| Agent 平台 | 工具调用 | Function Calling + 流式 + 长连接 |

## 8. 与其他技术关系

- [[01_模型服务化]] 提供 API 背后的推理引擎实现
- [[05_监控与日志]] 依赖 API 层的结构化日志和指标暴露
- [[06_版本管理与回滚]] 通过 API 版本控制实现灰度发布
- [[07_成本控制与配额]] 通过 API 限流和配额管理实现成本治理
- [[08_数据隐私与合规]] 要求 API 层实施数据脱敏和审计日志

## 9. 前沿发展

- **MCP (Model Context Protocol)**：Anthropic 提出的模型上下文协议，标准化 LLM 与外部工具的交互
- **结构化输出**：JSON Schema 约束 LLM 输出格式，提升 API 可靠性
- **API 即代码**：声明式 API 配置，自动生成文档和 SDK
- **智能限流**：基于 ML 的自适应限流，根据流量模式动态调整阈值
- **多模态统一接口**：文本、图像、音频的统一 API 设计
- **Agent 协作协议**：多 Agent 间的标准化通信协议

返回 [[00_模型部署与工程化]]
