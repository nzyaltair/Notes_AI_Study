# LLM 适配器与流式协议

## 一句话理解

LLM 适配器把不同提供方协议归一为 DSH 的消息和 `StreamChunk` 流；Consumer 应依赖规范化终止事件，而非猜测异常来自提供方还是中间件。

---

## 内容块与消息类型

### ContentBlockMap（可合并扩展）

一段对话由 `Message` 组成；一条消息是一个类型化内容块的数组：

| 类型键 | 含义 |
|---|---|
| `text` | 可见文本（`TextBlock { text }`） |
| `reasoning` | 推理过程（`ReasoningBlock`，区别于可见文本） |
| `image` | 图片附件（`ImageBlock`，持久化附件引用） |
| `tool-call` | 工具调用请求（`ToolCallBlock { id: CallId, name, arguments: string }`，**始终是原始 JSON 字符串**） |
| `tool-result` | 工具调用结果（`ToolResultBlock { toolCallId, content: ContentBlock[], isError? }`） |

> 新增模态只有适配器、UI、压缩和持久回放路径全部支持时，才可纳入可合并扩展的 map。

### Message（不可变）

```ts
interface Message {
  id: MessageId          // 跨表示边界保持稳定的唯一身份
  role: 'system' | 'user' | 'assistant'
  content: ContentBlock[]
  source: MessageSource  // 可合并扩展的来源和类型
}
```

**AssistantProvenance**（助手消息来源）：

| 字段 | 说明 |
|---|---|
| `provider` | 产生该消息的提供方路由 |
| `model` | 提供方的模型 id |
| `replayState?` | 适配器私有的无损 JSON 回放状态（只在历史提供方与目标提供方同属同一适配器实例时由 `LlmRuntime` 传递） |

### ContextForm（消息来源呈现形式）

`kind` 回答"由谁产生"，`form` 回答"这是什么类型的信息"——两轴独立：

| form | 含义 |
|---|---|
| `'instructions'` | 工作区文件中的指令（期望模型遵循） |
| `'catalog'` | 本次会话中可用项目的目录（随变化重新发布） |
| `'snapshot'` | 当前状态（同一生产者的更新快照取代旧快照） |
| `'notice'` | 刚刚发生的一次性说明（不取代任何内容） |
| `'relay'` | 另一个 agent 转发给此 agent 的消息 |
| `'recall'` | 从另一会话日志中提取的内容 |

> `form` 是**语义**，不是视觉样式；颜色、图标、折叠默认值是消费方的事，不得进入此联合。

---

## StreamChunk：原始流协议

```ts
type StreamChunk =
  | { type: 'block-start'; index: number; blockType: ContentBlockType }
  | { type: 'text-delta'; index: number; text: string }
  | { type: 'reasoning-delta'; index: number; text: string }
  | { type: 'tool-call-delta'; index: number; id: CallId; name?: string; argumentsDelta: string }
  | { type: 'block-end'; index: number; block: ContentBlock }
  | { type: 'usage'; usage: TokenUsage }
  | { type: 'finish'; reason: FinishReason; replayState?: unknown }
```

**设计要点**：
- `index` 将每个 delta 关联到其所属块，支持多工具调用交错；
- `block-end` 携带完整组装好的 `ContentBlock`，消费方无需自行重新组装 delta；
- 这是**封闭的**可辨识联合：`switch (chunk.type)` 末尾有 `assertNever`，新增变体会在所有消费方处触发编译错误；
- `usage` 在 `finish` 之前发出，`finish` 之后不再有任何分片。

---

## TokenUsage：token 记账

各字段**互不重叠**：

| 字段 | 含义 |
|---|---|
| `inputTokens` | 未缓存输入 |
| `outputTokens` | 输出（含 `reasoningTokens`） |
| `cacheReadTokens?` | 缓存命中 |
| `cacheWriteTokens?` | 缓存写入 |
| `reasoningTokens?` | 推理 token（信息性，已包含在 `outputTokens` 中，**不得重复相加**） |

> 计费输入 = `inputTokens` + `cacheReadTokens` + `cacheWriteTokens`。提供方把缓存命中折入单一提示词总数时（如 DeepSeek 的 `prompt_tokens`），适配器会将其扣除。

---

## 失败与取消

### LlmFailure（可序列化的失败事实）

```ts
interface LlmFailure {
  message: string               // 人类可读说明
  code: string                  // 稳定的提供方无关路由 code
  status?: number               // HTTP 状态码
  providerRetryAfterMs?: number // 提供方请求的延迟（正数有效值）
  requestId?: ProviderRequestId // 诊断用不透明字符串
}
```

### 两条错误路径

| 路径 | 场景 |
|---|---|
| `stream()` 抛出异常 | 传输/协议错误 |
| `finish { kind: 'error'\|'aborted', failure }` | 无法在流中途抛异常时，提供方带内错误 |

`LlmRuntime.stream()` 将两条路径**规范化**为终止型 finish 后暴露给消费方。

### 重要错误规范

| Code | 含义 |
|---|---|
| `CONTEXT_WINDOW_EXCEEDED` | 上下文溢出（唯一规范 code，消费方按 code 路由，不依赖提供方文本） |
| `EMPTY_RESPONSE` | 空 completion（可重试错误，不是静默成功） |
| `TIMEOUT` | 流空闲超时（watchdog 在 `next()` 未完成时启动） |
| `ABORTED` | 调用方取消（更早发生时保留，不被 TIMEOUT 覆盖） |

---

## 适配器约定（实现者必须遵守）

1. **`usage` 在 `finish` 之前**：将两者都推迟到提供方流结束标记，防止顺序违反；
2. **工具调用 arguments 全程保持原始 JSON 字符串**：delta 用 `argumentsDelta` 流式传输；提供方返回已解析对象时，适配器在 `block-end` 时重新序列化；
3. **两条错误路径共用 `LlmFailure` 类型**；
4. **一次适配器调用就是一次提供方尝试**：适配器禁用库重试，重试由 agent 层在持久轮次中处理；
5. **提供方停顿在传输层受时限约束**：`streamIdleTimeoutMs` 默认 5 分钟；
6. **每个提供方 HTTP 请求都携带 `User-Agent` 应用归属头**（`AppIdentity`，不含 secret 或逐用户标识）；
7. **回放状态归适配器所有**：成功 `finish.replayState` 只在历史提供方与目标提供方同属同一适配器实例时传递。

### ResolvedRetryPolicy

提供方配置在路由注册前解析为不可变可辨识联合（`mode: 'normal'` 或 `mode: 'always'`）；选定注册后，策略固定不变，后续释放或替换路由不会改变进行中失败的恢复策略。

---

## 开发要点

实现新适配器时，先定义：
- 模型路由、认证方式；
- 输入模态（图片、工具调用等）；
- 流式映射（`StreamChunk` 分片）；
- 用量字段对齐（`TokenUsage` 各字段去重）；
- 取消和错误分类（两条路径归一）；

再用真实请求与回放快照验证，不只 mock HTTP 响应。

---

## 相关知识

- [[01_入门与运行/03_模型提供方与凭据配置]]
- [[03_Agent运行时与会话/01_Agent_Turn与Step生命周期]]
- [[06_安全可靠性与工程化/03_测试诊断与开发工作流]]

## References

- `D:\_Projects\deepseek-harness\docs\cookbook\adding-an-llm-adapter.zh.md`
- `D:\_Projects\deepseek-harness\docs\subsystems\llm-streaming.zh.md`
