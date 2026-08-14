# 会话投影与 Conversation Node

## 一句话理解

Conversation Node 是客户端对会话事件的可回放呈现；新增 UI 功能应先判定它只是已有事件的纯投影，还是需要先新增持久事实。

---

## 三层模型

1. **持久事实**：以 JSON 可序列化 `SessionEventMap` 事件追加；模型需要看见时还必须参与 surface 历史。
2. **投影**：纯函数地从事件流折叠节点状态，保证刷新、恢复和 fork 一致。
3. **呈现**：客户端通过 `ConversationNodeDefinition` 和 keyed renderer 显示投影。

UI 格式、随机值和实时 I/O 不能作为无法回放的事实来源。仅改展示时不应新增事件 schema；只有领域事实改变才扩展事件、投影和持久化兼容性。

---

## 会话投影 seam（session-projection）

### 三节点结构

| 角色 | 包/服务 | 职责 |
|---|---|---|
| Service Definition + 注册表 | `dsh-session-projection`（`ctx.sessionProjections`） | 订阅 `session/event`，驱动所有 unit 的 fold；负责缓存、变更流和快照 |
| 领域贡献方 | 各领域包注册 `ProjectionDefinition` | 纯计算单元（三个同步函数），不持有任何订阅 |
| 载体（Carrier） | `dsh-host-apiproxy`（历史尾页 + `session/projection` 推送帧） | 消费快照与变更流，向 Client 发送已物化的全量值 |

### ProjectionDefinition（投影定义）

每个领域注册一个对应其 `SessionProjectionMap` key 的纯计算单元：

| 字段 | 类型 | 说明 |
|---|---|---|
| `key` | `keyof SessionProjectionMap` | 该 unit 拥有的投影键 |
| `schema` | `ZodType` | 在发出前校验 wire payload（`view` 输出） |
| `init()` | `() => S` | 返回空日志时的初始状态 |
| `apply(state, event)` | `(S, SessionEvent) => S` | 纯转换：前一状态 + 一个已提交事件 → 下一状态；**无关事件必须返回同一个引用**（`Object.is` 不变则下游零工作） |
| `view(state)` | `(S) => SessionProjectionMap[K]` | 状态 → wire payload（全量值，只读侧） |
| `stateVersion` | `number` | 持久缓存失效版本号；语义或序列化字段变化时 bump，防止旧缓存行被前向应用成垃圾 |

**全量值事件规则（承重结构）**：携带状态的日志事件携带的是**变更后的完整状态**，绝不是裸增量——这让每次状态转移始终廉价，也让每个被供给的值自描述（对消费方即 last-wins）。

### ProjectionSnapshot（快照）

```ts
interface ProjectionSnapshot {
  asOfSeq: number                          // 所有值所反映的最后事件 seq；空日志为 -1
  values: Partial<SessionProjectionMap>   // 每个注册 key 的当前全量值
}
```

快照提供**一致性读取切面**（consistent cut）——所有值共享同一个 seq 水位线，携带者（载体）无需处理跨字段的时序问题。

### 变更流监听器

```ts
interface ProjectionChangeFeedEntry {
  key: keyof SessionProjectionMap
  value: SessionProjectionMap[keyof SessionProjectionMap]
  seq: number   // 触发该变更的事件 seq（该 unit 的水位线）
}
```

注册表仅订阅一次 `session/event`，各 unit 不持有任何订阅；注册表发现状态引用发生变化时才触发变更流通知。

---

## Conversation Node 在 Web Client 中

### 注册方式

```ts
// 在 Host 侧或 Client 侧注册 ConversationNodeDefinition
ctx.conversationNodes.register({
  key: 'my-node',
  // 纯函数：从 SessionEvent 流折叠节点状态
  fold: (state, event) => { ... },
  // 纯函数：从状态渲染 JSX
  render: (state) => <MyNodeView ... />
})
```

### 三条设计规则

1. **呈现只是投影**：`render` 函数是纯函数，只依赖已折叠的状态，不持有 React 状态；
2. **回放一致性**：所有输入必须来自会话事件流，不能在 render 中引入随机值、实时 I/O 或当前时间；
3. **按需选择机制**：仅改变界面呈现（如颜色、折叠）不需要新事件；只有领域事实变化才需要扩展 `SessionEventMap`。

---

## UI 的输入输出

- **客户端消费**：通过 `session/event` 消费助手分片（`assistant/chunk`）、工具活动（`tool/call`/`tool/result`）与边界（`turn/start`/`step/start`/...）；
- **用户输入**：通过 `agent.followup()` 或 `agent.steer()` 将输入驱动回 Agent；
- **不要**把临时组件状态当作会话权威状态；
- **不要**依赖 React 渲染顺序表达会话逻辑——一切必须可从日志重建。

---

## 相关知识

- [[03_Agent运行时与会话/02_会话事件日志与模型历史]]
- [[01_插件开发与调试]]
- [[06_安全可靠性与工程化/02_持久化恢复与运行时不变式]]

## References

- `D:\_Projects\deepseek-harness\docs\cookbook\adding-a-conversation-node.zh.md`
- `D:\_Projects\deepseek-harness\docs\subsystems\session-projection.zh.md`
