# Agent Turn 与 Step 生命周期

## 一句话理解

Turn 是一次已接纳输入的完整排空过程，Step 是一次模型请求及其工具调用；一个 Turn 可含零到多个 Step，持久日志记录事实，`agent/*` 管理实时控制。

## 主流程（完整路径）

```text
followup(content)
  └─► agent/inbox/inserted（UI 通知）
  └─► agent/inbox/spliced（UI 通知）
  └─► 唤醒 Driver

Driver → agent/status: running
Driver → session: turn/start

─── 领取 inbox：pending next-step input + 一条队列消息 ───
  └─► agent/inbox/spliced（纯删除）
  └─► agent/inbox/claimed（每条消息，携带 turn）

agent/pre-step（waterfall）
  ├─► 被 reject 或被改写为空
  │    └─► 领取的批次已移除；轮次不含 Step，但仍持久（记录尝试）
  │    └─► session: turn/end
  └─► enter(messages) 进入 Step

session: step/start
session: user/message（每条 entered 消息）

system-prompt/assemble（waterfall，组装提示词段与工具 schema）

agent/request（waterfall）→ llm/stream（waterfall）
  └─► StreamChunk* → session: assistant/chunk*（UI 实时流）

─── 模型请求失败路径 ───────────────────────────────────────
  session: step/end
  agent/request-error（waterfall）
    ├─► compaction（context 溢出时触发）：先工具结果剪枝 → 再摘要
    │    只有当剪枝或摘要推进了 surface replacement generation 时
    │    才开启全新重试轮次；否则仍以原始请求错误为准
    └─► retry（由 llm-retry 提供）：返回 retry action 或保留原始错误

─── 模型请求成功路径 ───────────────────────────────────────
session: assistant/message（含用量；空内容不进派生历史，但事件保留）

工具调用分类（按 executionMode）→ 工具执行流水线循环
  （含 barriers、有界滚动池、重新分类）
  └─► 每个 call/result 对: tool/call → [流水线] → tool/result

session: step/end

─── Step 自然停止，且 inbox 为空 ───────────────────────────
agent/turn-stopping（serial，最终检查点）
  └─► 钩子可通过 steering 触发下一个 Step（不是 next()，而是直接注入）

─── next-step inbox 仍有待处理输入 ─────────────────────────
  └─► 重新领取 → 新一轮 agent/pre-step waterfall → 下一 Step

session: turn/end
agent/status: idle
```

## 关键路径说明

### `agent/pre-step` 被 reject 的语义

`agent/pre-step` 是 waterfall 事件，监听器可以用两种方式响应：
- **authoritative reject**：返回拒绝决策，Step 不开始；已领取的输入批次从 inbox 移除，但 Turn 仍然持久化（`turn/start` → `turn/end`，不含任何 `step/start`），用于审计这次尝试。
- **enter(messages)**：返回进入消息，Step 正式开始。

steering 注入的上下文和在后续领取操作取得其下一步骤批次后，会重新经过同一 waterfall——这意味着 `pre-step` 监听器需要正确处理不同来源的输入。

### `agent/request-error` 与 Compaction

`agent/request-error` 是用于处理模型请求失败（包括 context 溢出）的 waterfall：
- **compaction（上下文压缩）**：`dsh-compaction-basic` 监听此事件；只有规范的上下文溢出才触发摘要压缩。流程是先做工具结果剪枝（可选），再做摘要生成；只有这两步中至少一步推进了 `surface replacement generation`（表层替换代次），才会开启一个全新的重试轮次——否则仍以原始请求错误返回给上层，不无谓地消耗轮次。
- 压力（pre-step 阶段）与请求错误（request-error 阶段）是两种不同的触发时机，对应不同的场景和处理策略。

### `agent/turn-stopping` 的位置与性质

`agent/turn-stopping` 是 **serial** 事件（不是 waterfall），运行于 Step 自然结束且 inbox 为空之后：
- 钩子插件（如 `hooks-claude-code`、`hooks-codex`）在此处做最终检查；
- 若需要续跑下一 Step，不通过 `next()` 实现，而是直接向 agent 注入 steering；
- 此事件的存在确保 Turn 在最终关闭前有一个对外可见的同步检查点。

### `assistant/message` 的空内容语义

每次成功的提供方调用（包括返回空内容或以 `max-tokens` 结束的调用）都会记录 `assistant/message` 事件：
- 空内容**不会**进入 `deriveMessages()` 派生的模型历史；
- 但 `assistant/message` 事件**仍然**持久化，保留用量信息和对应 `assistant/chunk` 序列的 `sourceEventSeqs`（包括显式空列表）。

## 控制与持久化的分工

| 通道 | 用途 |
|---|---|
| `session/event` | 持久事实：回放、UI、fork、transcript、遥测应消费此通道 |
| `agent/*` | 实时协调：inbox 状态、运行状态、请求改写、取消、steer、重试 |

**异步状态规则**：不要把 `agent/status: idle` 或 `whenIdle()` 解释成某一条 `followup()` 的单独结果。多条排队消息、steering 和 inject 可能共享同一 `running` 区间。拥有一次运行的自动化调用方必须显式定义区间（从入队回执到整个 agent 下一次进入 idle），并将选取的输出描述为整个区间的输出，而非归因于某条消息。

## 相关知识

- [[02_会话事件日志与模型历史]]
- [[03_系统提示词与上下文管理]]
- [[04_工具与可替换能力/02_工具执行流水线与策略钩子]]
- [[04_Agent控制_目标计划与调度]]

## References

- `D:\_Projects\deepseek-harness\docs\agent-lifecycle.zh.md`
- `D:\_Projects\deepseek-harness\docs\architecture.zh.md`
