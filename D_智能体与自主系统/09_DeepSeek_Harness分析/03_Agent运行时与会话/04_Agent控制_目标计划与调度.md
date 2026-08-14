# Agent 控制：目标、计划与调度

## 一句话理解

目标（Goal）、计划模式（Plan Mode）、调度（Schedule）和人类命令（Commands）是围绕 Agent Loop 组合的策略层：它们各自记录必要状态并通过 followup/steer/事件监听续跑，而非把业务控制硬编码进循环。

## 三种控制面

| 控制面 | ctx 服务 | 作用范围 | 可见性 |
|---|---|---|---|
| 人类命令 | `ctx.commands` | 解释 `/...` 斜杠命令，不成为模型消息 | UI 状态，不进入 transcript |
| 目标（Goal） | `ctx.goals` | 附着于会话的持久完成状态；Goal Round 是接纳的续行周期 | 持久化 `goal/change` 事件 |
| 计划模式 | `ctx.planMode` | 记录协作状态并向模型提供策略段落 | 持久化 `plan/mode` 事件（log-only） |
| 调度（Schedule） | （包内服务） | 到期后向原 live Session 提交 followup | 持久化 `schedule/change` 事件 |

---

## 目标（Goal）

### 什么是目标

目标（Goal）是附着在现有会话上的**单个持久完成目标**：
- 不是调度器，也不是独立对话；
- 通过 `goal/change` 会话事件持久化，支持恢复、fork 和压缩；
- 进程本地的激活状态（`armed/disarmed`）不持久化，恢复或 fork 后须经人类授权（`resume`/`create`）才能重启自动工作。

### 目标生命周期状态机

```
active  <--resume-- paused
  |                  ^
  +--pause----------+
  |
  +--block--> blocked
  |            |
  +--clear--> (tombstone)   complete
  |
  +--complete--> complete
```

| 持久阶段（GoalPhase） | 含义 |
|---|---|
| `active` | 正在进行，续行消费方可以触发 Goal Round |
| `paused` | 暂停，自动续行已撤销 |
| `blocked` | 因问题而停止；携带 `GoalBlockReason`（code + message） |
| `complete` | 目标已完成；只有 `complete` 状态可被新目标替换 |

**激活状态**（GoalActivation）与持久阶段分离，只有 `armed` 时续行消费方才接纳下一个 Goal Round：
- `create()` 和 `resume()` 会 arm；
- `pause()`、`block()`、`complete()` 和 `disarm()` 会 disarm；
- 进程退出后激活状态丢失（下次须显式 resume）。

### Goal Round 机制

**Goal Round** 是为当前目标接纳的一次续行周期：
- 每个获准的 user-role Goal 消息标注正数且连续的 Round 编号；
- 同会话驱动器将 Goal Round 具体化为一个由目标触发的 Turn；
- 同一会话中与目标无关的人类 Turn 不消耗 Goal Round 上限（`maxGoalRounds`）；
- 回放会拒绝编号缺口、陈旧修订号、已停止阶段和超出上限。

### 版本一致性（CAS）

每次变更都是 compare-and-set 操作：调用方持有 `GoalRef`（`{ id, revision }`），每次获准变更递增修订号；使用过期 `ref` 的变更会被拒绝，防止并发写入冲突。

### `ctx.goals` 主要操作

| 操作 | 前提条件 |
|---|---|
| `create(agent, request)` | 无当前目标，或当前目标已 `complete` |
| `pause(agent, ref)` | 目标处于 `active` 状态 |
| `resume(agent, ref)` | 目标处于 `paused`/`blocked` 状态，且 Round 预算仍有余量 |
| `complete(agent, ref)` | 目标非 `complete` |
| `block(agent, ref, reason)` | 目标处于 `active` 状态 |
| `clear(agent, ref)` | 目标处于任意状态；留下 tombstone |
| `disarm(agent)` | 任意状态；进程本地操作，不改变持久阶段 |

---

## 计划模式（Plan Mode）

### 定位

计划模式是**软性指引**，不是执行约束。它的作用是：
- 激活期间为每个模型请求注入一段部署持有的策略文本（`plan:policy` 段落，order 50）；
- 注册 `exit_plan_mode` 工具（**始终可见**，在计划模式之外执行才报错，防止工具集在进入/离开时变化）；
- 可选地注册 `/plan` 命令。

沙箱模式与审批策略才是真正的执行强制；两者不读写计划状态。

### 状态持久化

`plan/mode`（`{ active: boolean }`）是仅记日志、整值替换的会话事件：
- **持久且可回放**，绝不进入模型 transcript；
- `foldPlanMode(events)` 返回日志中最后一条记录值，没有时返回 `false`；
- 恢复、fork 与压缩无需实时镜像即可复原状态。

### 待生效选择机制

`set(agent, active)` 在轮次内只记录**待生效选择**，不立即追加事件：
- 唯一的追加点是 `agent/pre-step` 监听器（前置注册）；只在下游接受该步骤后追加；
- 在某轮最后一个被接受的 pre-step 之后作出的选择只存在于进程内；如果进程在下一个 pre-step 之前退出，该选择**会丢失**；
- `get(agent)` 返回 `{ active: boolean; pending?: boolean }`：当前已记录状态 + 等待追加的选择。

### `exit_plan_mode` 工具的评审流程

1. 模型调用 `exit_plan_mode`，并提供以 `#` 标题开头的完整 markdown 计划；
2. 工具通过 `ctx.userQuestions`（用户交互 seam）呈交用户评审；
3. 批准：返回 `{ approved: true }`，记录静默的待生效退出（由下一个 pre-step 追加）；
4. 拒绝：携带用户反馈的失败调用，模型据此修订并再次呈交；
5. 评审期间交互通道缺失或服务重载使调用失败，而不是静默离开计划模式。

因此，**计划指引在 assistant 当前这批工具调用的剩余部分继续生效**，工具结果本身报告这次转换。

---

## 调度（Schedule）

### 定位

Schedule 拥有持久提醒，这些提醒在到期后作为普通后续对话轮次返回**原 live Session**（不提供外部通知渠道，cold Session 不执行任何工作）。

### 三类提醒记录

| 类型 | 描述符 | 最小时间 |
|---|---|---|
| `after` | 延时一次性（`afterSeconds`） | 任意正整数秒 |
| `at` | 绝对时间一次性（严格 RFC 3339 UTC 或本地日历对象） | 未来时刻 |
| `every` | 固定速率重复（`everySeconds`，以创建时间为锚点） | 最小 300 秒（5 分钟） |

每条记录存储规范化后的 UTC `scheduledAt`，回放绝不依赖环境时区。

### 持久化协议

`schedule/change` 会话事件是唯一权威：
- `create`：保存完整记录；
- `delete`：终结性且仅含 id；
- `dispatch`（一次性）：终结性且仅含 id；
- `dispatch`（every）：携带 `acceptedAt` 判断时刻，直接将记录推进到下一个锚点对齐目标，**跳过错过的间隔**而非逐一补发。

### 交付语义

- 进程内 owner 根据持久 fold 派生最早的 timer；
- 到期时先等待 Agent 完全 idle，再将 `followup()` 入队并追加 dispatch 变更；
- **绝不调用 `steer()`，绝不中断当前轮次**；
- 一次性提醒优先，每次只进入一个后续轮次；所有 overdue 的 every 记录组成单个批次；
- dispatch 表示 followup 已同步入队，不表示模型答复成功；
- 窄崩溃窗口可能导致重复触发：提供的是**至少一次**交付语义，而非恰好一次。

---

## 调度规则与异步约束

1. **自动化调用方必须显式定义完成区间**：从消息入队回执（inbox 准入）到整个 agent 下一次进入 idle；不应声称某条消息单独造成了某个 idle 或结果。
2. **Schedule 到期后 followup，当前轮次已完成才执行**：不会抢占或中断当前工作。
3. **进程本地激活在恢复或 fork 后丢失**：只有随后通过 `/goal`、模型工具或显式 API 执行一次经人类授权的 resume 变更，自动工作才能重启。
4. **计划模式是软性指引**：真正的执行约束由审批策略（`ctx.userApproval`）和沙箱（`ctx.sandbox`）负责，不依赖 `ctx.planMode`。

## 相关知识

- [[02_Cordis插件框架/06_作用域隔离与注册表]]
- [[01_Agent_Turn与Step生命周期]]
- [[05_扩展与客户端集成/04_子代理_工作流与Skills]]
- [[06_安全可靠性与工程化/01_审批权限与沙箱]]

## References

- `D:\_Projects\deepseek-harness\docs\subsystems\goal.zh.md`
- `D:\_Projects\deepseek-harness\docs\subsystems\plan.zh.md`
- `D:\_Projects\deepseek-harness\docs\subsystems\schedule.zh.md`
- `D:\_Projects\deepseek-harness\docs\glossary.zh.md`（目标、Ralph 等术语定义）
