# 子代理、工作流与 Skills

## 一句话理解

子代理（Subagent）负责将工作委派给另一个 Agent 或外部产品；工作流（Workflow）将模型编写的脚本以受记录的结构化引擎执行；Skills 则是按需发现并加载的知识/工作流说明。三者解决不同问题，不能互相替代。

## 三者对比

| 机制 | ctx 服务 | 解决的问题 | 是否有命名提供方注册表 |
|---|---|---|---|
| Subagent | `ctx.subagents` | 委派执行给另一个 Agent 或外部产品，管理其继续执行 | 是（多提供方共存） |
| Workflow | `ctx.workflowEngine` | 将模型编写的脚本以 worker-thread 引擎执行，支持并发子 agent | 否（单引擎替换） |
| Skills | `ctx.skills` | 按需发现、加载 Markdown 指令，注入提示词供模型调用 | 是（分层多提供方） |

---

## 子代理（Subagent）

### 设计定位

Subagent seam 是一项可选能力，不属于 agent loop。与 bash 执行器不同，**同一上下文中可共存多个提供方实现**，并按名称注册（`ctx.subagents`）——这遵循 LLM 适配器注册表的模式，而非单服务的 bash 执行器。

六个内置提供方（`dsh-subagent-spawn-in-process`、`-fork`、`-acp`、`-codex`、`-claude-code`、`-dsh-sdk`）覆盖进程内 spawn、fork 注入父级历史、外部协议三类场景。

### 两类子 agent 模式

| 模式 | 特征 | 生命周期 |
|---|---|---|
| **一次性（one-shot）** | 启动后单次运行，有一个 `SubagentRun` 句柄 | 调用方 await `result`，必须调用 `dispose()` |
| **可继续（continuable）** | 持久化子会话，可多轮次继续；由继续执行管理器持有 `AgentHandle` | `followup()` 路由到 Activation 或冷恢复 |

### 启动时能力校验

每个提供方在其描述符中声明启动时能力标志（`SubagentCapabilities`）：

| flag | 含义 |
|---|---|
| `outputSchema` | 支持结构化输出 JSON Schema |
| `depthLimit` | 支持绝对委派深度上限（`maxDepth`） |
| `toolFilter` | 支持子 agent 工具过滤（在创建窗口应用 `tools.restrict()`） |
| `persona` | 支持按 agent 定制 persona（作为 shadow 区段注入） |

服务在 `start()` **之前**校验请求的能力，不支持的请求直接以 `UNSUPPORTED_CAPABILITY` 拒绝，绝不会被接受后静默忽略。

### 委派深度

委派深度通过持久化的 `SessionHeader.delegationDepth` 和可合并扩展的运行时字段 `AgentOptions.subagentDepth` 共同表示。进程内子 agent 持久化 parent 深度 + 1；冷恢复无法降低深度；超出安全整数域或超出 `maxDepth` 时每次 start 都会拒绝。

### 可继续子 agent 的激活状态机

```text
persisted Session
  -> optional live Activation
       -> one retained AgentHandle
       -> Agent inbox as the only turn FIFO
       -> zero or more owned child Activations
```

`followup()` 路由规则：

| Activation 状态 | `followup` 行为 |
|---|---|
| `running` | 在同一 Activation 中入队 |
| `waiting` | 唤醒同一 Activation |
| 无 Activation | 冷恢复一个新的 Activation |

`settled`（完全停稳且子级都已 dispose）时，管理器 dispose AgentHandle 并移除 Activation。

### `interrupt()` 语义

`interrupt(targetSessionId, authority)` 同步完成鉴权，对在线目标发出 `Agent.cancel(cause, { keepInbox: true })`，**不等待完全停稳即返回**。已被领取进入中断轮次的工作不会重新入队；被中断的 driver 进入 idle 后，一次唤醒发送可恢复被暂停的 FIFO 队列。

### `reportFrom()` 语义

子 agent 主动上报选中内容给直接父级（调用方不能指定接收方）：
- 使用 `Agent.inject()`（静默投递）或 `Agent.followup()`（唤醒投递），不结束子 agent 当前轮次；
- Activation 结算时管理器向父级投递终态通知（`subagent-settled` kind），与子 agent 自己的 report 使用不同的 kind，防止把运行时记账呈现为子 agent 写下的内容。

### 终态结果

`SubagentResult` 中：
- `output`：子 agent 最后一条非空 assistant 消息的内容；
- `structured`：请求了 `outputSchema` 且成功满足时才存在；
- `stopReason`：`completed | aborted | error | max-tokens | refusal`（可合并扩展联合）；非 `completed` 时消费方映射为 `isError` 工具结果。

### 可继续子 agent 的 Descriptor 与 Inbox

**`subagent/descriptor` 持久化事件**（log-only，不进入模型历史）由建立提供方在子 agent 的**初始轮次内**、第一次请求之前写入一次，携带身份与生命周期模式（一次性/可继续）及可继续子 agent 的可恢复组合。这是子 session 的"出生证明"，跨重启/恢复后仍可重建相同组合。

**收件箱（inbox）作为唯一的继续执行 FIFO**：每条 `followup` 调用成为 `Agent.followup()` 队列的一个轮次，因此所有已接受消息共享**同一个**可观测顺序，后来的消息无法改变已在进行中的轮次。`agent/inbox/spliced` 持久化事件记录收件箱的每次规范化 splice，并在 live dispatch 之前触发，使同步观察者可在 splice 前读取旧收件箱。

**`startContinuable()` 准入协议**：
1. 预留稳定子 agent id；
2. 对 `subagent/descriptor` payload 建立快照；
3. 向提供方索取分离的 `ContinuableCreateSpec`；
4. 通过私有 activation-owner 作用域创建子 Agent；
5. 建立任何可继续父级的所有权；
6. 提交初始提示词；
7. 当收件箱准入产出消息 id 时，以 `{ childId, messageId }` resolve（无需等待轮次开始或消息进入日志）。

准入前任何失败均以两个 id 都不返回的方式 reject，并 dispose 已创建的 handle、回滚 Activation 与父级所有权。

**所有权图**：每个 Activation 拥有：
- 一个 `AgentHandle`
- 一个 `ownedChildren: Set<SessionId>`

只要 `ownedChildren` 非空，父级无法 settle。顶层或非继续执行 Agent 没有 Activation，处于 waiting 图之外。子 agent 完全释放条件：子 Agent 完全停稳 **且** 每个子级都已 dispose **且** 最终会话 flush 结算 **且** 子 agent 的 `AgentHandle` dispose。

**结算通知语义**：
- `subagent-report` kind：子 agent 自己选择的上报内容（`reportFrom()` 触发）；
- `subagent-settled` kind：管理器记账，说明该 epoch 如何结束，携带最终 assistant 内容；两者使用**不同 kind**，防止把运行时记账呈现为 child 自己写下的内容。

---

## 工作流（Workflow）

### 设计定位

工作流 seam 允许 agent 运行**模型编写、由脚本驱动的编排**：脚本可调用 `agent()` 启动子 agent、`parallel()`/`pipeline()` 组合并发，以及 `phase()`/`log()` 记录进度。与 bash 一样，**每个上下文只允许一个引擎实现**（`ctx.workflowEngine`），第二个引擎通过配置替换第一个，而不与它同时运行。

内置引擎：`dsh-workflow-worker-thread`（每个 run 一个 `node:worker_threads` worker，脚本的 vm 上下文位于其中）。

### 启动请求与 meta 块

`WorkflowStartRequest` 核心字段：
- `script`：脚本正文（允许顶层 await，以 `return <json-value>` 结束）；
- `meta`：`WorkflowMeta` 身份块（`name`、`description`、可选 `phases`），由引擎在正文运行前校验；
- `parent`：必填，脚本启动的每个子 agent 都归属于它；
- `args`：可选输入，以全局变量暴露给脚本；
- `signal`：可选取消信号。

**`meta` 和 `args` 都是普通 JSON 数据**，引擎不会通过对脚本文本求值来获取它们。

### 失败纪律：`WorkflowError.fatal`

`fatal: true` 错误（错误参数、超出结构化输出子集、超出上限、seam 启动失败、取消）在 `parallel()`/`pipeline()` 组合器中**直接重新抛出**，而非映射为 `null`。逐项的 `null` 仅保留给子运行失败（非 `completed` 的 stop reason）和普通脚本错误。**一个拼写错误的选项必须明确报错并终止脚本，绝不能消融为看似普通子 agent 失败的结果**。

### `WorkflowRun` 句柄与取消

- `result` 永远不会 reject：脚本失败兑现为 `stopReason: 'error'`；
- 取消后，即使脚本永不结算，结果也会在引擎规定的**有界宽限期**内强制结算为 `cancelled`，随后 worker-thread 引擎终止 worker；
- `dispose()` 执行取消、等待有界结算并等待子 agent 完全停稳，不会因脚本卡死而挂起。

### 事件与持久化 Chat 记录

`workflow/*` 事件（`workflow/start`、`workflow/phase`、`workflow/log`、`workflow/agent-start`、`workflow/agent-end`、`workflow/end`）是**仅供观察的 emit**：每个 payload 以 `WorkflowRunInfo`（id + meta）开头，而非活跃的 `WorkflowRun`，防止订阅者获得 `cancel`/`dispose`。监听器异常被隔离，不会阻止后续监听器。

`dsh-tool-workflow` 把展示事实投影到父 Session 的会话日志中（`tool-workflow/run-start`、`tool-workflow/agent-start`/`agent-end`、`tool-workflow/run-end`），且只在 result 已取得且 dispose 完全停稳后写 `run-end`。

---

## Skills

### 设计定位

Skills 是可选的指令而非会话事件。注册表（`ctx.skills`）组合本地、内嵌、远程或其他提供方的目录，Consumer（`dsh-tool-skill`）拥有初始目录和替换目录以及面向模型的 `skill` 工具。

### 本地发现优先级（rank 越小优先级越高）

| Rank | 来源 | 根目录 |
|---|---|---|
| 100 | `project-dsh` | `<projectRoot>/.dsh/skills` |
| 200 | `project-agents` | `<projectRoot>/.agents/skills` |
| 300 | `custom` | `Config.customSkillDirs` |
| 400 | `user-dsh` | `<dshHome>/skills` |
| 500 | `user-agents` | `<agentsHome>/skills` |
| 600 | `bundled` | `Config.bundledSkillDir`（可选） |

本地提供方接受目录包（`<name>/SKILL.md`）和扁平 Markdown 文件（`<name>.md`）；**嵌套递归的 `**/SKILL.md` 发现不受支持**。

### 分层注册与 shadowing

注册表采用宿主 + 按 scope 的分层结构（与工具注册表一致）：
- 宿主行和 repository 插件落入全局层；
- agent preset 常驻组合挂载的插件落入该 preset 的层；
- 读取时将全局层与观察 scope 的链合并：**最近层的条目直接赢得重名 skill**，rank 顺序只在单层内裁决重名。

### 调用策略（SkillInvocationPolicy）

`SkillSummary` 携带规范化的调用控制（正向布尔值）：

| 状态 | 含义 |
|---|---|
| `{ modelInvocable: true, userInvocable: true }` | 模型和用户都可调用（默认） |
| `{ modelInvocable: true, userInvocable: false }` | 仅模型可调用 |
| `{ modelInvocable: false, userInvocable: true }` | 仅用户可调用 |
| `{ modelInvocable: false, userInvocable: false }` | 仅受信的 `ctx.skills.get()` 可获取 |

本地提供方通过 frontmatter 键 `disable-model-invocation` 和 `user-invocable` 控制，省略的字段默认为 `true`。

### 会话目录注入机制

`dsh-tool-skill` 的会话目录注入规则：
1. 存活会话中第一个观察到非空完整视图的 `agent/pre-step` 注入初始 `<system-reminder>`；
2. 目录**只包含 `name` 和 `description`**，不包含正文、路径、来源；
3. 后续每个步骤之前计算 digest，变化时通过 `agent.inject()` 追加完整替换；不完整快照保留上一份可用视图；
4. 这些目录消息属于会话历史，而非 World State（压缩不会破坏，下次完整快照可重建）。

面向模型的 `skill({ name })` 工具在返回内容前会**再次**检查 `isModelInvocable` 策略，并据 agent 的 cwd 重新读取完整定义；**注册表不缓存完整定义**，每次 `get()` 都重新读磁盘。

## 设计约束

- 子代理的 scope **不从父 Agent 自动继承**；应显式设置工具、提示词、权限与 lineage；
- 工作流应记录关键状态和结果到会话日志（`tool-workflow/*` 事件），避免把可恢复业务状态藏在内存回调中；
- Skills 适合渐进披露：发现阶段只暴露 `name`+`description`，调用时才加载完整正文；
- 三者都是可选 seam，缺少提供方时对应能力不存在，但不会导致 Agent Loop 崩溃。

## 相关知识

- [[03_Agent运行时与会话/04_Agent控制_目标计划与调度]]
- [[02_Cordis插件框架/06_作用域隔离与注册表]]
- [[04_工具与可替换能力/05_执行能力_文件Shell终端与Web]]
- [[05_扩展点选择决策树]]

## References

- `D:\_Projects\deepseek-harness\docs\subsystems\subagent.zh.md`
- `D:\_Projects\deepseek-harness\docs\subsystems\workflow.zh.md`
- `D:\_Projects\deepseek-harness\docs\subsystems\skills.zh.md`
