# 工具定义、Schema 与执行契约

## 一句话理解

Tool 是模型可见的受控动作：Schema 决定参数和返回值形状，`execute()` 只实现领域操作，注册表负责校验、快照、错误归一化和呈现。

---

## ToolDefinition 完整结构

`ToolDefinition` 继承自 `ToolSchema`（面向模型的字段），并追加执行与呈现字段：

### ToolOutputDefinition（规范输出声明）

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema` | `JsonSchemaNode` | 每次成功规范值都要校验的 JSON Schema |
| `render(args, value)` | `ContentBlock[]` | 纯函数：从参数和值投影为模型可见的内容块 |
| `presentationMeta?(args, value)` | `JsonValue` | 纯函数：可回放的展示元数据，仅对顶层调用计算 |

### execute(args, exec)

```ts
execute(args: unknown, exec: ToolRunContext): Promise<unknown>
```

- `args`：注册表冻结的无损快照（只读）；
- `exec`：执行身份、取消信号和上下文延迟；
- **必须尊重 `exec.signal`**：取消和超时才能真正终止底层工作；
- 异步工作必须在 signal abort 后停止并到达完全停稳；
- 返回 `output.schema` 声明的规范 JSON 值；
- 基础设施失败**抛异常**；成功但"不理想"的业务结果作为规范值返回。

### finalizeContent（最后一道同步内容检查）

```ts
finalizeContent?(exec: Readonly<ToolExecution>, result: Readonly<ToolExecutionResult>): ContentBlock[] | undefined
```

- 在注册表外层归一化（快照固定）**之后**，物化**之前**调用；
- 对每个规范化结果（包括 pipeline 失败）调用恰好一次；
- 返回 `undefined` 保留内容，返回 `ContentBlock[]` 替换；
- **必须 total**，不能抛出；
- 接收不可变的执行对象（含无效输入和外层失败），可施加工具自有内容限制同时保留 `isError`、结构化错误身份等。

### timeoutMs（协作超时预算）

```ts
timeoutMs?: number   // 毫秒；省略表示无时限
```

- 由 `dsh-tool-call-timeout-policy`（`tools/execute` wrapper）执行；
- **绝不**暴露给模型（`schemas()` 白名单只含 `name`/`description`/`parameters`）；
- 声明即断言：该工具在 signal abort 后能到达完全停稳。

### isConcurrencySafe（并发安全标记）

```ts
isConcurrencySafe?(args: unknown): boolean
```

- **只有** `true` 才允许并发；省略、异常、非 `true` 均视为独占；
- 选择并发的执行**不得**修改父级拥有的状态；
- 共享状态必须容忍并发分派；记录竞态只在可交换（commute）或 fail closed 时允许；
- 此元数据**不对模型可见**。

### presentCall / presentResult（UI 展示钩子）

```ts
presentCall?(args: unknown): ToolCallView | undefined
presentResult?(args: unknown, result: ToolResult): ToolResultView | undefined
```

- 纯函数且无副作用（UI 在实时流式和会话日志回放时都可调用）；
- 省略或返回 `undefined` 时回退到通用展示（标题=工具名，原始参数作为输入）；
- `presentCall` 用于 **pending 状态**（仅依赖 `args`）；
- `presentResult` 用于 **completed 状态**（依赖 `args` + `ToolResult` 中的 `content`、failure 状态、`meta`）。

---

## Schema DSL（ValueSchemaSpec）

统一的工具参数与输出值描述词汇：

| 类型 | 说明 |
|---|---|
| `string`, `number`, `integer`, `boolean`, `null` | 标量类型 |
| `array` | 数组 |
| `object` | 显式对象，必须声明 `additionalProperties: true \| false` |
| `json` | 仅作者侧可用（any JSON） |
| `oneOf` | 恰好命中一个分支的联合 |

- 标量 `enum` 和 `const` 值必须与节点类型匹配；
- 参数定义是隐式的开放对象属性映射，每个必填属性附带 `required: true`；
- `defineTool` 代为校验并收窄参数类型、推导返回类型。

---

## 执行约定速查

| 场景 | 正确做法 |
|---|---|
| 基础设施失败 | 抛异常（registry 归一化） |
| 成功但"不理想"的结果 | 作为规范值返回（不是 `isError`） |
| 取消 | 监听 `exec.signal`，abort 后到达停稳 |
| 并发 | 只在 `isConcurrencySafe` 返回 `true` 时自动启用 |
| 工具参数改变模型上下文 | 必须追加 `SessionEvent`，不能只在内存拼接 |

---

## 注册方式

```ts
// 方式一：defineTool（类型安全，推荐）
ctx.tools.register(defineTool({
  name: 'my-tool',
  description: 'Do something',
  parameters: { ... },
  output: { schema: ..., render: (args, val) => [...] },
  execute: async (args, exec) => { ... }
}))

// 方式二：原始 ToolDefinition（MCP 来源工具直接用此方式）
ctx.tools.register({ name: ..., description: ..., parameters: ..., output: ..., execute: ... })
```

---

## 何时不用 Tool

| 场景 | 应该用 |
|---|---|
| 只供人操作的斜杠命令 | `ctx.commands` |
| 只改变界面呈现 | 客户端投影（`ConversationNodeDefinition`） |
| 可替换后端的执行能力 | 先设计为 Capability Seam，Tool 充当 Consumer |

---

## 相关知识

- [[02_工具执行流水线与策略钩子]]
- [[03_Capability_Seam_设计模式]]
- [[03_Agent运行时与会话/03_系统提示词与上下文管理]]

## References

- `D:\_Projects\deepseek-harness\docs\cookbook\adding-a-tool.zh.md`
- `D:\_Projects\deepseek-harness\docs\subsystems\tools.zh.md`
