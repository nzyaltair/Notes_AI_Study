# Capability Seam 设计模式

## 一句话理解

Capability Seam 是完整的可替换能力，由 Service Definition、一个或多个 Provider 和 Consumer 组成；它不是 TypeScript 接口或其中任一单独角色。

## 三种角色

| 角色 | 职责 |
|---|---|
| Service Definition | 定义 `ctx.<key>`、请求/结果类型和稳定语义 |
| Service Provider | 以本地、远程或沙箱机制实现服务 |
| Consumer | 注入服务并将其暴露为 Tool、UI 或其他产品行为 |

以 Shell 为例：`dsh-shell` 定义 Bash 请求与结果（Service Definition），`dsh-bash-local`/`dsh-bash-sandbox` 是 Provider，面向模型的 `dsh-tool-bash` 是 Consumer。替换 Provider 不应迫使 Consumer 改动。

## 角色分类：`core` vs `seam` vs `bundle`

DSH 中的 `ctx.*` 服务按角色分为三类：

| 类型 | 含义 | 示例 |
|---|---|---|
| `core` | 唯一实现，不可替换；定义包即实现 | `ctx.sessions`、`ctx.tools`、`ctx.agents`、`ctx.goals` |
| `seam` | 可替换：定义包与实现包分离，应用在组合时选择后端 | `ctx.llm`、`ctx.shell`、`ctx.fs`、`ctx.sandbox`、`ctx.sessionPersistence` |
| `bundle` | 唯一的具体插件；扩展包依赖其事件/服务而非此包 | `ctx.agentLoop` |

## 主要 Seam 一览

### 模型与执行

| ctx 键 | 类型 | 已知 Provider | Consumer |
|---|---|---|---|
| `ctx.llm` | seam | `llm-deepseek`、`llm-pi-ai`、`llm-replay` | `agent-loop`、`compaction-basic` |
| `ctx.shell` | seam | `bash-local`、`bash-sandbox`、`pwsh-local` | `tool-bash`、`tool-pwsh`、`hooks-*` |
| `ctx.subprocess` | seam | `subprocess-local`、`subprocess-e2b` | `bash-local`、`bash-sandbox`、`terminal-bash`、`lsp-stdio`、`subagent-*` |
| `ctx.fs` | seam | `fs-local`、`fs-sandbox`、`fs-e2b` | `tool-fs` |
| `ctx.codeRuntime` | seam | `code-runtime-worker` | `tools`（Code Mode） |
| `ctx.lsp` | seam | `lsp-local` | `tool-lsp` |

### 会话与存储

| ctx 键 | 类型 | 已知 Provider | Consumer |
|---|---|---|---|
| `ctx.sessions` | core | — | `agent-loop`、`agent`、`session-persistence` |
| `ctx.sessionPersistence` | seam | `session-persistence-jsonl`、`session-persistence-sqlite` | `agent-loop`、`tool-bash`、`session-query` |
| `ctx.sessionQuery` | seam | `session-query-sqlite` | `session-reference`、`tool-session-query` |
| `ctx.storage` | seam | `storage-json`、`storage-sqlite` | `storage-domain` |
| `ctx.compaction` | seam | `compaction-basic` | `compaction-basic`（自身既是 Provider 也是 Consumer） |
| `ctx.spillStore` | seam | `spill-local` | `spill-policy` |

### 安全与策略

| ctx 键 | 类型 | 已知 Provider | Consumer |
|---|---|---|---|
| `ctx.approval` | seam | `acp`（通过 `approval/request` waterfall） | `tools`、`tool-bash` |
| `ctx.sandbox` | seam | `sandbox-local` | `bash-sandbox`、`terminal-bash` |
| `ctx.sandboxPolicy` | core | — | `bash-sandbox`、`fs-sandbox`、`terminal-bash` |
| `ctx.permissionPresets` | core | — | — |

### 子代理与工作流

| ctx 键 | 类型 | 已知 Provider | Consumer |
|---|---|---|---|
| `ctx.subagents` | seam | `subagent-spawn-in-process`、`subagent-fork-in-process`、`subagent-acp`、`subagent-codex`、`subagent-claude-code`、`subagent-dsh-sdk` | `tool-subagent`、`tool-subagent-control`、`tool-ralph` |
| `ctx.workflowEngine` | seam | `workflow-worker-thread` | `tool-workflow`、`tool-ralph` |
| `ctx.jobs` | seam | `jobs-local` | `tool-bash`、`tool-terminal`、`tool-subagent`、`tool-jobs` |

### 用户交互与 UI

| ctx 键 | 类型 | 已知 Provider | Consumer |
|---|---|---|---|
| `ctx.userQuestions` | seam | （UI 前端提供当前生效的人工回答方） | `tool-ask-user` |
| `ctx.skills` | seam | `skill-badge`、`skill-filesystem` | `tool-skill` |
| `ctx.sessionTitle` | seam | `session-title-first-prompt-llm`、`session-title-all-prompts-llm` | — |
| `ctx.directoryPicker` | seam | `directory-picker-native`、`directory-picker-browse` | `apiproxy` |
| `ctx.web` | seam | `web-search-exa`、`web-search-perplexity`、`web-search-deepseek`、`web-fetch-http` | `tool-web` |

### 凭据与设置

| ctx 键 | 类型 | 已知 Provider | Consumer |
|---|---|---|---|
| `ctx.credentials` | seam | `credentials-local` | `llm-deepseek`、`llm-pi-ai`、`apiproxy` |
| `ctx.settings` | seam | `settings-file` | `llm-deepseek`、`llm-pi-ai`、`apiproxy` |
| `ctx.sessionTelemetry` | seam | `session-telemetry-otel` | — |

## 何时拆分

当服务契约、后端实现和产品暴露方式需要独立演进时拆为三层；简单专用工具不应预防性拆包。接口包命名应反映稳定能力，实现包再附加机制或厂商名。

**不拆分的理由**：`ctx.llm` 同时承担 Service Definition 和 Consumer 角色（`dsh-llm` 注册适配器同时也调用适配器）——角色合并是允许的，只要角色可识别且替换点清晰。

## 设计检查

1. 是否有明确且可替换的后端边界？
2. Provider 与 Consumer 是否只依赖 Definition，而不互相依赖？
3. 失败、取消、输出限制和安全语义是否在契约中定义？
4. 更换 Provider 后，工具展示、执行和日志是否仍然一致？

## 相关知识

- [[01_工具定义_Schema与执行契约]]
- [[05_执行能力_文件Shell终端与Web]]
- [[02_Cordis插件框架/03_服务与依赖注入]]
- [[07_子系统参考地图]]

## References

- `D:\_Projects\deepseek-harness\docs\capability-seams.zh.md`
- `D:\_Projects\deepseek-harness\docs\glossary.zh.md`
- `D:\_Projects\deepseek-harness\docs\architecture.zh.md`
