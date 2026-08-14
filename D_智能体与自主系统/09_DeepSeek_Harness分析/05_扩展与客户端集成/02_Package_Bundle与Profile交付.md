# Package、Bundle 与 Profile 交付

## 一句话理解

Package 是代码与类型的工作区单元，Bundle 是可安装的插件配置发行物，Profile 是用户侧的运行组合；三者边界清晰才能安全升级、覆盖和分发。

## 三层职责

| 层 | 职责 |
|---|---|
| workspace package | 服务、工具或 Provider 的源码、类型与测试 |
| Bundle | 通过 `dsh.bundle.patch` 声明的一组可挂载配置 |
| Profile | `$DSH_HOME/profiles/<name>` 下指定 Bundle 与用户 patch 的运行方案 |

**组合层叠顺序**：Profile 中按顺序列出的每个 Bundle → Profile 的 `cordis.patch.yml` → home 级的那份 → 任意 `--patch` overlay。

内置 Bundle：
- `dsh-base`：每个 Profile 的第一层，包含模型适配器、工具、持久化、沙箱与审批策略、设置、凭据、遥测；
- `dsh-web-app`：增加浏览器应用；
- `dsh-headless`：增加一次性运行器，完全不带服务器。

## Package 设计

小型一次性扩展优先单包；只有 seam 的 Definition、Provider、Consumer 需独立演进时才分包。Host 与 Client 端不兼容的 TypeScript 声明不能复用同一 `Context` key，即使运行时隔离，声明合并仍会相遇。

## 交付安全

Git 依赖安装源码，`prepare` 构建会在用户机器执行。安装方必须显式授权 build 并固定可信 commit；发布预构建 npm 包或 tarball 能缩小这项安装时执行边界。

## Vendor 包命名规则

DSH 以 vendor 方式引入 Cordis 框架及其基础库，发布时以 `@deepseek-ai` scope 重命名，以避免在 npm registry 上占用上游名称。

| 上游名 | 发布名 | 角色 |
|---|---|---|
| `cordis` | `@deepseek-ai/cordis` | 框架核心：`Context`、`Service`、`Fiber`、事件 |
| `cosmokit` | `@deepseek-ai/cosmokit` | 框架与 Schemastery 共用的基础工具 |
| `schemastery` | `@deepseek-ai/schemastery` | 配置 schema（`Schema`），每个插件的 `Config` 基于它 |
| `@cordisjs/plugin-loader` | `@deepseek-ai/cordis-plugin-loader` | `cordis.yml` 装载、插件解析、repository 缓存 |
| `@cordisjs/plugin-include` | `@deepseek-ai/cordis-plugin-include` | 配置包含与 patch 叠加 |
| `@cordisjs/plugin-group` | `@deepseek-ai/cordis-plugin-group` | 嵌套插件分组 |
| `@cordisjs/plugin-timer` | `@deepseek-ai/cordis-plugin-timer` | `ctx` 上随 disposal 回收的定时器 |
| `@cordisjs/plugin-hmr` | `@deepseek-ai/cordis-plugin-hmr` | 插件与配置的热替换 |
| `@cordisjs/plugin-logger-console` | `@deepseek-ai/cordis-plugin-logger-console` | 控制台日志导出 |

子路径导出保持原路径：`@cordisjs/plugin-loader/repository` → `@deepseek-ai/cordis-plugin-loader/repository`。

### 改名不碰什么

- 目录名与版本号（`vendor/hmr/` 仍是 `vendor/hmr/`）；
- 依赖 range（只换键，不换范围）；
- Loader 的 `cordis:` 内建前缀（`cordis:include`、`cordis:group`）；
- `cordis.yml` 配置文件家族；
- 名字里带这个词的 harness 包（如 `@deepseek-ai/dsh-tool-cordis`）；
- 上游运行时标识符（如 Schemastery 的 `Symbol.for('schemastery')`）。

### 代码中要改的地方

| 位置 | 改前 | 改后 |
|---|---|---|
| 模块 import | `import { Context } from 'cordis'` | `import { Context } from '@deepseek-ai/cordis'` |
| 类型声明合并 | `declare module 'cordis'` | `declare module '@deepseek-ai/cordis'` |
| `package.json` 依赖键 | `"@cordisjs/plugin-hmr": "^1.0.15"` | `"@deepseek-ai/cordis-plugin-hmr": "^1.0.15"` |
| `cordis.yml` 插件条目 | `name: '@cordisjs/plugin-include'` | `name: '@deepseek-ai/cordis-plugin-include'` |

改名由 `scripts/rescope-vendor.ts` 承载，不靠手工编辑。上游 sync 后重跑：

```sh
pnpm run rescope-vendor --apply
pnpm install                          # 重生成 lockfile
pnpm run gen-third-party-notices
```

## 相关知识

- [[01_插件开发与调试]]
- [[01_入门与运行/02_安装启动与运行配置]]
- [[06_安全可靠性与工程化/03_测试诊断与开发工作流]]

## References

- `D:\_Projects\deepseek-harness\docs\rescope.zh.md`
- `D:\_Projects\deepseek-harness\docs\cookbook\adding-a-package.zh.md`
- `D:\_Projects\deepseek-harness\docs\user\develop\basic\publish.zh.md`
- `D:\_Projects\deepseek-harness\docs\architecture.zh.md`
