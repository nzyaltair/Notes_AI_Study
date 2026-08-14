# Cordis 核心模型

## 一句话理解

Cordis 是 DSH 的插件运行时：插件在共享 `ctx` 上贡献服务、事件监听和可逆副作用，插件卸载时这些贡献一并撤销。

## 核心对象

| 对象 | 作用 |
|---|---|
| `Context` | 服务访问、事件注册和生命周期资源的统一入口 |
| Plugin | 导出 `apply(ctx)` 的功能单元 |
| Fiber | 一次插件加载实例及其资源所有权边界 |
| Service | 在 `ctx.<key>` 上提供的能力 |
| Effect | 随 Fiber 生命周期自动清理的副作用 |

## 最小插件

```ts
import type { Context } from '@deepseek-ai/cordis'

export const name = 'my-plugin'
export function apply(ctx: Context) {
  ctx.on('some-event', () => {})
}
```

插件不是“全局初始化脚本”。它的注册属于 Fiber；卸载时事件监听、工具和服务注册应自动消失。需要释放外部资源时使用 `ctx.effect()` 返回清理函数。

## 使用边界

- 简单行为优先作为单插件接入已有服务或事件。
- 需要独立替换的通用能力才设计为 [[04_工具与可替换能力/03_Capability_Seam_设计模式|Capability Seam]]。
- 任何模型可见的新信息都不能只在插件内临时拼接，必须按会话事件规则记录。

## 相关知识

- [[02_插件生命周期与资源管理]]
- [[03_服务与依赖注入]]
- [[04_事件系统与分发语义]]

## References

- `D:\_Projects\deepseek-harness\docs\cordis-primer.zh.md`
- `D:\_Projects\deepseek-harness\docs\cordis-tutorial\index.zh.md`
