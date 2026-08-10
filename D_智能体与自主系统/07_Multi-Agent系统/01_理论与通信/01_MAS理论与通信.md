# Multi-Agent 理论与通信

MAS 中每个 Agent 有局部观察、目标、能力和策略。系统行为来自通信协议、组织结构和激励机制，而非简单增加模型实例。

## 组织与协议

| 结构 | 适合场景 | 风险 |
|---|---|---|
| 层级式 | 需要统一目标与责任链 | 协调者瓶颈 |
| 对等式 | 分布式探索、容错 | 共识与重复劳动 |
| 黑板/共享状态 | 需要异步协作 | 状态冲突与权限泄露 |
| 市场/竞标 | 动态任务分配 | 激励错配与操纵 |

通信可交换消息、信念、承诺、证据或可执行产物。高质量协议需定义消息模式、来源、置信度、权限、时效、确认与冲突解决；自然语言对话本身不提供这些保证。

## References

- Wooldridge, *An Introduction to MultiAgent Systems*.
- FIPA Agent Communication Language Specification.
