# Agent安全综述

## 领域定义

Agent 安全研究当 LLM 被赋予工具调用、记忆积累、外部资源访问和持续自主行动能力时，如何防止其产生越权、不可逆或不可预期的后果。它是模型安全在"执行层"的延伸。

## 为什么会出现

单纯的语言模型只生成文本，危害有限；一旦赋予工具调用能力，风险从"生成错误文本"升级为"产生越权或不可逆后果"：
- 发邮件、访问账户、执行代码、操作文件、调用 API
- 读取检索内容时可能被外部注入指令
- 长期记忆积累后，早期污染影响持续

同时，LLM 作为推理核心的特殊性使传统访问控制不够：模型可能被说服"理解"了规则而获得权限，而不经过外部策略验证。

## 发展历史

| 阶段 | 重点 | 代表事件 |
|---|---|---|
| 工具调用原型 (2022–) | ReAct、Toolformer，初步工具安全意识 | ReAct (Yao 2022)、Toolformer (2023) |
| Agent 攻击研究 (2023–) | 提示注入分类化、AgentBench | Indirect Prompt Injection (Greshake 2023)、AgentBench |
| 行业安全实践 (2023–) | OWASP LLM Top 10、MCP 协议安全 | OWASP 2023、Anthropic MCP (2024) |
| Agent 评测体系 (2024–) | AgentHarm、MACHIAVELLI、行动轨迹评测 | AgentHarm (2024)、MACHIAVELLI |

## 核心攻击面

```text
不可信输入 → 解析/标记来源 → 策略决策 → 权限校验 → 受限执行 → 审计/监控 → 可撤销恢复
```

模型不得自行扩大权限；高影响行动应绑定可验证前置条件、人类批准或可逆事务机制。

## 重要分支

- [[01_工具权限与执行边界]]：最小权限、能力令牌、审批、沙箱与幂等执行
- [[02_提示注入与上下文隔离]]：直接/间接注入、数据-指令分离和外部内容净化
- [[03_记忆目标与多智能体安全]]：记忆投毒、目标漂移、委派风险与协作治理
- [[04_Agent评测与运行时防护]]：场景测试、轨迹审计、监控、熔断和事件响应

## 学习路径

1. 从 [[../03_模型与生成式AI安全/02_生成式AI攻防]] 理解提示注入的基础
2. 学习 [[01_工具权限与执行边界]] 建立执行层防御的基本设计
3. 学习 [[02_提示注入与上下文隔离]] 了解 Agent 场景下的注入路径
4. 学习 [[03_记忆目标与多智能体安全]] 了解长程和多体场景的新风险
5. 学习 [[04_Agent评测与运行时防护]] 了解如何测试和监控

## 当前发展状态

- **间接提示注入**已从理论转为实际 PoC，RAG 系统和网页浏览 Agent 均有公开案例
- **MCP（Model Context Protocol）**标准化了工具调用接口，带来统一的权限设计机会
- **Agent 评测基准**（AgentBench、AgentHarm）开始覆盖安全维度的工具调用行为
- **多智能体系统**的安全性研究仍在早期，责任归因和通信安全尚无通行标准
- 行业实践仍缺乏统一的 Agent 权限设计规范，大量系统依赖模型"自律"而非外部策略

## 与其他方向的关系

- [[../03_模型与生成式AI安全/00_模型与生成式AI安全_综述]]：本方向是其在 Agent 执行层的延伸
- [[../02_AI对齐理论/00_AI对齐理论_综述]]：对齐约束了Agent的目标，但不能替代执行层权限控制
- [[../06_安全评测与安全工程/00_安全评测与安全工程_综述]]：Agent 安全需要专门的轨迹评测方法
- [[../09_超级智能安全/00_超级智能安全_综述]]：Agent 权限控制是超级智能遏制问题的实践预演

## References

- Greshake, K. et al. "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection." arXiv:2302.12173, 2023.
- Liu, Y. et al. "AgentBench: Evaluating LLMs as Agents." ICLR 2024. arXiv:2308.03688
- Debenedetti, E. et al. "AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents." arXiv:2410.09024, 2024.
- OWASP Top 10 for LLM Applications, 2023. https://owasp.org/www-project-top-10-for-large-language-model-applications/
- Perez, E. et al. "Ignore Previous Prompt: Attack Techniques For Language Models." NeurIPS 2022 ML Safety Workshop. arXiv:2211.09527
