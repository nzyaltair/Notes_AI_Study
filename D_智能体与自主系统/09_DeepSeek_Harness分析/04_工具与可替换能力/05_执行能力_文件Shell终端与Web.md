# 执行能力：文件、Shell、终端与 Web

## 一句话理解

文件、命令、持久终端和 Web 访问是不同的 capability seam；它们可共享执行世界，但安全策略、生命周期与结果形态不可混为一谈。

## 能力分工

| 能力 | 适合的任务 | 关键约束 |
|---|---|---|
| 文件系统 | 读取、写入、结构化编辑 | 读后编辑、路径与写意图策略 |
| Shell / 子进程 | 一次性命令和脚本 | 清理环境变量、取消并等待退出 |
| PTY 终端 | 需要持续状态的交互式命令 | 会话所有权、有限读取、POSIX 环境 |
| Web | 搜索和抓取外部信息 | 提供方可用性、网络错误和内容边界 |
| LSP | 代码导航与语言服务查询 | 将语言服务作为独立 Provider |

## 选择原则

一次性命令用 Shell；需保留 cwd、导出变量和交互状态才用终端；模型不应直接获得比 Tool 契约更广的底层能力。后端从本地改为远程沙箱时，应通过 Provider 替换保持 Consumer 不变。

## 相关知识

- [[03_Capability_Seam_设计模式]]
- [[02_工具执行流水线与策略钩子]]
- [[06_安全可靠性与工程化/01_审批权限与沙箱]]

## References

- `D:\_Projects\deepseek-harness\docs\subsystems\filesystem.zh.md`
- `D:\_Projects\deepseek-harness\docs\subsystems\shell.zh.md`
- `D:\_Projects\deepseek-harness\docs\subsystems\terminal.zh.md`
- `D:\_Projects\deepseek-harness\docs\subsystems\web.zh.md`
