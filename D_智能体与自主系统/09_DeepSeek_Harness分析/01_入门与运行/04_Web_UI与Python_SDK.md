# Web UI 与 Python SDK

## 一句话理解

Web UI 适合交互式观察、审批和工作区操作；Python SDK 适合把同一套 Agent 组合嵌入自动化程序，两者都以会话日志作为可延续状态的依据。

## Web UI

启动服务后，在界面中先配置模型，再添加并选中工作区。输入框只会在工作区选中后启用；任务执行过程中，UI 根据会话事件渲染流式文本、工具活动与审批请求。

## Python SDK

SDK 的 `DeepSeekHarness` 在上下文管理器内复用内置运行时。`cwd` 决定可操作工作区，`session_root` 保存 JSONL 会话日志；复用同一 session id 会延续对话与其拥有的持久 Bash 状态，独立任务应使用新 id。

```python
with DeepSeekHarness(cwd=str(workspace), session_root=str(sessions), cordis=str(config)) as harness:
    result = harness.run("Inspect the repository.", session_id="task-001")
```

## 安全边界

官方最小 Python 组合使用裸本地文件系统和 `danger-full-access`，Bash 与编辑器可访问运行时进程有权访问的路径。它只适合可丢弃 checkout 或容器，不应直接连接生产主机或敏感目录。

## 相关知识

- [[02_安装启动与运行配置]]
- [[03_Agent运行时与会话/02_会话事件日志与模型历史]]
- [[06_安全可靠性与工程化/01_审批权限与沙箱]]

## References

- `D:\_Projects\deepseek-harness\docs\user\guide\index.zh.md`
- `D:\_Projects\deepseek-harness\docs\user\guide\python-sdk.zh.md`
