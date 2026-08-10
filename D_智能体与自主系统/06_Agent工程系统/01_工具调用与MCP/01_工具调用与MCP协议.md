# 工具调用与MCP协议

## 1. 概述

工具调用（Tool Use / Function Calling）使 `大语言模型` 能够调用外部API和工具，是从"聊天机器人"到 `Agent` 的关键能力跃迁。LLM不直接执行操作，而是根据用户意图选择合适的工具并生成结构化调用参数，由外部执行器执行后返回结果。

MCP（Model Context Protocol）是Anthropic于2024年提出的开放标准，旨在统一LLM与外部工具/数据源的连接方式，解决工具集成碎片化问题——一个协议连接所有工具，避免N×M集成难题。

## 2. 发展历史

| 时间 | 里程碑 | 意义 |
|------|--------|------|
| 2023.03 | OpenAI Function Calling | LLM原生工具调用能力，GPT-4首次支持 |
| 2023.05 | Toolformer（Schick et al.） | LLM自学习何时、如何使用工具 |
| 2023.06 | OpenAI JSON Mode | 强制输出合法JSON |
| 2023.08 | Anthropic Tool Use | Claude原生工具调用 |
| 2023.10 | 并行工具调用 | 单次返回多个工具调用，并行执行 |
| 2024.06 | OpenAI Structured Output | 严格Schema约束，100%格式合规 |
| 2024.11 | MCP协议发布（Anthropic） | 工具调用标准化，解耦工具与应用 |
| 2024.10 | Computer Use（Anthropic） | Agent可直接操作GUI界面 |
| 2024.12 | OpenAI Operator | 浏览器操作Agent |
| 2025 | MCP生态爆发 | 社区共享MCP Server，工具复用生态形成 |
| 2025 | Computer Use 2.0 | GUI操作精度和速度大幅提升 |

## 3. 核心概念

### 工具调用流程

```
1. 用户请求 → LLM分析意图
2. LLM选择工具，生成调用参数（JSON）
3. 执行器调用实际API
4. 返回结果给LLM
5. LLM基于结果继续推理或生成最终回答
```

### 关键术语

- **Function Calling**：LLM根据函数描述选择并调用函数的能力
- **Tool Schema**：工具的定义，包括名称、描述、参数类型（JSON Schema）
- **Tool Executor**：接收LLM生成的调用参数并实际执行的外部组件
- **MCP Server**：暴露工具和数据给LLM的服务端
- **MCP Client**：LLM应用中的MCP连接器
- **Computer Use**：LLM操作GUI界面（点击、输入、截图）的能力
- **Parallel Tool Calling**：单次返回多个工具调用，并行执行

### Function Calling vs MCP

| 维度 | Function Calling | MCP |
|------|-----------------|-----|
| 定位 | 模型能力（LLM选择工具） | 通信协议（标准化工具连接） |
| 层级 | 模型层 | 协议层 |
| 标准化 | 各厂商API不同 | 统一开放标准 |
| 工具开发 | 与应用耦合 | 独立开发，解耦 |
| 复用性 | 低（每应用重新定义） | 高（MCP Server可共享） |

## 4. 技术原理

### Function Calling

LLM根据工具描述选择工具并生成调用参数：

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "北京天气如何？"}],
    tools=tools
)

# LLM返回工具调用决策
tool_call = response.choices[0].message.tool_calls[0]
# {"name": "get_weather", "arguments": {"city": "北京"}}

# 执行工具调用
result = execute_tool(tool_call)

# 将结果返回给LLM继续生成
final_response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "北京天气如何？"},
        response.choices[0].message,
        {"role": "tool", "tool_call_id": tool_call.id, "content": result}
    ]
)
```

### 并行工具调用

现代LLM支持单次返回多个工具调用，并行执行以提升效率：

```
LLM → [get_weather("北京"), get_news("科技"), search_db("用户数据")]
   → 并行执行 → 汇总结果 → LLM生成回答
```

### MCP（Model Context Protocol）

MCP采用客户端-服务端架构，标准化LLM与外部资源的连接：

```
LLM应用 (MCP Client) ←→ MCP协议 ←→ MCP Server ←→ 数据源/工具
                                         ↓
                                    Resources（数据源）
                                    Tools（可调用函数）
                                    Prompts（提示模板）
```

**MCP核心概念**：

| 概念 | 作用 | 示例 |
|------|------|------|
| MCP Server | 暴露工具和数据给LLM | 文件系统Server、数据库Server |
| MCP Client | LLM应用中的MCP连接器 | Claude Desktop中的MCP客户端 |
| Resources | 可读取的数据源 | 文件、数据库、API |
| Tools | 可调用的函数/API | 搜索、代码执行、数据库操作 |
| Prompts | 预定义的提示模板 | 代码审查模板、分析模板 |
| Sampling | Server请求LLM生成 | Server端需要LLM辅助时使用 |

**MCP核心优势**：

- **标准化**：一个协议连接所有工具，避免N×M集成问题
- **解耦**：工具开发与LLM应用独立演进
- **生态**：社区共享MCP Server，工具复用
- **安全**：工具权限由Client控制，Server不直接访问LLM

### Computer Use

Agent通过截图+坐标点击的方式操作GUI界面：

```
1. 截取屏幕截图
2. LLM分析截图，识别UI元素
3. LLM生成操作指令（点击坐标、输入文本、按键）
4. 执行器执行操作
5. 截取新截图，观察结果
6. 循环直到任务完成
```

### 工具定义最佳实践

| 原则 | 说明 |
|------|------|
| 描述清晰 | LLM根据description选择工具，描述要准确完整 |
| 参数Schema严格 | 用JSON Schema约束参数类型和取值范围 |
| 错误处理 | 工具失败时返回有用的错误信息，帮助LLM调整策略 |
| 工具数量适中 | 过多工具降低选择准确率，建议<20个 |
| 语义化命名 | 函数名和参数名应自解释 |
| 提供示例 | 在description中包含调用示例 |

## 5. 关键方法与模型

| 方法/协议 | 核心贡献 | 来源 |
|-----------|---------|------|
| Function Calling | LLM原生工具调用 | OpenAI, 2023 |
| Tool Use | 通用工具调用能力 | Anthropic, 2023 |
| Toolformer | LLM自学习工具使用 | Schick et al., 2023 |
| MCP | 工具调用标准化协议 | Anthropic, 2024 |
| Computer Use | LLM操作GUI界面 | Anthropic, 2024 |
| Parallel Tool Calling | 并行工具调用 | OpenAI, 2024 |
| Structured Output | 严格Schema约束 | OpenAI, 2024 |

## 6. 优势与局限

### 优势

- **能力扩展**：LLM从纯文本生成扩展到执行实际操作
- **实时信息**：通过工具获取实时数据（搜索、数据库查询）
- **精确计算**：通过代码执行器解决LLM数学计算不准的问题
- **系统集成**：通过API调用集成到企业系统
- **标准化**：MCP协议使工具开发与LLM应用解耦

### 局限

- **工具选择错误**：LLM可能选择错误工具或生成错误参数
- **安全风险**：工具执行可能带来安全隐患（误操作、数据泄露）
- **延迟增加**：工具调用增加端到端延迟
- **依赖外部服务**：工具可用性影响整体系统可靠性
- **成本**：多次LLM调用+工具执行的成本累积
- **调试复杂**：工具调用链路长，问题定位困难

## 7. 应用场景

- `Agent系统`：工具调用是Agent的核心能力
- `RAG`：RAG检索可作为工具被动态调用
- **API编排**：LLM根据需求选择和组合多个API
- **代码执行**：生成并执行代码进行计算验证
- **数据库查询**：自然语言转SQL并执行
- `自动化工作流`：多工具组合完成复杂任务
- **Computer Use**：Agent操作浏览器、桌面应用
- **文件操作**：读写文件、目录管理

## 8. 与其他技术关系

- 工具调用是 `Agent系统` 的行动能力
- `结构化输出` 是工具调用的基础（工具参数依赖结构化输出）
- MCP标准化了Agent与工具的连接
- `提示工程` 影响工具选择的准确性
- 工具调用是 `自动化工作流` 的执行层核心
- `RAG` 可作为工具被Agent调用
- Computer Use扩展了Agent的操作范围到GUI

## 9. 前沿发展

- **MCP生态化**：社区共享MCP Server，形成工具复用生态，类似npm/pip的包管理
- **Computer Use 2.0**：GUI操作精度和速度大幅提升，向通用自动化迈进
- **自主工具发现**：Agent自主发现和学习新工具，无需预定义工具列表
- **工具组合推理**：LLM不仅选择单个工具，还能推理出工具组合策略
- **安全工具执行**：沙箱化工具执行、权限分级、行为审计
- **工具调用优化**：减少工具调用次数、并行执行、结果缓存
- **多模态工具调用**：工具不仅接受文本参数，还接受图像、视频等多模态输入
- **工具调用标准化**：MCP之外，更多标准化方案出现（如OpenAI的Function Calling标准化）
- **端侧工具调用**：在设备端执行工具调用，保护隐私和降低延迟

## 相关知识

- 前置：[[01_提示工程|提示工程]]
- 平级：[[02_结构化输出|结构化输出]]、[[04_Agent系统/00_Agent系统|Agent系统]]
- 延伸：[[06_自动化工作流|自动化工作流]]、[[../18_模型部署与工程化/01_模型服务化|模型服务化]]
