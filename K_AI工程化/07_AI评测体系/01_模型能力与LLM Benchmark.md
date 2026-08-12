---
tags:
  - AI评测
  - LLM Benchmark
  - 能力评测
  - 模型评测
  - AI工程
created: 2026-08-10
updated: 2026-08-10
---

# 模型能力与 LLM Benchmark

## 一句话理解

LLM Benchmark 是用标准化测试集对模型的通用能力（知识、推理、代码、数学、长上下文等）进行量化打分的尺子，用于横向比较和版本回归，但它只能反映"考试能力"，不能替代真实业务场景验证。

## 1. 核心能力维度

### 1.1 知识与推理

| 基准 | 测量能力 | 题型 |
|:---|:---|:---|
| **MMLU** | 大学水平多学科知识 | 57 个学科多项选择 |
| **MMLU-Pro** | MMLU 增强版，更难更少猜测 | 10 选项，需推理 |
| **HellaSwag** | 常识推理 | 选择最合理续写 |
| **ARC** | 科学推理 | 小学科学多项选择 |
| **BBH (Big-Bench Hard)** | 高难度推理 | 23 个挑战任务 |
| **GPQA** | 研究生水平问答 | 物理/化学/生物 |

### 1.2 数学

| 基准 | 测量能力 | 特点 |
|:---|:---|:---|
| **GSM8K** | 小学应用题 | 8,500 题，多为 2-8 步 |
| **MATH** | 竞赛数学 | 高中竞赛水平，含证明 |
| **MathVista** | 视觉数学 | 图表/几何/函数图像 |

### 1.3 代码

| 基准 | 测量能力 | 评测方式 |
|:---|:---|:---|
| **HumanEval** | 函数级代码生成 | 执行单元测试 pass@k |
| **MBPP** | 基础 Python 编程 | 974 道入门题 |
| **HumanEval+ / MBPP+** | 增强测试覆盖 | 更多边缘 case |
| **SWE-Bench** | 真实仓库级修 Bug | Docker 环境执行测试 |
| **LiveCodeBench** | 最新竞赛题 | 防数据污染 |
| **MultiPL-E** | 多语言代码生成 | 18 种语言 |

### 1.4 长上下文

| 基准 | 测量能力 | 特点 |
|:---|:---|:---|
| **LongBench** | 长文本理解 | 中英双语，13 任务 |
| **∞Bench** | 超长上下文 | 平均 100K+ token |
| **RULER** | 长上下文综合 | 可变长度生成 |
| **Needle in a Haystack** | 长文中检索特定信息 | 不同位置插入"针" |

### 1.5 指令遵循

| 基准 | 测量能力 | 特点 |
|:---|:---|:---|
| **IFEval** | 指令格式遵循 | 可验证规则指令 |
| **MT-Bench** | 多轮对话 | GPT-4 评分 |
| **AlpacaEval** | 指令跟随质量 | 对比胜率 |

### 1.6 多语言

| 基准 | 覆盖语言 | 特点 |
|:---|:---|:---|
| **MGSM** | 数学应用题 | 250 题 × 11 语言 |
| **XCOPA** | 常识推理 | 11 语言 |
| **FLORES** | 翻译 | 200+ 语言 |

## 2. Benchmark 分类体系

### 2.1 按评测方式分

```text
客观评测（Exact Match / 执行结果）
├── 选择题：MMLU, ARC, HellaSwag
├── 数学：GSM8K (数字匹配)
└── 代码：HumanEval (测试通过率)

主观评测（LLM Judge / 人工）
├── 对比：MT-Bench, AlpacaEval
├── 打分：G-Eval, FairEval
└── Elo 排名：Chatbot Arena

综合评测框架
├── HELM：整体性评测，多维度归一化
├── OpenCompass：开源评测工具链
└── lm-evaluation-harness：HuggingFace 评测框架
```

### 2.2 按任务类型分

| 类型 | 典型基准 | 指标 |
|:---|:---|:---|
| 判别式 | MMLU, ARC | Accuracy |
| 生成式 | HumanEval, GSM8K | pass@k, EM |
| 对话式 | MT-Bench | Elo, Win Rate |
| 检索式 | LongBench | F1, EM |

## 3. 评测执行规范

### 3.1 提示工程

- **Few-shot vs Zero-shot**：MMLU 默认 5-shot，HumanEval 默认 zero-shot
- **Chain-of-Thought**：推理任务是否允许 CoT 影响巨大
- **指令格式**：不同模型对 prompt 格式敏感度不同
- **建议**：报告所有 prompt 模板，使用各模型推荐格式

### 3.2 采样参数

```python
# 标准评测采样配置示例
{
    "temperature": 0.0,    # 贪心解码，保证可复现
    "top_p": 1.0,
    "max_tokens": 1024,
    "n": 1,               # 如需 pass@k 可设置 n=k
    "seed": 42
}
```

- **温度**：客观题用 T=0；生成任务可适当提高
- **pass@k**：生成 k 个样本，至少一个通过即算成功
- **重复次数**：T>0 时需多次运行报告均值和方差

### 3.3 评分协议

| 任务类型 | 评分方式 | 注意事项 |
|:---|:---|:---|
| 选择题 | 提取 A/B/C/D | 处理"答案是 B" vs "B" |
| 数学 | 数字提取匹配 | 处理分数、单位、LaTeX |
| 代码 | 执行测试用例 | 沙箱隔离，超时控制 |
| 生成 | LLM Judge | 校准 judge 偏见 |

## 4. 数据污染与防泄漏

### 4.1 污染检测方法

- **N-gram 重叠检测**：检查评测集与预训练语料的 n-gram 重叠
- **困惑度异常**：模型对测试题的困惑度异常低 → 可能见过
- **时间戳分割**：只用训练截止日期后的新题
- **变体测试**：修改题目数字/人名后观察性能是否骤降

### 4.2 缓解策略

```text
┌─────────────────────────────────────────────┐
│             防污染策略                        │
├─────────────────────────────────────────────┤
│  1. 维护私有评测集（不公开）                   │
│  2. 定期刷新公开基准                          │
│  3. 使用动态生成基准（如 LiveCodeBench）       │
│  4. 构造等价变体题                            │
│  5. 对比 train/test 性能差异                  │
└─────────────────────────────────────────────┘
```

## 5. 常见陷阱与误区

### 5.1 Benchmark 榜单的局限

| 陷阱 | 说明 | 应对 |
|:---|:---|:---|
| **分数崇拜** | 只追求榜单分数，忽略真实能力 | 关注业务指标 |
| **数据污染** | 训练时见过测试题 | 污染检测 + 私有集 |
| **提示敏感性** | 同一模型不同 prompt 差异大 | 标准化 prompt |
| **采样偏差** | 只报最好的一次 | 报告均值+方差 |
| **评估偏差** | LLM Judge 偏爱某模型 | 多 judge 交叉验证 |
| **基准饱和** | 分数已接近上限 | 引入更难基准 |

### 5.2 真实能力 vs 考试能力

- **能力引出（Capability Elicitation）**：好的 prompt 可能"引出"模型已有能力，但不代表模型在其他场景也能做到
- **真实能力（True Capability）**：模型在无特殊提示、无 few-shot 下的自然表现
- **建议**：同时报告 zero-shot 和优化 prompt 的结果

## 6. 主流评测框架对比

| 框架 | 维护方 | 特点 | 适用场景 |
|:---|:---|:---|:---|
| **lm-eval-harness** | HuggingFace/EleutherAI | 200+ 基准，社区驱动 | 学术研究 |
| **OpenCompass** | 上海 AI Lab | 中文友好，模型适配全 | 中文模型评测 |
| **HELM** | Stanford | 整体性评测，归一化 | 系统性对比 |
| **BIG-bench** | Google | 200+ 任务，含挑战题 | 极限能力测试 |
| **EvalPlus** | 北京大学 | 代码评测增强 | 代码能力评测 |

## 7. 评测报告最佳实践

一份可信的模型评测报告应包含：

1. **模型信息**：版本、参数量、训练截止日期
2. **评测集信息**：版本号、样本数、是否公开
3. **提示模板**：使用的完整 prompt
4. **采样参数**：温度、top_p、max_tokens、n
5. **评分方式**：自动/人工/LLM Judge 及具体协议
6. **统计信息**：均值、标准差、置信区间
7. **复现方式**：代码仓库和配置文件
8. **失败案例**：典型错误分析
9. **污染检测**：检测结果和方法

## 8. 子主题导航

- [[02_多模态评测]]
- [[03_Agent评测]]
- [[04_RAG评测]]
- [[06_人类偏好评测]]
- [[08_评测方法论]]

## 9. 相关知识

- [[00_AI评测体系_综述]]
- [[../03_推理工程/05_量化与模型压缩]]（量化对 benchmark 的影响）
- [[../01_数据工程与合成数据/08_数据质量评估]]（评测集质量保障）

## References

- Hendrycks et al., *Measuring Massive Multitask Language Understanding* (MMLU, 2021)
- Chen et al., *Evaluating Large Language Models Trained on Code* (HumanEval, 2021)
- Cobbe et al., *Training Verifiers to Solve Math Word Problems* (GSM8K, 2021)
- Bai et al., *HELM: Holistic Evaluation of Language Models* (2022)
- Suzgun et al., *BBH: Challenging BIG-Bench Tasks* (2022)
- Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* (2024)
