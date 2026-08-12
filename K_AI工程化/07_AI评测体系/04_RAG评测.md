---
tags:
  - AI评测
  - RAG评测
  - 检索质量
  - 生成质量
  - AI工程
created: 2026-08-10
updated: 2026-08-10
---

# RAG 评测

## 一句话理解

RAG 评测将系统拆分为检索和生成两段独立评估——检索看是否找到了正确的信息，生成看是否忠实地使用了检索到的信息，端到端分数不能掩盖任何一段的缺陷。

## 1. RAG 评测的独特挑战

### 1.1 为什么 RAG 需要专门评测

| 挑战 | 说明 |
|:---|:---|
| **两段式架构** | 检索和生成都可能出错，需独立定位 |
| **依赖外部数据** | 知识库质量直接影响系统质量 |
| **动态性** | 知识库更新、文档变更影响结果 |
| **多组件耦合** | Chunking、Embedding、重排、Prompt 任一变更都需回归 |
| **幻觉与引用** | 模型可能忽略检索结果或编造引用 |

### 1.2 RAG 评测层次

```text
┌─────────────────────────────────────────────┐
│             RAG 评测三层架构                  │
├─────────────────────────────────────────────┤
│                                             │
│  层次 1：组件级评测                          │
│  ├── Embedding 质量（检索召回率）            │
│  ├── Chunking 策略（边界合理性）             │
│  └── Reranker 效果（排序提升）               │
│                                             │
│  层次 2：管道级评测                          │
│  ├── 检索质量（召回/精确/排序）              │
│  ├── 生成质量（忠实/相关/完整）              │
│  └── 引用质量（引用准确/覆盖）               │
│                                             │
│  层次 3：端到端评测                          │
│  ├── 用户满意度                              │
│  ├── 任务完成率                              │
│  └── 延迟与成本                              │
│                                             │
└─────────────────────────────────────────────┘
```

## 2. 检索质量评测

### 2.1 核心指标

| 指标 | 定义 | 适用场景 |
|:---|:---|:---|
| **Recall@k** | 前 k 个结果中包含正确文档的比例 | 评估是否"找得到" |
| **Precision@k** | 前 k 个结果中正确文档的比例 | 评估结果"干不干净" |
| **MRR** | 第一个正确结果的倒数排名 | 单一正确答案场景 |
| **NDCG@k** | 考虑排序位置的归一化增益 | 多相关文档场景 |
| **Hit Rate@k** | 至少有一个正确结果的比例 | 是否命中 |
| **Context Recall** | 检索内容覆盖参考答案的比例 | RAG 专用 |

### 2.2 检索评测数据集

```text
检索评测集构建：
├── Query 集
│   ├── 来源：真实用户日志 / 人工构造 / LLM 生成
│   ├── 类型：事实型/推理型/比较型/多跳
│   └── 难度：简单（直接匹配）→ 困难（需推理）
│
├── 相关性标注
│   ├── 文档级：相关 / 部分相关 / 不相关
│   ├── 片段级：标注具体相关 chunk
│   └── 多标注者一致性检查
│
└── 查询变体
    ├── 同义改写
    ├── 拼写错误
    ├── 口语化表达
    └── 多语言
```

### 2.3 Embedding 质量评估

| 评估维度 | 方法 | 指标 |
|:---|:---|:---|
| **语义相似度** | STS-B / MTEB | Spearman 相关 |
| **检索能力** | 给定 query 找正确 doc | Recall@k |
| **跨语言** | 多语言 query-doc 对 | 跨语言 Recall |
| **领域适应** | 领域专用语料 | 领域 Recall |
| **长文本** | 长 doc 的检索 | Long Context Recall |

### 2.4 Reranker 评估

```python
# Reranker 效果评估示例
{
    "before_rerank": {
        "recall@10": 0.85,
        "precision@5": 0.60,
        "ndcg@10": 0.72
    },
    "after_rerank": {
        "recall@10": 0.85,       # Recall 不变（只是重排序）
        "precision@5": 0.82,     # 精确率提升
        "ndcg@10": 0.88          # 排序质量提升
    },
    "improvement": {
        "precision@5": "+22pp",
        "ndcg@10": "+16pp"
    },
    "latency_added_ms": 45       # 重排延迟
}
```

## 3. 生成质量评测

### 3.1 核心维度

| 维度 | 定义 | 指标 |
|:---|:---|:---|
| **Faithfulness（忠实度）** | 答案是否基于检索内容，无幻觉 | 引用准确率、幻觉率 |
| **Relevance（相关性）** | 答案是否回答了用户问题 | 语义相关分 |
| **Correctness（正确性）** | 答案是否事实正确 | 准确率 |
| **Completeness（完整性）** | 是否覆盖了所有要点 | 覆盖率 |
| **Citation（引用质量）** | 引用来源是否准确 | 引用验证率 |

### 3.2 RAG 专用指标

| 指标 | 提出者 | 评测内容 |
|:---|:---|:---|
| **Faithfulness** | RAGAS | 答案中的每个声明是否可由检索上下文支持 |
| **Answer Relevancy** | RAGAS | 答案与问题的相关程度 |
| **Context Precision** | RAGAS | 检索上下文中相关内容的比例 |
| **Context Recall** | RAGAS | 检索上下文覆盖参考答案的比例 |
| **Context Entities Recall** | RAGAS | 检索上下文覆盖参考答案实体的比例 |
| **Answer Similarity** | RAGAS | 答案与参考答案的语义相似度 |
| **Answer Correctness** | RAGAS | 答案的事实正确性 |

### 3.3 RAGAS 评测框架

```text
RAGAS 评测流程：

输入：Question, Answer, Context(s), Ground Truth
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
   Faithfulness  Context   Answer
   (LLM Judge)   Metrics   Relevancy
        │           │           │
        └───────────┼───────────┘
                    │
                    ▼
            综合评分报告

Faithfulness 计算逻辑：
1. 将答案拆分为独立声明（claims）
2. 逐个检查每个 claim 是否可由 context 支持
3. Faithfulness = 被支持的 claims / 总 claims
```

### 3.4 幻觉检测

| 幻觉类型 | 说明 | 检测方法 |
|:---|:---|:---|
| **上下文外幻觉** | 答案包含检索上下文中没有的信息 | 逐句对比上下文 |
| **事实性幻觉** | 答案与事实不符 | 外部知识验证 |
| **矛盾幻觉** | 答案内部自相矛盾 | 一致性检查 |
| **引用幻觉** | 引用了不存在的来源 | 引用验证 |

## 4. 端到端评测

### 4.1 端到端指标

| 指标 | 定义 | 说明 |
|:---|:---|:---|
| **End-to-End Accuracy** | 端到端答案正确率 | 综合指标 |
| **Task Completion** | 用户任务是否完成 | 业务指标 |
| **Latency** | 首字延迟 + 完整延迟 | 用户体验 |
| **Cost per Query** | 每次查询的成本 | 经济指标 |
| **Token Efficiency** | 检索 token 利用率 | 成本优化 |

### 4.2 分段诊断

```text
RAG 问题诊断决策树：

端到端分数低？
├── 检索分数低？
│   ├── Recall 低 → 改善 Chunking / Embedding / 增大 k
│   ├── Precision 低 → 增加 Reranker / 减小 k
│   └── 排序差 → 优化 Reranker / 调整分数融合
│
├── 生成分数低？
│   ├── Faithfulness 低 → Prompt 约束 / 换更遵循指令的模型
│   ├── Relevance 低 → 改善 Prompt / 增加 query 改写
│   └── Correctness 低 → 检查检索内容质量 / 模型能力
│
└── 引用质量低？
    ├── 引用缺失 → Prompt 要求引用 / 后处理添加
    ├── 引用错误 → 引用验证 / 后处理校验
    └── 引用冗余 → Prompt 优化 / 引用精简
```

## 5. 评测数据集构建

### 5.1 数据集要素

```text
RAG 评测集完整要素：
{
    "query_id": "q_001",
    "query": "公司2024年Q3的营收是多少？",
    "query_type": "factoid",          // factoid/reasoning/comparative/multi-hop
    "ground_truth": "15.2亿元",       // 参考答案
    "ground_truth_chunks": ["doc_03_chunk_07"],  // 正确 chunk ID
    "difficulty": "medium",
    "requires_multi_hop": false,
    "expected_citations": ["doc_03"],
    "metadata": {
        "domain": "finance",
        "language": "zh",
        "created": "2024-12-01"
    }
}
```

### 5.2 Query 生成方法

| 方法 | 优势 | 局限 |
|:---|:---|:---|
| **人工编写** | 质量高，贴合业务 | 成本高，规模有限 |
| **用户日志挖掘** | 真实分布 | 需脱敏，覆盖窄 |
| **LLM 生成** | 可大规模，可控制类型 | 可能不自然 |
| **文档逆向生成** | 从答案生成问题 | 可能过于简单 |
| **合成+人工审核** | 平衡规模和质量 | 审核成本 |

### 5.3 多跳与推理 Query

- **多跳推理**：需要组合多个文档的信息
- **比较型**：比较多个实体的属性
- **时间推理**：理解时间关系
- **数值推理**：需要计算
- **否定推理**：处理"不包含""除了"等

## 6. 评测工具与框架

| 工具 | 特点 | 适用场景 |
|:---|:---|:---|
| **RAGAS** | 开源 RAG 评测框架 | 通用 RAG 评测 |
| **TruLens** | RAG 追踪与评测 | 开发期调试 |
| **LangSmith** | LangChain 生态评测 | LangChain 用户 |
| **DeepEval** | 单元测试式评测 | CI/CD 集成 |
| **Arize Phoenix** | 可观测+评测 | 生产监控 |

## 7. 回归测试策略

### 7.1 变更触发回归

```text
RAG 变更回归矩阵：

变更类型              必须回归的层次
──────────────────────────────────────
Chunking 策略变更     → 检索 + 端到端
Embedding 模型替换    → 检索 + 端到端
Reranker 调整         → 检索 + 端到端
LLM 模型替换          → 生成 + 端到端
Prompt 修改           → 生成 + 端到端
知识库更新            → 检索 + 端到端
检索参数调整(k等)     → 检索 + 端到端
```

### 7.2 金标集管理

- **金标集规模**：100-500 个精选标注样本
- **覆盖度**：覆盖所有 query 类型和难度
- **更新频率**：每月补充新 case，每季全面审核
- **版本化**：每次变更前用同一金标集对比

## 8. 子主题导航

- [[01_模型能力与LLM Benchmark]]
- [[03_Agent评测]]
- [[07_在线评测与A_B测试]]
- [[08_评测方法论]]

## 9. 相关知识

- [[00_AI评测体系_综述]]
- [[../04_AI系统架构/02_RAG系统架构]]
- [[../04_AI系统架构/06_检索系统与Embedding]]
- [[../06_LLMOps与AgentOps/03_RAG与Embedding管理]]

## References

- Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation* (2023)
- Chen et al., *Benchmarking Large Language Models in Retrieval-Augmented Generation* (2024)
- Gao et al., *Retrieval-Augmented Generation for Large Language Models: A Survey* (2024)
- Wang et al., *Searching for Best Practices in Retrieval-Augmented Generation* (2024)
