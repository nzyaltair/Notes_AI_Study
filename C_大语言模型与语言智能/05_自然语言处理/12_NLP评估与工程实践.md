# NLP 评估与工程实践

## 1. 概述

评估与工程实践是 NLP 从研究走向应用的关键环节。评估指标衡量模型性能，工具链支撑开发效率，部署方案决定可用性。本章覆盖 NLP 中常用的评估指标、主流工具框架、工程实践流程和部署注意事项。

NLP 评估经历了从单一指标（BLEU/ROUGE）到多维度评估（事实一致性、流畅性、有用性）的演进。大模型时代，传统自动指标面临挑战——LLM 生成质量难以用 n-gram 匹配衡量，催生了基于模型的评估方法（如 LLM-as-a-Judge）和更全面的人类评估框架。

## 2. 发展历史

| 年代 | 里程碑 | 作者 | 意义 |
|:---|:---|:---|:---|
| 2002 | BLEU | Papineni et al. | 机器翻译自动评估 |
| 2003 | ROUGE | Lin | 摘要评估指标 |
| 2004 | NIST | Doddington | 改进的 MT 评估 |
| 2014 | SQuAD | Rajpurkar et al. | 阅读理解评估基准 |
| 2015 | GLUE | Wang et al. | NLP 多任务评估基准 |
| 2018 | spaCy 2.0 | Explosion | 工业级 NLP 工具成熟 |
| 2019 | Transformers | Hugging Face | 预训练模型生态 |
| 2020 | SuperGLUE | Wang et al. | 更难的 NLP 基准 |
| 2020 | BARTScore | Yuan et al. | 基于模型的生成评估 |
| 2021 | BLEURT | Sellam et al. | 预训练评估模型 |
| 2022 | HELM | Stanford | 全面 LLM 评估框架 |
| 2022 | Chatbot Arena | LMSYS | 人类偏好评估 |
| 2023 | MT-Bench / Arena | — | LLM-as-a-Judge 评估 |
| 2024+ | 多维度评估 | — | 事实性、安全性、有用性 |

## 3. 核心概念

### 3.1 评估方法分类

| 方法 | 说明 | 适用场景 |
|:---|:---|:---|
| 自动评估 | 算法计算指标 | 快速迭代、大规模评估 |
| 人工评估 | 人类打分 | 质量最终判定 |
| LLM 评估 | 用 LLM 评估 LLM | 大规模质量评估 |
| 在线评估 | A/B 测试 | 生产环境效果 |

### 3.2 评估维度

| 维度 | 说明 | 指标/方法 |
|:---|:---|:---|
| 准确性 | 答案是否正确 | Exact Match、F1 |
| 流畅性 | 语言是否通顺 | PPL、人工评分 |
| 忠实性 | 是否忠于原文/事实 | 事实一致性检测 |
| 多样性 | 生成是否多样 | Distinct-n、Self-BLEU |
| 有用性 | 是否满足用户需求 | 人类偏好 |
| 安全性 | 是否有害/偏见 | 安全分类器 |

## 4. 技术原理

### 4.1 分类评估指标

**准确率（Accuracy）**：

$$\text{Accuracy} = \frac{\text{正确预测数}}{\text{总样本数}}$$

**精确率（Precision）与召回率（Recall）**：

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

**F1 分数**：

$$F1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

适用于类别不平衡场景。NER 等任务使用 entity-level F1。

### 4.2 生成评估指标

#### BLEU（翻译评估）

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

- $p_n$：n-gram 精确率（生成中匹配参考的 n-gram 比例）
- BP（Brevity Penalty）：$\min(1, e^{1-r/c})$，惩罚短译文

**局限**：只看精确率不看召回率；与人工评估相关性有限。

#### ROUGE（摘要评估）

$$\text{ROUGE-N} = \frac{\sum \text{matching n-grams}}{\sum \text{reference n-grams}}$$

- ROUGE-1：unigram 召回率
- ROUGE-2：bigram 召回率
- ROUGE-L：最长公共子序列（LCS）

**ROUGE 偏向召回率**，BLEU 偏向精确率。

#### METEOR

考虑同义词匹配和词序，比 BLEU 更接近人工评估。

#### CIDEr

专为图像描述设计的评估指标，基于 TF-IDF 加权的 n-gram 共现。

### 4.3 基于 LLM 的评估

#### BERTScore

使用 BERT 嵌入计算生成文本和参考文本的相似度：

$$F_{\text{BERT}} = 2 \cdot \frac{P_{\text{BERT}} \cdot R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}$$

其中 $P_{\text{BERT}}$ 和 $R_{\text{BERT}}$ 基于 token 嵌入的余弦相似度。

#### BLEURT / COMET

微调的预训练评估模型，直接预测文本质量分数。

#### LLM-as-a-Judge

用 GPT-4 等强模型评估其他模型的输出：

```
提示：评估以下回答的质量（1-10分），从准确性、完整性、清晰度三个维度打分：
问题：...
参考答案：...
待评估答案：...
```

#### Chatbot Arena

人类盲比较两个 LLM 的回答，选择更好的一个，使用 Elo 评分系统排名。

### 4.4 NLP 基准测试

| 基准 | 评估任务 | 说明 |
|:---|:---|:---|
| GLUE | 8 项 NLP 任务 | 通用语言理解评估 |
| SuperGLUE | 更难的 8 项任务 | GLUE 的升级版 |
| SQuAD | 阅读理解 | 从文档中抽取答案 |
| MMLU | 57 个学科 | 多领域知识评估 |
| HELM | 多维度 | 斯坦福全面 LLM 评估 |
| MT-Bench | 多轮对话 | LLM 对话能力评估 |
| HumanEval | 代码生成 | 编程能力评估 |

## 5. 工具链与框架

### 5.1 传统 NLP 工具

| 工具 | 语言 | 功能 | 适用场景 |
|:---|:---|:---|:---|
| NLTK | Python | 全功能 NLP | 教学/研究 |
| spaCy | Python | 工业级 NLP 管线 | 生产环境 |
| Stanford CoreNLP | Java | 句法/NER/POS | 学术研究 |
| jieba | Python | 中文分词 | 中文处理 |
| HanLP | Java/Python | 中文 NLP 全栈 | 中文生产环境 |
| LTP | Python | 中文 NLP | 哈工大中文工具 |

### 5.2 深度学习 NLP 框架

| 框架 | 功能 | 特点 |
|:---|:---|:---|
| Hugging Face Transformers | 预训练模型 | 最大的模型生态 |
| Hugging Face Datasets | 数据集 | 统一的数据加载接口 |
| Hugging Face Tokenizers | 分词器 | 高性能 Rust 实现 |
| AllenNLP | 研究 NLP | 模块化研究框架（已停止维护） |
| Fairseq | Seq2Seq | Meta 开源翻译框架 |

### 5.3 评估工具

| 工具 | 功能 |
|:---|:---|
| `sacrebleu` | 标准化 BLEU 计算 |
| `rouge-score` | ROUGE 计算 |
| `bert-score` | BERTScore 计算 |
| `evaluate` (HF) | 统一评估接口 |

## 6. NLP 工程实践流程

### 典型开发流程

```
需求分析 → 数据收集 → 数据标注 → 模型选择 → 训练/微调 → 评估 → 部署 → 监控
```

| 阶段 | 工具/方法 | 注意事项 |
|:---|:---|:---|
| 数据收集 | 爬虫、公开数据集 | 数据质量 > 数据量 |
| 数据标注 | Label Studio、Prodigy | 标注一致性 |
| 数据预处理 | 清洗、分词、标准化 | 领域特异性 |
| 模型选择 | Hugging Face Hub | 预训练模型选择 |
| 训练/微调 | PyTorch、Transformers | 超参数调优 |
| 评估 | 自动+人工 | 多维度评估 |
| 部署 | ONNX、TorchServe、vLLM | 延迟与吞吐 |
| 监控 | 日志、A/B 测试 | 数据漂移 |

### 部署优化

- **模型量化**：FP16/INT8/INT4 减少内存和加速推理。参见 [[12_大模型推理与优化/02_量化技术]]。
- **模型蒸馏**：用大模型蒸馏小模型。参见 [[21_知识蒸馏与模型压缩]]。
- **批处理**：动态批处理提高吞吐量
- **缓存**：缓存常见查询结果
- **ONNX/TensorRT**：推理引擎优化

## 7. 应用场景

- **搜索引擎**：查询理解、结果排序
- **推荐系统**：内容理解、用户画像
- **客服系统**：自动问答、工单分类
- **内容审核**：有害内容检测
- **金融分析**：财报分析、舆情监控
- **医疗**：病历分析、药物发现

## 8. 与其他技术关系

- **与模型部署与工程化的关系**：NLP 模型部署是 MLOps 的重要场景。参见 [[18_模型部署与工程化/01_模型服务化]]。
- **与大模型推理与优化的关系**：LLM 推理优化直接影响 NLP 系统的延迟和成本。参见 [[12_大模型推理与优化/00_大模型推理与优化]]。
- **与 LLM 应用工程的关系**：RAG、提示工程等是 NLP 在大模型时代的新实践。参见 [[19_LLM应用工程/00_LLM应用工程]]。
- **与知识蒸馏与模型压缩的关系**：蒸馏是 NLP 模型部署的关键技术。参见 [[21_知识蒸馏与模型压缩]]。

## 9. 前沿发展

- **LLM 评估体系**：从单一指标到多维度全面评估（HELM、MT-Bench、Arena）
- **自动化评估**：LLM-as-a-Judge 大规模替代人工评估
- **持续评估**：模型更新后持续监控性能退化
- **红队测试**：主动发现 LLM 的安全漏洞和失败模式
- **多语言评估**：超越英语中心的评估体系
- **任务对齐评估**：评估模型在具体应用场景的实际效果而非通用基准
- **可解释评估**：不仅评估"对不对"，还分析"为什么对/错"
