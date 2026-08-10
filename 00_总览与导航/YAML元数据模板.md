# YAML 元数据模板

> 每篇笔记的 Front Matter 统一格式。按优先级分批添加。

## 完整模板

```yaml
---
title: "笔记标题"
aliases: ["别名1", "别名2"]
domain: "所属方向（如 C-09）"
level: "基础 | 核心 | 进阶 | 实践"
status: "草稿 | 初稿 | 审阅 | 稳定"
last_reviewed: 2026-08-10
review_cycle: 180
prerequisites:
  - "前置知识1"
  - "前置知识2"
depends_on:
  - "[[依赖笔记路径]]"
related:
  - "[[相关笔记路径]]"
canonical: "[[权威正文路径]]"  # 如果本笔记不是权威正文
evidence_level: "理论 | 实验 | 综述 | 实践 | 共识 | 争议"
source_of_truth: true  # 是否权威正文
tags:
  - 标签1
  - 标签2
created: 2026-01-01
updated: 2026-08-10
---
```

## 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| title | 是 | 笔记标题，与文件名一致 |
| aliases | 否 | 其他常用名称，便于搜索 |
| domain | 是 | 所属方向编码 |
| level | 是 | 知识层级 |
| status | 是 | 内容成熟度 |
| last_reviewed | 是 | 最近审阅日期 |
| review_cycle | 是 | 审阅周期（天） |
| prerequisites | 否 | 前置知识列表 |
| depends_on | 否 | 依赖的其他笔记 |
| related | 否 | 相关笔记 |
| canonical | 否 | 若本笔记不是权威正文，指向权威正文 |
| evidence_level | 是 | 证据等级 |
| source_of_truth | 是 | 是否权威正文 |
| tags | 是 | 标签 |
| created | 是 | 创建日期 |
| updated | 是 | 最后更新日期 |

## 优先级

### 第一批（P0 新增/修改的笔记）

C-10 推理算法域的 5 篇笔记、I-08 研究方法域的 8 篇笔记、K-08 新增的 4 篇笔记、D-06 新增的 1 篇笔记、J-06/J-09 新增的 2 篇笔记、K-07 新增的 1 篇笔记。

### 第二批（现有综述页）

所有 `00_*_综述.md` 文件。

### 第三批（现有核心概念笔记）

核心概念笔记逐步添加。

## 示例

### 概念笔记

```yaml
---
title: "解码与采样策略"
aliases: ["解码策略", "采样策略", "Decoding & Sampling"]
domain: "C-10"
level: "核心"
status: "初稿"
last_reviewed: 2026-08-10
review_cycle: 180
prerequisites:
  - "自回归语言模型"
  - "概率分布"
depends_on:
  - "../C-02/大语言模型核心架构"
related:
  - "../C-09/KV-Cache机制"
  - "../C-08/测试时计算"
canonical: null
evidence_level: "综述"
source_of_truth: true
tags:
  - 推理算法
  - 解码
  - 采样
created: 2026-08-10
updated: 2026-08-10
---
```

### 综述笔记

```yaml
---
title: "推理算法综述"
aliases: ["Inference Algorithms"]
domain: "C-10"
level: "综述"
status: "初稿"
last_reviewed: 2026-08-10
review_cycle: 180
prerequisites:
  - "C-08 推理与思考"
  - "C-09 模型压缩"
depends_on:
  - "../C-08/推理与思维链_综述"
  - "../C-09/模型压缩与高效推理_综述"
related:
  - "../K-03/推理工程_综述"
canonical: null
evidence_level: "综述"
source_of_truth: true
tags:
  - 推理算法
  - 综述
created: 2026-08-10
updated: 2026-08-10
---
```