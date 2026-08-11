# 开放世界视觉与视觉 Grounding

## 一句话理解

开放世界视觉从固定类别识别转向"由自然语言定义对象和区域"——核心任务包括开放词汇检测/分割、文本指代表达理解和可提示分割，使视觉系统从闭集分类器升级为可交互、可查询的视觉理解引擎。

## 1. 概述

开放世界视觉（Open-World Vision）和视觉 Grounding 是计算机视觉从"闭集识别"向"开放世界理解"演进的关键技术。传统视觉系统只能识别训练时见过的固定类别，而开放世界视觉通过自然语言作为桥梁，使视觉系统能够理解任意概念。

视觉 Grounding 指将自然语言描述（短语、句子）与图像中的对应区域（像素、框、掩码）建立对应关系，是连接视觉与语言的核心能力。

## 2. 发展历史

| 年代 | 里程碑 | 核心意义 |
|:---|:---|:---|
| 2016 | Referring Expression 数据集 (RefCOCO) | 标准化指代表达理解任务 |
| 2017 | MAttNet | 模块化注意力网络，指代分割 |
| 2020 | MDETR | 基于 Transformer 的端到端指代检测 |
| 2021 | CLIP | 对比学习对齐图文，零样本分类突破 |
| 2021 | GLIP | 将检测视为短语 grounding，统一检测与 grounding |
| 2022 | OWL-ViT | 开放词汇目标检测，CLIP 视觉编码器 |
| 2023 | Grounding DINO | 开放词汇检测 SOTA，结合 DINO 与 GLIP |
| 2023 | SAM | 可提示分割基础模型，点/框/文本提示 |
| 2024 | SAM 2 | 视频分割基础模型，流式记忆 |
| 2024 | Grounding DINO 1.5 | 改进的开放词汇检测，更广的类别覆盖 |

## 3. 核心概念

### 3.1 开放词汇检测（Open-Vocabulary Detection）

以文本嵌入替换固定分类器，使检测器能识别任意文本定义的类别：
- 利用 CLIP 等预训练图文模型的文本-视觉对齐能力
- 代表模型：OWL-ViT、Grounding DINO、GLIP
- 关键挑战：训练数据中未出现的类别（zero-shot 检测）

### 3.2 视觉 Grounding

将"红色杯子左侧的按钮"等语言短语定位到图像中的具体区域：
- **短语定位（Phrase Grounding）**：定位到边界框
- **指代分割（Referring Segmentation）**：定位到分割掩码
- **指代表达理解（Referring Expression Comprehension）**：结合物体、属性和空间关系消解指代

### 3.3 可提示分割（Promptable Segmentation）

以点、框、掩码或文本提示获得任意对象的分割掩码：
- SAM（Segment Anything）是代表性基础模型
- 支持交互式分割（用户点击修正）
- 可扩展到视频分割（SAM 2）

## 4. 技术原理

### 4.1 开放词汇检测架构

```text
图像 → 视觉编码器 → 区域特征 → 与文本嵌入匹配 → 输出类别/框
                    ↑
类别名 → 文本编码器 → 文本嵌入
```

- 视觉编码器：CLIP 视觉编码器或 DINO 等自监督编码器
- 文本编码器：CLIP 文本编码器，支持任意类别名
- 匹配策略：区域特征与文本嵌入的余弦相似度

### 4.2 Grounding DINO 架构

- 结合 DINO（DETR 改进版）的端到端检测架构
- 引入文本特征作为 Transformer 解码器的查询
- 语言引导的跨模态注意力机制

## 5. 关键方法与模型

| 模型 | 年份 | 核心创新 | 任务 |
|:---|:---|:---|:---|
| CLIP | 2021 | 对比学习对齐 4 亿图文对 | 零样本分类 |
| GLIP | 2021 | 统一检测与 phrase grounding | 开放词汇检测 |
| OWL-ViT | 2022 | CLIP 编码器 + 轻量检测头 | 开放词汇检测 |
| Grounding DINO | 2023 | DINO + 文本模态融合 | 开放词汇检测 |
| SAM | 2023 | 1B 掩码数据训练 | 可提示分割 |
| SEEM | 2023 | 统一可提示分割 | 多模态提示分割 |
| SAM 2 | 2024 | 流式记忆架构 | 视频分割 |

## 6. 优势与局限

### 优势
- **零样本能力**：无需为每个类别标注训练数据，通过自然语言直接定义
- **灵活性**：支持多种交互方式（点、框、文本、涂鸦）
- **可组合性**：可与 VLM、多模态推理系统结合

### 局限
- **细粒度不足**：对精细属性（颜色、纹理、小物体）的 grounding 仍有困难
- **空间关系推理**：复杂空间关系（"A 左边第二个 B"）的 grounding 准确率低
- **领域偏移**：在未见过的领域（医学、遥感）中 grounding 性能下降
- **计算成本**：大规模视觉基础模型推理成本较高

## 7. 应用场景

| 场景 | 使用方式 | 说明 |
|:---|:---|:---|
| 交互式图像编辑 | 文本提示 + SAM 分割 | 用户用自然语言指定要编辑的区域 |
| 机器人操作 | 指代分割 | 机器人抓取"红色杯子" |
| 自动驾驶 | 开放词汇检测 | 检测"正在过马路的行人" |
| 视觉问答 | Grounding 作为证据 | 为"桌子上的猫"提供区域定位 |
| 文档理解 | 文本指代表达 | "第三行的表格"定位 |

## 8. 与其他技术关系

- **与自监督学习的关系**：CLIP 等视觉-语言对齐模型是开放词汇视觉的基础。参见 [[07_自监督视觉学习]]。
- **与多模态推理的关系**：Grounding 输出为多模态推理提供视觉证据。参见 [[03_多模态理解与对齐/03_多模态理解与推理]]。
- **与感知行动接口的关系**：Grounding 是 VLA 中指令理解的关键。参见 [[07_感知与行动接口/03_视觉语言行动接口]]。
- **与视觉 Transformer 的关系**：开放词汇检测和 SAM 依赖 ViT 架构。参见 [[06_视觉Transformer]]。

## 9. 前沿发展

- **统一检测-分割-grounding 架构**：Grounding DINO + SAM 的端到端融合
- **视频级开放世界理解**：SAM 2 将开放世界分割扩展到视频
- **3D Grounding**：将指代理解扩展到三维空间
- **多模态提示**：结合文本、图像、音频的联合提示分割
- **高效推理**：开放词汇模型的推理加速与部署

## 相关知识

- **前置知识**：[[03_图像分类]]、[[04_目标检测]]、[[05_图像分割]]、[[07_自监督视觉学习]]
- **平级主题**：[[03_多模态理解与对齐/01_视觉语言模型]]、[[03_多模态理解与对齐/02_多模态对齐与融合]]
- **后续延伸**：[[03_多模态理解与对齐/03_多模态理解与推理]]、[[07_感知与行动接口/03_视觉语言行动接口]]

## References

- Radford, A. et al. (2021). Learning Transferable Visual Models from Natural Language Supervision. *ICML 2021*. (CLIP)
- Li, L. H. et al. (2021). Grounded Language-Image Pre-training. *CVPR 2022*. (GLIP)
- Minderer, M. et al. (2022). Simple Open-Vocabulary Object Detection with Vision Transformers. *ECCV 2022*. (OWL-ViT)
- Liu, S. et al. (2023). Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. *arXiv:2303.05499*.
- Kirillov, A. et al. (2023). Segment Anything. *ICCV 2023*.
- Kamath, A. et al. (2021). MDETR: Modulated Detection for End-to-End Multi-Modal Understanding. *ICCV 2021*.
- Zou, X. et al. (2023). Segment Everything Everywhere All at Once. *NeurIPS 2023*. (SEEM)
- Ravi, N. et al. (2024). SAM 2: Segment Anything in Images and Videos. *arXiv:2408.00714*.