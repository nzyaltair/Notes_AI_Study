# OCR

## 一句话理解

光学字符识别（OCR）将文档或场景图像中的文字转换为结构化文本，通常包含文本检测、文本识别与文档结构理解。

## 任务拆分

- **文本检测**：定位文字区域，可为水平框、旋转框或多边形。
- **文本识别**：将裁剪文本图像转为字符序列，常用 CTC 或注意力/Transformer 解码。
- **结构理解**：恢复阅读顺序、表格、段落和键值关系；其输入往往结合版面与文本内容。

## 技术要点

场景文本受透视、模糊、字体、多语言和背景干扰影响；文档 OCR 还需处理扫描质量与版式变化。端到端模型可减少流水线误差传播，但模块化系统在诊断、替换和领域适配上更直接。

## 评估

检测常用 precision、recall、F1 与区域匹配规则；识别使用字符错误率（CER）或词错误率（WER）。评测必须声明语言、词典约束、大小写和标点规范。

## 相关知识

- 前置：[[02_目标检测]]、[[../../04_序列模型/06_CTC与序列标注|CTC]]
- 对比：[[01_图像分类]]
- 延伸：[[../../../E_感知与多模态智能/01_视觉表示与视觉理解/00_视觉智能综述|视觉智能]]

## References

- Shi, Bai & Yao. An End-to-End Trainable Neural Network for Image-Based Sequence Recognition. *IEEE TPAMI*, 2017.
- Baek et al. What Is Wrong With Scene Text Recognition Model Comparisons? *ICCV*, 2019.
