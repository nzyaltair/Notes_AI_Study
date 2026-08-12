# Hebbian 学习与 STDP：局部可塑性的基本语言

## 一句话理解

Hebbian 学习用神经元共同活动解释联结增强；STDP 将这一思想扩展为对脉冲先后时序敏感的更新规则。它们适合表达局部可得信号下的自组织，但单独不足以解决复杂任务的长期信用分配。

## 基本思想

简化的 Hebbian 更新可写作：

\[
\Delta w_{ij} = \eta x_i x_j
\]

其中 \(x_i\) 与 \(x_j\) 表示前、后突触活动。实际系统通常加入归一化、竞争、稳态或调制项，防止权重无界增长。

STDP 根据前后脉冲相对时间 \(\Delta t\) 调整联结：通常前突触脉冲先于后突触脉冲时更可能增强，反向时更可能减弱；具体曲线与生物系统和模型设定有关。

## 常用变体

- **Oja 规则**：在 Hebbian 更新中加入归一化项，使权重向量趋向单位范数，防止无界增长，可实现主成分式特征提取。
- **BCM 理论**：突触增强/减弱由滑动阈值决定，阈值随近期活动调整，形成选择性与稳态之间的平衡。
- **三因素规则**：在“前—后突触相关性”之外引入第三因素（神经调质或误差信号），使局部可塑性受全局任务信号调制——这是把 Hebbian/STDP 与强化学习、预测误差结合的桥梁。
- **权值归一化与竞争**：横向抑制与权值重归一化让神经元分化成专门检测器，是自组织的基础。

## 优势与局限

- 优势：局部、在线、适合无监督相关性与时序结构学习。
- 局限：难以直接传递跨多层、长时程的任务误差信号；需要与奖励调制、预测误差、全局目标或其他机制结合。

## 相关知识

- [[00_脑启发学习_综述|脑启发学习综述]]：学习路线总览。
- [[../01_大脑与认知基础/03_学习与记忆的神经基础|学习与记忆的神经基础]]：LTP/LTD 等生物证据。
- [[03_持续学习与局部学习|持续学习与局部学习]]：稳定与可塑性的权衡。
- [[../09_神经形态计算/01_脉冲神经网络与神经可塑性|脉冲神经网络与神经可塑性]]：SNN 中的实现。

## References

- Hebb, *The Organization of Behavior* (1949).
- Bi & Poo, *Synaptic Modifications in Cultured Hippocampal Neurons* (1998).
- Oja, *Simplified Neuron Model as a Principal Component Analyzer* (Journal of Mathematical Biology, 1982).
- Bienenstock, Cooper & Munro, *Theory for the Development of Neuron Selectivity* (Journal of Neuroscience, 1982).
- Gerstner et al., *Neuronal Dynamics*.
