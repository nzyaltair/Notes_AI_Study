# 能量模型与 Score 模型

## 1. 概述

能量模型（Energy-Based Models, EBM）和 Score 模型是两类基于**未归一化概率密度**的生成模型。能量模型通过学习能量函数 $E_\theta(x)$ 定义未归一化分布 $p_\theta(x) \propto e^{-E_\theta(x)}$；Score 模型则通过学习对数概率密度的梯度 $\nabla_x \log p(x)$（即 score 函数）来建模分布。

两者的核心共性是**避免了难以计算的归一化常数**（配分函数），从不同角度绕开了这一难题：EBM 通过对比训练目标，Score 模型通过分数匹配。Score 模型尤其是理解现代扩散模型的理论基础。

## 2. 发展历史

| 年代 | 里程碑 | 意义 |
|------|--------|------|
| 2006 | RBM / DBN（Hinton） | 早期能量模型 |
| 2009 | Score Matching（Hyvärinen） | 无需归一化的密度估计 |
| 2011 | Denoising Score Matching | 通过加噪实现可扩展的 score matching |
| 2019 | NCSN（Song & Ermon） | 噪声条件分数网络 |
| 2020 | Score-Based SDE（Song 等） | 统一连续时间框架 |
| 2020 | 扩散模型 = Score 模型 | DDPM 噪声预测等价于 score 预测 |
| 2021+ | EBM 复兴 | 用于组合优化、文本控制等 |

## 3. 核心概念

### 3.1 能量模型

定义未归一化概率分布：

$$p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z(\theta)}, \quad Z(\theta) = \int e^{-E_\theta(x)} dx$$

其中 $E_\theta(x)$ 为能量函数，$Z(\theta)$ 为配分函数（通常不可解析计算）。

**能量低 = 概率高**：真实数据应具有低能量。

### 3.2 Score 函数

Score 函数定义为对数概率密度的梯度：

$$s(x) = \nabla_x \log p(x)$$

它指向概率密度增加的方向。如果能学习到 score 函数，就可以通过 Langevin 动力学等采样方法从分布中生成样本。

### 3.3 为什么直接学习 Score 困难

- **归一化常数未知**：$\log p(x) = -E(x) - \log Z$，$\log Z$ 未知
- **高维空间稀疏**：真实数据在高维空间中稀疏分布
- **低密度区域 score 估计不准**：数据稀少的地方难以学习准确的梯度

### 3.4 噪声条件分数网络（NCSN）的解决思路

通过**多尺度加噪**解决低密度区域问题：
- 使用多个噪声尺度 $\sigma_1 > \sigma_2 > \cdots > \sigma_L$
- 每个尺度训练一个 score 模型 $s_\theta(x, \sigma)$
- 大噪声填充低密度区域，小噪声保留细节
- 采样时退火 Langevin 动力学从粗到细

## 4. 技术原理

### 4.1 Score Matching

**原始 Score Matching**（Hyvärinen 2005）：

$$\mathcal{L}_{SM} = \mathbb{E}_{x \sim p_{data}}\left[\frac{1}{2}\|s_\theta(x) - \nabla_x \log p_{data}(x)\|^2\right]$$

由于 $\nabla_x \log p_{data}(x)$ 未知，经过分部积分可化为：

$$\mathcal{L}_{SM} = \mathbb{E}_{x \sim p_{data}}\left[\frac{1}{2}\|s_\theta(x)\|^2 + \nabla_x \cdot s_\theta(x)\right]$$

**问题**：需要计算散度 $\nabla_x \cdot s_\theta(x)$，在高维空间中计算代价大（Hutchinson 迹估计器可近似），且在低密度区域不稳定。

### 4.2 Denoising Score Matching (DSM)

通过加噪简化目标：

给定加噪分布 $q(x_t | x_0) = \mathcal{N}(x_t; x_0, \sigma^2 I)$，训练 score 模型匹配加噪分布的 score：

$$\mathcal{L}_{DSM} = \mathbb{E}_{x_0, x_t}\left[\frac{1}{2}\|s_\theta(x_t, \sigma) - \nabla_{x_t} \log q(x_t | x_0)\|^2\right]$$

由于 $\nabla_{x_t} \log q(x_t | x_0) = -\frac{x_t - x_0}{\sigma^2} = -\frac{\epsilon}{\sigma}$，损失变为：

$$\mathcal{L}_{DSM} = \mathbb{E}_{x_0, \epsilon}\left[\frac{1}{2}\left\|s_\theta(x_t, \sigma) + \frac{\epsilon}{\sigma}\right\|^2\right]$$

**与 DDPM 的等价关系**：令 $s_\theta(x_t, \sigma) = -\frac{\epsilon_\theta(x_t, t)}{\sigma}$，则 DSM 损失等价于 DDPM 的噪声预测损失。

### 4.3 NCSN (Noise Conditional Score Network)

训练单个网络，条件化于噪声尺度：

$$s_\theta(x, \sigma) \approx \nabla_x \log q_\sigma(x)$$

其中 $q_\sigma(x) = \int q_\sigma(x | x_0) p_{data}(x_0) dx_0$。

**退火 Langevin 动力学采样**：
1. 从大噪声 $\sigma_1$ 开始，运行 Langevin 动力学
2. 逐步减小噪声尺度
3. 最终在小噪声下得到高质量样本

Langevin 动力学：

$$x_{t+1} = x_t + \frac{\alpha}{2} s_\theta(x_t, \sigma) + \sqrt{\alpha} z, \quad z \sim \mathcal{N}(0, I)$$

### 4.4 Score-Based SDE 统一框架

Song 等 (2021) 将离散的 NCSN 和 DDPM 统一为连续时间的 SDE 框架：

**前向 SDE**（加噪）：

$$dx = f(x, t) dt + g(t) dw$$

- VP-SDE（对应 DDPM）：$f(x, t) = -\frac{1}{2}\beta(t) x$, $g(t) = \sqrt{\beta(t)}$
- VE-SDE（对应 NCSN）：$f(x, t) = 0$, $g(t) = \sqrt{\frac{d\sigma^2(t)}{dt}}$

**逆向 SDE**（去噪）：

$$dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)] dt + g(t) d\bar{w}$$

需要 score 函数 $\nabla_x \log p_t(x)$，通过 DSM 训练的神经网络近似。

**Probability Flow ODE**：对应确定性 ODE，具有相同的边缘分布，支持精确似然计算。

### 4.5 能量模型训练方法

#### Contrastive Divergence (CD)

$$\nabla_\theta \mathcal{L} = \mathbb{E}_{x^+ \sim p_{data}}[\nabla_\theta E_\theta(x^+)] - \mathbb{E}_{x^- \sim p_\theta}[\nabla_\theta E_\theta(x^-)]$$

通过 MCMC（如 Langevin 采样）从模型分布中获取负样本。

#### Persistent Contrastive Divergence (PCD)

维护一组持久的负样本链，避免每次从随机初始化开始 MCMC。

#### Noise Contrastive Estimation (NCE)

将生成问题转化为二分类问题：区分真实数据与噪声分布。

## 5. 关键方法与模型

### 5.1 模型对比

| 模型 | 类型 | 采样方法 | 特点 |
|------|------|---------|------|
| RBM | EBM | Gibbs 采样 | 经典能量模型 |
| NCSN | Score | 退火 Langevin | 多尺度 score |
| DDPM | Score/扩散 | 反向扩散 | 等价于 DSM |
| Score SDE | Score | 反向 SDE | 连续统一框架 |
| EBM (现代) | EBM | Langevin MCMC | 灵活但训练难 |

### 5.2 经典论文

- Hyvärinen (2005): *Estimation of Non-Normalized Statistical Models by Score Matching*
- Vincent (2011): *A Connection Between Score Matching and Denoising Autoencoders*
- Song & Ermon (2019): *Generative Modeling by Estimating Gradients of the Data Distribution* — NCSN
- Song et al. (2021): *Score-Based Generative Modeling through Stochastic Differential Equations*
- Du & Mordatch (2019): *Implicit Generation and Modeling with Energy-Based Models*

## 6. 优势与局限

### 6.1 优势

- **无需归一化常数**：绕开了配分函数的计算难题
- **理论统一**：Score 框架统一了扩散模型和 NCSN
- **灵活的概率建模**：EBM 可定义任意能量函数
- **精确似然（PF ODE）**：Score 模型可通过 PF ODE 计算精确似然
- **组合性**：EBM 的能量可以相加实现组合推理

### 6.2 屺限

- **采样慢**：Langevin MCMC 需要很多步
- **EBM 训练困难**：MCMC 负样本质量影响训练效果
- **低密度区域**：原始 score matching 在低密度区域不准确（DSM 部分缓解）
- **混合模式**：在多模态分布中，Langevin 可能无法在模式间转换

## 7. 应用场景

- **图像生成**：扩散模型的 score 框架（主要应用）
- **组合推理**：EBM 的能量可加性用于组合多个约束
- **异常检测**：能量/概率密度评估
- **文本控制**：EBM 作为约束引导文本生成
- **物理建模**：统计物理中的能量模型
- **图生成**：分子图结构生成

## 8. 与其他技术关系

- Score 模型是 [[04_扩散模型]] 的理论基础——DDPM 的噪声预测等价于 denoising score matching
- 能量模型的对比训练思想与对比学习（SimCLR 等）有联系
- Score-Based SDE 的 PF ODE 与 [[03_归一化流]] 的连续归一化流等价
- EBM 的能量函数思想与物理中的玻尔兹曼分布一脉相承
- 分类器引导（Classifier Guidance）本质上利用分类器的 score 引导扩散采样

## 9. 前沿发展

- **Flow Matching**：从 score matching 发展到向量场回归，更稳定的训练
- **EBM + 扩散**：能量模型作为扩散模型的约束或引导
- **组合 EBMs**：多个简单 EBM 的能量相加实现复杂组合推理
- **离散 EBM**：在离散空间（文本、图）中的能量模型
- **Score-based 理论深化**：分数模型与最优传输理论的关系
