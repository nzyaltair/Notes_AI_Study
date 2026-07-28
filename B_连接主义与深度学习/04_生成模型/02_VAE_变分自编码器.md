# VAE 变分自编码器

## 1. 概述

变分自编码器（Variational Autoencoder, VAE）由 Kingma 和 Welling 于 2013 年提出，是一种**基于变分推断的生成模型**。它通过编码器将数据映射到潜在分布，再通过解码器从潜变量重建数据，以最大化证据下界（ELBO）为训练目标。

VAE 的核心思想是：假设数据由潜变量 $z$ 生成，通过学习推断网络 $q_\phi(z|x)$ 近似真实后验 $p(z|x)$，同时学习生成网络 $p_\theta(x|z)$。与标准自编码器不同，VAE 学习的是**潜变量的分布**而非确定性编码，因此能从潜空间中采样生成新数据。

## 2. 发展历史

| 年代 | 里程碑 | 意义 |
|------|--------|------|
| 2013 | VAE（Kingma & Welling） | 变分推断与深度学习结合 |
| 2014 | Auto-Encoding VB | 重参数化技巧使端到端训练可行 |
| 2016 | β-VAE（Higgins 等） | 通过 KL 权重实现解耦表示 |
| 2017 | VQ-VAE（van den Oord 等） | 离散潜空间，避免后验坍缩 |
| 2019 | NVAE（NVIDIA） | 层次化潜变量，高质量图像生成 |
| 2019 | VQ-VAE-2 | 多层离散编码，接近 BigGAN 质量 |
| 2021 | Stable Diffusion 中的 VAE | 作为潜空间压缩器用于扩散模型 |
| 2023+ | VAE 成为扩散模型的标准组件 | 在潜在扩散中扮演关键角色 |

## 3. 核心概念

### 3.1 概率模型定义

VAE 定义了以下概率模型：
- **先验**：$p(z)$，通常为标准正态 $\mathcal{N}(0, I)$
- **似然**：$p_\theta(x|z)$，由解码器网络参数化
- **后验**：$p_\theta(z|x) = \frac{p_\theta(x|z) p(z)}{p_\theta(x)}$，通常不可解析计算
- **推断网络**：$q_\phi(z|x)$，编码器，近似真实后验

### 3.2 ELBO 推导

由于 $p_\theta(x)$ 不可解析计算（需对所有 $z$ 积分），引入变分分布 $q_\phi(z|x)$：

$$\log p_\theta(x) = \log \int p_\theta(x, z) dz = \log \int \frac{p_\theta(x, z)}{q_\phi(z|x)} q_\phi(z|x) dz$$

由 Jensen 不等式：

$$\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)}\right] = \text{ELBO}$$

进一步展开：

$$\text{ELBO} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{重建项}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{正则项}}$$

- **重建项**：鼓励解码器从 $z$ 重建 $x$
- **正则项**：鼓励编码器输出的后验接近先验

且有：

$$\log p_\theta(x) = \text{ELBO} + D_{KL}(q_\phi(z|x) \| p_\theta(z|x))$$

最大化 ELBO 等价于同时最小化 KL 散度（使推断网络逼近真实后验）和最大化似然。

### 3.3 重参数化技巧

直接从 $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$ 采样不可微，无法反向传播。重参数化技巧将随机性移出计算图：

$$z = \mu_\phi(x) + \sigma_\phi(x) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

这样 $\mu_\phi$ 和 $\sigma_\phi$ 可以通过梯度优化。

## 4. 技术原理

### 4.1 高斯情况下的 KL 散度解析解

当 $q_\phi(z|x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$ 且 $p(z) = \mathcal{N}(0, I)$ 时：

$$D_{KL} = \frac{1}{2} \sum_{j=1}^{d} \left(\mu_j^2 + \sigma_j^2 - \ln \sigma_j^2 - 1\right)$$

这使得 KL 项有解析解，无需 Monte Carlo 估计。

### 4.2 重建损失的选择

- **连续数据**（图像像素 $[0,1]$）：MSE / 高斯似然 $\log p(x|z) = -\frac{1}{2}\|x - \hat{x}\|^2$
- **二值数据**：BCE / 伯努利似然
- **分类数据**：交叉熵 / 类别分布

### 4.3 β-VAE

引入权重 $\beta$ 控制 KL 项强度：

$$\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot D_{KL}(q_\phi(z|x) \| p(z))$$

- $\beta > 1$：更强的解耦，但重建质量下降
- $\beta < 1$：更好重建，但解耦能力减弱
- $\beta = 1$：标准 VAE

### 4.4 VQ-VAE（向量量化 VAE）

使用**离散潜空间**替代连续高斯：

1. 编码器输出连续向量 $z_e(x)$
2. 从码本 $\{e_k\}_{k=1}^{K}$ 中找最近邻：$z_q(x) = e_k, k = \arg\min_j \|z_e - e_j\|$
3. 解码器从量化后的 $z_q$ 重建

**训练损失**：

$$\mathcal{L} = \|x - D(z_q)\|^2 + \|\text{sg}[z_e] - e_k\|^2 + \beta \|z_e - \text{sg}[e_k]\|^2$$

其中 sg 为停止梯度操作，第三项（commitment loss）确保编码器输出靠近码本向量。

**优势**：
- 避免后验坍缩
- 离散表示更适合建模离散数据（文本、音频）
- 可与自回归先验结合（如 PixelCNN 在离散潜空间上建模）

### 4.5 NVAE（层次化 VAE）

通过多层级潜变量 $z = (z_1, z_L, \dots, z_L)$ 和残差参数化实现高质量图像生成，在 CelebA/CIFAR-10 上达到接近 GAN 的质量。

### 4.6 后验坍缩问题

当解码器足够强大时，$q_\phi(z|x)$ 可能坍缩为先验 $p(z)$，即编码器忽略输入数据，潜变量不携带信息。

**原因**：KL 项被过度优化，使后验退化
**缓解策略**：
- 降低 KL 权重（β < 1）
- 弱化解码器（如限制容量）
- Free-bits 策略：对每个维度保留一定量的 KL 不惩罚
- VQ-VAE 的离散化从根本上避免此问题

## 5. 关键方法与模型

### 5.1 VAE 家族

| 模型 | 创新点 | 适用场景 |
|------|--------|---------|
| VAE | 变分推断 + 重参数化 | 通用生成 |
| β-VAE | KL 权重控制解耦 | 解耦表示学习 |
| CVAE | 条件生成 | 条件生成 |
| VQ-VAE | 离散潜空间 | 音频/图像/视频 |
| VQ-VAE-2 | 多层层次化离散编码 | 高分辨率图像 |
| NVAE | 深层层次化连续潜变量 | 高质量图像生成 |
| Stable Diffusion VAE | 潜空间压缩 | 扩散模型组件 |

### 5.2 经典论文

- Kingma & Welling (2013): *Auto-Encoding Variational Bayes* — VAE 原始论文
- Rezende et al. (2014): *Stochastic Backpropagation and Approximate Inference in Deep Generative Models*
- Higgins et al. (2017): *beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*
- van den Oord et al. (2017): *Neural Discrete Representation Learning* — VQ-VAE
- Razavi et al. (2019): *Generating Diverse High-Fidelity Images with VQ-VAE-2*
- Vahdat & Kautz (2020): *NVAE: A Deep Hierarchical Variational Autoencoder*

## 6. 优势与局限

### 6.1 优势

- **训练稳定**：基于明确的概率目标（ELBO），梯度行为良好
- **有似然估计**：可计算数据的对数似然下界
- **潜空间结构化**：连续可微的潜空间支持插值和算术运算
- **理论优雅**：有坚实的变分推断理论基础
- **可扩展**：支持大规模数据和深层架构

### 6.2 局限

- **样本模糊**：与 GAN 相比，VAE 生成的样本通常偏模糊
- **后验坍缩**：强大的解码器可能导致潜变量失效
- **ELBO 与质量不完美对应**：更高的 ELBO 不一定意味着更好的样本质量
- **先验假设限制**：高斯先验假设可能不匹配数据的真实潜空间结构

## 7. 应用场景

- **图像生成**： CelebA 人脸生成、插值动画
- **潜空间压缩**：Stable Diffusion 中的 VAE 将 $512 \times 512 \times 3$ 压缩到 $64 \times 64 \times 4$
- **表示学习**：学习结构化的潜表示用于下游任务
- **异常检测**：重建误差大的样本判为异常
- **数据去噪**：从含噪输入重建干净数据
- **半监督学习**：结合标签和无标签数据
- **音频/语音**：VQ-VAE 用于音频编码和生成

## 8. 与其他技术关系

- VAE 的变分推断框架与 [[01_GAN_生成对抗网络]] 的对抗训练互补，可组合为 VAE-GAN
- VQ-VAE 的离散表示思想与 [[05_自回归生成模型]] 结合（在离散潜空间上用 PixelCNN 建模先验）
- VAE 是 [[04_扩散模型]] 中潜在扩散模型（LDM）的核心组件——VAE 编码器/解码器负责空间压缩
- 数学基础依赖 [[03_概率与统计|概率与统计]] 中的变分推断和 Jensen 不等式
- β-VAE 的解耦表示学习与可解释性研究相关

## 9. 前沿发展

- **作为扩散模型组件**：VAE 已从独立生成模型转变为潜在扩散模型的标准压缩器
- **改进的 VAE 训练**：如 diffusion-based prior 替代标准高斯先验
- **层次化潜空间**：更深层次的多尺度潜变量建模
- **连续-离散混合**：结合连续和离散潜变量的优势
- **跨模态 VAE**：多模态数据的统一潜空间学习
