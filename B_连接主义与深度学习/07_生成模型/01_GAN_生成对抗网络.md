# GAN 生成对抗网络

## 1. 概述

生成对抗网络（Generative Adversarial Network, GAN）由 Ian Goodfellow 于 2014 年提出，通过**生成器与判别器的对抗博弈**来学习数据分布。生成器试图生成逼真样本欺骗判别器，判别器试图区分真实样本与生成样本。二者在博弈中共同进步，最终生成器能产生与真实数据难以区分的样本。

GAN 的核心创新在于：**不需要显式建模概率密度函数**，而是通过对抗训练隐式地匹配数据分布。这使得 GAN 能够生成极其锐利、逼真的样本，但也带来了训练不稳定的代价。

## 2. 发展历史

| 年代 | 里程碑 | 意义 |
|------|--------|------|
| 2014 | GAN（Goodfellow 等） | 对抗生成范式诞生 |
| 2015 | DCGAN（Radford 等） | 卷积架构稳定 GAN 训练 |
| 2016 | InfoGAN | 最大化互信息获得可解释隐变量 |
| 2016 | Pix2Pix | 条件图像到图像翻译 |
| 2017 | CycleGAN（Zhu 等） | 无配对图像转换 |
| 2017 | WGAN（Arjovsky 等） | Wasserstein 距离替代 JS 散度 |
| 2017 | WGAN-GP | 梯度惩罚替代权重裁剪 |
| 2017 | StackGAN | 文本到图像的级联生成 |
| 2018 | SNGAN（Miyato 等） | 谱归一化稳定判别器 |
| 2018 | StyleGAN（NVIDIA） | 风格混合与高分辨率人脸生成 |
| 2019 | BigGAN（Brock 等） | 大规模训练突破 ImageNet 生成 |
| 2019 | StyleGAN2 | 消除伪影、改进风格注入 |
| 2020 | StyleGAN2-ADA | 自适应数据增强应对小数据集 |
| 2021+ | GAN 逐渐被扩散模型取代 | 在图像生成领域让位于扩散模型 |

## 3. 核心概念

### 3.1 生成器与判别器

- **生成器 $G(z; \theta_g)$**：从噪声分布 $p_z$（通常为高斯）映射到数据空间，目标是生成逼真样本
- **判别器 $D(x; \theta_d)$**：输出 $x$ 为真实数据的概率 $D(x) \in (0, 1)$
- **对抗关系**：$G$ 试图最小化 $\log(1 - D(G(z)))$，$D$ 试图最大化 $\log D(x) + \log(1 - D(G(z)))$

### 3.2 纳什均衡

理想状态下，当 $G$ 完美匹配数据分布时，$p_g = p_{data}$，此时 $D^*(x) = 1/2$（无法区分）。但在实践中，达到完美的纳什均衡非常困难。

### 3.3 关键挑战

- **模式崩溃（Mode Collapse）**：生成器只生成少数几种样本，缺乏多样性
- **训练不稳定**：$G$ 和 $D$ 能力失衡导致训练发散
- **梯度消失**：当 $D$ 过强时，$G$ 无法获得有效梯度
- **评估困难**：没有直接的似然度量

## 4. 技术原理

### 4.1 原始 GAN 目标函数

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**最优判别器**：给定 $G$，最优判别器为：

$$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$$

**代入后的价值函数**：

$$V(G, D^*) = -2\log 2 + 2 \cdot JSD(p_{data} \| p_g)$$

其中 $JSD$ 为 Jensen-Shannon 散度。当 $p_g = p_{data}$ 时取最小值 $-2\log 2$。

### 4.2 非饱和损失

实践中生成器改用以下目标以避免梯度消失：

$$\max_G \mathbb{E}_{z \sim p_z}[\log D(G(z))]$$

### 4.3 DCGAN 架构要点

- 使用步幅卷积（strided convolution）替代池化层
- 生成器使用转置卷积（transposed convolution）上采样
- BatchNorm 应用于双方（$G$ 最后一层和 $D$ 第一层除外）
- LeakyReLU 用于判别器，ReLU/Tanh 用于生成器

### 4.4 Wasserstein GAN (WGAN)

**动机**：JS 散度在两个分布不重叠时为常数（$\log 2$），导致梯度消失。

**Wasserstein 距离**（Earth Mover's Distance）：

$$W(p_{data}, p_g) = \inf_{\gamma \in \Pi(p_{data}, p_g)} \mathbb{E}_{(x,y) \sim \gamma}[\|x - y\|]$$

利用 Kantorovich-Rubinstein 对偶：

$$W(p_{data}, p_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p_{data}}[f(x)] - \mathbb{E}_{z \sim p_z}[f(G(z))]$$

其中 $f$ 需满足 Lipschitz 连续性约束 $\|f\|_L \leq 1$。

**WGAN 损失**：

$$\min_G \max_{D \in \mathcal{D}} \mathbb{E}_{x \sim p_{data}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

**Lipschitz 约束实现方式**：
- 权重裁剪（WGAN）：简单但导致优化困难
- 梯度惩罚（WGAN-GP）：$\mathcal{L}_{GP} = \lambda \mathbb{E}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$，更稳定
- 谱归一化（SNGAN）：$W \leftarrow W / \sigma_{max}(W)$，全局 Lipschitz 约束

### 4.5 条件 GAN (CGAN)

将条件信息 $c$ 输入生成器和判别器：

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x|c)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z|c)|c))]$$

### 4.6 CycleGAN

实现无配对的图像到图像翻译，引入循环一致性损失：

$$\mathcal{L}_{cyc} = \mathbb{E}_{x}[\|G_{B \to A}(G_{A \to B}(x)) - x\|_1] + \mathbb{E}_{y}[\|G_{A \to B}(G_{B \to A}(y)) - y\|_1]$$

### 4.7 StyleGAN 架构

核心创新：
- **渐进式增长**：从低分辨率逐步增加到高分辨率
- **风格注入**：通过 AdaIN（Adaptive Instance Normalization）在每层注入风格向量 $w$
- **噪声注入**：在每个分辨率层添加随机噪声实现细节变化
- **$W$ 空间**：将隐变量 $z$ 映射到中间空间 $w$，实现更好的解耦

## 5. 关键方法与模型

### 5.1 GAN 变体分类

| 类型 | 代表 | 特点 |
|------|------|------|
| 无条件 GAN | DCGAN, SNGAN, BigGAN | 无条件生成 |
| 条件 GAN | CGAN, Pix2Pix, StackGAN | 条件控制生成 |
| 图像翻译 | CycleGAN, Pix2Pix | 图像到图像转换 |
| 高分辨率 | ProGAN, StyleGAN 系列 | 渐进式高分辨率 |
| 文本条件 | StackGAN, AttnGAN | 文本到图像 |

### 5.2 训练技巧汇总

| 技巧 | 原理 | 适用场景 |
|------|------|---------|
| 谱归一化 | 全局 Lipschitz 约束 | 所有判别器 |
| 梯度惩罚 | 局部 Lipschitz 约束 | WGAN-GP |
| TTUR | 生成器和判别器不同学习率 | 通用 |
| EMA | 指数移动平均提升样本质量 | BigGAN 等 |
| 自适应数据增强 | 防止过拟合小数据集 | StyleGAN2-ADA |
| Minibatch StdDev | 增强多样性防模式崩溃 | BigGAN |

### 5.3 经典论文

- Goodfellow et al. (2014): *Generative Adversarial Nets* — GAN 原始论文
- Radford et al. (2015): *Unsupervised Representation Learning with DCGAN*
- Arjovsky et al. (2017): *Wasserstein GAN* — WGAN
- Gulrajani et al. (2017): *Improved Training of Wasserstein GANs* — WGAN-GP
- Miyato et al. (2018): *Spectral Normalization for GANs* — SNGAN
- Karras et al. (2018): *Progressive Growing of GANs* — ProGAN
- Karras et al. (2019): *A Style-Based Generator Architecture* — StyleGAN
- Brock et al. (2019): *Large Scale GAN Training with BigGAN*

## 6. 优势与局限

### 6.1 优势

- **样本质量高**：GAN 生成的图像极其锐利逼真
- **采样速度极快**：单次前向传播即可生成（1步）
- **隐式建模**：不需要假设概率分布形式
- **多样化架构**：适配图像翻译、风格迁移等多种任务

### 6.2 局限

- **训练不稳定**：需要大量调参技巧，难以复现
- **模式崩溃**：生成多样性不足
- **无似然估计**：无法直接评估数据概率
- **难以评估**：缺乏统一的定量评估标准
- **已被扩散模型超越**：在图像生成质量上已不如扩散模型

## 7. 应用场景

- **高分辨率人脸生成**：StyleGAN 系列
- **图像翻译**：CycleGAN（马↔斑马、冬↔夏等）
- **数据增强**：生成合成数据增强训练集
- **图像编辑**：风格混合、年龄变换、表情编辑
- **超分辨率**：SRGAN 等基于 GAN 的超分辨率
- **文本到图像**：早期方案（已被扩散模型取代）

## 8. 与其他技术关系

- GAN 的对抗思想影响了扩散模型中的**分类器引导**
- **VAE** 与 GAN 互补：VAE 稳定但模糊，GAN 清晰但不稳定，可结合使用（VAE-GAN）
- StyleGAN 的 $W$ 空间启发了扩散模型的潜空间控制
- GAN 的训练技巧（谱归一化、EMA）被扩散模型借鉴
- 在图像生成领域，GAN 已被 [[04_扩散模型]] 取代，但在快速推理场景仍有价值

## 9. 前沿发展

- **GAN 与扩散融合**：用 GAN 做扩散模型的快速蒸馏
- **轻量化 GAN**：移动端实时生成
- **视频 GAN**：时序一致的视频生成
- **StyleGAN3**：解决纹理粘连问题（aliasing）
- **GAN inversion**：将真实图像反演到 GAN 隐空间进行编辑
