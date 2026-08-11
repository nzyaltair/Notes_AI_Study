# 神经切线核（NTK）

## 一句话理解

NTK 将无限宽、适当参数化的神经网络在训练初期附近近似为核方法，从而把非线性参数训练转化为可分析的函数空间动力学。

## 定义与直觉

对网络输出 $f_\theta(x)$，核定义为：

$$K_\theta(x,x')=\nabla_\theta f_\theta(x)^\top\nabla_\theta f_\theta(x').$$

在无限宽极限和小参数位移条件下，该核在训练中近似不变；梯度下降的预测演化接近核回归。

## 价值与局限

NTK 可分析收敛、插值和部分泛化现象，也揭示初始化的重要性。但有限宽网络常出现显著特征学习，深度网络的表示会改变；因此 NTK 是有用近似而非完整机制。

## References

- Jacot, Gabriel & Hongler. Neural Tangent Kernel. *NeurIPS*, 2018.
- Lee et al. Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent. *NeurIPS*, 2019.
