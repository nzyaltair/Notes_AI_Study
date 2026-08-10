# MCMC：马尔可夫链蒙特卡洛

MCMC 构造以目标后验为平稳分布的马尔可夫链，以样本均值近似期望。常见方法包括 Metropolis-Hastings、Gibbs sampling 和 Hamiltonian Monte Carlo / NUTS。

关键诊断：链的混合、有效样本量、收敛诊断与自相关。MCMC 渐近准确但采样昂贵，在高维、强相关后验中常需更高效的参数化或改用变分推断。