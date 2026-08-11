# SLAM 与三维重建

## 一句话理解

SLAM 在未知环境中同时估计传感器位姿和构建地图——它是机器人自主导航、AR/VR 空间定位和场景数字化的核心技术，连接了"我在哪里"和"环境长什么样"两个根本问题。

## 概述

SLAM（Simultaneous Localization and Mapping）和三维重建是空间智能的核心技术。SLAM 解决"实时定位+建图"的在线问题，三维重建解决"多视图→稠密场景"的离线或在线问题。

## 核心方法

| 方法 | 类型 | 特点 | 代表 |
|:---|:---|:---|:---|
| 特征点 SLAM | 稀疏 SLAM | 快速稳定，依赖特征 | ORB-SLAM 系列 |
| 直接法 SLAM | 半稠密/稠密 | 利用像素亮度，弱纹理鲁棒 | LSD-SLAM, DSO |
| VIO | 视觉+惯性 | IMU 辅助，鲁棒性高 | VINS-Mono, ORB-SLAM3 |
| RGB-D SLAM | 深度 SLAM | 直接获取深度 | KinectFusion, RGB-D SLAM |
| 神经 SLAM | 学习型 SLAM | 端到端可微 | DROID-SLAM, DPVO |

## 关键挑战

- **动态场景**：运动物体干扰定位和建图
- **大场景**：城市级场景的存储和计算
- **长期漂移**：长时间运行后的位姿累积误差
- **光照变化**：室外光照变化影响特征匹配
- **闭环检测**：识别已访问区域以消除累积误差

## 相关知识

- **前置知识**：[[01_三维视觉与多视角几何]]、[[02_深度估计与立体视觉]]
- **平级主题**：[[05_神经场与GaussianSplatting]]
- **后续延伸**：[[06_空间推理与场景记忆]]、[[E_感知与多模态智能/07_感知与行动接口/02_感知状态与世界表示|感知状态]]

## References

- Mur-Artal, R. et al. (2015). ORB-SLAM: A Versatile and Accurate Monocular SLAM System. *IEEE TRO*.
- Campos, C. et al. (2021). ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multi-Map SLAM. *IEEE TRO*.
- Engel, J. et al. (2014). LSD-SLAM: Large-Scale Direct Monocular SLAM. *ECCV 2014*.
- Qin, T. et al. (2018). VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator. *IEEE TRO*.
- Teed, Z. & Deng, J. (2021). DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras. *NeurIPS 2021*.
- Newcombe, R. A. et al. (2011). KinectFusion: Real-Time Dense Surface Mapping and Tracking. *ISMAR 2011*.