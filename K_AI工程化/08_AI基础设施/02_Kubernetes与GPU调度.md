---
tags:
  - AI基础设施
  - Kubernetes
  - GPU调度
  - Gang Scheduling
  - AI工程
created: 2026-08-10
updated: 2026-08-10
---

# Kubernetes 与 GPU 调度

## 一句话理解

Kubernetes 是 AI 集群的"操作系统"，通过声明式 API 管理 GPU 资源；训练场景需要 Gang Scheduling 保证多节点同时启动，推理场景需要弹性伸缩和 GPU 共享来提高利用率，两者都要求拓扑感知调度以充分利用 NVLink 互联。

## 1. Kubernetes GPU 管理基础

### 1.1 GPU 设备插件

```text
K8s GPU 管理架构：

┌─────────────────────────────────────────┐
│              API Server                 │
│         (Pod 请求 GPU 资源)              │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│            Scheduler                     │
│    (选择有 GPU 的节点)                    │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│            Kubelet                       │
│    (请求 GPU 设备分配)                    │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│      NVIDIA Device Plugin               │
│    (管理 GPU 设备，分配/回收)             │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│         GPU 硬件 (nvidia.com/gpu)        │
└─────────────────────────────────────────┘

资源声明示例：
resources:
  limits:
    nvidia.com/gpu: 4  # 请求 4 个 GPU
```

### 1.2 GPU 资源类型

| 资源类型 | 说明 | 使用方式 |
|:---|:---|:---|
| **nvidia.com/gpu** | 整卡分配 | 一个 Pod 独占 GPU |
| **nvidia.com/mig-1g.10gb** | MIG 切分（1G 10GB） | GPU 硬件分区 |
| **nvidia.com/mig-2g.20gb** | MIG 切分（2G 20GB） | 中等粒度切分 |
| **nvidia.com/mig-7g.80gb** | MIG 切分（7G 80GB） | 近乎整卡 |
| **nvidia.com/gpu.shared** | 时间切片共享 | 多 Pod 共享 GPU（推理） |

### 1.3 设备插件扩展能力

| 能力 | 说明 | 工具 |
|:---|:---|:---|
| **GPU 发现** | 自动发现节点 GPU | nvidia-device-plugin |
| **MIG 管理** | 配置 MIG 分区 | nvidia-mig-manager |
| **GPU 共享** | 时间切片 | GPU Time-Slicing |
| **拓扑发现** | GPU 拓扑信息 | Node Feature Discovery |
| **健康检查** | GPU 健康状态 | DCGM Exporter |
| **监控指标** | GPU 利用率/温度 | DCGM + Prometheus |

## 2. 训练调度

### 2.1 Gang Scheduling

```text
Gang Scheduling（成组调度）：

问题：分布式训练需要 N 个 Pod 同时启动
      如果只有部分 Pod 被调度，训练无法开始
      已调度的 Pod 占着 GPU 空等

解决：Gang Scheduling
      所有 Pod 要么全部调度成功，要么全部等待

┌──────────────────────────────────────────┐
│  训练作业：8 Pod × 8 GPU = 64 GPU       │
│                                          │
│  Pod 0 ✅  Pod 1 ✅  Pod 2 ✅  Pod 3 ✅  │
│  Pod 4 ✅  Pod 5 ✅  Pod 6 ✅  Pod 7 ✅  │
│                                          │
│  → 全部就绪，开始训练                     │
│                                          │
│  如果只有 56 个 GPU 可用：               │
│  Pod 0-6 就绪，Pod 7 等待               │
│  → Pod 0-6 也等待（不占 GPU）            │
│  → 直到有 64 GPU 可用，全部一起启动      │
└──────────────────────────────────────────┘

实现工具：
├── Volcano（华为，最广泛使用）
├── Kueue（K8s 官方 batch 调度）
├── Koordinator（阿里巴巴）
├── KAIWO（Kaito + 自动化）
└── Scheduler Plugins（K8s Sig-scheduling）
```

### 2.2 Volcano 调度器

```yaml
# Volcano PodGroup 示例
apiVersion: scheduling.volcano.sh/v1beta1
kind: PodGroup
metadata:
  name: llm-training-job
spec:
  minMember: 8              # 最少 8 个 Pod 同时启动
  priorityClassName: high
  queue: training-queue
  plugins:
    gang:                   # Gang Scheduling
    - name: gang
    - name: binpack         # 尽量打包到同一节点
    - name: topology        # 拓扑感知
---
apiVersion: batch/v1
kind: Job
metadata:
  name: llm-training
spec:
  parallelism: 8
  template:
    spec:
      schedulerName: volcano  # 使用 Volcano 调度器
      containers:
      - name: trainer
        resources:
          limits:
            nvidia.com/gpu: 8
```

### 2.3 抢占与优先级

```text
训练作业优先级管理：

优先级层级：
├── P0：生产推理服务（不可抢占）
├── P1：关键训练作业（高优先级）
├── P2：常规训练作业（中优先级）
├── P3：实验性训练（低优先级，可被抢占）
└── P4：批处理任务（最低优先级）

抢占规则：
├── 高优先级作业资源不足时，抢占低优先级作业
├── 被抢占作业需要支持 checkpoint 恢复
├── 抢占需优雅终止（SIGTERM → SIGKILL）
└── 训练作业应定期保存 checkpoint

抢占保护：
├── gracePeriodSeconds: 300  # 5 分钟优雅退出
├── 保存最终 checkpoint
├── 通知作业管理器
└── 重新排队等待资源
```

### 2.4 拓扑感知调度

```text
GPU 拓扑感知调度：

目标：将分布式训练的 Pod 调度到网络距离最近的节点

拓扑层次（从优到劣）：
1. 同节点（NVLink/NVSwitch）
   └── 900 GB/s 带宽，最佳

2. 同机架（InfiniBand 同交换机）
   └── 400 Gb/s，延迟最低

3. 同 POD（同可用区）
   └── 跨交换机 IB，性能稍降

4. 跨可用区
   └── 延迟显著增加，不推荐

实现方式：
├── Node Feature Discovery 标注拓扑信息
├── Volcano binpack 插件优先打包
├── 自定义调度器评分插件
└── Pod anti-affinity 控制分布
```

### 2.5 弹性训练

```text
弹性训练调度：

传统训练：
├── 固定 GPU 数量
├── 节点故障 → 整个作业失败
└── 资源利用率不灵活

弹性训练：
├── GPU 数量可动态变化
├── 节点加入/退出时自动调整
├── 全局 batch size 随 GPU 数调整
└── 需要框架支持（Torch Elastic / Horovod Elastic）

K8s 弹性训练架构：
┌──────────┐
│ ETCD/RDZ │ ←──── 作业状态存储
└────┬─────┘
     │
┌────┴─────┐
│ Worker 0 │ ←── 可增减
├──────────┤
│ Worker 1 │ ←── 可增减
├──────────┤
│ Worker 2 │ ←── 可增减
└──────────┘

适用场景：
├── 抢占式训练（被抢占时缩减规模）
├── Spot Instance 训练（实例回收时缩减）
├── 混部场景（利用空闲资源）
└── 渐进式扩展（逐步增加 GPU）
```

## 3. 推理调度

### 3.1 弹性伸缩

```text
推理服务弹性伸缩：

伸缩触发器：
├── GPU 利用率 > 70% → 扩容
├── 请求队列深度 > 10 → 扩容
├── 延迟 P95 > 阈值 → 扩容
├── GPU 利用率 < 30% → 缩容
└── 定时伸缩（高峰/低谷）

冷启动优化：
├── 模型权重预加载（镜像含权重）
├── 模型权重分布式缓存
├── 预热请求（启动后先跑 dummy 请求）
├── 渐进式放量（新副本先承接 10% 流量）
└── 优先缩容最旧的副本

HPA 配置：
├── metrics: GPU utilization + custom metrics
├── scaleTargetRef: InferenceService / Deployment
├── minReplicas / maxReplicas
└── scaleDown stabilizationWindowSeconds: 300
```

### 3.2 GPU 共享

```text
GPU 共享方案：

方案 1：MIG（硬件隔离）
├── 物理隔离，无干扰
├── H100 可切 7 个 MIG 实例
├── 适合：不同租户/不同服务
└── 局限：切分粒度固定，不支持动态调整

方案 2：时间切片（Time-Slicing）
├── 多 Pod 轮流使用同一 GPU
├── 软件隔离，有上下文切换开销
├── 适合：推理负载低、延迟要求不高
└── 局限：无内存隔离，互相影响

方案 3：MPS（Multi-Process Service）
├── 多进程共享 GPU 上下文
├── 减少上下文切换开销
├── 适合：同租户多服务
└── 局限：无内存隔离（需信任租户）

选型决策：
├── 多租户隔离 → MIG
├── 低成本推理共享 → Time-Slicing
├── 同租户高吞吐 → MPS
└── 关键服务 → 独占 GPU
```

### 3.3 多租户配额

```yaml
# 多租户 GPU 配额管理
apiVersion: v1
kind: ResourceQuota
metadata:
  name: team-a-gpu-quota
  namespace: team-a
spec:
  hard:
    requests.nvidia.com/gpu: "32"      # 最多申请 32 GPU
    requests.memory: "2Ti"
    requests.cpu: "500"
---
# LimitRange 控制单 Pod 请求
apiVersion: v1
kind: LimitRange
metadata:
  name: gpu-limits
  namespace: team-a
spec:
  limits:
  - type: Container
    max:
      nvidia.com/gpu: "8"  # 单容器最多 8 GPU
```

## 4. 调度器对比

| 调度器 | 维护方 | 核心能力 | 适用场景 |
|:---|:---|:---|:---|
| **默认调度器** | K8s 社区 | 基础调度 | 简单推理部署 |
| **Volcano** | 华为 | Gang Scheduling + 队列 + 抢占 | 分布式训练 |
| **Kueue** | K8s WG | Job 排队 + 配额管理 | 训练作业排队 |
| **Koordinator** | 阿里 | 混部 + GPU 共享 | 训练+推理混部 |
| **YuniKorn** | Apache | 大规模队列调度 | HPC 场景 |
| **Karmada** | CNCF | 多集群调度 | 多集群 GPU 管理 |

## 5. 作业管理

### 5.1 训练作业模板

```text
训练作业管理需求：

├── 作业生命周期
│   ├── 提交 → 排队 → 调度 → 运行 → 完成/失败
│   ├── 挂起/恢复
│   └── 自动重试（节点故障后）
│
├── 检查点管理
│   ├── 定期保存到分布式存储
│   ├── 故障后自动从 checkpoint 恢复
│   └── 训练完成后归档
│
├── 资源管理
│   ├── GPU/CPU/内存配额
│   ├── 优先级和抢占
│   └── 弹性伸缩
│
├── 可观测性
│   ├── 训练日志收集
│   ├── 指标导出（loss/accuracy/GPU 利用率）
│   └── 事件通知
│
└── 多用户支持
    ├── 命名空间隔离
    ├── 配额管理
    └── 成本归因
```

### 5.2 推理服务管理

```text
推理服务管理需求：

├── 模型管理
│   ├── 模型权重存储与分发
│   ├── 版本管理
│   └── A/B 测试支持
│
├── 流量管理
│   ├── 负载均衡
│   ├── 流量切分（灰度）
│   ├── 自动伸缩
│   └── 限流与熔断
│
├── GPU 资源
│   ├── GPU 共享/MIG
│   ├── 多模型混部
│   └── 显存管理
│
├── 延迟优化
│   ├── 模型预加载
│   ├── 批处理聚合
│   └── GPU 亲和性
│
└── 监控
    ├── QPS / 延迟 / 错误率
    ├── GPU 利用率 / 显存
    └── 成本追踪
```

## 6. 实践建议

### 6.1 集群规划

```text
AI K8s 集群规划要点：

1. 节点池划分
   ├── 训练节点池（H100/A100，NVLink，IB）
   ├── 推理节点池（A10/L40S，成本优化）
   ├── CPU 节点池（数据处理/调度）
   └── GPU 管理节点池（控制面）

2. 网络规划
   ├── 训练网络：InfiniBand/RoCE（无阻塞）
   ├── 存储网络：高速以太网
   ├── 管理网络：普通以太网
   └── 公网：NAT/负载均衡

3. 存储规划
   ├── 训练数据：分布式文件系统（Lustre/GPFS）
   ├── Checkpoint：NVMe → 分布式存储分层
   ├── 模型仓库：对象存储 + CDN 分发
   └── 日志/指标：ELK + Prometheus

4. 容量规划
   ├── GPU 利用率目标：训练 > 50%，推理 > 80%
   ├── 训练排队时间 < 2 小时
   ├── 推理冷启动 < 5 分钟
   └── 预留 10% 资源用于弹性
```

### 6.2 常见问题

| 问题 | 原因 | 解决 |
|:---|:---|:---|
| GPU Pod 一直 Pending | GPU 不足或调度策略问题 | 检查配额/队列/节点资源 |
| GPU 掉卡 | 硬件故障/XID 错误 | 驱逐节点，修复 GPU |
| 训练死锁 | Gang Scheduling 失败 | 检查资源是否足够 |
| 推理冷启动慢 | 模型加载慢 | 预加载/镜像优化 |
| GPU 利用率低 | 数据加载瓶颈 | 增加 DataLoader workers |

## 7. 子主题导航

- [[01_GPU与CUDA生态]]
- [[03_分布式存储与网络]]
- [[04_云原生AI与云平台]]
- [[08_异构硬件与能源调度]]

## 8. 相关知识

- [[00_AI基础设施综述]]
- [[../02_训练工程/01_分布式训练与并行]]（分布式训练架构）
- [[../03_推理工程/00_推理工程_综述]]（推理部署架构）
- [[../05_MLOps/05_容器化与部署]]（容器化部署）
- [[../09_AI可靠性工程/02_故障处理与容错]]（故障恢复）

## References

- Kubernetes, *Scheduling GPU Workloads on Kubernetes* (2024)
- Volcano, *Volcano: A Cloud Native Batch System* (CNCF)
- NVIDIA, *NVIDIA GPU Operator Documentation* (2024)
- Kueue, *Kueue: Kubernetes Native Job Queueing* (2024)
- Meta, *Building Meta's GenAI Infrastructure* (2024)
