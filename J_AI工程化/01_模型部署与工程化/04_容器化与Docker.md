# 容器化与 Docker

## 1. 概述

容器化是现代 AI 部署的标准方式，将应用及其全部依赖打包为可移植的容器镜像，实现"一次构建，到处运行"。在 LLM 部署中，容器化将推理引擎、模型权重、CUDA 运行时统一打包，确保跨环境一致性。

容器化解决的核心问题：
- **环境一致性**：消除"开发环境能跑、生产环境报错"的问题
- **依赖隔离**：不同模型可使用不同版本的 CUDA、Python、PyTorch
- **快速部署**：秒级启动，支持弹性扩缩容
- **标准化交付**：容器镜像作为部署的标准交付物

传统部署中，模型依赖（CUDA 版本、Python 库、系统库）的环境不一致是生产事故的主要来源。容器化通过将应用和依赖打包为不可变镜像，从根本上解决了这一问题。

## 2. 发展历史

| 年代 | 里程碑 | 意义 |
|:---|:---|:---|
| 2013 | Docker 发布 | 容器镜像概念，应用打包标准化 |
| 2014 | Kubernetes 发布 | Google 开源容器编排系统 |
| 2015 | Docker Compose | 多容器编排工具 |
| 2017 | Kubernetes 1.0 | 成为容器编排事实标准 |
| 2018 | NVIDIA Container Toolkit | 容器内 GPU 访问支持 |
| 2019 | K8s GPU Operator | K8s 上 GPU 资源管理自动化 |
| 2020 | 容器化 ML 成为标准 | TF Serving/Triton 容器部署 |
| 2022 | vLLM 官方镜像 | LLM 推理服务容器化 |
| 2023 | Serverless 容器 | AWS Fargate / Google Cloud Run |
| 2024 | K8s AI 工作负载标准 | GPU 调度、多租户隔离成熟 |

## 3. 核心概念

### 3.1 Docker 核心概念

| 概念 | 说明 |
|:---|:---|
| **镜像（Image）** | 只读模板，包含应用和依赖，通过分层文件系统存储 |
| **容器（Container）** | 镜像的运行实例，具有独立的进程、网络和文件系统空间 |
| **Dockerfile** | 构建镜像的指令文件，每条指令对应一层 |
| **仓库（Registry）** | 镜像存储和分发服务（Docker Hub / Harbor） |
| **卷（Volume）** | 持久化存储，独立于容器生命周期 |
| **网络（Network）** | 容器间通信的虚拟网络 |

### 3.2 容器 vs 虚拟机

| 维度 | 容器 | 虚拟机 |
|:---|:---|:---|
| 隔离级别 | 进程级 | 硬件级 |
| 启动时间 | 秒级 | 分钟级 |
| 资源开销 | 低（共享内核） | 高（完整 OS） |
| 镜像大小 | MB 级 | GB 级 |
| 密度 | 单机数百容器 | 单机数十 VM |

### 3.3 Kubernetes 编排

| K8s 概念 | AI 部署中的作用 |
|:---|:---|
| **Pod** | 模型推理实例，通常一 Pod 一 GPU |
| **Deployment** | 管理 Pod 副本数和滚动更新 |
| **Service** | 负载均衡和服务发现 |
| **HPA** | 基于 GPU 利用率或 QPS 自动扩缩容 |
| **ConfigMap** | 模型配置和推理参数管理 |
| **PV/PVC** | 模型文件持久化存储 |

## 4. 技术原理

### 4.1 LLM 服务 Dockerfile

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install vllm torch transformers
COPY model/ /app/model/
EXPOSE 8000
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "/app/model/", "--port", "8000"]
```

### 4.2 GPU 容器化

NVIDIA Container Toolkit 使容器内可以访问宿主机 GPU。通过 Docker Compose 声明 GPU 资源：

```yaml
services:
  llm-server:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./model:/model
    command: --model /model --port 8000
```

### 4.3 UnionFS 分层存储

镜像通过 OverlayFS 等联合文件系统实现分层叠加：
- 相同层只存储一份，节省磁盘空间和拉取时间
- 每条 Dockerfile 指令生成一层，层可缓存复用
- 构建时从上到下叠加，运行时呈现统一文件系统

### 4.4 资源隔离机制

- **cgroups**：限制容器的 CPU、内存、GPU 等资源使用量
- **namespace**：为容器提供独立的进程、网络、挂载和用户空间
- **seccomp**：限制容器可用的系统调用，增强安全性

### 4.5 GPU 调度

- **NVIDIA Device Plugin**：K8s 感知和管理 GPU 资源
- **GPU 共享**：时间切片（vGPU）或 MIG（A100/H100 硬件分区）
- **显存管理**：通过 `gpu-memory-utilization` 参数限制单容器显存占用

### 4.6 镜像优化策略

- **多阶段构建**：编译阶段与运行阶段分离，减小最终镜像大小
- **模型分离**：模型文件用 Volume 挂载而非打入镜像，避免镜像膨胀
- **层缓存**：将频繁变更的指令放在 Dockerfile 末尾，利用构建缓存
- **基础镜像选择**：使用 `runtime` 而非 `devel` 镜像，减少体积

## 5. 关键方法与模型

### 5.1 滚动更新

Kubernetes 逐个替换 Pod，通过 `maxSurge` 和 `maxUnavailable` 控制更新速率，确保零停机部署。

### 5.2 金丝雀部署

通过 Ingress 注解控制流量分配，小流量验证新版本：

```yaml
annotations:
  nginx.ingress.kubernetes.io/canary: "true"
  nginx.ingress.kubernetes.io/canary-weight: "10"  # 10% 流量到新版本
```

### 5.3 GitOps 部署

声明式部署：Git 仓库存储期望状态，ArgoCD/Flux 自动同步集群状态到 Git 声明，实现可追溯的自动化部署。

## 6. 优势与局限

### 6.1 优势

- **环境一致性**：消除环境差异导致的问题
- **快速启动**：秒级启动，支持弹性扩缩容
- **资源高效**：共享内核，密度高
- **标准化交付**：镜像作为不可变交付物

### 6.2 局限

- **镜像体积大**：LLM 推理镜像可达数 GB（含 CUDA、PyTorch）
- **GPU 驱动依赖**：宿主机需安装正确版本的 NVIDIA 驱动
- **冷启动慢**：大模型加载到显存需数十秒
- **安全风险**：容器逃逸攻击（需通过 seccomp/AppArmor 缓解）

## 7. 应用场景

| 场景 | 典型方案 |
|:---|:---|
| LLM 服务部署 | vLLM/TGI 容器化部署到 K8s 集群 |
| 训练任务容器化 | 保证训练环境可复现 |
| 开发环境统一 | DevContainer 提供团队一致的开发环境 |
| CI/CD 制品 | 容器镜像作为部署的标准交付物 |
| 混合云部署 | 相同镜像跨云/本地运行 |

## 8. 与其他技术关系

- 容器化是 [[01_模型服务化]] 的标准部署方式
- [[03_云端与本地部署]] 依赖容器实现跨环境一致性
- [[05_监控与日志]] 需要收集容器内应用的指标和日志
- [[06_版本管理与回滚]] 通过镜像标签管理模型版本
- [[09_ONNX模型导出]] 模型通过容器部署到不同硬件环境

## 9. 前沿发展

- **Serverless 容器**：AWS Fargate / Google Cloud Run，无需管理节点
- **Wasm 容器**：WebAssembly 作为轻量容器替代，启动更快
- **机密容器**：硬件加密的容器执行环境，保护模型和数据
- **镜像加速**：Lazy Pulling（Stargz/Nydus）减少镜像拉取时间
- **K8s AI 工作负载标准**：GPU 调度、多租户隔离的标准化

返回 [[00_模型部署与工程化]]
