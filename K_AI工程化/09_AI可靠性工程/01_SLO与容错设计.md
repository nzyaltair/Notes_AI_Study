---
tags:
  - 可靠性工程
  - SRE
  - SLO
  - 错误预算
  - 容错设计
  - 降级策略
  - AI工程
created: 2026-08-10
updated: 2026-08-10
---

# SLO 与容错设计

## 一句话理解

通过为 AI 系统定义多维度 SLI/SLO 与错误预算，并在检索、工具、模型等依赖失效时采用分层容错模式（超时、重试、熔断、降级），保证系统在部分故障下仍能以可预测的方式安全降级——宁可告知能力受限，也不伪造确定答案。

## 1. 为什么 AI 系统需要特别的 SLO 框架

传统 Web 服务的 SLO 通常只关注**可用性**和**延迟**，因为请求要么成功要么失败，响应时间即可衡量质量。AI 系统则引入了新的不确定性维度：

| 维度 | 传统 Web 服务 | AI/LLM 服务 |
|:---|:---|:---|
| 输出质量 | 不适用（HTTP 200 即正确） | 模型可能返回格式正确但语义错误的答案 |
| 非确定性 | 幂等请求返回相同结果 | 相同输入可能产生不同输出 |
| 依赖链路 | 数据库、缓存 | 检索服务、工具 API、模型推理、安全过滤 |
| 故障模式 | 超时、5xx | 幻觉、拒答失效、工具误用、上下文溢出 |
| 成本 | 与 QPS 线性相关 | 与 token 数、推理时长、GPU 占用相关 |

因此，AI 系统的 SLO 必须覆盖**服务可用性、延迟、任务成功率、输出质量和安全拦截**五个维度，并针对 LLM 推理、RAG 检索、Agent 工具调用等子链路分别定义。

## 2. SLI / SLO / SLA / 错误预算

### 2.1 核心定义

| 概念 | 全称 | 含义 |
|:---|:---|:---|
| **SLI** | Service Level Indicator | 实际测量的服务指标，如"过去 5 分钟 P99 延迟" |
| **SLO** | Service Level Objective | SLI 的目标值，如"P99 延迟 < 2s（月度 99.9%）" |
| **SLA** | Service Level Agreement | 对外合同承诺，通常比 SLO 更宽松，违约需赔偿 |
| **错误预算** | Error Budget | 1 - SLO，允许的"不达标"空间 |

**错误预算公式**：

```text
错误预算 = 1 - SLO 目标
月度可用 SLO = 99.9% → 月度错误预算 = 0.1%
一个月 ≈ 43200 分钟 → 错误预算 ≈ 43.2 分钟
```

### 2.2 错误预算的使用原则

错误预算是**可靠性与发展速度的平衡机制**：

- **预算未耗尽** → 允许激进发布、新功能上线
- **预算即将耗尽** → 冻结高风险变更，专注于稳定性修复
- **预算已耗尽** → 必须停止所有非紧急变更，进行根因分析

```text
每月初：错误预算 = 100%
  ↓ 正常运行 → 预算缓慢消耗
  ↓ 故障发生 → 预算按故障时长扣减
  ↓ 预算 < 20%（告警阈值）
     → 通知团队，限制发布频率
  ↓ 预算 = 0%
     → 冻结所有非紧急变更
     → 必须完成 Post-mortem 并实施修复
```

### 2.3 基于 burn rate 的告警

直接监控错误预算的绝对值反应太慢，SRE 实践使用 **burn rate（消耗速率）** 实现快速告警：

```text
burn rate = 实际错误率 / 允许错误率
         = 实际错误率 / (1 - SLO)

示例：SLO = 99.9%（允许错误率 0.1%）
  实际错误率 1%  → burn rate = 10（消耗速度是正常的 10 倍）
  实际错误率 10% → burn rate = 100（严重过载）
```

| burn rate | 时间窗口 | 含义 | 告警级别 |
|:---|:---|:---|:---|
| 14.4 | 1 小时 | 快速消耗，2% 预算/小时 | P1（立即响应） |
| 6 | 6 小时 | 中速消耗，2% 预算/6 小时 | P2（1 小时内响应） |
| 3 | 3 天 | 慢速消耗，10% 预算/3 天 | P3（工作日内响应） |
| 1 | 28 天 | 正常消耗速率 | 仅仪表盘展示 |

> 多窗口策略（如 1h + 5h 同时触发）可减少误报。

## 3. AI 系统的多维度 SLO

### 3.1 五维度 SLO 模板

| SLO 类型 | SLI 定义 | 示例目标 | 测量方式 |
|:---|:---|:---|:---|
| **服务可用性** | HTTP 200 且非超时的请求比例 | > 99.9%（月均停机 < 43 分钟） | 网关日志 |
| **延迟** | P99 TTFT（首 token 时间）/ P99 总时长 | TTFT < 2s；总时长 < 30s | 推理引擎指标 |
| **任务成功率** | Agent 任务成功完成的比例 | > 95% | 任务追踪系统 |
| **输出质量** | 在线评测得分相对基线的比例 | > 基线 95% | 抽样 + 自动评测 |
| **安全拦截率** | 有害内容被正确拒答的比例 | > 99.5% | 安全测试集 |

### 3.2 分子链路 SLO

AI 服务通常由多个子链路组成，每个子链路需独立定义 SLO：

```text
用户请求
  ├─ 输入安全过滤   SLO: P99 < 100ms, 拦截准确率 > 99%
  ├─ RAG 检索       SLO: P99 < 500ms, 召回率 > 90%
  ├─ LLM 推理       SLO: P99 TTFT < 2s, 吞吐 > 1000 token/s
  ├─ 工具调用       SLO: 成功率 > 99%, P99 < 5s
  └─ 输出安全检查   SLO: P99 < 200ms, 漏检率 < 0.5%
```

**关键原则**：端到端 SLO 必须低于任何单链路 SLO，因为故障会叠加。

```text
端到端可用性 ≈ 各链路可用性的乘积
  0.999 × 0.999 × 0.995 × 0.999 × 0.999 ≈ 0.991
  → 端到端 SLO 最多设为 99%（而非 99.9%）
```

### 3.3 SLI 仪表盘设计

一个完整的 AI 服务监控仪表盘应包含：

- **RED 指标**：Rate（QPS）、Errors（错误率）、Duration（延迟分布）
- **USE 指标**：Utilization（GPU 利用率）、Saturation（队列深度）、Errors（推理错误）
- **质量指标**：在线评测得分趋势、幻觉率、拒答率
- **成本指标**：日均 token 消耗、单请求平均成本
- **错误预算**：剩余预算百分比、burn rate 趋势

## 4. 容错设计模式

### 4.1 基础容错模式总览

| 模式 | 作用 | 适用场景 | AI 典型应用 |
|:---|:---|:---|:---|
| **超时** | 防止慢请求阻塞资源 | 所有外部调用 | LLM 生成超时 → 截断输出或放弃 |
| **重试** | 处理瞬时失败 | 幂等或可接受重复的操作 | 推理 API 5xx → 指数退避重试 3 次 |
| **熔断** | 防止级联故障 | 依赖服务不稳定 | LLM API 错误率 > 10% → 暂停调用 |
| **限流** | 保护服务容量 | 防止过载 | 按 token/分钟限制用户请求 |
| **隔离** | 防止故障传播 | 不同租户/任务 | Agent 工具在沙箱中执行 |
| **缓存** | 减少下游依赖 | 重复或相似请求 | 语义缓存相似查询的结果 |
| **降级** | 用简化服务替代完整服务 | 主服务不可用 | 大模型不可用 → 小模型替代 |

### 4.2 超时设计

AI 系统的超时需区分**流式**和**非流式**：

```python
# 非流式推理：总超时
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    timeout=30  # 总超时 30 秒
)

# 流式推理：分阶段超时
# TTFT 超时：首个 token 必须在 3 秒内到达
# token 间隔超时：token 间间隔不超过 5 秒
# 总时长超时：总生成时间不超过 60 秒
```

**超时层级**：外层超时应大于内层超时之和，避免外层先触发导致内层无法完成重试。

```text
用户请求超时（60s）
  └─ RAG 检索超时（5s，可重试 2 次 → 最多 15s）
  └─ LLM 推理超时（30s）
  └─ 工具调用超时（10s/工具，最多 3 个工具 → 30s）
```

### 4.3 重试与指数退避

```python
import asyncio
import random

async def call_llm_with_retry(prompt, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            response = await llm_client.generate(prompt)
            if response.status == 200:
                return response
            if response.status >= 500:  # 服务端错误可重试
                raise Exception(f"Server error: {response.status}")
            if response.status == 429:  # 限流
                retry_after = int(response.headers.get("Retry-After", base_delay * 2 ** attempt))
                await asyncio.sleep(retry_after)
                continue
            return response  # 4xx 客户端错误不重试
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            # 指数退避 + 抖动（防止重试风暴）
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(delay)
```

**AI 特有重试注意事项**：
- 非确定性输出：重试可能返回不同结果，需判断是否可接受
- 成本敏感：每次重试都消耗 token，需限制重试次数
- 幂等性：Agent 工具调用（如发邮件）必须幂等才能安全重试

### 4.4 熔断器模式

熔断器有三个状态：**关闭（正常）→ 打开（熔断）→ 半开（探测恢复）**。

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60, success_threshold=3):
        self.failure_count = 0
        self.failure_threshold = failure_threshold  # 连续失败 5 次触发熔断
        self.recovery_timeout = recovery_timeout    # 60 秒后进入半开
        self.success_threshold = success_threshold  # 半开状态连续成功 3 次恢复
        self.state = "closed"
        self.last_failure_time = None

    async def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
            else:
                raise CircuitBreakerOpenError("服务熔断中，请稍后重试")

        try:
            result = await func(*args, **kwargs)
            if self.state == "half_open":
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = "closed"
                    self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            raise
```

### 4.5 限流策略

AI 服务的限流维度比传统服务更丰富：

| 限流维度 | 示例 | 说明 |
|:---|:---|:---|
| QPS | 100 req/s | 每秒请求数 |
| TPM | 100K token/min | 每分钟 token 数（OpenAI 标准） |
| 并发 | 10 concurrent | 并发推理请求数 |
| 用户级 | 50 req/user/hour | 防止单用户滥用 |
| 租户级 | 1M token/team/day | 团队级配额 |
| GPU 级 | 8 concurrent/batch | 单 GPU 并发批处理数 |

```python
# 基于 token bucket 的多维度限流
class TokenBucketLimiter:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity      # 桶容量
        self.refill_rate = refill_rate  # 每秒补充速率
        self.tokens = capacity
        self.last_refill = time.time()

    def consume(self, tokens=1):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
```

## 5. AI 特有降级策略

### 5.1 分层降级路径

AI 系统降级的核心原则：**保持透明，不伪造确定性**。

```text
正常路径:  用户请求 → 输入安全检查 → RAG 检索 → LLM 生成 → 输出检查 → 带引用的答案
                                    ↓ 检索服务故障
降级路径1: 用户请求 → 安全检查 → LLM 生成（无上下文）→ 声明"知识可能不完整"
                                    ↓ LLM 服务故障
降级路径2: 用户请求 → 安全检查 → 小模型生成或缓存答案 → 声明"服务受限"
                                    ↓ 完全不可用
降级路径3: 用户请求 → 静态提示"服务暂时不可用，请稍后重试"
```

### 5.2 降级策略对比

| 降级策略 | 触发条件 | 用户体验 | 安全性 | 实现复杂度 |
|:---|:---|:---|:---|:---|
| 语义缓存命中 | 相似查询已缓存 | 良好（快速） | 需验证缓存时效性 | 中 |
| 小模型替代 | 大模型不可用 | 可接受（质量略降） | 高（确定性输出） | 低 |
| 去除 RAG 上下文 | 检索服务故障 | 需声明限制 | 中（可能增加幻觉） | 低 |
| 静态回复 | 完全不可用 | 差但明确 | 最高 | 最低 |
| 人工接管 | 高风险或连续失败 | 最好但最慢 | 最高 | 高 |

### 5.3 Agent 特有容错

Agent 系统由于涉及多步推理和工具调用，容错设计更为复杂：

```python
class AgentFaultTolerance:
    """Agent 容错配置"""
    max_steps: int = 20               # 最大推理步数
    max_execution_time: int = 300     # 最大执行时间（秒）
    max_cost: float = 1.0             # 最大成本（美元）
    max_tool_errors: int = 3          # 单工具最大连续失败次数
    checkpoint_interval: int = 5      # 每 5 步保存检查点

    async def execute_with_limits(self, task):
        """带限制的任务执行"""
        checkpoints = []
        for step in range(self.max_steps):
            # 检查时间和成本限制
            if self.elapsed_time() > self.max_execution_time:
                return self.fallback_to_human(task, "超时")
            if self.total_cost() > self.max_cost:
                return self.fallback_to_human(task, "成本超限")

            try:
                result = await self.execute_step(step)
                if step % self.checkpoint_interval == 0:
                    checkpoints.append(self.save_checkpoint(step))
            except ToolError as e:
                if self.tool_error_count(e.tool) >= self.max_tool_errors:
                    # 工具连续失败 → 跳过或换用替代工具
                    result = await self.use_fallback_tool(e.tool, step)
                else:
                    continue
            except Exception:
                # 未知错误 → 尝试从最近检查点恢复
                if checkpoints:
                    self.restore_checkpoint(checkpoints[-1])
                else:
                    return self.fallback_to_human(task, "执行失败")
        return result
```

### 5.4 检查点与恢复

长时 Agent 任务必须支持检查点持久化，以便中断后恢复：

```text
检查点内容：
  ├─ 当前推理步数和状态
  ├─ 已完成的工具调用及结果
  ├─ 对话历史（或摘要）
  ├─ 中间变量和文件
  └─ 预估剩余成本

恢复流程：
  中断检测 → 加载最新检查点 → 验证上下文一致性 → 从断点继续
```

## 6. 超时与重试的参数调优

### 6.1 超时参数推荐

| 场景 | TTFT 超时 | 总时长超时 | token 间隔超时 |
|:---|:---|:---|:---|
| 短对话（< 500 token） | 2s | 10s | 3s |
| 长文本生成（> 2K token） | 3s | 60s | 5s |
| Agent 多步推理 | 5s | 300s | 10s |
| 流式嵌入向量 | N/A | 5s | N/A |

### 6.2 重试参数推荐

| 场景 | 最大重试次数 | 基础退避 | 抖动范围 |
|:---|:---|:---|:---|
| LLM 推理 API | 3 | 1s | 0-1s |
| RAG 检索 | 2 | 0.5s | 0-0.5s |
| 工具调用（幂等） | 3 | 2s | 0-2s |
| 工具调用（非幂等） | 0 | N/A | N/A |

> **非幂等工具**（如发送邮件、创建订单）不应自动重试，需通过幂等键或人工确认处理。

## 7. 监控与告警

### 7.1 告警分级

| 级别 | 触发条件 | 响应时间 | 通知方式 |
|:---|:---|:---|:---|
| P0 | 服务完全不可用 | 立即 | 电话 + 短信 + IM |
| P1 | SLO burn rate > 14.4 | 5 分钟 | 电话 + IM |
| P2 | 错误率 > 5% 或 P99 延迟翻倍 | 30 分钟 | IM |
| P3 | 质量指标下降 > 5% | 工作日 | IM 群通知 |
| P4 | 仅仪表盘展示 | 无 | 日报 |

### 7.2 告警规则示例

```yaml
# Prometheus 告警规则示例
groups:
  - name: ai_service_slo
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m]))
          / sum(rate(http_requests_total[5m])) > 0.01
        for: 2m
        labels:
          severity: P1
        annotations:
          summary: "AI 服务错误率超过 1%"
          runbook: "https://wiki/runbooks/ai-high-error-rate"

      - alert: HighTTFT
        expr: |
          histogram_quantile(0.99, rate(llm_ttft_seconds_bucket[5m])) > 3
        for: 5m
        labels:
          severity: P2
        annotations:
          summary: "P99 TTFT 超过 3 秒"

      - alert: QualityDegradation
        expr: |
          avg_over_time(online_eval_score[1h]) < 0.85
        for: 10m
        labels:
          severity: P3
        annotations:
          summary: "在线评测得分低于基线 85%"
```

## 8. 设计检查清单

在部署 AI 服务前，应逐项确认以下容错设计：

- [ ] **SLO 定义**：五个维度（可用性、延迟、成功率、质量、安全）均有明确 SLO
- [ ] **错误预算**：已计算月度错误预算并设置 burn rate 告警
- [ ] **超时**：所有外部调用（LLM、检索、工具）均有超时设置
- [ ] **重试**：可重试调用使用指数退避 + 抖动，非幂等调用不自动重试
- [ ] **熔断**：依赖服务有熔断器，熔断后返回降级响应
- [ ] **限流**：按 QPS、TPM、并发、用户、租户多维度限流
- [ ] **降级路径**：至少定义 3 级降级路径，每级有明确触发条件
- [ ] **Agent 限制**：最大步数、最大时间、最大成本均已配置
- [ ] **检查点**：长时任务支持检查点保存与恢复
- [ ] **监控**：RED + USE + 质量 + 成本指标全覆盖
- [ ] **告警**：每条告警绑定 Runbook URL
- [ ] **混沌测试**：已模拟 LLM 不可用、检索故障、工具超时等场景

## 9. 子主题导航

- [[00_AI可靠性工程综述]]：可靠性工程总览
- [[02_模型失败与漂移处理]]：漂移检测与质量回归
- [[03_事件响应与恢复]]：On-Call、Runbook、Post-mortem

## 10. 相关知识

- [[../05_MLOps/00_MLOps综述]]：MLOps 运维基础
- [[../06_LLMOps与AgentOps/00_LLMOps与AgentOps综述]]：LLM 运行管理
- [[../07_AI评测体系/00_AI评测体系_综述]]：在线质量监控方法
- [[../08_AI基础设施/00_AI基础设施综述]]：硬件层可靠性（GPU 故障、RDMA）

## References

- Beyer et al., *Site Reliability Engineering* (Google, 2016) — Ch. 4 Service Level Objectives
- Beyer et al., *The Site Reliability Workbook* (Google, 2018) — Ch. 5 Alerting on SLOs
- Kleppmann, *Designing Data-Intensive Applications* (O'Reilly, 2017) — Ch. 12 The Future of Data Systems
- Netflix, *Hystrix: Latency and Fault Tolerance for Distributed Systems* (2012)
- Microsoft, *Azure OpenAI Service Quotas and Limits* (2024)
- OpenAI, *Production Best Practices for Rate Limits and Retries* (2024)
