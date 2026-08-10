# LLM Serving

LLM Serving 将生成模型作为多租户、流式、可扩缩服务交付。与传统推理不同，预填充（prefill）和逐 token 解码具有不同的计算/访存特性。

核心设计：请求队列、流式协议、调度器、KV Cache、模型副本、鉴权限流和容量隔离。关键指标为首 token 延迟（TTFT）、每 token 时延（TPOT）、端到端延迟、tokens/s、GPU 利用率和错误率。

服务容量必须按输入/输出长度分布、并发、缓存命中率和 SLA 估算，而非只按 QPS。