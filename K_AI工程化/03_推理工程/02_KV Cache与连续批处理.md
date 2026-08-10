# KV Cache 与连续批处理

KV Cache 缓存注意力层已处理 token 的 key/value，避免每步解码重复计算前缀；代价是显存随层数、上下文长度、并发持续增长。

连续批处理允许请求在不同生成时刻动态加入/退出批次，提高 GPU 利用率。调度器需在吞吐、公平性、TTFT 和缓存碎片间权衡。常见优化包括 paged attention、前缀缓存、chunked prefill、缓存逐出和租户配额。