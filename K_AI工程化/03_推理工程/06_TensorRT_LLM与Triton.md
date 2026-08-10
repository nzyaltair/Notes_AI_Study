# TensorRT-LLM 与 Triton

TensorRT-LLM 面向 NVIDIA GPU，通过图优化、定制 kernel、量化和并行策略提升 LLM 性能；Triton Inference Server 提供多模型服务、批处理、模型仓库和监控能力。

二者常结合：TensorRT-LLM 负责高性能执行，Triton 负责服务编排。需验证模型/硬件兼容、构建时间、引擎可移植性和版本回滚路径。