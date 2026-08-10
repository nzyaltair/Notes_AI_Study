# GPU 与 CUDA 生态

GPU 性能受计算能力、显存容量/带宽、互联（NVLink/PCIe）和 kernel 效率共同影响。CUDA 生态包含驱动、运行时、cuDNN、NCCL、编译器和推理库。

工程上需固定驱动/容器兼容矩阵，监控显存、功耗、温度、ECC 与利用率，并按训练或推理负载选择 GPU 型号。