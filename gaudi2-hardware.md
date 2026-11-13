# Gaudi2 硬件能力与软件路径边界

硬件规格复核于2026-11-09，AIP口径更新于2026-11-13。原文将 INT8 写成与 FP8 同量级峰值，且把所有数据类型都归为 MME 原生计算；此推断没有依据，已更正。

| 项目 | 已确认能力 | 对本项目的含义 |
|---|---|---|
| BF16 MME | 公开峰值 432 TFLOP/s | 原生矩阵计算；不是单 token/全模型必达速度 |
| FP8 MME | 公开峰值 865 TFLOP/s；支持 E4M3/E5M2 | 当前栈 `fp8_gemm_v2` 可用，需要匹配 scale/layout/output dtype |
| TPC | 24 个；支持 FP32/BF16/FP16/FP8 和 INT32/INT16/INT8 | 可写融合/解量化等 kernel，不代表拥有 INT8 MME GEMM |
| MME | 2 个；白皮书列 FP32/TF32/BF16/FP16/FP8，累加为 FP32 | 当前资料不能推出有与 FP8 同峰值的 INT8 MME 接口 |
| HBM | 96 GB；公开标称 2.45–2.46 TB/s | 小批量 decode 要同时评估权重流量、状态流量和提交开销 |
| SRAM | 48 MB | 全文本模型和 lm_head 不可能整体驻留 SRAM |

本机 Habana 1.24.2 / torch 2.13.0a0 的 `torch._int_mm` 已实测 CPU fallback；不能以支持 int8 tensor 或 TPC INT8 指令替代这一证据。W4A8/GPTQ 同样必须核实解包与矩阵计算的实际 backend。

Gaudi2 的 FP8 数值范围和跨平台 checkpoint 的字节/scale 约定需要单独处理。本项目曾遇到 E4M3FN 高编码转 float 的 inf/nan；生产解量化使用完整解码表，原生 FP8 计算则必须按 Gaudi 的量化 API 和 scale contract 校准。不能仅凭 dtype 名称假定与其它平台逐字节等价。

本机1.24.2驱动的`hl-smi utilization.aip`已通过加载模块反汇编及531组功率配对确认采用功率公式，详见 [AIP与计数器](comfyui-hpu/optimization/BOTTLENECKS.md)。此前按通用文档活动窗口解释的判断据此修正。`utilization.memory`描述内存占用，算力与HBM效率使用实际shape吞吐、设备时间线和经校准字节计算。

来源：

- [Gaudi2 白皮书，第 6–10 页](https://habana.ai/wp-content/uploads/pdf/2021/gaudi2-whitepaper.pdf)
- [Intel Gaudi 产品对照，第 3 页](https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf)
- [TPC User Guide](https://docs.habana.ai/en/latest/TPC/TPC_User_Guide/index.html)
- [hl-smi 指标说明](https://docs.habana.ai/en/latest/Management_and_Monitoring/Embedded_System_Tools_Guide/System_Management_Interface_Tool.html)
- [FP8 推理 API 与校准](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html)
