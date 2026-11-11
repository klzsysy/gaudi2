# Gaudi2 INT8 / FP8 kernel 支持调查

日期：2026-11-10（Asia/Shanghai）。目标环境为本项目 Gaudi2 HL-225、SynapseAI/Habana 1.24.2、eager + `hpu_backend`。本轮读取官方文档、公开 GitHub 源码和本仓库已有实测，没有执行新的 HPU benchmark、停止服务、安装软件或修改产品 checkout。

## 结论

1. **FP8 有官方原生计算路径，且本机已有验证。** 覆盖量化/反量化、GEMM、FP8 attention；源码中还包含 FP8 convolution、softmax、MoE 等接口。不能把接口存在直接当成每种 Gaudi2 shape 都已通过。
2. **INT8 有 TPC 数据类型/指令及部分张量操作支持，但未找到可直接用于当前模型的原生高吞吐 INT8 GEMM。** 官方整数算子表将 `mm/matmul` 标为不支持；vLLM Gaudi 显式使用 BF16 fallback；本机 `torch._int_mm` 已证实 CPU fallback。这是对已审查软件栈的判断，不是对所有未公开硬件能力的绝对断言。
3. **GitHub 上既有厂商算子封装，也有真正的 TPC-C/Triton kernel 源码。** `gaudi-pytorch-bridge`、INC、vLLM 的 FP8 Linear 主路径最终调用 Synapse 算子；`1Cat-vLLM-Gaudi` 则有 Gaudi2 FP8 动态量化、SiLU+量化和低比特权重解码源码，但不能由此推断它提供了通用 INT8 GEMM 或独立的高性能 FP8 MME 实现。
4. 当前 ComfyUI 最值得复用的是 **`cast_to_fp8_v2` + `fp8_gemm_v2`，配合模型校准和权重生命周期管理**。不建议为了当前 INT8 ConvRot 文件先重写通用 INT8 GEMM。

后续按用户指定仓库补查 JChen 当前 `6da6fb16`：确认有自定义 TPC FP8 MAC 稀疏 attention，以及新增的直接 MME/TPC BF16/FP16 attention。其能力超过最初第 3.7 节对输出头的局部介绍，完整修正见 [JChen 专项报告](JCHEN_GAUDI2_KERNEL_REVIEW_2026-11-10.md)。

## 1. 硬件、公开 API、实际执行必须分开

| 层次 | INT8 | FP8 |
|---|---|---|
| 硬件公开资料 | TPC 明确支持 INT8；不能据此套用 MME 峰值 | MME/TPC 支持 E4M3、E5M2；白皮书描述 MME 使用 FP32 累加 |
| 官方 PyTorch 支持表 | 整数 `mm/matmul` 为 No；部分逐元素/数据搬运操作为 Yes，按算子而异 | `torch.mm` 为 Yes；同表 `torch.matmul` 的 FP8 为 No，不能泛化所有入口 |
| 框架量化 | 能保存 INT8 权重、做 Q/DQ，不等于乘法使用 INT8 | 厂商 cast/GEMM/SDPA 为实际低精度执行入口 |
| 本项目实测 | `_int_mm` CPU fallback；生产用 INT8→BF16 解量化计算 | E4M3/E5M2 `mm`、`fp8_gemm_v2` probe、FP8 SDPA 已通过各自算子测试 |

依据：[Gaudi2 白皮书，第 6、10 页](https://habana.ai/wp-content/uploads/pdf/2021/gaudi2-whitepaper.pdf)、[官方算子表](https://docs.habana.ai/en/latest/PyTorch/Reference/Pytorch_Operators/Pytorch_Operators.html)、[TPC User Guide](https://docs.habana.ai/en/latest/TPC/TPC_User_Guide/index.html)。TPC 是可编程向量处理器，MME 是矩阵计算引擎；TPC-C 程序不是给 MME 写 CUDA 风格的 microkernel。

## 2. 官方 kernel / 算子清单

以 `HabanaAI/gaudi-pytorch-bridge` 的 `v1.24.2` 默认分支快照 `27eccbcffeab10842c9f29f64c6bf50d8f65d78f` 为依据。

| 入口 | 用途 | 证据与边界 |
|---|---|---|
| `hpu.cast_to_fp8_v2`、`hpu.cast_from_fp8` | 带 scale 的 FP8 转换；提供 tensor/scalar 等重载 | schema 和 backend 公开；本机前者已用于 FP8 SDPA/head probe |
| `hpu.cast_to_fp8_hybrid` | 生成 E4M3/E5M2 两种量化结果 | 源码存在，未在本项目验证该入口 |
| `hpu.fp8_gemm_v2` | 带 A/B scales、transpose、bias、可选 accumulation 的矩阵乘 | 本项目已调用；bridge 构造 `fp8_gemm_<output precision>` Synapse 节点 |
| `torch.mm` 的 FP8 路径 | 普通矩阵乘入口 | 官方表和本机实测支持；不是完整的带 scale Linear 适配 |
| `torch._scaled_mm` | PyTorch scaled FP8 GEMM 入口 | bridge 已注册，但本项目此前的配置报 `synStatus 26`；不能当成全面可用，也不能据此否定 `fp8_gemm_v2` |
| `Fp8FusedSDPA` / `fp8_fused_sdpa` | FP8 attention；含 Q/K/V 与中间值 scale | 官方 Python wrapper 和底层 SDPA 连接代码公开，本机已做真实 decode shape 对照 |
| `hpu.conv2d_fp8`、`hpu.softmax_fp8` | FP8 卷积、softmax | schema 与 INC wrapper 存在；本轮未做 Gaudi2 全参数/shape 验证 |
| `hpu.mixture_of_experts` 的 FP8 重载 | FP8 MoE | 官方 custom-op 文档和 INC wrapper 提供路径；不外推所有重载都适合 Gaudi2/MOSS |
| `quantized_decomposed.quantize/dequantize_per_tensor/per_channel` | 通用量化/反量化图节点 | 有官方 bridge 实现；具体 dtype、模式与参数仍需测试，不等价于 INT8 GEMM |

源码入口：[注册表](https://github.com/HabanaAI/gaudi-pytorch-bridge/blob/27eccbcffeab10842c9f29f64c6bf50d8f65d78f/scripts/hpu_op.yaml#L8483)、[FP8 GEMM backend](https://github.com/HabanaAI/gaudi-pytorch-bridge/blob/27eccbcffeab10842c9f29f64c6bf50d8f65d78f/hpu_ops/backend/fp8_gemm_gen.cpp#L286)、[FP8 casts/backend](https://github.com/HabanaAI/gaudi-pytorch-bridge/blob/27eccbcffeab10842c9f29f64c6bf50d8f65d78f/hpu_ops/backend/fp8_ops.cpp)、[FP8 SDPA wrapper](https://github.com/HabanaAI/gaudi-pytorch-bridge/blob/27eccbcffeab10842c9f29f64c6bf50d8f65d78f/python_packages/habana_frameworks/torch/hpex/kernels/Fp8FusedSDPA.py)、[Q/DQ backend](https://github.com/HabanaAI/gaudi-pytorch-bridge/blob/27eccbcffeab10842c9f29f64c6bf50d8f65d78f/hpu_ops/backend/quant_dequant_gen.cpp)。

这里的公开 backend 代码负责参数、shape、scale、建图和 lowering，底层还依赖安装好的 Synapse/TPC 库。不能把这个仓库描述成包含所有厂商设备 kernel 的完整、独立实现。运行时分层见 [PyTorch Gaudi Theory of Operations](https://docs.habana.ai/en/latest/PyTorch/Reference/PyTorch_Gaudi_Theory_of_Operations.html)。

官方 [FP8 推理指南](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html) 提供 INC 的校准、scale 和动态量化配置；[Custom Ops](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Custom_Ops_PyTorch.html) 提供 MoE 等接口说明。它们是应用官方算子的依据，不是当前 ComfyUI 已完成这些集成的证明。

## 3. GitHub 项目逐项核查

### 3.1 vLLM Gaudi：FP8 真计算，INT8 明确转 BF16

快照 `vllm-project/vllm-gaudi@57cec1ec789a9eebe5338312039133e94f58fc47`。

- [extension/ops.py::apply_fp8_linear_hpu](https://github.com/vllm-project/vllm-gaudi/blob/57cec1ec789a9eebe5338312039133e94f58fc47/vllm_gaudi/extension/ops.py#L994)：静态或逐行动态量化 activation，再调用 `hpu.fp8_gemm_v2`，输出原 activation dtype。适合参考 scale、bias、布局与 weight preparation。
- [HPUCompressedTensorsW8A8Int8_BF16Fallback](https://github.com/vllm-project/vllm-gaudi/blob/57cec1ec789a9eebe5338312039133e94f58fc47/vllm_gaudi/ops/hpu_compressed_tensors.py#L327)：明确说明 INT8 GEMM kernel 不可用；加载后将 `int8_weight * weight_scale` 变为 BF16，运行 `F.linear`。这个 fallback 是 **HPU 上的 BF16 路径**，与本机 `_int_mm` 的 **CPU fallback** 不是同一件事。
- [GPTQ apply](https://github.com/vllm-project/vllm-gaudi/blob/57cec1ec789a9eebe5338312039133e94f58fc47/vllm_gaudi/ops/hpu_gptq.py#L237)：`convert_from_uint4` 后调用普通 `torch.matmul`。支持 INT4 checkpoint 不证明存在 INT4/INT8 MME GEMM。

### 3.2 Intel Neural Compressor：校准/量化与官方算子封装

快照 `intel/neural-compressor@8667aadc26452b39a01436b010e7bcaf3230e833`。

[HPU quantized-function wrappers](https://github.com/intel/neural-compressor/blob/8667aadc26452b39a01436b010e7bcaf3230e833/neural_compressor/torch/algorithms/fp8_quant/_core/quantized_func_wrappers/hpu/hpu_quantized_func_wrapper.py) 将 Matmul、Conv、Softmax、Cast 和 MoE 分别路由到 `fp8_gemm_v2`、`conv2d_fp8`、`softmax_fp8`、FP8 cast 和厂商 MoE。它提供模型校准和替换机制，未在所审查 HPU 实现中发现独立 INT8 GEMM。项目首页同时列 INT8/FP8/多个硬件，不能把 CPU/XPU 的 INT8 支持套到 HPU。

### 3.3 bitsandbytes：Gaudi 支持不等于原生 LLM.int8 GEMM

快照 `bitsandbytes-foundation/bitsandbytes@a616886cacf86701514f0cb1d6be760f967481c0`。

- [HPU backend](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/a616886cacf86701514f0cb1d6be760f967481c0/bitsandbytes/backends/hpu/ops.py) 的专用注册是 NF4 dequant，调用 `hpu.dequantize_nf4`，没有 HPU 专用 `int8_linear_matmul`。
- [默认 int8 matmul 实现](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/a616886cacf86701514f0cb1d6be760f967481c0/bitsandbytes/backends/default/ops.py#L134) 把输入转 FP32，做浮点 matmul，再转 INT32。因此该代码可提供接口语义，不能作为原生 INT8 吞吐依据。本轮没有在本机安装或测试它。

### 3.4 Comfy Kitchen：HPU eager 路由不补齐底层 INT8 kernel

快照 `Comfy-Org/comfy-kitchen@7d6a22ae8281cd1053a8449530fa0e764a2e51cb`。

[eager 注册](https://github.com/Comfy-Org/comfy-kitchen/blob/7d6a22ae8281cd1053a8449530fa0e764a2e51cb/comfy_kitchen/backends/eager/__init__.py#L144) 包括 `hpu`；[Triton 注册](https://github.com/Comfy-Org/comfy-kitchen/blob/7d6a22ae8281cd1053a8449530fa0e764a2e51cb/comfy_kitchen/backends/triton/__init__.py#L130) 的设备是 CUDA/XPU。其 [eager INT8 累加](https://github.com/Comfy-Org/comfy-kitchen/blob/7d6a22ae8281cd1053a8449530fa0e764a2e51cb/comfy_kitchen/backends/eager/quantization.py#L750) 调 `torch.int8_mm`（存在时）或 `_int_mm`，没有在这里实现独立的 Gaudi INT8 矩阵 kernel。[W4A8](https://github.com/Comfy-Org/comfy-kitchen/blob/7d6a22ae8281cd1053a8449530fa0e764a2e51cb/comfy_kitchen/backends/eager/w4a8_int8.py#L629) 则分为先解码 INT8 再调用 INT8 Linear，以及带 correction 时解量化后 `F.linear`。

上游当前快照不等于本机安装版本。本机的实际行为以本仓库 `_int_mm` fallback 和 ConvRot→BF16 工作流实测为准；`torch.int8_mm` 新名称没有在本轮硬测。

### 3.5 Habana 官方 TPC 示例和编译器

`HabanaAI/Habana_Custom_Kernel@06e73bbac4f50debaf67609a0dbd49aa87cb39a4` 的完整目录共 173 个文件。本轮核对 `kernels/`：Gaudi2 示例包括 add、BF16 softmax、ReLU、F16→I16 cast、LUT 等；矩阵乘示例为 `gaudi/matrix_mul_fwd_f32.c`，**未发现可直接拿来用的 Gaudi2 INT8/FP8 通用 GEMM 示例**。

[Habana_Custom_Kernel](https://github.com/HabanaAI/Habana_Custom_Kernel/tree/06e73bbac4f50debaf67609a0dbd49aa87cb39a4/kernels) 提供 TPC-C、host glue 和测试框架；[tpc_llvm](https://github.com/HabanaAI/tpc_llvm) 提供编译器源码；[TPC special functions/intrinsics](https://docs.habana.ai/en/latest/TPC/TPC_C_Language_Spec/Built_in_Special_Functions.html) 描述整数能力。它们适合开发 INT8 解包、转换和融合操作，不能当作现成 INT8 GEMM 库。

### 3.6 1Cat-vLLM-Gaudi：确实包含 Gaudi2 设备 kernel 源码

快照 `1CatAI/1Cat-vLLM-Gaudi@e2266e0efa381e6649e0f25b789cbb61b21b0e1f`。本轮完成目录检索和以下相关文件的静态审查，没有编译或验证其性能。

| 源码 | 实际做什么 | 不能据此宣称什么 |
|---|---|---|
| [dynamic_quant_bf16_gaudi2.c](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/e2266e0efa381e6649e0f25b789cbb61b21b0e1f/csrc/deepseek_v4/kernels/deepseek_v41_dynamic_quant_bf16_gaudi2.c) | 真正的 TPC-C：逐行 absmax、scale、BF16→FP8 转换和写出 | 不是 GEMM；含固定局部数组和 tile 假设，移植需同步审查 host shape guard |
| [triton_gaudi/kernels.py](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/e2266e0efa381e6649e0f25b789cbb61b21b0e1f/vllm_gaudi/ops/triton_gaudi/kernels.py#L517) | `@triton.jit` 动态 FP8 量化、SiLU+量化；明确编译到 Gaudi2 TPC target | 不代表普通 CUDA Triton kernel 可以直接在本机 Habana 栈运行 |
| [expert_paired_decode_fp8_gaudi2.c](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/e2266e0efa381e6649e0f25b789cbb61b21b0e1f/csrc/deepseek_v4/kernels/deepseek_v41_expert_paired_decode_fp8_gaudi2.c) / [expert_n256.h](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/e2266e0efa381e6649e0f25b789cbb61b21b0e1f/csrc/deepseek_v4/include/deepseek_v41_expert_n256.h) | 专用低比特专家权重的解包、查表、缩放和 FP8 输出；这里 decode 是权重解码 | 名称含 FP8/expert 不证明实现了 FP8 GEMM，更不是通用 INT8 ConvRot 解码 |

[runtime.py](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/e2266e0efa381e6649e0f25b789cbb61b21b0e1f/vllm_gaudi/ops/triton_gaudi/runtime.py#L264) 对动态量化限制为连续二维 BF16、行数 ≤32、列数 ≤16384；该候选在 hybrid 模式仍返回厂商路径，strict 才允许实验。SiLU+量化另有宽度 ≤4096 的限制。还依赖配套 Gaudi Triton backend 和 bridge launch ABI。

因此可以把它列为 **FP8 周边融合 kernel 的开源参考**，但当前证据不足以直接部署。若只关注“GitHub 是否真的包含 HPU kernel 源码”，答案是有；若问“是否已有可无缝替换 `_int_mm` 的 Gaudi2 INT8 GEMM”，本轮未找到。

### 3.7 JChen、Optimum Habana、TGI Gaudi

- **JChen 后续专项复核已补全，见 [当前 `6da6fb16` 报告](JCHEN_GAUDI2_KERNEL_REVIEW_2026-11-10.md)。** 最初本节只读取 FP8 输出头，未覆盖整个 main。该仓库除厂商 FP8 Linear 外，还包含 TPC FP8 MAC 稀疏 attention、RMSNorm/量化融合、GDN，以及相对 `e08710f7` 新增的直接 MME/TPC BF16/FP16 attention 实验。不能把整个仓库概括成封装；通用 INT8 GEMM 仍未找到，W8A8 仍转 BF16。旧 lm_head probe 和历史审查见 [既有调查](MOSS_HPU_PERFORMANCE_RUN_2026-11-08.md)。
- [Optimum Habana text-generation](https://github.com/huggingface/optimum-habana/blob/main/examples/text-generation/README.md)、[TGI Gaudi](https://github.com/huggingface/tgi-gaudi) 提供 FP8 模型应用/INC 集成路径，可参考校准与模型覆盖。这里核实的是官方项目的使用说明，不宣称已审完它们全部源码，也不把它们列为独立 INT8 kernel 实现。

## 4. FP8 使用中的 Gaudi2 特有限制

- **范围与编码不能想当然。** vLLM 的 [FP8_MAX](https://github.com/vllm-project/vllm-gaudi/blob/57cec1ec789a9eebe5338312039133e94f58fc47/vllm_gaudi/extension/ops.py#L25) 在 Gaudi2 使用 `float8_e4m3fnuz` 的 max（240），但设备 API 仍传 `float8_e4m3fn`。不能仅据 dtype 名称认定 CPU/GPU 上的 E4M3FN 文件可逐字节搬运；常见 E4M3FN max 为 448。
- vLLM 包含 [weight/scale adjustment](https://github.com/vllm-project/vllm-gaudi/blob/57cec1ec789a9eebe5338312039133e94f58fc47/vllm_gaudi/extension/ops.py#L1039)，且有按模型量化格式关闭重复调整的判断。本项目 Krea2 已实测部分 FP8 字节直接 HPU cast 出现 inf/nan，并用 CPU 语义 LUT 解码修复；那是读取兼容修复，不是原生 FP8 GEMM 适配。
- 官方 scale 文档针对 Gaudi2 列出 exponent bias 3/7/11/15，Gaudi3 范围更宽。不能把 Gaudi3 校准和 block-scale 支持无条件外推给 Gaudi2。
- INT8 ConvRot 权重包含 scale/旋转语义；若走 FP8，需要正确还原/转换权重或保留其数学变换，再建立 FP8 scale。不能 reinterpret INT8 存储字节作为 FP8，也不能将这种二次量化当成从原始 BF16 直接校准的质量。

## 5. 本机已有验证与下一步

| 已有测试 | 观察 | 支持的结论 |
|---|---|---|
| 4096³ FP8 E4M3 / E5M2 `torch.mm` | 约 525.88 / 519.63 TFLOP/s；BF16 同 shape 约 275.76 | FP8 原生矩阵路径存在；不能据此承诺整个模型快 2 倍 |
| INT8 `_int_mm` | 记录到 103 次 CPU fallback | 3.17 的测量值不能用作 HPU INT8 TOPS |
| `fp8_gemm_v2` lm_head probe | fallback 为空；32-token 单次约 9.580 s | 可运行；未建立端到端重复收益 |
| FP8 FusedSDPA 真输入 | 0.1064 ms/call；默认 BF16 0.0848，BF16/FP32-softmax 0.1085 | FP8 attention 可用，但该 shape 未优于最佳 BF16；FP8 初步量化误差也更大 |

这些是复用的历史结果，不是本轮重新测量。出处：[性能调查](MOSS_HPU_PERFORMANCE_RUN_2026-11-08.md)、[attention 算子实测](ATTENTION_IDEAS_REVIEW_2026-11-10.md)、[执行记录](PROGRESS.md)。SDPA 的量化/计时边界和未完成校准见原记录。

工程建议：继续保持当前 BF16 产品基线；若开启量化专项，优先在真实 DiT Linear 和文本 M=1/2/4/8 上接 `fp8_gemm_v2`，统一 BF16 输出，分别计入 activation 量化、GEMM、scale/layout 和权重转换成本。权重一次转换常驻，模型卸载/LoRA/patch 时正确失效；对提示词与 diffusion timestep 做质量校准。只有测得量化准备成为热点后，再试 1Cat 的 TPC/Triton 周边融合方案。INT8 checkpoint 的首选仍是已验证的 BF16 解量化路径，或另做正确转换后的 FP8 实验。

## 调查可复核性与范围

- [sources.json](kernel-audit-2026-11-10/sources.json) 保存 7 个主要仓库的固定 commit、目录是否完整及已下载源码的 Git blob SHA、SHA-256、永久链接；外部源代码只放临时目录，没有复制进入产品仓库。
- 官方文档使用检索当日的 latest（页面显示 1.24.2）；旧 TPC special-functions 页面索引显示 1.24.0。文档、源码存在、本项目已实测三种证据分别表述。
- 全网搜索不构成“不存在任何实现”的证明。负面结论限定为本轮官方文档、指定版本源码和本机已有测试；没有把 CUDA、XPU、CPU 的同名实现算作 Gaudi2 kernel。
- 检索中猜测的 `HabanaAI/Intel_Gaudi3_Software` API 返回 404，未将其作为依据；1Cat 的 Triton 路径最初按单文件请求返回 404，后由完整目录确认是 package 并读取实际文件；Comfy Kitchen 页面读取超时后改用固定 commit 的 raw 源码。没有下载模型或使用/记录凭据。
