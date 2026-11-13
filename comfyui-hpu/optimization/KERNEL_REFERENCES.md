# Gaudi2算子接口与外部源码参考

本篇合并2026-11-10的官方接口调查及JChen补充审查，并标出后续本机验证。外部结论绑定以下源码快照，后续升级应重新核对schema、shape与ABI。目录检索仅证明已审查文件的内容。

## 官方入口与本机执行

| 入口 | 能力与实测边界 |
|---|---|
| `cast_to_fp8_v2` / `cast_from_fp8` | scale量化/反量化，当前产品使用前者 |
| `fp8_gemm_v2` | A/B scales、transpose、bias及输出dtype，已接入DiT和文本Linear |
| FP8 `torch.mm` | 本机4096³ E4M3/E5M2约525.88/519.63TF/s，BF16同shape275.76TF/s |
| `torch._scaled_mm` | bridge注册存在，本机旧配置报`synStatus26`，与已通过的`fp8_gemm_v2`分别评价 |
| `Fp8FusedSDPA` | 真实decode和DiT已测，结果依shape/scale变化，见对应主题 |
| `softmax_fp8` | 当前BF16输入、FP8输出用于DiT，scale128，概率误差边界见Attention |
| `conv2d_fp8`、FP8 MoE、hybrid cast、通用Q/DQ | 源码/schema可见，当前项目缺全参数与模型验收 |
| INT8 `mm/matmul` / `_int_mm` | 审查时整数支持表受限，本机`_int_mm` CPU fallback，量化权重使用BF16解码路径 |

固定bridge为`27eccbcffeab10842c9f29f64c6bf50d8f65d78f`（v1.24.2）。公开backend处理参数、shape、scale和Synapse建图，实际设备执行还依赖厂商库。[算子注册](https://github.com/HabanaAI/gaudi-pytorch-bridge/blob/27eccbcffeab10842c9f29f64c6bf50d8f65d78f/scripts/hpu_op.yaml) · [FP8 GEMM](https://github.com/HabanaAI/gaudi-pytorch-bridge/blob/27eccbcffeab10842c9f29f64c6bf50d8f65d78f/hpu_ops/backend/fp8_gemm_gen.cpp) · [FP8 cast](https://github.com/HabanaAI/gaudi-pytorch-bridge/blob/27eccbcffeab10842c9f29f64c6bf50d8f65d78f/hpu_ops/backend/fp8_ops.cpp) · [softmax lowering](https://github.com/HabanaAI/gaudi-pytorch-bridge/blob/27eccbcffeab10842c9f29f64c6bf50d8f65d78f/hpu_ops/backend/softmax_fp8_gen.cpp)

Gaudi2 E4M3原生范围240，常见checkpoint可达448，API仍使用`float8_e4m3fn`名称。权重/scale需配套转换，exponent bias有3/7/11/15限制。INT8 ConvRot还包含旋转/scale语义。读取兼容、二次量化来源和原生计算分别见 [模型适配](../ADAPTATION.md)、[DiT FP8](DIT_FP8.md) 与 [文本精度](TEXT_PRECISION.md)。

## 可复用工程快照

| 项目与commit | 已确认内容 | 移植边界 |
|---|---|---|
| vllm-gaudi `57cec1ec` | FP8 Linear量化后调用`fp8_gemm_v2`，W8A8 INT8加载后转BF16，GPTQ先解uint4 | 参考weight preparation、scale和bias，layer contract需适配 |
| INC `8667aadc` | HPU cast/matmul/conv/softmax/MoE wrappers与校准 | 硬件/后端支持按实际wrapper核对 |
| bitsandbytes `a616886c` | HPU专用NF4 dequant，默认INT8 matmul转FP32后计算再转INT32 | 未找到所审查版本的HPU原生INT8 Linear，未在本机测试 |
| comfy-kitchen `7d6a22ae` | eager注册HPU，INT8调用`int8_mm`或`_int_mm`，W4A8先解码 | 该上游快照与本机安装版本分开核对 |
| Habana_Custom_Kernel `06e73bba` | Gaudi2 add、BF16 softmax、ReLU、cast、LUT及host glue | 所读目录未找到即用通用Gaudi2 INT8/FP8 GEMM示例 |
| 1Cat-vLLM-Gaudi `e2266e0` | TPC-C动态FP8量化、Gaudi Triton SiLU/量化、专用低比特专家权重解码 | Gaudi backend与bridge ABI、shape guard、host glue需共同适配 |

固定文件哈希与永久链接集中于 [sources.json](../kernel-audit-2026-11-10/sources.json)。主要调用入口为vLLM `extension/ops.py::apply_fp8_linear_hpu`、`hpu_compressed_tensors.py`，INC `hpu_quantized_func_wrapper.py`，Kitchen `backends/eager/quantization.py`及`w4a8_int8.py`。

1Cat动态量化runtime要求连续二维BF16、行数≤32、列数≤16384，hybrid仍选择厂商路径，strict才允许实验。SiLU＋量化宽度≤4096，专用专家decode的含义为权重解码。应连同这些门控读取设备kernel。[固定runtime](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/e2266e0efa381e6649e0f25b789cbb61b21b0e1f/vllm_gaudi/ops/triton_gaudi/runtime.py) · [动态量化kernel](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/e2266e0efa381e6649e0f25b789cbb61b21b0e1f/csrc/deepseek_v4/kernels/deepseek_v41_dynamic_quant_bf16_gaudi2.c)

## JChen的补充审查与修正

固定`6da6fb16892f0e6ec3a7e1900942dec4e47a8706`。此前只审查`e08710f7`的输出头，曾把整体概括为厂商封装，补查native及csrc后修正。两版新增直接MME/TPC路线，同时原有csrc已包含真实TPC FP8 MAC、量化融合和GDN。[固定目录及关键哈希](../kernel-audit-2026-11-10/jchen-sources.json)

| 路径 | 实际实现与限制 |
|---|---|
| `hpu_lm_head_fp8.py` | dense BF16/FP16权重逐输出通道量化，激活动态量化，调用厂商GEMM，保留原权重 |
| `add_rmsnorm_quant_bf16_gaudi2.c` | 残差、RMSNorm、absmax/scale及FP8写出，宽度256–17408、128对齐，含normalized/residual输出 |
| `silu_mul_quant_bf16_gaudi2.c` | 读取已算SiLU和up，合并乘法/量化，bridge另建厂商SiLU |
| `deepseek_v4_paged_sparse_attn_fp8.h` | QK调用`v_f8_mac_acc32_b`、FP32累加，head_dim512中448维FP8及专用scale |
| `gdn_mtp_fused_f32_gaudi2.c` | slot、norm、decay和8行递推，固定B≤16、H16/HV48、K/V128及`head/3`映射 |
| `qwen4exp-nvfp4-moe` | packed NVFP4在寄存器解码BF16，再BF16 MAC，含专家scatter等，所读独立库只过host/模拟器 |
| `mtp-sampling` / `tpc-kernel-shim` | 拒绝采样TPC与GUID多库转接，分别处理数学和库注册 |

真实融合源码可用于重新设计当前shape的候选，已有本机TPC实验及撤回结果见 [GDN](TEXT_GDN.md)。作者Qwen图片记录在线FP8从3.59→3.2秒但构图变化，未启用，`qwen_worker.py/image_api.py`未包含在该仓库。源代码范围与作者性能记录分别保存。[作者固定优化记录](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/docs/OPTIMIZATION.md)

### 直接MME/TPC attention

裸hl-thunk原型使用MME描述符、WREG、DMA/SOB/monitor，固定128³ BF16、小整数测试，作者报告12项通过。H3框架互操作实现则在Synapse内使用私有SCAL流，QK→exp/行和→PV→归一化，中间S/P复用SRAM，含地址缓存、流水、同步和输出所有权。[固定实现](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/native/mme-direct-attention/README.md)

H3交付路径Q/K/V BF16、score FP16、probability BF16，CPU `sim_fp8pv.py`是另一个数值实验。Python门控batch1、seq37750、D128、同shape、无mask、非因果，部分敏感head回原入口。作者完整请求100.45→84.41秒包含attention、head streaming和compiled gates，再加常驻/锁页/VAE tile到79.7秒。逐层最大RMS约0.78%，视频约18dB PSNR、生产质量未完成，45/48/49层使用FusedSDPA。[runtime](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/native/mme-direct-attention/attention/runtime_quad.hpp)

这提供跨MME/TPC片上协作的源码线索。移植到MOSS约4K/8K序列需重建descriptor、SDK互操作、allocator及同步条件。作者要求同进程没有其它并发设备任务。作者栈SDK1.24.1/Python3.12/torch2.11，本机1.24.2/Python3.10/torch2.13，二进制与私有ABI须按本机重建。

## Bridge与Synapse的修改范围

本机bridge1.24.2.481的`compilers.py`和`_passes/fuse_attention.py`已与官方`27eccbcf`逐blob核对一致。已读fusion pass注册FP32/BF16普通SDPA，混合BF16 QK→FP8 softmax→FP8 AV的整体匹配/lowering还需专项调查。源代码可从这一版本建立最小复现。[bridge编译入口](https://github.com/HabanaAI/gaudi-pytorch-bridge/blob/27eccbcffeab10842c9f29f64c6bf50d8f65d78f/python_packages/habana_frameworks/torch/dynamo/compile_backend/compilers.py)

量产Synapse图规划包含布局、融合、流水和内存管理。公开SynapseAI_Core仅提供受限参考实现，官方UserCustomOpDescriptor描述单个TPC核。norm/量化融合可沿此接口，MME→TPC→MME驻留需要进一步lowering、bundle或运行时接口。[SynapseAI_Core限制](https://github.com/HabanaAI/SynapseAI_Core#limitations) · [CustomOp接口](https://docs.habana.ai/en/latest/PyTorch/Reference/PyTorch_CustomOp_API/page_index.html#api-limitations)

当前优先复用已验厂商原语及匹配的bridge源码，外部融合代码作为具体shape候选。通用INT8 GEMM在已审查版本中尚无可替换本机`_int_mm`的实现，模型专用稀疏attention、NVFP4和dense MOSS各自验证。
