# Moody MOSS FP8主模型适配

2026-11-11，Gaudi2 HL-225、Habana 1.24.2、eager。测试文件为`Moody-Moss-Mix-V4.1_00001__clean-fp8.safetensors`，185个E4M3权重张量，其余378个BF16张量包含scale、norm和bias。

## 问题与修复

用户原请求222.10s，25步DiT约205.8s，稳定步约7.89s。日志明确把E4M3列为emulated，既有执行在每次Linear使用时解码FP8权重并做浮点乘法。FP8文件仅占约4.85GB，不能直接据此判断计算精度或速度。

- 加载警告的根因是`load_diffusion_model_state_dict`在去除`model.diffusion_model.`前后两次注入同一份JSON量化元数据。前一次的键随权重正确改名，后一次重新加入185个带旧前缀的`comfy_quant`键。修复避免第二次注入，保留legacy scaled-FP8转换。
- 复用既有MixedPrecision Linear与CastBiasWeightContext，增加Gaudi E4M3布局和官方`cast_to_fp8_v2`/`fp8_gemm_v2`执行。已有BF16 attention、norm、RoPE、残差和区域编译继续组合执行。
- Gaudi2原生E4M3范围为240，文件含448。加载时将超范围权重除2、scale乘2，保留单份量化参数，由现有加载/卸载/patch生命周期持有。已在范围内的重载权重保持编码，避免反复缩小。微小subnormal会舍入，不能宣称逐bit等价。
- 激活按行动态absmax量化，计入运行时间。文件已有input scale时扩大2倍适配范围。权重更新使用安全范围重新量化，保存仍使用标准E4M3格式及scale。
- BF16/FP32激活走原生路径。当前bridge的FP16输入`cast_to_fp8_v2`报不支持，FP16、梯度和动态patch使用既有浮点路径。文本编码器量化策略保留。

调查依据沿用[INT8/FP8 kernel调查](GAUDI2_INT8_FP8_KERNEL_SURVEY_2026-11-10.md)，本轮复核[官方FP8指南](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html)。

## 测量契约

- 使用用户实际FP8主模型。参考分支把同一文件权重在CPU解码为常驻BF16，检验原生FP8相对已消除重复解码开销的收益。
- 1024²、25步、CFG2.5、er_sde/simple，同caption与seed。固定caption重复6次，首次含加载/编译和同步诊断，后5次只重算KSampler/VAE并报告中位数。
- 原用户工作流另测含TextGenerate的完整流程，扩写token数会变化，单独报告。它使用的文本文件实际为`qwen3.5_9b_bf16_from_int8.safetensors`、BF16运行。按用户优先处理主模型的意图沿用该配置。
- 三组语义样本覆盖茶壶、MOSS海报与人物，另做768²/512²及卸载重载。质量结论限于这些样本。

## 首轮算子证据

取Moody第0层真实四类权重，N=4119/4403、随机BF16激活，编译BF16与编译FP8各3 warmup、5×10次同步计时。FP8包含逐行量化与GEMM。

| Linear | BF16 ms，两种N | FP8 ms，两种N |
|---|---:|---:|
| QKV | 0.615 / 0.650 | 0.418 / 0.397 |
| output | 0.217 / 0.228 | 0.143 / 0.151 |
| FFN up/gate | 1.063 / 1.111 | 0.575 / 0.737 |
| FFN down | 0.550 / 0.592 | 0.356 / 0.361 |

结果有限、fallback为空，输出相对RMS误差约2.67%。这是随机激活、单层权重重复的局部筛选，真实扩散质量与端到端收益由完整流程验收。四类权重缩范围重编码相对误差为数个1e-6至2.54e-5。

## 回归与部署

产品提交`2227f7d3`，已部署8189。四份文件SHA-256与本地一致，正常入口cold70.970s、缓存13.600s，队列为空。正常入口图片已检查MOSS招牌、白色蓬袖大衣与湿地反光。HPU专项、混合精度和模型检测合计240 passed，最终量化/加载补充回归29 passed（与前者有重叠）。覆盖CPU梯度、HPU eager/compile、1D/2D/3D输入、bias、BF16/FP32与FP16浮点分支、静态input scale、权重patch、保存重载、legacy字节存储、显式full-precision、文本及MoE策略。完整LoRA工作流、训练、其它FP8格式和其它模型尚未验收。

原生入口只扩展Linear，MoE和文本的原量化策略保留。既有bridge的FutureWarning和第三方custom-node环境警告仍可出现，本轮修复的是重复`comfy_quant`加载警告。

## 完整流程结果

| 场景 | 常驻BF16参考 | 原生FP8 |
|---|---:|---:|
| 固定caption缓存端到端，5 warm中位数 | 13.993s | 12.810s |
| 同组KSampler中位数 | 13.749s | 12.565s |
| 首次含同步诊断、加载与编译 | 42.420s | 46.084s |
| 多提示词/形状验收allocator峰值 | 39.510GiB | 34.842GiB |
| 卸载稳定占用 | 2,685,928,832 bytes | 2,685,928,832 bytes |

固定工作量端到端降低8.46%，DiT降低8.61%，峰值减少4.67GiB。FP8首轮编译/加载开销仍高于常驻BF16参考。

原用户含自然TextGenerate的工作流首次诊断70.292s，随后20.666/21.984/20.593s，中位数20.666s。三次DiT为13.319/13.917/13.897s。用户原222.10s包含当时的加载与缓存状态，需与上述cold/warm分别阅读。自然扩写工作量与seed变化，因此该组用于验证实际入口和排除严重慢路径，固定caption组用于归因FP8收益。

所有已测请求fallback为空，诊断latent/VAE均有限。三组图片并排检查通过主体语义与指定元素，红茶壶和人物接近，海报保留MOSS、蓝花瓶与三朵黄花，小字、夹子等细节改变。FP8/参考PNG MAE分别为5.480/12.383/4.391（0–255），非像素等价。512重载、768海报和1024人物均完成，质量范围限于该集合。

当前最大的改善来自消除每次Linear重复解码并接入原生计算。相比正确的常驻BF16参考，实际收益为约1.18s。attention等剩余成本仍在，后续研究应先按真实timestep输入和42层轮换统计量化、GEMM与attention占比，再决定校准、量化融合或attention方案。当前证据不支持把FP8文件体积或单GEMM加速直接折算为整图2倍。

## 复现与证据

[结构化验收证据](evidence/moss-fp8-adaptation-2026-11-11.json)包含各请求计时、finite、fallback、内存、图片误差、测试与部署结果。

服务器原始证据根目录为`/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-11-fp8`。本地副本在忽略目录`results/2026-11-11-fp8`。

- `probes/moss_fp8_probe.py`验证真实权重尺寸与原生算子。
- `probes/moss_fp8_server.py`建立同文件的常驻BF16参考。
- `probes/moss_fp8_workflows.py`与`moss_fp8_jobs.py`串行执行固定工作量、图片和生命周期验收。
- `probes/moss_fp8_summary.py`从request/node事件计算结果，核对finite和fallback。

初始接线试验发现模型基类调用`pick_operations`时省略load_device，必须在operation选择处解析默认设备，否则仍落入emulated路径。该试验中断后修正。跨激活dtype回归还发现默认cast会解量化weight，原生入口需显式保留权重存储dtype并单独设置bias dtype。以上失败均保留在原始日志。
