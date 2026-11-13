# FP8 DiT接入、区域融合与量化准备

主模型为`Moody-Moss-Mix-V4.1_00001__clean-fp8.safetensors`，185个E4M3权重及378个BF16张量。`2227f7d3`接入原生FP8 Linear，`0995638a`采用native FP32 Q/K norm及更大block编译区。混合精度attention另见 [Attention](ATTENTION.md)。

## 加载与原生计算

初始用户请求222.10秒，DiT约205.8秒、稳态约7.89秒/步。原路径每次Linear都解码FP8权重再做浮点计算。修复复用MixedPrecision Linear和CastBiasWeightContext，使用`cast_to_fp8_v2`与`fp8_gemm_v2`。

- 去除`model.diffusion_model.`前后重复注入metadata，留下185个旧前缀`comfy_quant`键，引发加载warning。去掉第二次注入并保留legacy scaled-FP8转换。
- 文件E4M3可达448，Gaudi2原生范围240。加载时超范围权重除2、scale乘2，使用224安全范围，重载已安全编码不重复缩放。subnormal可能舍入，缩范围重编码RRMSE为数个1e-6至2.54e-5。
- 激活逐行动态absmax量化，已有input scale相应扩大2倍。权重更新、保存和卸载通过原参数生命周期管理。
- operation选择曾遗漏load_device，仍走emulated。默认cast又会先解量化weight，原生入口需保留权重存储dtype并单独处理bias dtype。
- 本栈BF16/FP32输入已验，FP16 cast不支持，FP16、梯度及动态patch保留浮点计算入口。

同一文件CPU解码后常驻BF16作为对照，排除重复解码成本。1024²/25步、同caption/seed、五轮warm仅重算KSampler/VAE。

| 指标 | 常驻BF16参考 | 原生FP8 |
|---|---:|---:|
| 缓存整图 | 13.993s | 12.810s |
| DiT | 13.749s | 12.565s |
| 多提示词/shape allocator峰值 | 39.510GiB | 34.842GiB |
| 卸载 | 2,685,928,832 bytes | 相同 |

第0层真实四类权重、随机激活、N4119/4403，FP8含量化后QKV约0.397–0.418ms，BF16约0.615–0.650ms，up/gate约0.575–0.737ms对1.063–1.111ms。局部RRMSE约2.67%，完整三组图像MAE5.480/12.383/4.391，主体通过有限样本目视。240项广泛回归及29项重叠专项通过，含patch、保存重载、静态scale、legacy字节和多输入维度。

## Native FP32 norm与编译区域

从`f67e7f3f`继续，Q/K norm采用`hpu.rms_norm`的FP32输入/权重，再按原BF16/RoPE转换。代表层约5.59→5.27ms，RRMSE0.017%–0.075%。直接BF16 native norm的敏感层变化可达3.66%，BF16 RoPE可达3.34%，均未采用。

FP8 block扩大编译范围，同时保留attention和FFN调制私有方法之间的graph break边界。将调制直接内联到总forward后DiT10.682秒、代表层约5.21ms，恢复边界后代表层约5.00ms。最终继续由既有dispatcher及cast/offload管理，hooks、override、reference、mask、梯度/训练按对应分区执行。

同Moody/FP8文本、固定298-token、1024²/25步，各五轮。

| 指标 | 同轮基线 | `0995638a` |
|---|---:|---:|
| 完整流程 | 17.202s | 15.925s |
| DiT | 11.361s | 10.211s |
| cold | 73.861s | 77.273s |
| allocator累计高水位 | 27.402GiB | 31.285GiB |
| 请求结束常驻 | 17.987GiB | 17.987GiB |

其它kernel预算3.668→2.651秒，FP8 MME约3.045秒，launch1984→640、runtime run3632→776，但launch_recipe336→504。扩大区域后生成多个recipe，单个接口计数不能独立代表总提交成本。

18组Moody真实输入相对前版RRMSE0.306%–4.066%，对相同解码量化权重的FP32参考误差比0.72808–1.29272、中位1.00704，最差绝对误差1.380%→1.784%。融合FFN调制、残差和整block的误差比最高依次1.07530、1.17892、1.29272。历史配对图MAE32.792/2.938/8.662，光影/构图细节变化。287项回归及自然图、重载、原INT8通过，质量结论限定于记录集合。

## V量化的速度与精度取舍

| 实现 | 完整流程 / DiT | 高水位 | 结论 |
|---|---|---:|---|
| 初版24-head批量V量化 | 15.675 / 9.847s | 31.293GiB | 随机测试发现四维BF16广播引入约1.5%独立AV参考差异 |
| FP32量化输入修正 | 16.107 / 10.307s | 33.916GiB | 通过原0.5% AV阈值及288项回归，收益被精度修正抵消 |
| 同轮逐head量化 | 16.079 / 10.197s | 31.299GiB | 吞吐接近、峰值较低，采用 |
| 共享amax、逐head cast | 局部慢4.54%，整层差异最高1.58% | — | 未采用 |

初版三个真实输入相同仍未覆盖随机广播压力问题。后续检验保留原阈值，并用FP32缩放与完整区域性能共同决定取舍。

## SiTU、scale和其它失败候选

SiTU公式为`4*tanh(gate/4)*sigmoid(gate) * 25*tanh(hidden/25)`，绝对值上界100。当前两个tanh、sigmoid及乘除已融合为单个TPC kernel，独占预算约1.01秒。

| 候选 | 实测 | 判断 |
|---|---|---|
| 重用tanh表达sigmoid | block约4.84→8.01ms | 代数操作减少未转化为速度 |
| SiTU FP32 / sigmoid / exp改写 | 独立0.524→1.112/2.984/7.086ms | 更慢 |
| TPC SiTU | 三层0.523–0.527→0.768–0.778ms | 更慢 |
| SiTU＋逐行FP8量化 | 独立区域1.134–1.146→1.173–1.178ms | 无净收益，完整block已有融合 |
| BF16 Padé近似 | 约0.525→1.35ms | 耗时和误差均增加 |
| 静态activation scale | 部分Linear节省0.02–0.14ms，输出差异1.4%–3.8% | 仅用当前输入amax，跨prompt/timestep校准未完成 |
| QK scale移入softmax参数 | 5.589→5.581ms，敏感层差异3.90% | 未采用 |
| V二维逐行scale或显式乘scale | 局部约5.10/5.02ms，差异约0.21% | 沿用逐head方案 |

手写TPC中的BF16/FP32 lane转换曾混用，修正默认转换后恢复误差，`SW_LINEAR`方案触发编译错误。本轮VLM和glue接入经验见 [GDN](TEXT_GDN.md)。

## 思路与证据

大Linear活动阶段已约806TF/s，纯算术空间较小。剩余候选应瞄准norm/调制/量化/布局的实际读写，并保留FP32归约和BF16舍入位置。量化独占预算约0.20秒，SiTU和QK数据流具有更大绝对预算，融合后需重测引擎重叠。

额外72组输入将已有18组扩至90组，覆盖短提示、四物体计数、低对比纹理和复杂书店。FP32参考RRMSE中位1.267%、最高4.200%，末层较敏感。原始资料为`performance-2026-11-11-fp8`、`performance-2026-11-12-fp8-fusion`、`performance-2026-11-12-optimization`及`performance-2026-11-12-evidence`。入口为`moss_fp8_probe.py`、`moss_fp8_fusion_screen.py`和`moss_fp8_fusion_release.py`。[适配证据](../evidence/moss-fp8-adaptation-2026-11-11.json) · [融合证据](../evidence/moss-fp8-fusion-2026-11-12.json) · [TPC筛选](../evidence/moss-optimization-execution-2026-11-12.json) · [扩展精度](../evidence/moss-evidence-boundaries-2026-11-12.json)
