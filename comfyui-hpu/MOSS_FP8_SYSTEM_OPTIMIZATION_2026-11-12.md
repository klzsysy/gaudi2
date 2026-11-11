# MOSS FP8全流程与Qwen3.5文本适配

本轮从2026-11-11执行至2026-11-12。主模型固定为`Moody-Moss-Mix-V4.1_00001__clean-fp8.safetensors`，单卡Gaudi2、Habana 1.24.2、eager。原始证据位于服务器`/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-11-fp8-system`。

## 文件与精度来源

已生成`/home/sonny/ComfyUI-Models/models/text_encoders/qwen3.5_9b_fp8_from_int8.safetensors`，11,377,021,988 bytes。源文件为现有19,306,312,416-byte BF16文件，由INT8 ConvRot解码得到。本次属于二次量化，来源写入模型metadata。

转换201个语言Linear，包括输出头。使用E4M3、逐tensor权重scale和224安全范围，激活逐行动态量化。embedding、vision、GDN门控/卷积/norm、MTP保持源精度，递推状态保持FP32。文件由临时文件原子发布，已有目标文件与源文件均受覆盖保护。脚本在产品仓库`scripts/gaudi/convert_qwen35_fp8.py`。

## 测量契约

- 完整性能使用既有1970-input/298-output-token固定MTP程序，保留实际draft、sampling和计算，并核对130个控制步骤。DiT固定正负条件307/23、1024²、25步、CFG2.5、er_sde/simple。
- 各完整对照为1 cold＋5 warm＋单独diagnostic。warm不导出logits、不运行profiler。最后一次才采集logits、同步节点、文本两步trace及DiT第0/12/24步正负条件六段trace。
- 最初把logits保存放入每次warm使基线增加约1.3s，相关目录标为`baseline-logits-profile-interrupted`，正式基线重新测量。profiler导出阶段约235s，该耗时独立于性能结果。
- 自然扩写和图片质量单独验证。自然生成token数和caption会变化，固定轨迹用于相同工作量归因。

## 基线耗时分布

| 环节 | 5 warm中位数 |
|---|---:|
| 完整流程 | 18.954s |
| TextGenerate | 5.307s |
| 两次CLIPTextEncode合计 | 0.317s |
| KSampler / DiT | 13.071s |
| VAEDecode | 0.132s |
| PreviewImage | 0.109s |

节点中位数独立计算，和完整中位数不要求逐位相等。优化优先级集中在DiT与自回归文本，VAE占比约0.7%。

## DiT设备预算

六段trace按正负条件和25步外推，重叠硬件lane通过时间区间并集计算，避免直接累加多核kernel duration。分类优先级为FP8 MME、其它MME、softmax、其它kernel、间隙。

| 互斥分类 | 外推耗时 |
|---|---:|
| FP8 Linear MME | 2.515s |
| 其它MME，主要QK/AV | 3.508s |
| softmax独占及与其它TPC重叠部分 | 2.700s |
| 其它kernel独占 | 3.604s |
| kernel区间间隙 | 1.041s |
| 合计 | 13.367s |

外推预算与无profiler DiT13.071s接近。softmax与MME的重叠已记入MME分类，因此2.700s为可见的互斥成本。

42层四类Linear理论运算量约2.027PFLOP，除以FP8 MME活动并集约为806TF/s。这是基于trace的估算，约为公开FP8峰值865TF/s的93%。这些大矩阵的有效执行阶段已有很高吞吐，整个DiT只有约19%的时间落在这类FP8矩阵上。其余attention、softmax、量化、norm、RoPE、数据搬运和提交间隙决定当前端到端速度。[Intel产品对照，第3页](https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf)

`hl-smi`按warm节点窗口采样，去除首尾各1秒，DiT AIP均值约56.0%、功率343.8W，文本AIP约24.2%、功率157.7W。采样trace中有kernel的时间约92%，与AIP读数存在口径差异，本轮尚未确定固件计数和trace差异的具体来源。硬件上限需结合算子吞吐、区间并集和端到端时间判断。[hl-smi官方指标说明](https://docs.habana.ai/en/latest/Management_and_Monitoring/Embedded_System_Tools_Guide/System_Management_Interface_Tool.html)

## 已试路径

| 候选 | 测试 | 结果与判断 |
|---|---|---|
| 文本FP8首次接线 | 完整298-token程序 | 文本740.685s。CLIP的force-cast标志阻断原生FP8，回到重复解量化。修复原生布局在保留请求输出dtype的同时使用量化权重 |
| 文本FP8＋原有区域编译 | 5 warm完整流程 | 18.899s，文本5.096s、编码0.390s。总收益约0.3%，当前波动内，显存节省更明显 |
| 小M FP8 Linear区域编译 | 同输入298-token，16线程，3 warm | 普通4.728s，编译4.597s，约2.8%。后续回归出现最大0.0703的额外输出差异及重载兼容问题，候选已撤回 |
| 厂商FP8 fused SDPA | 真实SFT捕获激活，N4403、24×128，量化计入 | BF16按head3.229ms，FP8整组14.748ms、按head15.320ms，RRMSE3.13%，淘汰 |
| FP8 QK/softmax量化/AV分解 | 同输入 | 3.520ms，BF16 3.230ms，RRMSE2.07%，淘汰 |
| 全tensor激活动态scale | Moody真实四类权重、两种N，随机激活 | 8组局部时间外推3.382s，逐行3.273s，整体慢约3.3%，保留逐行量化。局部有反向结果，未据单一shape改变产品策略 |

attention筛选使用既有SFT真实捕获激活，属于相同尺寸与数据分布的局部试验。上述候选未进入Moody产品路径。

文本固定轨迹比较1033×248320 logits，top-1一致974/1033，平均KL0.011884、最大KL0.25926、相对RMS误差6.72%。该结果量化了相同输入下的偏差，自然生成仍需单独检查。

## 自然质量与兼容性

BF16/FP8各三组完整自然扩写和1024²图片已通过finite/fallback检查并并排目视。茶壶、MOSS海报三朵黄花、红围巾白杯人物均保留指定元素，caption和图像细节有明显变化。对应输出token数为343/332、307/266、243/453，量化引发的自然扩写长度变化会影响实际等待时间。

切换到MTP off并重载时发现Habana `synStatus 26`。异步堆栈先后指向DeltaNet、MLP及同步点，单行clone、等价公式和区域旁路未解决完整流程，均已撤回。同步执行可完成该流程，进一步数值回归又发现新增动态FP8 Linear编译存在额外偏差，最终撤回整个小M动态编译候选。基础FP8 Linear继续复用既有模型区域编译与原生eager执行。底层vendor根因尚未确认。

## 最终产品验收

产品提交`148b252a`已部署8189，8份文件哈希一致。普通入口cold74.786s、缓存13.466s，已检查白色蓬袖大衣、MOSS招牌与湿地反光的输出，队列为空。251项回归通过。默认eager pipeline异步设置下，三组FP8自然扩写/1024²图片、卸载重载后MTP off/512²、原INT8/MTP auto均通过，finite检查通过、fallback为空。

| 相同298-token完整工作量 | BF16文本＋FP8 DiT | FP8文本＋FP8 DiT |
|---|---:|---:|
| 5 warm完整中位数 | 18.954s | 18.933s |
| cold完整流程 | 68.610s | 75.146s |
| TextGenerate | 5.307s | 5.154s |
| CLIPTextEncode合计 | 0.317s | 0.411s |
| DiT | 13.071s | 13.074s |
| warm前已达allocator峰值 | 34.777GiB | 27.267GiB |
| 请求结束allocator常驻 | 25.372GiB | 17.987GiB |

完整时间差约0.1%，当前视为持平。文本扩写的局部节省被普通编码增加的开销及其它波动抵消。显存峰值减少7.51GiB，常驻减少7.38GiB。最终无profiler/logits保留的卸载占用为2,685,928,832 bytes。带logits/profile的基线卸载3,711,490,432 bytes包含调查进程保留对象，不能与最终值直接比较。

## 下一步有证据的优化方向

1. 优先研究attention与softmax的数据流。现有QK/AV及softmax互斥预算约6.21s，目标是减少N²中间结果与softmax读写。已测试的厂商FP8 SDPA和直接FP8 GEMM替换均更慢，新方案需要真实输入、数值与完整25步验证。
2. 细分其它kernel的3.60s，尤其FP8量化前后的reduce/cast、norm/modulation及FP32 RoPE。优先评估量化与上游激活融合，以及同一激活在多个投影间复用量化。优化须保留原有舍入、patch及模型卸载契约。
3. 文本短步profile中，FP8减少矩阵活动时间，同时launch数从950升至992。短窗口受profiler影响显著，不能按其间隙比例直接推算全流程收益。下一阶段重点是四行GDN投影、词表投影和采样控制的提交粒度，沿用既有clone与回滚状态约束。
4. 大Linear剩余纯算术空间已较小，后续以完整流程缩短、固定轨迹误差和自然质量作为验收条件。

## 复现入口

[结构化证据](evidence/moss-fp8-system-2026-11-12.json)保留计时、设备预算、候选、质量、回归、部署和代码哈希。

- `moss_fp8_system_jobs.py`与`moss_fp8_system_server.py`运行固定MTP流程、节点计时和trace。
- `moss_fp8_system_summary.py`计算节点中位数及互斥设备区间。
- `moss_fp8_text_linear.py`复现真实小M权重延迟，`moss_fp8_text_candidate.py`保存实验接线。
- `moss_fp8_logits_compare.py`分块比较固定轨迹logits。
- `moss_fp8_text_quality.py`运行BF16/FP8自然扩写、图片、重载及INT8回归。
