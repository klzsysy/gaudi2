# Qwen3.5文本权重与量化路径

本篇涵盖INT8 ConvRot、BF16常驻、FP8文本及W4A8的2026-11-08至11-12实验。当前普通CLIPLoader支持原INT8、解码BF16和原生FP8三种文件。FP8接入提交为`148b252a`。

## 从存储格式定位实际计算

初始INT8 ConvRot使用`QuantizedTensor`，`full_precision_mm=True`使每层先解量化为BF16再在HPU执行Linear。日志中的BF16描述计算dtype。强制关闭该选项后进入`torch._int_mm`，2-token约1.47→12.54秒并产生CPU fallback。独立INT8矩阵只有约3.17TF/s、103次fallback，本机1.24.2栈未找到可用于该格式的原生INT8 GEMM路径。

输出头曾直接调用`F.linear`绕过量化层策略，普通生成和MTP verify每步触发CPU fallback。`0690d18f`改为调用lm_head层自身，普通短生成10.19→4.47秒且token相同，42项decode回归通过、1项CUDA跳过。337.013→134.586秒的完整重放包含生命周期差异，保留为使用观察。

## BF16常驻

复用已有INT8文件，将317个二维量化权重还原为BF16并移除对应量化metadata。输出`qwen3.5_9b_bf16_from_int8.safetensors`为19,306,312,416 bytes，原INT8为9,748,121,532 bytes。这是已有量化权重的解码版本。

MTP auto、相同prompt/seed、2-token warmup后各5次32-token测试，INT8中位10.097秒，BF16为9.407秒，描述性差异约6.8%。两侧各自token稳定，跨路径末尾token不同。完整自然图的token长度、加载和缓存状态存在差异，未据此计算严格端到端收益。

| 局部候选 | 结果 | 取舍 |
|---|---|---|
| 单独缓存约2.03GiB BF16 lm_head | MTP off 16.963→15.920s，Auto 10.068→10.622s | 方向相反，未采用 |
| 强制embedding驻留 | 短probe约3%差异 | 噪音内，未采用 |
| 两阶段精确top-k | 9.407→9.349s，首个采样token改变 | BF16同分值的tie policy改变，撤回 |
| JChen式逐通道FP8输出头 | 单次32-token9.580s，BF16五轮中位9.407s，末尾token不同 | 无可确认收益，不能据此外推FP8原语上限 |
| 早期561模块FP8文件 | 2-token16.23s，原路径约1.47–1.87s | 当时接线仍有解量化慢路径，后续重新做原生适配 |
| 早期仅lm_head FP8 | 2-token4.71s | 无收益 |

早期转换误处理167个一维norm权重，触发HPU `add_stub`错误。修正为保留norm精度后可加载。下载BF16源文件时已有HF端点/Xet持续重试，随后使用现有INT8本地转换，来源记录到metadata。

## 原生FP8产品接入

最终文件`qwen3.5_9b_fp8_from_int8.safetensors`为11,377,021,988 bytes。由上述BF16解码文件再次量化，转换201个语言Linear，包括输出头，采用E4M3、逐tensor权重scale、224安全范围及逐行激活动态量化。embedding、vision、GDN门控/卷积/norm、MTP保留源精度，递推状态FP32。便携转换入口为产品`scripts/gaudi/convert_qwen35_fp8.py`，使用临时文件原子发布及覆盖保护。

初次完整程序文本耗时740.685秒，定位到CLIP force-cast阻断原生FP8。修复量化权重的计算入口并保留请求输出dtype后，复用已有Qwen区域编译完成验收。

同一Moody FP8 DiT、固定1970输入/298输出、1024²/25步，各1 cold＋5 warm。

| 指标 | BF16文本 | FP8文本 |
|---|---:|---:|
| 完整warm中位数 | 18.954s | 18.933s |
| TextGenerate | 5.307s | 5.154s |
| 正负普通编码 | 0.317s | 0.411s |
| DiT | 13.071s | 13.074s |
| cold | 68.610s | 75.146s |
| allocator峰值 | 34.777GiB | 27.267GiB |
| 请求结束常驻 | 25.372GiB | 17.987GiB |

完整延迟处于波动范围，显存峰值减少7.51GiB。1033×248320固定logits的top-1为974/1033，平均KL0.011884，RRMSE6.72%。三组自然扩写输出长度分别343/332、307/266、243/453，图片主体通过该组目视，文字和细节变化明确记录。

251项回归通过，默认异步eager下自然图、512²重载/MTP off和原INT8通过，fallback为空，卸载2,685,928,832 bytes。

## 未采用的后续候选

- 小M FP8 Linear额外动态编译在16线程三轮局部测试4.728→4.597秒，但出现最大0.0703额外输出差异和重载`synStatus26`。单行clone、等价公式和局部旁路未解决完整流程，候选整体撤回。底层vendor原因尚未确认。
- 全tensor激活scale在Moody八组真实shape外推3.382秒，逐行3.273秒，整体慢约3.3%，采用逐行策略。
- 文本共享量化的完整流程17.166秒、文本5.157秒，与同期产品持平。减少一次量化还需计入缓存所有权、多个投影的scale和完整调用成本。
- W4A8 GPTQ文件为7,343,591,088 bytes，主体`asym_w4a8_int8`、group16、ConvRot256，lm_head仍是INT8 ConvRot。已有eager backend先解4-bit再走INT8或BF16解量化，原生INT8仍会fallback。重新评估需要可用的HPU backend及完整质量对照。

## 可复用判断与证据

存储变小、原生量化算子启用和端到端加速需要分别验证。FP8文本这一轮的主要收益是容量，文本短步提交优化见 [解码执行](TEXT_DECODE.md) 与 [GDN](TEXT_GDN.md)。原生FP8主模型的矩阵形状与量化成本见 [DiT FP8](DIT_FP8.md)。

原始资料位于`performance-2026-11-08`和`performance-2026-11-11-fp8-system`。固定图为`workflows/moss-user-textgen-bf16.json`，复现入口为`moss_fp8_system_jobs.py`、`moss_fp8_text_linear.py`、`moss_fp8_logits_compare.py`及`moss_fp8_text_quality.py`。[结构化证据](../evidence/moss-fp8-system-2026-11-12.json)
