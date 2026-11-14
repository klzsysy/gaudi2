# Qwen3.5文本权重与量化路径

本篇涵盖INT8 ConvRot、BF16常驻、FP8文本及W4A8的2026-11-08至11-12实验。当前普通CLIPLoader支持原INT8、解码BF16和原生FP8三种文件。FP8接入提交为`148b252a`。

2026-11-14原生INT8 MME能力核验阴性，安装库MME类型掩码排除INT8，直接Synapse的9组INT8建图均失败，3组BF16对照成功。用户要求阴性即停止，通用INT8开发已停止并放弃。[验证结果](INT8_NATIVE_MME.md) · [历史计划](INT8_DEVELOPMENT_PLAN.md)

## 2026-11-14 最后一次 Qwen Image 2.1 INT8 扩写调查

普通8189服务部署标记为`ed9552e581acfa37c6be2ac6f1a86a629f21d542`。读取实际API历史、日志、模型header及已部署源码，定位用户截图对应请求`4da456e9-127e-40ba-9082-aa494b08678e`，北京时间22:45:45开始，总时长120.523秒。调查时队列为空，`hl-smi`显示单个服务进程、HPU占用约49GiB、瞬时利用率0%。这些硬件值采于任务结束后。

### 现场请求与工作量

| 请求开始时间（Asia/Shanghai） | PE模型 | 文本进度末值 | 文本平均速度 | 整图 |
|---|---|---|---|---|
| 22:43:03 | 同源BF16解码文件 | 456/4096，00:11 | 39.12 tokens/s | 52.325s，含模型加载、1.2MP图 |
| 22:44:11 | 同源BF16解码文件 | TextGenerate命中缓存 | — | 10.492s，2.0MP图 |
| 22:45:45 | 原INT8 ConvRot文件 | 737/4096，01:41 | 7.23 tokens/s | 120.523s，2.0MP图 |

表中token数及秒数来自tqdm最终进度，停止token与迭代器更新可造成计数边界差异。按末尾平均速度折算，两个文本循环约11.7秒和101.9秒。中段稳定速度分别约42和7.27 tokens/s，单token耗时约相差5.8倍。INT8进度计数又约为BF16的1.62倍，两项共同放大文本耗时。

两次PE计算的输入图逐字段比较确认，提示词、系统提示词、文本seed=1及采样设置相同。thinking=false，temperature=1.0、top_k=20、top_p=0.95、presence_penalty=1.5、max_length=4096、mtp=auto。切换INT8的请求相对前一条只改了PE模型文件和图像KSampler.seed。4096是上限，实际请求成功提前结束，18%为已生成数量相对上限的比例。

2.0MP两次采样均为8步，最终进度约7秒。最后一次请求的绝大多数时间集中于文本生成循环，其他约11秒包含PE模型加载、基础文本编码、VAE和执行调度等成本，日志未提供这些项的独立计时。原INT8加载日志为`9032.15 MB loaded, full load: True`，当前设备为`hpu:0`。

### 代码定位

1. `comfy/text_encoders/qwen35.py:1326`的自定义HPU量化operations入口只接入`float8_e4m3fn`。INT8通过`comfy/sd1_clip.py:114`创建`mixed_precision_ops(..., full_precision_mm=True)`。
2. `comfy/ops.py:1610`在量化Linear每次forward时调用`weight.dequantize()`，随后执行`F.linear`。当前INT8文件的310组量化配置均为`int8_tensorwise`、`convrot=true`，group256/64/16分别为202/81/27组。310组包含视觉和embedding参数，文生图逐token计算涉及其中的语言Linear及输出头。
3. 安装环境中的`comfy_kitchen/tensor/int8.py:170`将该格式送入ConvRot权重反量化。其eager实现`backends/eager/quantization.py:916`依次做缩放解码、Hadamard逆旋转，再在`:926`转换成目标dtype。反量化权重在此调用中临时生成，已有BF16文件将这部分工作提前完成并持久保存。
4. `comfy/ops.py:1558`的`hpu_compile`门控限定HPU原生FP8 operations。INT8 operations的标记为false，因此`qwen35.py:615/628`的短步MLP区域编译和`:522/538`的完整attention区域编译被跳过。GDN内部递推和公共norm等仍有各自优化入口。
5. 两个PE文件的header均无MTP张量。`qwen35.py:969`在`self.mtp is None`时使用普通自回归生成，每步预测一个token。该条件同时适用于本次两种权重文件。

以上三个产品源码文件的本地/服务器SHA256完全一致。模型header确认BF16文件为同一个INT8文件的本地解码产物，18,819,722,496 bytes，INT8原文件9,471,072,252 bytes。BF16路径以额外权重常驻空间换取预先完成反量化及已有编译入口。实际自然采样内容和长度可以随路径改变，本次相同文本seed也出现了长度变化。

### 结论与调查范围

当前INT8选择触发了逐层重复反量化/ConvRot还原、BF16矩阵计算及部分区域编译缺失的组合路径。稳定约7 tokens/s与11月13日原INT8增强器验收一致。已有快速PE文件为`qwen3.5_9b_qwen_image_2.1_pe_t2i.bf16_from_int8.safetensors`，同源模型已在本次历史中达到约42 tokens/s的稳定速度。

本轮为只读调查，使用实际历史执行作对照，未新增推理、逐算子profile或产品修改。两项底层开销各自的耗时占比尚待固定token的profile/消融测量，最近三次请求也有加载、文本缓存及分辨率差异。此前强制原生INT8矩阵路径出现CPU fallback的实验见下节，该历史结果不足以推定本次发生CPU fallback。服务绑定`192.168.10.21:8189`，最初访问loopback被拒绝后使用实际绑定地址完成读取。

[本次结构化证据](../evidence/qwen21-int8-slow-2026-11-14.json)保存请求参数、提示词hash、字段差异、日志摘要、模型header摘要及产品源码hash。

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
