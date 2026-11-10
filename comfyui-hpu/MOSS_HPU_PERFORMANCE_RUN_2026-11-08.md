# MOSS HPU 性能计划恢复与可观测性验证

更新时间：2026-11-09（Asia/Shanghai）

本文只保留本轮性能调查的可复现实验、结果、失败路径和限制。原始大 trace 留在服务器 performance-2026-11-08 目录，不进入 Git。

## 恢复与工具验证

- 服务器恢复后，5.15.0-202 和 5.15.0-186 的 Habana DKMS 均为 installed，设备为 Operational，独立服务版本为 0690d18f，/system_stats 返回 hpu:0 GAUDI2。
- 队列空闲后停止独立服务再运行 probe；BF16 matmul 与 CPU 参考一致。调查期间没有重新下载模型，也没有修改基础 torch、模型或产品推理代码。
- sysstat/linux-tools、HPU torch profiler activity、synprof_parser、Perfetto Python/native trace_processor 和 fallback 计数均完成 smoke。真实 .hltv 转换和 Perfetto SQL 通过。
- TensorBoard Habana 插件和 habana_perf_tool 的高层摘要在短 eager smoke 缺 ProfilerStep/bridge track；真实 MOSS trace 虽有 ProfilerStep，仍因 MMEEvents 不可用失败。该限制保留为工具结论，当前验收路径是原始 HPU trace + Perfetto SQL。

## eager 校准与低精度边界

- 4096³ eager FP8 E4M3/E5M2 torch.mm 原生 HPU、无 fallback，约 525.88/519.63 TFLOP/s；同 shape BF16 约 275.76 TFLOP/s。相对公开 BF16/FP8 理论值 432/865 TFLOP/s，约为 63.8%/60.8%/60.1%，仅是形状和路径特定的微基准。
- INT8 torch._int_mm 约 3.17 TFLOP/s，fallback 103 次；当前 Habana 1.24.2 PyTorch 栈没有可用于该 ConvRot 权重的原生 INT8 linear/mm HPU 路径。Gaudi2 有 INT8 TPC 指令能力，但硬件能力不等于 PyTorch 量化 GEMM 接口。
- E4M3FNUZ/E5M2FNUZ 探针不支持；FP8 custom op schema 存在，但缺 scale/layout contract 时没有硬测。公开 827 TFLOP/s 数据使用不同版本、形状、lazy、scaled_mm、BF16 输出、cache/output reset 和 device event；本机相同外部形状的 eager torch.mm 为 564.11 TFLOP/s，scaled_mm 复核报 synStatus 26，不能直接比较或外推到 MOSS。

## 严格 B0/B1 基线

冻结工作流为 workflows/moss-user-textgen.json，SHA256 为 c283f15bcae32cceebd51c9a8863400f421f46354be32b3c52c4a16743be7bd6；扩散 seed 649029532689846，文本 seed 0。它与用户历史 143.34 s 的 seed 不同，因此本轮不计算相对用户基线的严格提速。

执行控制：

- B0：1 次 process cold + 5 次真实 warm；
- B1：1 次 transition + 5 次真实 warm，首轮 caption 固定；
- 13 个请求均成功，目标计算节点执行断言通过，warm 保留 loader，不重复加载；
- 正常计时无 profiler/逐步同步；另有一次同步诊断请求。

| Case | 样本 | median | min–max |
|---|---:|---:|---:|
| B0 process cold | 1 | 145.521 s | — |
| B0 warm | 5 | 137.282 s | 135.893–137.526 s |
| B1 transition | 1 | 40.305 s | — |
| B1 warm | 5 | 40.183 s | 39.690–40.314 s |
| B1 synchronized diagnostic | 1 | 40.313 s | — |

- B0 warm 节点计时中位数：TextGenerate 96.966 s，文本编码合计 2.013 s，KSampler 37.960 s，VAE 0.132 s，PNG 0.117 s。6 次 B0 都是 327 tokens，token ID 一致，fallback 始终为空，图片语义和 MOSS 灯牌/白大衣/剧院/湿地反光目视通过。
- B1 同步诊断：文本编码 2.034 s、KSampler 38.006 s、VAE 0.136 s、PNG 0.122 s；分段覆盖 99.963%。逐步同步间隔 median 1.5212 s；诊断请求比 B1 warm median 高 0.32%，仅作为扰动参考。
- latent [1,128,64,64] 与 VAE 输出 [1,1024,1024,3] finite；PNG 像素 MAE 为 0–0.1226/255，最大单像素差最高 11/255。AIP/功耗旁采样只作状态证据，不能当 MFU/MBU。
- 首次切换文本 probe 时旧服务尚未完全释放，出现 Device acquire failed；旧 PID 退出、hl-smi Compute Processes 为空后重试成功。该错误是切换次序问题，不是恢复后的驱动故障。

P1 尚欠 B2 模式/长度矩阵、B4 尺寸/步数和跨进程 cold；本轮不宣称 P1 全部完成或已有 P3 收益。

## P2/P3 候选与拒绝

### lm_head 解量化缓存

把 INT8 ConvRot lm_head 一次性解量化为约 2.03 GiB BF16 并复用：

- MTP off、32 tokens 单次样本：16.963 s → 15.920 s，token ID 一致，fallback 为空；
- 目标 MTP auto：10.068 s → 10.622 s，方向相反；
- 额外显存和收益不稳定，候选拒绝，未改产品代码。

### 强制 embedding 驻留 HPU

调查进程中固定 vbar_above_watermark=false，MTP auto 8-token 两次约 4.052 s/3.929 s，差异约 3%，低于短 probe 噪音。未作为产品配置或代码改变。

### 当前 host/device 线索

MTP auto 8-token trace 的主要线索是 aten::_copy_from，约 2.14 s/profile；ConvRot dequant 约 0.21 s，fallback 为空。没有新的 device/host 调用归因前，不缓存权重、不强行改变 embedding 驻留、不切 lazy。

## dtype、INT8 与低 HPU 利用率复核

日志中的 dtype: torch.bfloat16 是计算 dtype；权重仍是 QuantizedTensor。当前 full_precision_mm=true 会逐层解量化到 BF16，再在 HPU 执行 F.linear。

独立探针强制 full_precision_mm=false 后，2-token 从约 1.47 s 变为 12.54 s，调用栈进入 fast_int8_mm → torch._int_mm，并发生 CPU fallback。因此当前栈不能把 native INT8 开关作为优化。

同一 eager、MTP auto、短 prompt 的 32-token 旁采样为 10.10 s，26 个 AIP 样本平均 12.65%、最高 47%。低负载与小批量自回归、.tolist() 控制同步、重复解量化和临时张量搬运一致；不能解释为 HPU 没有加载。

## 外部方案审查

审查 JChen-318/gaudi2-single-card-vllm 后确认：

- native/ 主要是 GDN、拒绝采样、TPC shim 和 NVFP4 MoE，没有通用 INT8 GEMM、INT8 Linear 或 ConvRot 解码 kernel；
- hpu_lm_head_fp8.py 面向 BF16 lm_head → FP8，不是已有 INT8 ConvRot 的直接替换；模型 shape、校准、token/top-1 和 fallback 都需独立验证；
- 该仓库 SDK/torch/ABI/权重布局与本机不同，不能直接替换生产 GC_KERNEL_PATH。

它只能提供 FP8 vocabulary projection、speculative decode、lookahead 和 host/device overlap 的设计参考。

## 2026-11-09：JChen 仓库 FP8 路径源代码复核

本次复核针对仓库当前 `main`（提交 `e08710f7fe4c953529c728ec0b3a661f922d07c5`）。其目标是 Qwen3.8-27B FP8 target + 独立 DFlash2 FP16 draft，验证环境为 SDK 1.24.1、`torch 2.11.0a0`、`habana-torch-plugin 1.24.1.482`；本项目使用 Habana 1.24.2、`torch 2.13.0a0+git2469d18`，因此不能把编译产物或运行时 ABI 直接混用。

### 实际使用的 FP8 路径

- `vllm_gaudi/ops/hpu_lm_head_fp8.py` 只实现 vocabulary projection 的替换。它要求 `ParallelLMHead` 的权重是 BF16/FP16 的二维 dense tensor；按输出通道分块计算 amax，调用 `torch.ops.hpu.cast_to_fp8_v2` 生成 E4M3 权重和每输出通道 scale，并保留 BF16 原权重。
- 每次投影先对 BF16 activation 做逐 token 动态 FP8 量化，然后由 `apply_fp8_linear_hpu` 调用厂商 `torch.ops.hpu.fp8_gemm_v2`，输出 BF16。该文件没有自带一个可脱离 Habana runtime 的 FP8 GEMM kernel。
- `vllm_gaudi/ops/hpu_fp8.py` 的 per-tensor/per-channel FP8 linear 也走 `fp8_gemm_v2`；部分 block-FP8 路径会先解量化成 BF16 再 `F.linear`。仓库 `native/` 中的自写 kernel 主要是 GDN、拒绝采样、TPC shim 和 NVFP4 MoE，没有通用的 INT8 GEMM、INT8 ConvRot 解码或独立 FP8 GEMM。TPC shim 只是把多个 TPC 库按 GUID 转接成单一 `GC_KERNEL_PATH` 文件。
- 该仓库的性能收益并非来自 FP8 投影单项。记录中的词表投影约从 1.13 ms 降到 0.59 ms，top-1 与 BF16 一致率为 98.4%、平均 KL 为 0.0016；其余收益还来自 DFlash2 speculative decode、GDN 融合、快速 top-k、预提交和 host/device overlap。

### 与 MOSS 当前权重的差异

- MOSS 使用 `qwen3.5_9b_int8_convrot.safetensors`，线性权重是 `QuantizedTensor`/ConvRot INT8 布局；当前安全路径是逐层解量化为 BF16 后在 HPU 执行 `F.linear`。因此 JChen 的 `hpu_lm_head_fp8.install()` 不会直接接受 MOSS 的 `lm_head`，因为它明确要求 BF16/FP16 dense weight。
- 可行的实验路线只有两种：先把 MOSS 的 `lm_head` 一次性解量化为 BF16，再按 Gaudi2 的 scale/range 约定转换为 FP8 并调用 `fp8_gemm_v2`；或者编写 ConvRot INT8 解码与 FP8 GEMM 融合 kernel。JChen 仓库没有后者，直接复制其 `native/` 不能解决问题。
- 本项目先前的全模型/局部 FP8 A/B 是 Comfy/Kitchen 的自制 FP8 权重路径，没有移植这套 `fp8_gemm_v2` wrapper，因此“自制 FP8 变慢”不能严格等同于“JChen 的 vendor-op 路径也一定变慢”。但此前 MOSS 的 BF16 lm_head 缓存在 MTP auto 下已从 10.068 s 变为 10.622 s，说明仅改变 lm_head 不能预设端到端收益。
- 仍需严格复用 Gaudi2 的 FP8 range、E4M3 存储字节、每通道权重 scale、逐 token activation scale、转置布局和 BF16 输出约定。MOSS 之前的 FP8 解码 `inf/nan` 修复已经证明，不能只替换 dtype 或复用 CUDA 的 scale 语义。

### 可移植性结论

- **不能直接移植“kernel 二进制”**：该仓库没有独立的通用 FP8 GEMM 二进制可替换 `GC_KERNEL_PATH`；其代码依赖不同版本的 Habana runtime、PyTorch、vLLM layer contract 和权重布局。
- **可以移植“lm_head 的调用策略”**：本机已观察到 `torch.ops.hpu.fp8_gemm_v2` schema，但尚未在 MOSS 的 ConvRot scale/layout 上完成硬测。最合理的下一步是只做调查 probe：一次性把 `lm_head` 解量化到 HPU BF16，按 JChen 的 per-output-channel weight FP8 + per-token activation FP8 方式转换，再调用 `fp8_gemm_v2`。
- **不建议当前直接做全模型 FP8 移植**：需要把 561 个量化模块从 ConvRot INT8 重新定义为可被 FP8 linear 消费的权重格式，且文本扩写还受小批量自回归、`_copy_from`、控制同步和 MTP 路径影响。DFlash2 的独立 draft、状态快照和 lookahead 也不是一个 kernel patch 能带过来的。

若继续实施，验收应固定 2/8/32-token 与完整用户图，至少同时检查 token ID/top-1/KL、文本和图片质量、fallback、显存、HPU trace、5 次 warm median 及端到端时间；只有在 vendor-op 路径本身稳定快于当前 INT8→BF16 路径时，才考虑产品化。源代码参考：[hpu_lm_head_fp8.py](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/main/vllm-gaudi/vllm_gaudi/ops/hpu_lm_head_fp8.py)、[hpu_fp8.py](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/main/vllm-gaudi/vllm_gaudi/ops/hpu_fp8.py)、[优化记录](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/main/docs/OPTIMIZATION.md)。

## FP8 文本编码器 A/B

下载 BF16 文件时，已有 HF_ENDPOINT/代理下 Xet 单块长时间重试；未更改端点，停止下载并清理临时文件。随后从现有 INT8 权重在 HPU 解量化生成独立 FP8 E4M3FN 文件，保持 MOSS key layout 和量化元数据。

- 全模型 FP8：561 个量化模块可加载 HPU，但 MTP auto 2-token warmup 16.23 s，慢于原 INT8→BF16 的约 1.47–1.87 s。
- 仅 lm_head FP8：2-token warmup 4.71 s，仍慢于原路径；32-token 在第 17–27 token 间约 1.1–1.3 s/token，未继续跑完。
- 全局 full_precision_mm=false：主体 INT8 层进入 CPU fallback，2-token 约 17.42 s。
- 初版误量化 167 个 1-D norm weight，触发 HPU add_stub 错误；改为保留 norm BF16 后重新加载通过。

结论：当前 Habana 1.24.2 / torch 2.13.0a0 / comfy-kitchen 组合下，自制 FP8 文本编码器没有性能收益，反而变慢；文件未替换生产模型。

## 2026-11-09：BF16 常驻与 W4A8 GPTQ 候选复核

服务器只读检查确认共享模型目录已有：`qwen3.5_9b_int8_convrot.safetensors`（9,748,121,532 bytes）和 `qwen3.5_9b_w4a8_gptq.safetensors`（7,343,591,088 bytes）。W4A8 文件的 `lm_head` 仍然是 `[248320,4096]` 的 `int8_tensorwise + ConvRot`；主体线性层使用 `asym_w4a8_int8`，`group_size=16`、`convrot_groupsize=256`，带 FP8 相对 scale、FP32 通道 scale 和 4-bit codebook。

### 一次性解量化为普通 BF16

- 这是当前最值得先做的候选。已有 INT8 ConvRot 的 `QuantizedTensor.dequantize()` 会还原到原始 dense basis；将每个二维线性权重一次性解量化后保存为普通 BF16 safetensors，可以绕过运行时每次 forward 的解量化和 `torch._int_mm` 分支。
- MOSS 的 `SD1ClipModel` 对带 quant metadata 的文本模型固定创建 `mixed_precision_ops(..., full_precision_mm=True)`；当前安全路径正是“每层解量化到 BF16，再执行 `F.linear`”。换成没有 `comfy_quant` 元数据的普通 BF16 文件，才能把这个结果真正常驻并复用。
- 该路线使用现有 INT8 权重，数值风险低于换成 W4A8；主要风险是启动峰值显存和同时保留原量化副本。应在独立进程中只加载 BF16 文件，记录 HBM 峰值，避免把原 INT8 和 BF16 模型同时留在 HPU。
- 用户确认 96 GiB 有余量后，空间换时间在工程上可行，但仍需用固定 2/8/32-token 和完整 B0 图验证 token ID、文本、latent/PNG finite、fallback、warm median 和端到端时间。单看 lm_head 缓存此前没有收益，不能替代整模型 BF16 A/B。
- 按当前 INT8 文件的张量形状粗估，完整普通 BF16 文本权重约 **18.0 GiB**（不含运行时 workspace、模型对象和扩散模型）；这支持先做 BF16 常驻试验，但不能代替实际 HBM 峰值测量。

### W4A8 GPTQ 的当前限制

- 服务器上的 `comfy_kitchen` 只注册了 eager backend；CUDA/Triton/HPU 专用 W4A8 backend 均不可用。eager W4A8 路径先把 4-bit code 解成 INT8，再调用 `int8_linear`，最终使用 `torch._int_mm`（当前 Habana torch 没有 `torch.int8_mm`）。
- 因此，保持 MOSS 默认的 `full_precision_mm=True` 时，W4A8 会在每次线性层调用中解量化为 BF16；强制 `full_precision_mm=False` 则会回到已验证会 CPU fallback 的 native INT8 路径。它目前只能降低磁盘/HBM 权重存储，不能视为可用的 HPU W4A8 加速路径。
- W4A8 还没有改变 `lm_head` 的格式，词表矩阵仍是 INT8 ConvRot；所以它不会自动解决输出头和自回归阶段的主要问题。除非新增 HPU W4A8/ConvRot kernel，否则优先级低于普通 BF16 常驻。

### 当前排序

1. 从已验证的 INT8 ConvRot 文件生成一个独立的普通 BF16 文本模型，先做冷启动和 2/8/32-token A/B。
2. 若 BF16 稳定降低 TextGenerate 时间，再做完整 B0/B1 重复基线；不要把模型加载时间混入 warm 结果。
3. 暂不把 `qwen3.5_9b_w4a8_gptq.safetensors` 作为性能优化方案；只有在 HPU W4A8 backend 或融合 kernel 出现后再复测。

## 2026-11-09：BF16 常驻实测与可移植优化复核

### BF16 文件与 token probe

- 从现有 INT8 ConvRot 文件转换了独立的 `qwen3.5_9b_bf16_from_int8.safetensors`：317 个二维量化权重还原为 BF16，删除 317 组 `.comfy_quant`/scale 元数据；文件大小 **19,306,312,416 bytes**。源文件未覆盖。
- HPU 常驻加载量约 **9.75 GB（INT8）→19.31 GB（BF16）**；BF16 模型的 `embed_tokens`/`lm_head` 均为普通 BF16 Parameter，不再是 `QuantizedTensor`。
- 同一进程、MTP auto、固定 prompt/seed、先 2-token warmup 后 5 次 32-token warm：

| 路径 | 32-token median | min–max | token ID（同一路径） | fallback |
|---|---:|---:|---|---|
| INT8 ConvRot | **10.097 s** | 9.778–10.200 s | 一致 | `{}` |
| 普通 BF16 | **9.407 s** | 9.013–9.747 s | 一致 | `{}` |

BF16 相对 INT8→BF16 运行路径约快 **6.8%**。跨权重格式的采样序列在末尾出现差异，属于 BF16 与量化还原值的数值/采样差异；两条路径的文本均可读，固定路径内重复结果稳定。

### 完整 MOSS 工作流

- BF16 workflow 首次冷请求成功：1024²、25 步、图片 finite、语义正确，wall **150.20 s**；日志中的文本模型加载约 18.4 GB。
- 模型缓存后改 TextGenerate seed 强制重算，BF16 请求 wall **125.22 s**，日志 prompt execution **124.91 s**；INT8 同 seed 切换请求 wall **130.18 s**，日志 prompt execution **125.97 s**。两次扩写长度不同且 INT8 请求包含重新加载量化模型，因此这组完整 wall 时间只作正确性/生命周期证据，不作严格端到端收益结论。
- 可复现 BF16 工作流保存在 `workflows/moss-user-textgen-bf16.json`；实测图片保存在 `results/moss-bf16-workflow.png` 和 `results/moss-bf16-workflow-seed1.png`。
- 固定英文 caption 的 B1 1024²/25-step 对照也成功：BF16 与 INT8 图片均 finite、目视语义一致；PNG MAE **0.01046/255**、PSNR **33.33 dB**，均值/标准差分别为 BF16 125.31/73.20、INT8 124.74/72.39。BF16 B1 wall **50.08 s**，INT8 B1 wall **40.06 s**，后者命中了已有服务/节点缓存，不能用来推算纯模型算子差异。

### JChen 优化点的映射

- 当前 Qwen3.5 MOSS 实现有 MTP auto 自适应树和 deferred 状态管理语义，但不能据此判断已具备 JChen 的 HPU 加速。后续源码复核确认：GDN 在本机走 PyTorch 参考实现，HPU 的异步预取和 Comfy 图重放关闭，详见下方“实际执行路径纠正”。先前将这些接口视为已经启用的优化不成立。
- 尝试把 JChen 的两阶段精确分块 top-k 移植到 HPU 采样。该路径在 BF16 32-token probe 中 median **9.349 s**，只比普通 BF16 的 9.407 s 快约 0.6%；由于 BF16 词表中大量相同分值，新的 tie policy 改变了首个采样 token（不是可接受的 token ID 保持），因此已回滚产品 checkout 的实验改动，并将该路径标记为拒绝。当前产品文件恢复到实验前版本。
- JChen 的 guard skip、RNG pool、lookahead/predraft、DFlash2 独立草稿、FP8 KV 和 vLLM graph/调度器依赖 vLLM 请求批处理或不同状态 contract；本轮没有把它们伪装成 Comfy 可移植优化。FP8 lm_head 仍是独立候选，不能与 BF16 常驻收益混算。

### JChen FP8 vocabulary projection probe

- 在普通 BF16 文本模型上按 JChen 的布局做了调查 probe：`lm_head` 权重一次性转换为 per-output-channel E4M3（约 1.017 GiB）并调用当前 Habana `torch.ops.hpu.fp8_gemm_v2`，逐 token activation 动态量化；没有改产品代码或默认服务。
- 2-token warmup 为 1.732 s，32-token 为 **9.580 s**；普通 BF16 同进程对照 median 为 **9.407 s**。fallback 始终为空，但采样末 token 出现差异，未形成收益，因此拒绝该候选。

## 2026-11-09：实际执行路径纠正

本次只读复核服务器已安装源码，没有重启服务或运行新的性能测试。重点区分“接口存在”“原生融合 kernel 启用”和“无 CPU fallback”三者。

- `comfy_kitchen/gated_delta.py:62` 的 `deferred_is_available` 只接受 CUDA/HIP；HPU 返回 False。`deltanet_conv_step_deferred` 调用 `_conv_deferred_torch`；`gated_delta_decode_deferred` 调用 `_decode_deferred_torch`（约 224 行起）。它们是由普通 torch 操作组成的参考实现，张量仍可在 HPU，不代表算子回退到 CPU。
- `_decode_deferred_torch` 固定循环 `SLOT_MAX=8`，即使待提交 token 少于 8，也通过中性门控跑完全部槽位；随后按当前 verify 行再次递推。每轮有状态乘法、einsum、索引、where 和拷贝等操作。它是融合/编译和减少无效槽位计算的具体候选，尚无分段计时证明其占总时间的比例；直接改循环还需保证 MTP tree/rollback 状态语义。
- `comfy/model_management.py:1374` 对 HPU 的 `device_supports_non_blocking` 返回 False；约 1406 行将 HPU 的 `NUM_STREAMS` 固定为 0。`comfy/model_prefetch.py:337` 因而返回空预取队列。当前未获得这里的异步预取收益。
- `comfy/model_prefetch.py:31` 的 `malloc_graph_enabled` 和约 235 行的图重放门控要求 CUDA。当前 HPU 以 eager 模式运行，未接入等价 HPU 图/区域编译路径。不能把 CUDA graph 接口存在当成 HPU graph 已启用。
- 由此修正判断：现有证据更支持“Comfy/MOSS 的 HPU 执行路径尚缺融合和提交优化”，而非已经证明 Gaudi2 的基础算子或硬件算力不足。已有大矩阵微基准不代表小 M decode；低 AIP 也不能单独区分 host gap、HBM 带宽和 TPC kernel 限制。
- BF16 32-token 的 6.8% 是两个独立进程、各自五次重复的描述性差异；跨路径 token 末尾不同，不能当严格相同解码轨迹的收益。完整图的文本长度和加载状态不同，也尚未证明端到端固定快几秒。FP8 head 9.580 s 是单次样本，与 BF16 median 并非同进程交替 A/B，不能据此证明 FP8 原语性能差。

后续已执行 GDN/conv 区域编译、相同输入重放、状态回归及完整图重复验证，详见 [MOSS_GDN_COMPILE_2026-11-09.md](MOSS_GDN_COMPILE_2026-11-09.md)。参考：[Gaudi 执行模式](https://docs.habana.ai/en/latest/PyTorch/Reference/PyTorch_Gaudi_Theory_of_Operations.html)、[主机开销与推理优化](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Optimization.html)。

## 真实 HPU trace

- 真实用户 prompt、文本 seed 0、MTP auto、sampling on：2-token warmup 后，8-token 无 profiler 4.1018 s，profile 内 5.2150 s（含 prefill），token ID 相同，fallback {}。
- Perfetto SQL 读到 414,226 个 cpu_op、1,030,554 个 kernel 类事件和 ProfilerStep#0 5.216 s；事件存在多 engine/lane、嵌套和重叠，不等于一百万次 kernel launch，duration 不能直接相加。
- kernel 标签包含 memcpy_f32、mult_fwd_f32、cast_i8_to_f32、cast_f32_to_bf16 等，只能作为 ConvRot 解码、prefill/GDN 和 host gap 的线索，不能凭名称认定 PCIe copy 或瓶颈。
- 后续 trace 应拆分 prefill/稳态 decode、缩短范围并默认关闭 shape 记录，先完成事件去重、区间并集和 host/device 关联。

## 当前状态

后续 GDN/conv 区域编译已验证并部署为 `60da4e33`；日常 ComfyUI-Gaudi 服务恢复，队列为空，`/system_stats` 为 `hpu:0 GAUDI2`。最终性能、质量边界及源码提交见 [MOSS_GDN_COMPILE_2026-11-09.md](MOSS_GDN_COMPILE_2026-11-09.md)。B2/B4 和其余 host/device 归因仍是进一步工作，不影响本次已通过的区域编译验收。

## 参考

- Gaudi PyTorch profiling：https://docs.habana.ai/en/latest/Profiling/Profiling_with_PyTorch.html
- Gaudi profiling workflow：https://docs.habana.ai/en/latest/Profiling/Profiling_Workflow.html
- Perfetto Python API：https://perfetto.dev/docs/analysis/trace-processor-python
