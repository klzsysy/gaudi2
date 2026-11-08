# MOSS HPU 性能计划恢复与可观测性验证

时间：2026-11-08（Asia/Shanghai）。所有调查脚本和过程文档在本仓库；服务器调查副本及产物位于 `/home/sonny/gaudi2/comfyui-hpu/`。产品 checkout 只作为执行目录，未改产品源码；原始 `/home/sonny/ComfyUI` 未操作。

## 恢复检查

- SSH 正常，内核 `5.15.0-202-generic`；Habana `1.24.2-481` 在 `186` 和 `202` 两个内核均 DKMS installed；设备 sysfs 为 Operational。
- 原独立服务 PID 2103，版本 `0690d18f4b0157a14387a12afcf833eec924bb9e`，8189 队列为空，`/system_stats` 返回 `hpu:0 GAUDI2`。
- 三个共享模型文件大小与既有记录一致；未重新下载模型。home 剩余约 1.1 TiB，根分区剩余 69 GiB。
- 队列空闲后通过 `stop-hpu.sh` 停止该独立服务，再串行执行 HPU probe。HPU BF16 matmul 对 CPU FP32 参考通过。

## 工具安装与验证

安装命令：`sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends sysstat linux-tools-5.15.0-202-generic`。

新增 5 个包、无升级/移除：sysstat `12.5.2-2ubuntu0.2`、linux-tools `5.15.0-202.212`、对应 generic 包、linux-tools-common `5.15.0-203.213`、libslang2。`perf version` 输出 `5.15.222`，实际二进制来自匹配当前内核的 tools。安装自动启用的 sysstat timers/service 已 disable/stop，按需旁采样。

| 工具 | 实测结果 |
| --- | --- |
| pidstat / iostat / sar | 均为 12.5.2；pidstat、iostat 已用于基线旁采样 |
| perf | 普通用户因 `perf_event_paranoid=4` 被拒绝；单次 `sudo perf stat -e task-clock,cycles,instructions -- true` 成功；未改 sysctl |
| synprof_parser | 初次缺 `libhw_trace.so`；修正进程级动态库路径后，help 与真实 .hltv → JSON 离线解析均成功 |
| TensorBoard / Habana torch_tb_profiler | 基础环境已含 2.21.0 / 0.4.0，未 pip 重装；临时 loopback 6006 服务启动且 `pytorch_profiler.enabled=true` |
| Perfetto | 基础环境已含 Python 包 0.59.1；首次缺原生 trace_processor，直连 Google storage 约 130 s 超时，随后通过已有 `http_proxy_enable` 下载成功；fixture SQL 与真实 HPU trace SQL 均通过 |
| torch.profiler CPU + HPU | BF16 smoke 导出成功，实际含 78 个 kernel 事件（含多个 engine/lane 事件），示例 `gemm`、`relu_fwd_bf16`；fallback `{}` |

Perfetto 原生程序位于 `/home/sonny/.local/share/perfetto/prebuilts/trace_processor_shell-2434129a3af4dc9e`，可通过 `TraceProcessorConfig(bin_path=...)` 或 `habana_perf_tool --binary ...` 指定。torch trace 导出沿用已有 wrapper 兼容调用 `prof.profiler.export_chrome_trace(path)`。

**高层摘要限制：** TensorBoard 的 Habana 插件和 `habana_perf_tool` 对短 eager smoke 出现 `ProfilerStep#`/bridge forward track 缺失导致的 SQL/KeyError。随后 MOSS trace 已有真实 `ProfilerStep#0`，工具仍因 MMEEvents 数据不可用抛出 `DataNotAvailableError`。UI 插件 enabled 不代表 HPU 总览分析通过；不修改厂商包或切换 lazy 来掩盖。当前验收路径为原始设备 trace + Perfetto SQL；厂商高层自动摘要列为兼容性限制。

另用独立 `raw-profiler-config.json`（Gaudi2、multi-enq 1–4）及进程级 `HABANA_PROFILE=1`、`HABANA_PROF_CONFIG=<该文件>` 采集 4 次 BF16 matmul/relu，原生进程退出 0。得到约 98 KB 的 `raw/raw-smoke_89572.hltv`；`synprof_parser <文件> --json --gaudi2 --outdir <raw-parsed>` 退出 0，生成含 1024 个事件的 JSON。parser 有 `Unknown unit for STM #0` warning；没有将其写成完全无 warning。一次 `hl-prof-config --help` 的 `-short-help` 子进程在输出后未退出，已清理本轮遗留 help 进程；实际配置命令和采集/转换均成功。

## eager 校准

`probes/hpu_observability.py`：3 次暖机、5 组同步边界计时；copy 每组 20 次、GEMM 每组 10 次。copy 的两个 buffer 各 256 MiB，工作集超过 48 MB SRAM；有效字节按读+写计算，包含 eager dispatch 开销。

| 项目 | median 实测 |
| --- | ---: |
| copy，读+写合计 | 1949.35 GB/s |
| BF16 GEMM M=N=K=4096 | 260.40 TFLOP/s |
| BF16 GEMM M=1,N=K=4096 | 0.464 TFLOP/s |
| BF16 GEMM M=16,N=K=4096 | 7.72 TFLOP/s |
| BF16 lm_head 形状 M=1,N=248320,K=4096 | 2.144 TFLOP/s |

这些是具体 shape/实现下的有效值，不是模型 MFU/MBU。尚未分别校准 HBM 纯读/纯写、实际 ConvRot 路径或原生 FP8/INT8。原始数据 `results/performance-2026-11-08/calibration.json`，该目录被 gitignore；完整副本留在服务器同名目录。

## FP8 / INT8 能力与理论上限

按用户建议补测 `probes/hpu_low_precision.py`。测试在停止 8189 服务、HBM 约 768 MiB、空卡状态下进行；4096×4096×4096，3 次暖机、5 组同步计时，每组 20 次。结果保存在服务器 `results/performance-2026-11-08/low-precision/low-precision-isolated.json`，小 JSON 也备份到本机同名目录。

| 路径 | 实测 | 公开理论上限 | 达到理论值 | fallback | 结论 |
| --- | ---: | ---: | ---: | --- | --- |
| BF16 `torch.mm` | 275.76 TFLOP/s | 432 TFLOP/s | 63.8% | `{}` | 原生 HPU MME 基线 |
| FP8 E4M3 `torch.mm` | 525.88 TFLOP/s | 865 TFLOP/s | 60.8% | `{}` | 原生 HPU FP8 路径可测 |
| FP8 E5M2 `torch.mm` | 519.63 TFLOP/s | 865 TFLOP/s | 60.1% | `{}` | 原生 HPU FP8 路径可测 |
| INT8 `torch._int_mm` | 3.17 TFLOP/s | 未公开可套用的 INT8 MME 峰值 | 不计算 | `aten::_int_mm: 103` | CPU fallback/桥接路径，不是 HPU INT8 峰值 |

运行时支持 E4M3/E5M2 的 float8 storage/`torch.mm`；E4M3FNUZ/E5M2FNUZ allocation 失败。`torch.ops.hpu` 暴露 `fp8_gemm_v2` schema，包含 scale、transpose 和输出 dtype 参数；没有在缺少模型量化布局/scale contract 时手工调用它作为性能结果。Transformer Engine 模块可 import，但本轮没有把未经 MOSS 校准的 TE 层改造成产品路径。

理论值的来源是 Intel 的 Gaudi2 对照规格：BF16 MME 432 TFLOP/s、FP8 MME 865 TFLOP/s、HBM 2.46 TB/s、48 MB SRAM；Gaudi 文档确认 Gaudi2 支持 FP8 E4M3/E5M2。公开资料没有给出可直接套用的 Gaudi2 INT8 MME dense peak，因此不能以 BF16 或 FP8 倍数推算 INT8 峰值。[Gaudi FP8 inference](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html)；Intel 的 [Gaudi3 white paper](https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf) 里的 Gaudi2 comparison table 给出上述峰值。

按公开峰值计算的 compute roofline arithmetic intensity 约为 BF16 `432/2.46 = 175.6 FLOP/byte`、FP8 `865/2.46 = 351.6 FLOP/byte`。这是 roofline 分界，不是 MOSS 每层实际强度；实际 ConvRot INT8 权重的读写、scale 和反量化字节必须从模型路径/trace 单独计算。当前 FP8 GEMM 约 60%、BF16 约 64% 的峰值，说明该 eager shape benchmark 还有调度、实现和时钟余量，但不能直接推断 MOSS 可获得同等比例提升。

只读解析 safetensors header：DiT 有 5,082,308,608 个 BF16 存储元素及 16 个 F32；文本文件有 9,567,236,096 个 I8、85,864,432 个 BF16、2,240,624 个 F32 和量化 U8 元数据；VAE 有 84,046,371 个 F32 元素和一个 I64。这里是存储元素统计，量化 scale/元数据不等于可训练参数。lm_head 和 embedding 均为 `[248320,4096]`，单份 I8 payload 1,017,118,720 bytes。完整 dtype/shape 摘要为 `model-static.json`。

## 计划 review 与首轮基线

review 修正已进入 [原计划](MOSS_HPU_PERFORMANCE_PLAN_2026-11-06.md)：节点缓存、cold/warm 数量、两个 seed、正常计时与诊断同步分组、逻辑字节与 HBM 事务区分、原生后端计数、工具能力验收。

本轮冻结 `workflows/moss-user-textgen.json`，SHA256 `c283f15bcae32cceebd51c9a8863400f421f46354be32b3c52c4a16743be7bd6`；扩散 seed `649029532689846`，文本 seed `0`。这与 143.34 s 那次计划采集的 seed 不同，本轮建立新基线，不计算旧值相对提速。

调查 launcher 在 loopback 8190 启动，保持原运行参数；仅在内存中替换计算节点 fingerprint，保留 loader 缓存。正常计时无 profiler、无逐步同步；诊断请求另开同步阶段和逐步观测。每条事件同时写 wall_ns 和 mono_ns，hl-smi/pidstat/iostat 同机旁采样。baseline client 检查每次确实执行 TextGenerate（B0）、CLIPTextEncode、KSampler、VAEDecode，并保存 API、history、token ID/实际 caption、PNG 和像素统计。

已完成 B0 1 cold + 5 warm；B1 固定首轮 caption、1 transition + 5 warm；另有同步 B1 诊断。B1 transition 不是独立进程 cold，不计入 cold 结论。所有 13 个请求成功、fallback 累计计数始终 `{}`，每次所需计算节点执行断言通过；warm 未重新执行 loader。

| Case | 样本 | 请求内 median | min–max | p10 / p90 |
| --- | ---: | ---: | ---: | ---: |
| B0 process cold | 1 | 145.521 s | — | — |
| B0 warm | 5 | 137.282 s | 135.893–137.526 s | 136.178 / 137.451 s |
| B1 transition | 1 | 40.305 s | — | — |
| B1 warm | 5 | 40.183 s | 39.690–40.314 s | 39.873 / 40.278 s |
| B1 synchronized diagnostic | 1 | 40.313 s | — | — |

请求内时间从 PromptExecutor 入口到末尾 HPU 同步；客户端含提交/轮询约多 0.3 s，history/服务日志另有交叉验证。process cold 没有清文件系统页缓存；本轮未设置磁盘 recipe cache 环境变量。5 个 warm 样本的分位数仅作描述，不能当尾延迟置信估计。

B0 warm 的节点计时中位数（无显式逐节点同步）为 TextGenerate 96.966 s、正负文本编码合计 2.013 s、KSampler 37.960 s、VAE 0.132 s、PNG 0.117 s。B0 的 6 次生成均为 327 tokens，ID 留在事件文件；文本可读，MOSS 灯牌、白色大衣、剧院和湿地反光目视通过。文本节点约占 70.6% 的请求时间，是下一步详细归因的优先段，而非已经确定的硬件瓶颈标签。

B1 同步诊断：文本编码 2.034 s、KSampler 38.006 s、VAE 0.136 s、PNG 0.122 s；节点之和解释 99.963% 的请求时间。逐步同步间隔 median 1.5212 s、min/max 1.5055/1.5308 s；诊断请求比 B1 warm median 高 0.32%，仅作本轮扰动参考。latent `[1,128,64,64]` 与 VAE 输出 `[1,1024,1024,3]` finite 检查均通过，VAE 范围 0–1。所有 PNG 与首轮图的像素 MAE 为 0–0.1226/255，最大单像素差最高 11/255；不要求逐像素相同。

主机采样：前两次 warm 的 TextGenerate 平均 AIP 约 24.0%/25.9%、功耗 158.4/169.5 W；KSampler 平均 AIP 44.7%/46.9%、功耗 282.7/283.7 W。单段 10 s `sudo perf stat -p` 采集通过。这些不是 HBM 带宽或 MFU，不能据此断言算力/带宽受限。

切换文本 probe 时首次在旧调查服务完全释放设备之前初始化，出现 `Device acquire failed`；失败日志保留为 `moss-text-hpu-acquire-failed.log`。随后验证旧 PID 消失、hl-smi Compute Processes 为空、HBM 回到约 768 MiB，再串行重试。这是切换次序错误，不是恢复后的驱动故障。

P1 仍在进行中：尚欠完整 B2 模式/长度控制矩阵、B4 的尺寸/步数及跨进程 cold 控制、实际模型 bytes/FLOPs 的可检验估算和更完整设备归因；本轮不宣称 P1 全部完成或 P3 优化收益。

## 2026-11-08：P2/P3 首轮优化候选与停止

按“先预测、单变量、重复验证、低于噪音则停止”的规则，正式测试了两个调查候选，均未进入产品代码：

1. **一次性缓存 lm_head 的 BF16 解量化权重。** 探针在 2-token 暖机后，把 `[248320,4096]` 的 INT8 ConvRot lm_head 解量化为约 2.03 GiB BF16，并在同一进程后续生成中复用。MTP off、32 token 的一次对照从 16.963 s 降到 15.920 s，token ID 完全一致、fallback `{}`；但实际产品使用的 MTP auto 从 10.068 s 变为 10.622 s，方向相反。额外显存约 2.03 GiB，收益没有在目标 MTP 路径复现，候选拒绝。
2. **强制 embedding 不走 host row 路径。** 只把 `vbar_above_watermark` 在调查进程中固定为 false；MTP auto 8-token 两次无 profiler 对照约 4.052 s 与 3.929 s，差异约 3%，低于当前短 probe 噪音，且 profiler 总体仍约 3.9 s。没有作为产品配置或代码改变。

MTP auto 8-token HPU trace 的主要 host 侧线索仍是 `aten::_copy_from`，约 2.14 s/8-token profile；ConvRot dequant kernel 约 0.21 s，fallback 为空。当前证据更像频繁 host/device 控制与同步开销，需要新的 device/host 时间线关联才能安全改动；缓存权重或强行改变 embedding 驻留都没有稳定收益。因而本轮在此停止性能改动，保留探针和失败/回退证据，恢复正常服务 PID **109952**。后续只有在新 trace 能把 `_copy_from` 归因到具体调用并给出单变量预测时再继续。

## 2026-11-08：复核 CLIP INT8 dtype 与低 HPU 利用率

用户日志中的 `CLIP/text encoder ... dtype: torch.bfloat16` 与权重文件名 `qwen3.5_9b_int8_convrot.safetensors` 并不矛盾。该字段来自 `load_clip` 的计算 dtype；模型模块实际仍保留 `comfy_kitchen.tensor.base.QuantizedTensor`（由探针打印的 `weight` 类型确认）。当前产品代码在 `comfy/sd1_clip.py` 对带量化元数据的文本编码器调用 `mixed_precision_ops(..., full_precision_mm=True)`，`MixedPrecisionOps.Linear.forward_comfy_cast_weights` 因此对每个线性层把量化权重解量化为 BF16 后执行 `F.linear`。所以它是“INT8 存储、BF16 计算”的安全路径，不是误把模型加载成普通 BF16 文件。

为验证是否可直接打开量化 matmul，在独立探针中仅把该开关强制改为 `False`，没有改产品仓库、模型或服务配置。结果：首个 2-token 请求从基准约 1.47 s 变为 12.54 s；调用栈明确进入 `comfy_kitchen...fast_int8_mm -> torch._int_mm`，该版本的 `torch._int_mm` 在 HPU 上发生 CPU fallback，随后生成明显变慢，HPU 利用率采样为 0%。因此当前 Habana 1.24.2/torch 组合不能把这个开关作为优化。

低负载的直接证据来自同一 eager、MTP auto、固定短 prompt 的 32-token 旁采样：10.10 s，`hl-smi utilization.aip` 26 个样本平均 **12.65%**、最高 **47%**。这符合 Qwen3.5 自回归扩写的执行形态：每次只生成一个很小的 token/GEMV 工作单元，CPU 负责采样/树路径控制并把少量 token 结果通过 `.tolist()` 拉回；每一步还会触发量化权重的解量化与临时张量搬运。已有 HPU trace 中 `aten::_copy_from` 在 8-token profile 自身约 2.14 s，ConvRot dequantize 约 0.21 s，fallback 计数为空。DiT 阶段的 25 步约 33 s 也与该阶段不同，当前 135 s 级端到端时间主要由 TextGenerate 扩写的约 97 s 决定。

本轮结论：`dtype=bfloat16` 是当前实现的明确计算策略；直接切换 native INT8 会更慢且落 CPU；低 HPU 负载主要是小批量自回归、host/device 控制同步和重复解量化的组合，而不是 HPU 没有被加载。下一步若继续，必须先在 device/host 时间线上把 `_copy_from` 分成采样结果回传、权重解量化和缓存/预取三类，再分别评估批量 speculative verify、常驻权重或原生 INT8 kernel 版本升级；在此之前不提交猜测性的产品改动。

## 2026-11-08：复核外部 827 TFLOP/s FP8 数据

用户提供的图片与公开 benchmark 记录可以对应上：Gaudi2 FP8 E4M3 **827 TFLOP/s**，理论 865，95.6%；矩阵形状是 **M×N×K = 6144×11264×5120**，软件版本 `2.6.0+hpu_1.21.2-76`，并注明 `PT_HPU_LAZY_MODE=1`。该数字是 MAMF（短时 boost burst），公开表的 MSMF 栏为空，不能当持续吞吐。

我们的首轮低精度探针是当前 `2.13.0a0+git2469d18` / Habana `1.24.2-481`、`PT_HPU_LAZY_MODE=0`、4096³、通用 `torch.mm(FP8, FP8)`，输出 dtype 也是 FP8；因此不能与该 827 数字直接比较。按对方矩阵形状在当前 eager 栈重测 `torch.mm` 得 **564.11 TFLOP/s**（约 65.2% 理论），说明矩阵形状有影响，但仍没有复现外部 headline。

外部 benchmark 的 FP8 代码路径是 `torch._scaled_mm(A, B, scale, scale, out=C)`：输入来自随机数据转换为 FP8，输出 `C` 是 BF16；每轮先写 256 MiB cache buffer、重写输出矩阵，然后用 HPU device event 计时。它还会扫描候选形状、区分 MAMF/MSMF，并采集功耗/时钟。我们的首轮脚本用 host wall time、每次分配输出，没有 cache flush/output rewrite、没有 device event 或 boost clock 验证，测量目标不同。

当前 Habana 1.24.2 栈对 `torch._scaled_mm` 的 1024² 和外部 6144×11264×5120 形状均出现 `synStatus 26 / Current graph is nullptr`；FP8 storage allocation 与普通 `torch.mm` 可用，但这条 scaled FP8 API 未通过。因此不能用当前结果断言硬件只有 520–560 TFLOP/s，也不能盲目把外部 827 当作 MOSS 可达到的吞吐。lazy 模式下若没有正确 `mark_step`、输出消费和 device event，host wall time 会得到虚假的 PFLOP/s；临时复现脚本已拒绝 lazy 模式，避免误用。

结论：此前结果的主要问题是 benchmark 定义不一致，包含矩阵形状、PyTorch/Habana 版本、eager/lazy、`torch.mm` 与 `_scaled_mm` kernel、输出 dtype、cache/output 处理、MAMF 与持续吞吐、host 与 device 计时等差异。MOSS 当前性能基线仍应保持 eager 与实际模型路径；硬件峰值基准另建兼容的 Gaudi MAMF/MSMF 复现。外部方法说明见 [benchmark 结果表](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/README.md#maximum-achievable-and-sustainable-matmul-flops-comparison-table)、[计时定义](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/README.md#how-mamf-and-msmf-are-defined) 和 [FP8 实现](https://github.com/stas00/ml-engineering/blob/master/compute/accelerator/benchmarks/mamf-finder.py)。

## 真实 MOSS HPU trace 与最终状态

`moss_performance.py` 新增可选真实工作流输入、结果目录、CPU+HPU activity、step 标记和同参数无 profiler 对照。独立 probe 使用 16 个 CPU threads，与服务默认线程配置分开，不能把它的 8-token 吞吐直接代替完整 B0。

- 真实用户 system/user prompt，文本 seed=0、MTP auto、sampling on；2-token 暖机后，8-token 无 profiler 对照 4.1018 s，profile 内生成 5.2150 s（含 prefill），约增加 27.1%；8 个 token ID 相同，fallback `{}`。
- `moss-p1-user-hpu-trace.json` 约 930 MiB，保留在服务器，不复制到 Git。最终停止/整理/导出花费数分钟，不能只报告生成段 5.215 s 而隐去 profiler 全流程开销。
- Perfetto SQL 成功读出 414,226 个 cpu_op 事件、1,030,554 个 kernel 类事件、`ProfilerStep#0` 5.216 s；其中有 MME/TPC 多 engine/lane 和嵌套/重叠事件，**不等于一百万次 kernel launch**。各事件 duration 不能直接相加当 wall time。
- kernel 标签包含 `memcpy_f32`、`mult_fwd_f32`、`cast_i8_to_f32`、`cast_f32_to_bf16`、`MMEH_WB1_MON_TS_BIT1` 等。这是后续定位 ConvRot 解码、prefill/GDN 和 host gap 的线索，不能只凭名称认定主机 PCIe copy 或已确定瓶颈。
- 后续采集应拆分 prefill 和稳态 decode，缩短范围并默认关闭 shape 记录；先用本次 trace 建立事件去重/区间并集和关联方法，再扩大采集。

本轮结束已停止调查服务和 hl-smi/pidstat/iostat 旁采样，正常 `start-hpu.sh` 恢复日常服务：PID **90386**，http://192.168.10.21:8189 可用，队列为空，`/system_stats` 为 `hpu:0 GAUDI2`，部署版本仍为 `0690d18f`。原始 ComfyUI、基础 torch、模型与推理代码均未修改。

产物完整保存在服务器 `/home/sonny/gaudi2/comfyui-hpu/results/performance-2026-11-08/`；除大型 trace 外的 API/history、PNG、事件、摘要、工具日志已备份到本机同名 `results/`。本仓库只提交调查脚本与文档。优先下一步为 B2 的同 prompt 固定 token、MTP off/auto 与 sampling/greedy 控制，以及 B4；P2/P3 依赖这些证据继续推进。

## 参考

- [Gaudi PyTorch profiling](https://docs.habana.ai/en/latest/Profiling/Profiling_with_PyTorch.html)：厂商高层插件、CPU/HPU activity、memory profile。
- [Gaudi profiling workflow](https://docs.habana.ai/en/latest/Profiling/Profiling_Workflow.html)：分层定位与低层硬件 profiler。
- [Perfetto Python API](https://perfetto.dev/docs/analysis/trace-processor-python)：离线 SQL 分析及自定义原生程序路径。
