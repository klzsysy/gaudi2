# MOSS-Image HPU 性能优化计划

更新时间：2026-11-08（Asia/Shanghai）

本文是 MOSS-Image 在单卡 Gaudi2 HPU 上的性能调查和优化计划。2026-11-08 用户授权服务器恢复后继续推进：先补齐可观测性、review 本计划，通过后开始 P1；无需重复请求停止空闲独立服务的许可。所有后续结论都应回填本文或 `PROGRESS.md`；产品代码、测试和过程记录继续分属各自仓库。

## 2026-11-08 执行前 review

方向通过，正式采样前修正以下测量问题：

1. **节点缓存会跳过真实计算。** 相同输入重复 `/prompt` 不能直接算 warm run。`--cache-none` 同时丢弃 loader 结果，可能把每次加载混入 warm。首轮使用调查仓库的独立 launcher，仅令 TextGenerate、CLIPTextEncode、KSampler、VAEDecode 的输入 fingerprint 随请求变化，保留原 loader 结果；不修改产品文件。记录每次真实执行的节点，并用加载事件验证模型常驻。`--cache-none` 留作全图重算控制组，默认缓存服务请求另列。
2. **冷启动与 warm 样本数分开。** 每组至少 1 次进程冷启动 + 5 次真实 warm 计算；文件系统页缓存和磁盘 recipe cache 状态单独记录。5 个样本的 p10/p90 仅作描述，不能当高置信尾延迟估计。跨进程重复 cold 留作后续控制组。
3. **冻结两个 seed。** KSampler seed 与 TextGenerate 的 `sampling_mode.seed` 分开记录，不能只固定扩散 seed。当前保存的 `moss-user-textgen.json` 与旧性能重放图扩散 seed 不同；P1 以用户图为冻结来源，同时保存 JSON SHA256。随机生成的 token 数/内容可能变化，必须同时给 token/s 和实际 token ID，不能只比较总秒数。
4. **分离正常计时与诊断计时。** 首先使用无 profiler 的 wall time；同步分段、逐步同步、shape/memory profile、硬件 trace 都标为诊断运行，并用同配置无采集对照估计扰动。同步段总和覆盖率 ≥95% 只评价该诊断请求；其时间不能替换正常 B0 延迟。
5. **字节和 FLOPs 不冒充硬件计数。** 静态逻辑流量与实际 HBM 事务分别报告；`utilization.memory` 不是 bytes/s，不允许积分推算真实字节。没有经过校准的硬件计数器时，MBU/MFU 只报明确假设下的估算/边界，不能作为 P1 阻塞条件。CPU copy 的等待时间不能等同 PCIe 搬运时间。
6. **GEMM 与带宽校准必须可执行。** 产品默认设备策略仍不把 INT8/FP8 声明为通用 compute；独立探针已验证本机原生 FP8 E4M3/E5M2 `torch.mm`，并验证 INT8 `torch._int_mm` 会 fallback。FP8/INT8 原生微基准仅做能力探测，不改 torch/驱动。copy 计读+写字节，工作集大于 48 MB SRAM；校准范围和实现决定可比性。Gaudi2 公开 BF16/FP8 理论值分别为 432/865 TFLOP/s；没有公开 INT8 MME 峰值，不做倍数推导。
7. **使用实际后端和计数来源。** TextGenerate 是 ComfyUI 原生 Qwen3.5，不是 vLLM。MTP 接受率若没有直接计数，记为未测；进度条递增数不能冒充草稿/验证次数。阶段事件用 wall-clock + monotonic 成对时间戳与 hl-smi CSV 对齐。
8. **工具可调用不等于观测能力已通过。** TensorBoard/HPU 插件、Perfetto Python 包已在 Habana 基础环境；不重复 pip 安装。补装 sysstat 和当前内核 linux-tools；验证设备 activity、trace 导出、解析和查看。`synprof_parser` 需要进程级 `LD_LIBRARY_PATH=/opt/habanalabs/habana_plugins:/usr/lib/habanalabs`。perf 使用单次 sudo，保留系统 `perf_event_paranoid=4`。hl-smi 本机 `power.limit=550000` 按 mW 归一化为 550 W，不能与 `power.draw` 的 W 直接比较。

首轮冻结文件为 `workflows/moss-user-textgen.json`，SHA256 `c283f15bcae32cceebd51c9a8863400f421f46354be32b3c52c4a16743be7bd6`，实际 KSampler seed=`649029532689846`、TextGenerate seed=`0`。它与 11-06 计划只读采集到的 seed=`438829726369282` 不同，也与旧性能重放 seed=`133401810542576` 不同。因此本轮是新建的可复现 B0，不能据此计算相对用户 143.34 s 的严格提速率。

执行证据与工具复现见 [本轮记录](MOSS_HPU_PERFORMANCE_RUN_2026-11-08.md)。先完成工具 smoke、低精度能力探针与首轮计算基线，再按 B2/B4 控制矩阵推进；P2/P3 必须等待相应证据。

## 1. 目标与边界

目标工作流是用户提供的 ComfyUI MOSS-Image 工作流：

`http://192.168.10.21:8189/#e8f30439-267f-45de-b260-db0d114b2c85`

用户给出的当前基线是 1024×1024、25 步、端到端 **143.34 s**。日志中可见采样为 25/25、约 33 s、1.33 s/步；模型和 VAE 的加载信息如下：

```text
MOSSImage:      78134.40 MB usable, 9693.73 MB loaded
AutoencoderKL:  73054.90 MB usable, 160.31 MB loaded
Prompt executed in 143.34 seconds
```

这个数字先作为“用户基线”，不能直接与已有的 134.586 s 重放结果混为同一基准：后者使用了不同请求历史、服务生命周期和缓存状态。第一阶段的验收目标是建立可重复的、分阶段计时的严格基线，然后再谈提升比例。

本计划先覆盖：

- 单卡 Gaudi2、batch=1、文生图、1024×1024；
- MOSS-Image SFT BF16 DiT、Qwen3.5 9B INT8 ConvRot 文本模型、已有 Flux2 VAE；
- eager 模式下的正确性和性能观测；
- prompt enhancer、文本自回归、DiT 采样、VAE 解码四段的独立归因。

暂不把 MOSS Base、编辑/控制 LoRA、SLG、并发服务、跨节点比较和换模型量化版本纳入第一轮结论。

## 2. 已收集的环境事实

采集时间为 2026-11-06 16:31（Asia/Shanghai），所有操作为只读，未停止现有用户服务。

| 项目 | 当前事实 |
| --- | --- |
| 主机 | `llm-machine`，Ubuntu 内核 `5.15.0-202-generic` |
| HPU | 1× Habana Gaudi2 HL-225，PCI `1da3:1020`，96 GiB HBM；`hl-smi` 显示 98304 MiB |
| 驱动 | Habana `1.24.2-481`，驱动/NIC `1.24.2-52a555d`，Operational |
| Python | 3.10.12，独立 `/home/sonny/ComfyUI-Gaudi/.venv`，继承 Habana 基础 site-packages |
| PyTorch | `2.13.0a0+git2469d18` |
| ComfyUI | `0.44.0`，frontend `1.58.5`，workflow templates `0.11.102` |
| 部署版本 | ComfyUI-Gaudi `0690d18f4b0157a14387a12afcf833eec924bb9e` |
| 运行模式 | `PT_HPU_LAZY_MODE=0`（eager）；`GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so` |
| 服务参数 | `--disable-dynamic-vram --disable-pinned-memory --disable-async-offload --disable-xformers --use-pytorch-cross-attention --enable-manager` |
| 服务状态 | PID 5202，监听 `192.168.10.21:8189`，队列为空 |
| 当前显存 | 采集时 `hl-smi` 使用 22480 MiB / 98304 MiB；`/system_stats` 报 HPU 总量 101601996800 bytes、空闲 78835170816 bytes |
| 当前功耗/温度 | 约 83 W；`hl-smi` 当前设备 power limit/max 为 550 W，官方 HL-225/Gaudi2 TDP 资料为 600 W；板上温度约 43–51°C；采集时 AIP 利用率 0% |

当前服务已经独占单卡。独立 Python 进程可以看到 `torch.hpu.is_available()` 和设备数量，但尝试初始化设备会得到 `Device acquire failed`；后续微基准和 profiler 必须在队列为空、服务按流程停止后串行执行。不能把这个失败误判为驱动故障。

模型文件位于共享目录，当前目标组合为：

| 组件 | 文件 | 大小 | 角色 |
| --- | --- | ---: | --- |
| DiT | `moss_image_sft_bf16.safetensors` | 10,164,655,808 bytes | 25 步扩散主模型 |
| 文本模型 | `qwen3.5_9b_int8_convrot.safetensors` | 9,748,121,532 bytes | CLIP/TextGenerate；包含 MTP 路径 |
| VAE | `flux2-vae.safetensors` | 336,213,556 bytes | latent → RGB 解码 |

## 3. 严格基线的定义

工作流 URL 的 UUID 是前端工作流 ID，不是一次请求 ID。每次正式测试都必须保存提交用的 API JSON、ComfyUI history JSON、服务日志和采集配置，不能只抄 UI 上显示的总秒数。

### B0：用户目标端到端基线

这是与 143.34 s 对齐的基线：

- 1024×1024，batch=1，25 steps，`er_sde/simple`，`cfg=2.5`，`denoise=1.0`；
- `moss_image_sft_bf16.safetensors`；
- `qwen3.5_9b_int8_convrot.safetensors`，`type=moss_image`；
- `flux2-vae.safetensors`；
- prompt enhancer 开启：`mtp=auto`、sampling on、temperature 0.7、top_k 64、top_p 0.95、min_p 0.05、`max_length=4096`、thinking off；
- SLG 关闭；固定原始 prompt、system prompt、negative prompt、seed（当前采集到的工作流 seed 为 `438829726369282`，若与用户基线不同必须以用户记录为准）。

B0 至少 1 次进程 cold + 5 次真实 warm 计算，节点结果缓存控制见执行前 review；记录 median、p10/p90、min/max。首轮不改模型/采样参数，图片必须有限、可读且语义符合提示。

### B1：无扩写的图像路径基线

将 prompt enhancer 关闭，直接使用同一个最终英文 caption，保持 DiT/VAE、seed、尺寸、步数完全相同。B1 用来剥离 TextGenerate 的成本，不能拿来宣称 B0 的端到端提升。

### B2：文本生成单独基线

只执行同一 system prompt、user prompt、采样参数和文本模型，分别测：

- `mtp=off`、greedy；
- `mtp=off`、sampling on；
- `mtp=auto`、sampling on；
- 固定输出 token 数的短序列和实际自然结束序列。

记录输入 token、输出 token、总耗时、token/s、每步接受 token 数、MTP 草稿/验证次数和 fallback 计数。B2 不包含图像采样时间。

### B3：DiT/VAE 分段基线

使用固定最终 caption，绕过 TextGenerate，分别记录：文本编码、25 步 KSampler、VAE 解码、保存的阶段时间。KSampler 需要按步骤记录时间序列，至少保留首步、稳态步和末步；不能把 HPU 异步提交时间当作完成时间，必要时在测量边界显式同步。

### B4：尺寸、步数和缓存控制组

在 B1 的固定 prompt/seed 下建立最小矩阵：

| Case | 尺寸 | 步数 | 目的 |
| --- | ---: | ---: | --- |
| S512-25 | 512² | 25 | 分离固定开销与空间规模 |
| S1024-25 | 1024² | 25 | 用户目标图像基线 |
| S1024-8 | 1024² | 8 | 估算每步固定开销，后续仅作性能诊断 |
| S1024-15 | 1024² | 15 | 步数线性度和调度开销 |

每个 case 都要有 cold（新服务/首次编译）和 warm（同一服务重复）标签。冷启动时间、编译缓存、模型加载不允许混入 warm 的单步性能。

## 4. 可观测性契约

指标分成不可变/可校准上限、模型静态属性和运行时观测三类。HPU 的总利用率只能作为旁证，不能直接当作 MBU 或 MFU。

### 4.1 硬件上限与校准

在停止服务、确认 HPU 空闲后做一次独立微基准；结果写入版本化 JSON，后续优化复用，不在每次模型测试中重复。

- HBM 读、写、copy 带宽：用已知字节数校准实际有效带宽。
- BF16、FP8、INT8 GEMM：使用与 MOSS/Qwen 实际矩阵形状相近的 M/N/K，另保留大矩阵峰值和小矩阵启动下限。
- 单 kernel/单图启动下限：测出小算子在本机 eager 模式下的最短时间。
- 功耗、温度、AIP 利用率、时钟：`hl-smi --query-aip=timestamp,utilization.aip,utilization.memory,memory.used,power.draw,power.limit,temperature.aip,clocks.current.soc --format=csv,noheader,nounits --loop=1`；本机实测语法，1 s 状态时间序列，power.limit 原始 mW 需归一化。
- 记录 PCIe 链路、驱动、固件和内核，防止换机或重启后把硬件变化误当优化收益。

计划使用的 roofline 关系为：

```text
T_step >= max(bytes_step / B_peak, flops_step / F_peak)
MBU = bytes_step / T_step / B_peak
MFU = flops_step / T_step / F_peak
images/s = 1 / T_end_to_end
```

实测微基准值优先于规格宣传值；规格值、实测值和测量工具的定义必须同时记录。若 HPU profiler 的设备 activity 能力与当前 Habana wrapper 不兼容，先记录能力检查结果，再选择能导出设备时间线的接口，不能把 CPU-only trace 当成完整 HPU kernel 证据。

### 4.2 模型和图静态属性

- 每个 checkpoint 的文件大小、dtype、量化布局、参数量、层数和词表大小。
- DiT 每步逻辑 FLOPs、权重读写字节、latent token 数；按 512²/1024² 分开计算。
- Qwen3.5 每个自回归 token 的权重字节、lm_head 形状 `[4096, 248320]`、MTP 草稿和验证的计算/字节量。
- VAE 的输入输出 shape、dtype、每次解码的估算 FLOPs/字节。
- 这些属性由模型和硬件组合决定；代码版本、kernel 名称和 profiler 输出属于观测结果，不能作为上限定义。

### 4.3 运行时观测

每个 case 统一采集下面的时间序列和事件：

| 层次 | 指标 | 主要路径 | 交叉验证 |
| --- | --- | --- | --- |
| 请求 | queue wait、load、text、DiT、VAE、save、端到端 | history、服务日志、API 客户端单调时钟 | 日志时间戳与客户端时间 |
| TextGenerate | 输入/输出 token、token/s、MTP 接受数、草稿长度、采样参数 | ComfyUI/Qwen3.5 原生生成计数与日志 | 固定 token 数的独立 probe |
| HPU 状态 | AIP/HBM 利用率、显存、功耗、温度、时钟 | `hl-smi` CSV 1 s 采样 | `/system_stats`、阶段边界显存 |
| 算子 | fallback 数、CPU 等待、kernel 序列、设备耗时 | Habana fallback 计数、torch/Habana profiler | CPU profiler、阶段 wall time |
| 模型 | 每步逻辑 bytes/FLOPs、估算 MBU/MFU、步耗时 | 静态解析 + profiler | 经校准的硬件计数器；缺失时明确标注估算 |
| 主机 | CPU 使用率、RAM、I/O、进程线程数 | `/proc`、`pidstat`/系统工具 | 服务日志和客户端时间 |

关键时间用至少两条独立路径交叉验证：请求 history 与客户端时钟、同步阶段 wall time 与设备 trace；fallback 用计数器与 CPU profile。DiT 步耗时与自回归 ITL 分别统计。实际每步字节仅在硬件计数器含义、范围和校准明确后报告；HBM 活跃度不作字节积分。

### 4.4 Gaudi2 架构与 MOSS 的映射

本机是 HL-225 Gaudi2。官方架构资料把 Gaudi2 分为计算、内存和网络三个子系统：

- **计算**：一个面向矩阵乘的 MME/GEMM 引擎，以及 24 个第四代 TPC。MME 适合线性层、卷积和 batched GEMM；TPC 是可编程 VLIW SIMD 核，适合 attention 之外的逐元素、归一化、激活、索引、量化/反量化和胶水操作。TPC 的 SIMD 宽度为 2048 bit。
- **内存**：96 GB HBM2e，官方峰值带宽 2.45 TB/s；另有 48 MB 片上 SRAM。MME、TPC、DMA 和 RDMA 可以并行访问内存。性能分析时必须区分 HBM 带宽上限、实际可达到的带宽和 kernel 的有效字节数。
- **数据类型**：Gaudi2 官方支持 FP32、TF32、FP16、BF16、FP8 E4M3/E5M2；MME 内部使用 FP32 累加。INT8/ConvRot 在当前 Qwen3.5 路径中还要用实际 trace 确认其落到 MME、TPC 或解量化后路径，不能凭文件名推断硬件执行单元。
- **主机与互联**：主机接口为 PCIe Gen4 x16；芯片集成 24×100 GbE RoCE v2 NIC。当前单卡文生图主要关注 PCIe/CPU↔HPU 边界和单卡执行，网络互联暂不列入首轮瓶颈。
- **当前功耗边界**：官方资料的 Gaudi2 TDP 是 600 W；本机 `hl-smi` 对 HL-225 报告的 54 V 电源 power max/limit 是 550 W，分析是否触碰功耗上限时必须用本机实时读数和 throttle reason，而不是直接套 600 W。

对 MOSS 的第一版硬件映射假设如下，必须由 trace 验证：

| MOSS 区段 | 主要硬件假设 | 首要观测问题 |
| --- | --- | --- |
| Qwen3.5 线性层、lm_head | MME/GEMM；INT8/ConvRot 可能包含 TPC 解码或额外转换 | 是否每 token 重复解量化/搬运约 1 GB lm_head；是否有 host gap/fallback |
| Qwen3.5 attention、GDN、norm、采样 | TPC、DMA 和小矩阵 MME 混合 | 小 kernel 启动、标量回读、shape 特化和 TPC 利用率是否主导 |
| MOSS DiT 的大矩阵乘/卷积 | MME，权重主要从 HBM 读取 | 25 步每步的有效 HBM 字节、矩阵形状、MME 利用率和带宽上限差距 |
| DiT attention、norm、激活、量化 | TPC/融合 kernel | kernel 数、TPC busy、空闲间隙和融合收益 |
| Flux2 VAE | MME + TPC，受 shape 和中间激活影响 | 是否已经接近 HBM/算力下限，是否只是固定 launch 开销 |

上述是可检验的工作假设，不是已完成的硬件归因。官方参考：[Gaudi 架构](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html)、[TPC User Guide](https://docs.habana.ai/en/latest/TPC/TPC_User_Guide/index.html)。

### 4.5 可观测性工具盘点

11-06 只读盘点已于 11-08 补齐安装与实机 smoke，当前结论如下：

| 工具/接口 | 当前状态 | 能回答的问题 | 限制和后续动作 |
| --- | --- | --- | --- |
| `hl-smi` | **已具备，查询成功** | HBM 使用量、AIP/HBM 利用率、功耗上限/读数、温度、时钟、ECC、固件 | 不能给每个 kernel 的 bytes/FLOPs；用 1 s CSV 旁采样，作为状态旁证 |
| `torch.hpu.memory_stats()` | **已具备** | 阶段边界显存、分配/缓存变化 | 不是 HBM 带宽计数器；需与 `hl-smi` 对齐 |
| `torch.profiler` HPU activity | **11-08 smoke 实采通过**，CPU + HPU，含 gemm/relu device kernel | host API、HPU activity、shape、memory profile、trace | MOSS 代表图与采集开销继续验证；导出需用当前 Habana wrapper 的内层接口 |
| Habana profiler：`hl-prof-config` / `synprof_configuration` | **已安装** | Gaudi host API、MME/TPC 硬件时间线、synLaunch、recipe、memory reuse | 需要停止用户服务、配置 `HABANA_PROFILE`；trace 应保存到服务器并控制采样范围 |
| `synprof_parser` | **11-08 实际 .hltv → JSON 通过** | 离线解析 `.hltv` 为 JSON/CSV 等 | 修正进程级动态库路径；1024 事件 smoke 有 STM warning，详见本轮记录 |
| `habana_perf_tool` | **已安装，高层摘要存在兼容性限制** | 解析 profiler JSON，生成性能摘要/建议 | smoke 缺 step；真实 MOSS 带 step 后仍因 MMEEvents 不可用失败。当前以 Perfetto SQL 分析原始 trace |
| `habana_frameworks.torch.utils.debug` | **已有使用证据** | `_get_fallback_op_count()` 检查 CPU fallback | 这是调试接口，不是完整设备时间线；每个 case 都需记录调用结果 |
| `PT_HPU_METRICS_FILE` | **运行库代码存在，尚未启用** | 编译/运行指标、重复编译等运行时线索 | 需单独 probe 验证格式、开销和与 profiler 的关系；不在用户服务上直接打开 |
| `HABANA_LOGS` / `LOG_LEVEL_*` | **可用但当前未开启诊断级别** | bridge、Synapse、graph compiler、profiler 的错误和警告 | 只在最小复现和短 profile 中开启，避免日志扰动性能 |
| ComfyUI `/system_stats`、`/queue`、`/history`、`logs/hpu.log` | **已具备** | 请求、队列、模型加载、端到端和工作流复现 | 目前缺少可靠的 Text/DiT/VAE 阶段事件；B1–B3 需由客户端和受控 probe 补齐 |
| `/proc`、`free`、`vmstat`、`htop`、`pidstat`、`iostat`、`sar`、`perf` | **已具备；11-08 补装 sysstat/linux-tools** | CPU、RAM、线程、I/O、CPU PMU | perf 使用单次 sudo；不改全局权限，sysstat 定时采集已关闭 |
| `nsys` | **存在于 `/usr/local/cuda-13.0/bin/nsys`，默认 PATH 未包含**；不列为主工具 | NVIDIA/CUDA 生态的 host trace | 当前没有 Gaudi 适配证据；不能用它替代 Habana profiler |
| TensorBoard `torch-tb-profiler` | **基础环境已有；11-08 临时 UI/插件启动通过** | 官方高层 HPU profile 总览、MME/TPC kernel view、memory view | 短 eager trace 的高层摘要失败，详见本轮记录；未替换基础 torch 或厂商插件 |
| Perfetto Python + native trace_processor | **11-08 完整验证通过** | 原始 device trace 的离线 SQL 分析 | 首次 native 下载直连超时，使用已有代理完成；fixture 与真实 HPU trace SQL 通过 |

因此，工具链不是“完全缺失”：低开销状态监控、HPU profiler、离线 parser、HPU activity profiler 和 fallback 计数器都已在环境中或有运行证据。当前真正的缺口是：

1. 已采集真实 MOSS 8-token CPU+HPU trace，尚需拆分 prefill/decode 并扩展 DiT/VAE 的代表设备时间线；
2. TensorBoard/Perf Tool 自动摘要在真实 MOSS trace 上仍缺可识别的 MME 数据；Perfetto 原始 trace 分析已通过；
3. 已建立 `hl-smi` 与 ComfyUI 阶段事件的同机 wall/monotonic 对齐，待用完整请求和 profiler 范围验证；
4. 当前服务参数禁用了 dynamic VRAM、pinned memory 和 async offload，适合先做可重复基线，但不能直接代表其它执行配置的硬件上限。

官方 profiling 入口：[PyTorch profiling](https://docs.habana.ai/en/latest/Profiling/Profiling_with_PyTorch.html)、[Gaudi profiling workflow](https://docs.habana.ai/en/latest/Profiling/Profiling_Workflow.html)、[Gaudi profiler 入门](https://docs.habana.ai/en/latest/Profiling/Intel_Gaudi_Profiling/Getting_Started_with_Profiler.html)。

## 5. 归因顺序与优化候选

第一轮不改实现，只做观测和预测；第二轮每次只改变一个因素，并先写出预测。

1. **先分段**：确认 B0 的 143.34 s 中 TextGenerate、DiT、VAE、加载和保存各占多少。
2. **先排 fallback 和同步**：任何 CPU fallback、标量 `.item()`、设备到主机 copy、空闲间隙先单独归因；已有输出头 fallback 修复要以当前目标工作流重新计数确认。
3. **判断瓶颈**：对 TextGenerate 看每 token 权重字节/接受 token 与 host gap；对 DiT 看每步字节、FLOPs、MBU/MFU；对 VAE 看 shape、kernel 启动和带宽/算力占用。
4. **参数级实验**：prompt enhancer on/off、MTP off/auto、sampling/greedy、草稿长度（若节点暴露）、步数和尺寸；这些是可回退的诊断实验，不能直接当产品优化。
5. **实现级实验**：只在证据表明对应类别有余量时评估 HPU Graph/compile、算子融合、attention/线性层实现、量化计算路径和权重缓存。每项先做小 probe，再做完整图。
6. **容量和并发**：单图 warm 性能稳定后，再研究模型常驻、卸载、batch 和多请求；不把单卡 batch=1 的结果外推到服务吞吐。

每个候选必须写成：

```text
假设：优化会改变哪些静态/运行时属性？
预测：bytes、FLOPs、token 数、MBU/MFU、T_step、T_total 应如何变化？
实验：固定哪些变量，改变哪一个变量，跑多少次？
结果：实际改变了什么，是否超过噪音，质量是否通过？
结论：采用、回退或继续定位；保留证据路径。
```

## 6. 噪音、正确性和验收门槛

- 同一启动内先测稳定性；至少 5 次 warm run，以 median 和分位数比较。跨启动、跨节点、驱动或代码版本必须重新跑 B0/B1 对照。
- 记录首次编译、模型加载、显存占用和 cache 状态。不能用一次 cold run 与一次 warm run 计算提升率。
- 图像必须 finite，无 NaN/Inf、无黑图/异常纹理；固定 seed 和固定 prompt 下保存输出、尺寸、像素范围、均值/标准差，并进行目视检查。HPU 数值允许有明确记录的微小差异，不默认要求逐像素一致。
- 文本扩写必须可读、结束条件正常、输出 token 数可解释；固定 token probe 的 token ID 和 fallback 计数要留档。
- 任何声称“HPU 加速”的路径，目标 case 的 fallback 计数必须为零，或明确列出仍存在且已量化的 CPU 路径。
- 小于同一启动噪音的收益忽略。第一轮用 5 次测出噪音；后续只有延迟下降超过噪音并在重复实验中方向一致，才进入下一阶段。
- 优化报告必须同时给出：端到端秒数、阶段秒数、steps/s 或 token/s、HBM/算力利用率、功耗、显存、质量检查和回滚信息。

## 7. 阶段出口

### P0：资料和环境冻结

已完成：记录硬件、驱动、代码版本、模型文件、服务参数、用户 143.34 s 基线和当前工作流参数。本阶段没有启动新测试，也没有改服务器原始 `/home/sonny/ComfyUI`。

### P1：严格观测基线

当前状态：2026-11-08 服务器恢复，用户已授权按工具补齐 → review → 正式开始的顺序继续。先检查 SSH、内核/DKMS、`hl-smi` Operational、共享模型、服务版本和端口；队列为空后串行停止独立服务，验证 HPU 初始化和工具 smoke。完成 B0–B4、硬件微基准和观测 JSON；没有性能优化代码。每轮测试结束恢复正常用户服务。

11-08 已完成工具补齐、BF16 GEMM/copy 校准、B0 cold+5 warm、B1 transition+5 warm、同步 B1 诊断与真实 MOSS 短设备 trace。B0 warm median=137.282 s，B1=40.183 s，B1 同步段覆盖率 99.963%；原始证据与限制见本轮记录。**P1 尚未全部完成**：B2/B4 完整控制矩阵及更完整静态属性/设备归因继续待做；用户服务已恢复。

出口：能把 B0/B1 的总时间拆成 TextGenerate、DiT 每步、VAE、加载/保存，并能解释至少 95% 的端到端时间；关键指标有交叉验证。

### P2：瓶颈归因

按 roofline 和事件时间线给每段打标签：带宽受限、算力受限、kernel 启动/host gap、CPU fallback、模型加载或其它 I/O。只选择一个最高收益且风险最低的候选进入实验。

### P3：单变量优化

一次只改一个因素，保存 patch、配置、预测、结果和回退理由。每个优化项先通过 correctness，再通过重复性能和噪音门槛。

### P4：经验沉淀

把 MOSS 的每次实验压缩成“模型属性 → 瓶颈 → 指标变化 → 预测/实测 → 结论”的记录，形成可迁移到 Krea2、Qwen-Image 和其它模型的 HPU 性能数据集。

## 8. 当前禁止事项

- 不修改服务器原始 `/home/sonny/ComfyUI`。
- 不在用户服务持卡时启动独立 HPU probe、微基准或 profiler。
- 不先切 lazy、全模型 BF16 常驻、盲目缓存权重或替换量化数学。
- 不以 `hl-smi` 的单点利用率、ComfyUI 进度条或一次最好成绩推断硬件已达到上限。
- 不把 token/s、profiler 耗时或引擎日志数字本身当成不可变的硬件指标。
- 不记录 HF token、代理凭据或其它认证信息；模型已经存在时不重复下载 VAE 或其它组件。
