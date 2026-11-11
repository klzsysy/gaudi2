# 执行记录

本文保留每个问题的验证逻辑、核心证据、成功/失败结论和边界；服务 PID、重复启停、工具安装流水账和已完成计划不再逐次记录。

## 阶段一：设备后端（`8326476f`）

- 在独立 HPU checkout 和独立 venv 中增加 HPU 设备探测、设备/内存查询、同步、bf16 策略及 CUDA 专用路径的安全关闭；没有使用全局 CUDA monkeypatch。
- 实机 smoke：bf16 matmul 与 CPU 参考一致，ComfyUI attention 通过，ModelPatcher 可加载到 HPU 并卸载到 CPU；HTTP `/`、`/object_info`、`/system_stats` 返回成功，设备为 `hpu:0 GAUDI2`；CPU quick-test 不分配 HPU。
- 失败路径：非交互 SSH 不加载 Habana 环境，缺少 `GC_KERNEL_PATH` 时 attention 编译报 `synStatus 26`；显式设置 `/usr/lib/habanalabs/libtpc_kernels.so` 后通过。服务持卡时独立 probe 会 `Device acquire failed`，改为串行执行并等待旧进程退出。pip resolver 会尝试替换 Habana torch，改为继承基础 site-packages、只安装缺失依赖；`pip check` 仅剩已知 torchvision 声明冲突。
- 阶段边界：未验证模型工作流、其它实体加速器或所有自定义节点；阶段一不代表完整生图通过。

## 阶段二：Krea2（`fc1c891f`）

- 初次流程能返回 history，但文本改到 HPU bf16 后保存阶段出现 `invalid value encountered in cast`，图片异常，不能通过。
- 根因验证：HPU 将 E4M3FN 存储字节 `[240,256,288,320,384,448]` 转 float 得到 `inf/nan`，第一处非有限值在 Qwen3VL 第 0 层 q_proj；不是 prompt、采样或 VAE 问题。
- 修复 `comfy/quant_ops.py::_TensorCoreFP8LayoutBase.dequantize`：HPU 使用完整 256 值解码表，保留权重/scale/量化状态；其它设备走原路径。10 项回归通过，256 个 FP8 字节在 bf16/fp32 下与 CPU 逐值一致（合法 NaN 编码除外）。
- 修复后 512²、1024² 和卸载重载均正常出图，文本/DiT/VAE 均使用 HPU bf16。量化权重仍是解量化计算，未宣称原生量化 GEMM。

| 验证 | 配置 | 端到端耗时 | 结果 |
|---|---|---:|---|
| 修复后首次 | 512²，seed 42，8 steps | 15.02 s | 红茶壶正常，无 NaN warning |
| 大尺寸 | 1024²，seed 43 | 10.93 s | 图片正常，标准差 75.36 |
| 卸载重载 | 512²，seed 44 | 13.75 s | 文本/DiT/VAE 重新加载并成功 |

## 阶段三：Qwen-Image-2.1 Turbo（`6db162dd`）

- 初次 512²/1024² history 均 success，但图片异常纹理；禁用 prefix cache、改 CPU 文本编码器均无效。固定随机输入的 CPU/HPU DiT 中间层有限，不能解释异常图。
- 同一 latent 经 CPU fp32 VAE 得到正确红茶壶，证明文本和 DiT 已产生有效潜变量；HPU VAE 逐层对照显示，带卷积 layout metadata 的 `strip_apply` 原地残差累加在分支合并处产生大误差。取消分块、`contiguous()` 和 `copy_` 均不能修复；最小复现的 4/64 通道最大绝对误差分别为 48/993。
- 修复 Wan 2.2 `strip_apply` 的 HPU 分支：保留分块，但将 `out + residual` 改为返回新张量，避免写入卷积视图；CPU/CUDA 行为不变。VAE 分块/不分块和真实工作流回归通过。
- 修复后 512²、1024²、长提示词和卸载重载均通过；14 项回归通过。CPU fp32 与 HPU bf16 VAE PNG 对照 MAE 0.2102/255、PSNR 54.49 dB；这是 VAE 对照，不是全生成器逐像素等价声明。

## 阶段四：MOSS-Image SFT（`21e98390`）

- 组件为 MOSS SFT bf16 DiT、Qwen3.5 9B INT8 ConvRot 文本模型和复用的 Flux2 VAE；通过 header/shape/dtype 对照确认旧 VAE 与官方文件的 251 个张量一致，不重复下载 VAE。
- Qwen3.5 `torch_chunk_gated_delta_rule` 长度 17/65/128 与 CPU 参考输出有限，最大绝对差 0.0004883/0.0009766/0.0009766；VAE roundtrip 通过。
- 512²、1024²、长提示词和卸载重载均通过，配置为 25 步 `er_sde/simple`、cfg2.5；图片像素范围 0–255，无非有限值或异常纹理。验证只覆盖 SFT 文生图组合，不包括 Base、编辑/控制 LoRA、SLG、prompt enhancer 或其它量化变体。

## MOSS TextGenerate 采样修复（`0139b1da`）

- 官方模板启用 `mtp=auto`、sampling 后，HPU 在 top-k 采样的 `top_indices.gather(1, next_token)` 报 `gather_elements_fwd_i32` / `GLUE_INCOMPATIBLE_DATA_TYPE`。随机 logits 最小复现确认根因是索引 int64，不是 `.item()` 或 `GC_KERNEL_PATH`。
- 修复只在 HPU 将词表索引临时转 int32，gather 后恢复 `torch.long`；模型计算、随机采样和返回值仍在 HPU，没有关闭 MTP 或把文本模型搬到 CPU。
- 原样用户图重放成功，扩写约 251 tokens 后正常结束并完成 1024² 出图；诊断模式端到端 350.39 s，不作为性能基准。回归 **26 passed**，覆盖 CPU/HPU、bf16/fp32、top_k=0/1/64、batch、高位词表编号、固定 seed 和返回设备/类型。

## MOSS 输出头 CPU fallback 修复（`0690d18f`）

- 8-token 诊断发现 `BaseGenerate.logits` 和 MTP `verify_logits` 直接调用 `F.linear`，绕过量化 `Linear` 的 `_full_precision_mm` 策略；INT8 词表矩阵每步进入 `aten::_int_mm` CPU fallback。
- 改为在存在独立 lm_head 时调用该层本身，复用既有解量化/精度策略；普通生成从 10.19 s 降至 4.47 s，token ID 完全一致，fallback 为空。真实 MTP auto 32-token 为 10.412 s，fallback `{}`；CPU 解码回归 **42 passed, 1 skipped**。
- 同一完整用户请求从 337.013 s 重放到 134.586 s；新扩写约 307 tokens、约 89 s，图片语义正确。旧请求与新服务缓存/生命周期不同，因此不能作为严格多轮性能基准。
- HPU 利用率仍有 0% 间隙，84 个扩写阶段样本平均约 22%、最高 57%；本修复只消除了已定位的输出头 CPU fallback。

## 驱动恢复与多内核检查

- 故障原因：重启进入 `5.15.0-202` 时，Habana DKMS 只安装在 `5.15.0-186`；内核自带模块没有 Gaudi2 PCI alias，`hl-smi` 无设备。不是 ComfyUI、Python、模型或 PCI 枚举故障。
- 修复：为 `202` 安装现有 Habana DKMS，按既有模块顺序加载并刷新 initramfs；随后修复驱动安装脚本，使安装/verify 覆盖当前内核和所有有效 `/boot/vmlinuz-*`，校验模块、vermagic、路径、alias 和补丁字符串。
- 验证：`186`/`202` 双内核 DKMS installed，重启后 `hl-smi` Operational，BF16 HPU matmul 与 CPU 参考一致，ComfyUI-Gaudi `/system_stats` 返回 `hpu:0 GAUDI2`；19 项模拟回归、实际双内核 verify 和完整重启通过。审查补丁为 `patches/install-gaudi2-all-boot-kernels.patch`。
- 边界：未配置开机自启；驱动恢复后没有修改原始 `/home/sonny/ComfyUI`。

## 模型集中存储

- 两个 checkout 的模型已合并到 `/home/sonny/ComfyUI-Models/models`，去重 36 项、冲突 0；两个 checkout 的分类 symlink 和相同 `extra_model_paths.yaml` 已验证。
- Gaudi checkout `/system_stats` 返回 HPU，模型列表包含 Qwen 2.1、MOSS、Qwen3、T5 和 VAE；Qwen 2.1 Turbo 512² 实际出图成功。原始 ComfyUI 没有启动。
- 旧配置备份路径和最终目录结构见 `MODEL_STORAGE_PLAN_2026-11-06.md`；共享目录不包含代码、venv、output 或 custom nodes。

## 性能调查：当前证据与未完成项

- 2026-11-10 用户指定 JChen 仓库后补做 `6da6fb16` 专项审查：相对 `e08710f7` 新增 2 提交/297 文件，包含固定 H3 形状的直接 MME/TPC attention（BF16 QKV、FP16 score、BF16 probability）；还确认 `csrc` 内的 TPC FP8 MAC 稀疏 attention、残差/RMSNorm/量化和 GDN。更正前轮只看输出头造成的范围遗漏；INT8 W8A8 仍转 BF16，未发现通用 INT8 GEMM。作者 H3 100.45→84.41→79.7 s 含不同优化组合且质量未验收，不作为本机结果。见 [专项复核](JCHEN_GAUDI2_KERNEL_REVIEW_2026-11-10.md)；本轮无编译、硬测或服务器改动。
- 2026-11-10 INT8/FP8 kernel 调查：核对官方 1.24.2 文档/bridge 与 vLLM Gaudi、INC、bitsandbytes、Comfy Kitchen、官方 TPC 示例、1Cat 等固定源码。FP8 cast/GEMM/SDPA 有官方路径；INT8 TPC 能力不等于 INT8 MME GEMM，vLLM 明确使用 BF16 fallback，bitsandbytes 默认 INT8 matmul 实际转 FP32。1Cat 包含真实 TPC-C/Triton FP8 量化和低比特权重解码，但动态量化候选在 hybrid 模式仍关闭、需要配套 backend/bridge。没有新的硬测、部署或产品修改；记录 [调查报告](GAUDI2_INT8_FP8_KERNEL_SURVEY_2026-11-10.md) 和 [源码 manifest](kernel-audit-2026-11-10/sources.json)。
- 2026-11-10 按用户要求编写 [BF16 深挖计划](MOSS_BF16_DEEP_OPTIMIZATION_PLAN_2026-11-10.md)：从 `b1163be0` 开始，优先真实shape/关键路径归因，按证据推进矩阵布局、短链融合/提交、BF16 attention、文本prefill/decode和条件触发的执行模式实验。复用已有负面结果，明确固定工作量、局部高精度/完整质量、生命周期和端到端收益门槛；以真实算子可用上限和剩余热点判断BF16是否收敛。本轮只制定计划，未修改产品或运行HPU实验。
- 2026-11-10 只读完成 [BF16 硬件下界与 RTX 5090 对标](MOSS_HARDWARE_LIMIT_AND_5090_2026-11-10.md)：CPU 实际 tokenizer 为正307/负23，25步两分支 DiT 矩阵约2.496 PFLOP，下界5.78 s；当前等效矩阵峰值比例31.43%。沿用同输出历史 MTP 诊断的文本权重流量约2.700 TB，连同 DiT 得到约6.95 s的不完整乐观整图参照，明确未计状态/非线性/运行时/VAE。一手公开5090报告 BF16缓存出图14.7 s、INT8 DiT缓存出图5.8 s、另一作者采样约7 s；与我方18.764 s只能近似对标，完整扩写29.673 s不能直接比较。NVIDIA BF16 FP32累加稠密峰值核为209.5 TFLOP/s，不误用稀疏419或FP4 TOPS。新增纯CPU可复算脚本，没有改产品、跑HPU或停服务。
- 2026-11-10 DiT 产品 **`b1163be0`**：在本次 conditioning 中预计算静态文本特征与 FP32 RoPE，50 次降至 2 次；固定 caption 5 warm，DiT **19.581→18.384 s**、AIP **56.79%→60.97%**，图像 MAE 0.115/255，与基线自身重复噪声相当。完整用户图 5 warm **29.673 s**（原 30.786），298-token ID 与原轨迹一致。新 preparation 19 项、原 MOSS 21 项回归通过，512/768/加权提示词/重载/原 INT8 图 finite/fallback 检查通过。CFG 合批无稳定叠加收益；fast softmax 虽快但首层 FP64 参考误差从 0.314% 增至 0.721%，不合入。严格卸载阈值检查曾超时，随后同流程基线对照确认新旧稳定占用均为 2,685,928,832 bytes，无新增残留。详见 [DiT 预计算记录](MOSS_DIT_PRECOMPUTE_2026-11-10.md)。
- 2026-11-10 重新统计当前 `441d459d` pipeline：用户当前文本 seed=1、298 tokens、1024²/25 步；完整与文本缓存各 5 warm，median **30.786/19.900 s**。同一次完整代表请求中 TextGenerate 9.093、正负编码 1.817、KSampler 19.611 s。单独同步诊断保持相同 token ID 和 129 次 verify；DiT 主干 18.150 s、条件 text_embedder 0.797、RoPE 0.541，后二者各重复 50 次；文本 main decode 5.239、prefill 2.035 s。诊断与基准时间分开解释；fallback 为空、latent/VAE finite。恢复日常服务并回填原工作流缓存，无产品改动。详见 [MOSS_PIPELINE_TIMING_2026-11-10.md](MOSS_PIPELINE_TIMING_2026-11-10.md)。
- 2026-11-10 用户报告日常服务 `Host malloc failed`：部署哈希一致，最早故障是 `sendfile64` 触发 SIGPIPE，被 Habana 1.24.2 当作 fatal，随后加载权重和 worker sync 连带失败。产品 **`441d459d`** 在 HPU 初始化后恢复 Python 的 SIGPIPE 忽略策略；最小 broken-sendfile 复现由设备失效变为后续运算正常。实际 8189 服务通过 8 次中断下载、显式 SIGPIPE 和用户原 pending 图重放，1024² 出图成功，cold 48.152 s。前轮 API 性能验收未覆盖浏览器取消文件传输，此次已补。见 [故障记录](HPU_SIGPIPE_RECOVERY_2026-11-10.md)。
- 2026-11-10：产品 **`3fec1143`** 编译 KV attention/tree merge 和普通 BF16 Qwen3.5 的完整短步 attention，并修复 HPU MTP 分支行搬移失效。正式固定 32-token teacher 对照 **4.312 → 3.002 s**、top-1 32/32、KL 0.0002996；完整用户图 5 warm median **29.925 s**（305 tokens），相对旧 36.648 s 属不同自然轨迹。33 项专项与 42 项 decode 回归通过、1 CUDA 项跳过；512/768、MTP auto/off、BF16 重载、原 INT8 图检查通过，fallback 为空，文本模型卸载后 InUse 93.98 MiB。AIP 全图/文本/DiT **45.22%/22.20%/57.04%**。前缀分桶、扩大 GDN decode、FP8 attention 未获得足以合入的证据。详情见 [优化记录](MOSS_KV_OPTIMIZATION_2026-11-10.md) 和 [讨论组建议评估](ATTENTION_IDEAS_REVIEW_2026-11-10.md)。
- 2026-11-10：当前调查仓库先提交 `b6210ca`。复算既有 `full-complete-product` 五次 warm，以实际 median 请求 b0-warm-4 严格拆分 36.647989 s：TextGenerate 14.949898、正负编码 1.754006、KSampler 19.666598、VAE/图片输出 0.256466、其它 0.021022 s。只读源码确认当前 HPU KV attention/tree merge 使用 FP32 torch 参考路径、完整 capacity 计算后 mask；该区域尚未编译，实际耗时待 trace。输入 1970、生成 277、上限 4096，优先验证有效前缀/shape bucket 与现有算子编译，再查 MTP 成本估计及 DiT 剩余开销。没有新 HPU 性能结果或产品改动；详见 [MOSS_WORKFLOW_BREAKDOWN_2026-11-10.md](MOSS_WORKFLOW_BREAKDOWN_2026-11-10.md)。
- 2026-11-09 继续推进现有 BF16 算子：九组真实 block 的 CPU FP64 对照定位调制/残差的舍入差异，保留这两个边界，编译投影与 FFN（`8ebc3f40`），再启用局部 `force_static_compile`（`15c5b17c`）。固定 caption DiT **28.225 → 19.456 s**。新增普通 BF16 Qwen3.5 的 KV-cache 短步 MLP 编译（`05d2a0a5`），固定 teacher 轨迹五次 warm **4.876 → 4.223 s**；32-token logits top-1 32/32、平均 KL 0.0003535。编译后新增 hooks 的回归失败已通过 eager 分流修复（`b830b3b8`）。分阶段回归包括 79 passed/1 skipped、静态 recipe 20 项复验、最终 MLP/hook/MOSS 35 项复验；有重复，不累加。完整 BF16 用户图 warm median **36.648 s**，自然结束长度 267–362 tokens；全图/文本/DiT AIP 均值 **37.93%/16.79%/57.23%**。BF16/INT8 完整诊断、hooks、变长和重载通过，fallback 为空；卸载后 allocator InUse 约 94 MiB。服务器已部署 **`05d2a0a5`** 并恢复日常服务，本地未 push。详见 [MOSS_BLOCK_OPTIMIZATION_2026-11-09.md](MOSS_BLOCK_OPTIMIZATION_2026-11-09.md)。不承诺逐位复现，未达到 80% 目标。
- 2026-11-09 BF16 DiT attention 产品提交 **`f739cb93`**：固定 caption 各 1 cold + 5 warm，DiT **37.916 → 28.217 s**、整图 **40.093 → 30.258 s**、allocator 峰值 **44.35 → 40.11 GiB**。图像 MAE 0.12055/255，与基线重复波动相当；最终产品专项 **7 passed**。512/768/1024、加权提示词、卸载重载和原 INT8 两次用户图 finite/fallback 检查均通过。完整 BF16 扩写图 median **52.591 s**，token 数变化，属于实际工作流观察而非严格同轨迹提速。共享动态 SDPA 曾出现 SIGSEGV，整块编译有明显更大图像差异，两者均未合入。全部细节及失败证据见 [MOSS_DIT_OPTIMIZATION_2026-11-09.md](MOSS_DIT_OPTIMIZATION_2026-11-09.md)。按最新用户要求，继续复用现有算子，暂不自写 kernel。
- 2026-11-09 路线评估完成：只读核对官方 MME/TPC 能力、AIP 口径和真实 checkpoint shapes。文本主干+lm_head 静态权重约 15.87 GB；DiT 为 42 层、d=3072、FFN=8192、已融合 QKV/up-gate。建议 BF16 执行/DiT 优先、原生 FP8 试点其次、trace 证明后再做 TPC 热点融合；通用 INT8 GEMM 暂缓。更正 `gaudi2-hardware.md` 中无依据的“INT8 与 FP8 同峰值”推断。详见 [MOSS_OPTIMIZATION_STRATEGY_2026-11-09.md](MOSS_OPTIMIZATION_STRATEGY_2026-11-09.md)，其中估算不是新性能成绩。
- 2026-11-09 GDN/conv 区域编译：复用 Kitchen 公开操作，HPU 使用 `hpu_backend`，状态管理保持原接口。相同 BF16 模型两组各 1 cold + 5 warm，完整图 median **131.031 → 60.063 s**；TextGenerate **96.740 → 20.156 s**，token/s **3.88 → 15.03**。自然结束 caption 不同，不能声称等价 token 轨迹；全部 fallback 为零。59 passed/1 skipped，增强状态/权重复用后专项 5 passed。源码、性能、利用率和误差证据见 [MOSS_GDN_COMPILE_2026-11-09.md](MOSS_GDN_COMPILE_2026-11-09.md)。
- 已提交并同步服务器 **`60da4e33`**（本地未 push）；原 INT8 文件两次完整图也成功，诊断 latent/VAE 全部 finite，fallback 为空。日常 HPU 服务恢复、队列空闲；保持用户原模型选择，`--disable-comfy-compiler` 可回到原执行路径。
- 目标用户图为 1024²、25 步、MOSS TextGenerate 开启；严格 B0/B1 首轮分别完成 1 cold + 5 warm/1 transition + 5 warm。B0 warm median **137.282 s**（135.893–137.526），B1 warm median **40.183 s**（39.690–40.314），fallback 始终为空，诊断 B1 分段覆盖 99.963%。
- FP8 E4M3/E5M2 4096³ `torch.mm` 原生 HPU 约 525.88/519.63 TFLOP/s，BF16 275.76 TFLOP/s；INT8 `torch._int_mm` 3.17 TFLOP/s 且 103 次 fallback，不作为 HPU 峰值。当前外部 827 TFLOP/s 数据使用不同版本、形状、lazy、`_scaled_mm`、输出 dtype 和 MAMF 计时，不能直接比较；本机 `_scaled_mm` 复核报 `synStatus 26`。
- lm_head BF16 缓存：MTP off 单次 16.963→15.920 s，但 MTP auto 10.068→10.622 s，拒绝；强制 embedding 驻留 HPU 的差异约 3%，低于噪音，拒绝。强开 native INT8 2-token 约 1.47→12.54 s，确认会走 CPU fallback。
- 从现有 INT8 权重转换的全模型/局部 FP8 文本模型分别为 16.23 s/4.71 s 的 2-token warmup，均慢于 INT8→BF16；转换器误量化 1-D norm 后已修正，FP8 文件未进入生产。
- 从现有 INT8 ConvRot 权重生成普通 BF16 文本模型并完成 HPU 验证：约 19.31 GB，MTP auto 32-token 5 次 warm median **9.407 s**，INT8 路径 **10.097 s**，fallback 为空；完整 1024² MOSS workflow 成功。两阶段分块 top-k 移植因 BF16 tie policy 改变采样 token、收益仅约 0.6% 已回滚；当前 MTP tree、deferred GDN、host_rows/prefetch 已是 Qwen3.5 路径的一部分。
- JChen 风格 BF16 `lm_head`→`fp8_gemm_v2` 调查 probe 单次为 9.580 s/32-token，fallback 为空但末 token 有差异；尚未证明收益，未进入产品。
- 源码复核纠正此前“已有对应优化”的判断：GDN deferred 的原生融合入口只支持 CUDA/HIP，HPU 实际走 `_decode_deferred_torch`，包含固定 8 槽位 replay；HPU 非阻塞/异步预取关闭，Comfy 图重放门控只允许 CUDA。MTP 状态语义存在不等于 HPU 融合与图优化已经启用。尚未测得各部分占比，不据此宣称具体提速；BF16 的 6.8% 及完整图单次耗时亦不能视为严格相同解码轨迹的端到端收益。
- 真实 HPU trace + Perfetto SQL 通过：读到 414,226 个 CPU op 和 1,030,554 个 kernel 类事件；它们是后续归因线索，不等于 launch 次数，事件 duration 不能直接相加。厂商高层摘要因 MMEEvents 不兼容失败，作为工具限制保留。
- P1 尚欠 B2 模式/长度矩阵、B4 尺寸/步数和跨进程 cold；P2/P3 后续必须先把 `_copy_from` 区分为采样回传、解量化和缓存/预取，再做单变量优化。
