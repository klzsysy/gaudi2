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
