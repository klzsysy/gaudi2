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

## MOSS性能优化里程碑

2026-11-11按用户要求精简各轮记录。当前路线与完整结果索引见 [BF16复盘与FP8判断](MOSS_OPTIMIZATION_STRATEGY_2026-11-09.md)，各轮保留测量条件、数值/质量边界、失败原因和原始证据位置。

| 产品 | 已验收变化 | 记录 |
|---|---|---|
| `60da4e33` | GDN/conv区域编译，同输入重放显著减少小算子执行成本 | [GDN](MOSS_GDN_COMPILE_2026-11-09.md) |
| `f739cb93` | MOSS attention编译，固定caption DiT37.916→28.217s | [DiT](MOSS_DIT_OPTIMIZATION_2026-11-09.md) |
| `05d2a0a5` | 投影/FFN分区、静态recipe、MLP与hooks兼容，DiT28.225→19.456s | [block](MOSS_BLOCK_OPTIMIZATION_2026-11-09.md) |
| `3fec1143` | KV与完整短步attention编译、分支行clone，固定teacher4.312→3.002s | [KV](MOSS_KV_OPTIMIZATION_2026-11-10.md) |
| `441d459d` | 恢复Python SIGPIPE策略，浏览器取消下载后设备可继续执行 | [恢复记录](HPU_SIGPIPE_RECOVERY_2026-11-10.md) |
| `b1163be0` | 条件/RoPE预计算50→2次，DiT19.581→18.384s | [预计算](MOSS_DIT_PRECOMPUTE_2026-11-10.md) |
| `1412e1f7` | native norm/RoPE、prefill分桶、验证后BLOCKWISE，DiT至14.364s | [深挖](MOSS_BF16_DEEP_EXECUTION_2026-11-11.md) |
| `64155fe4` | 原norm区域编译，固定MTP完整22.461→21.060s，补全P0–P6证据 | [预算](MOSS_BF16_PLAN_COMPLETION_2026-11-11.md) |
| `a8f42df2` | 并行递推、八行GDN扩区、单head attention，固定完整19.978s/缓存14.386s | [最新验收](MOSS_BF16_ATTENTION_AND_SUBMISSION_2026-11-11.md) |

早期BF16常驻、FP8/INT8 probe及失败路径见 [初期调查](MOSS_HPU_PERFORMANCE_RUN_2026-11-08.md)。拆解36.648s工作流和30.786s节点耗时的证据分别见 [工作流拆解](MOSS_WORKFLOW_BREAKDOWN_2026-11-10.md) 和 [节点计时](MOSS_PIPELINE_TIMING_2026-11-10.md)。

## 当前验收与下一阶段

- 最新产品194 passed、1 CUDA skip，512/768/1024、MTP off/3/tree3s1、原INT8及卸载重载通过。自然文本与同seed图像细节会改变，cold64.665s，完整warm19.906–20.122s。
- 同工作量allocator峰值39.3754GiB，卸载2,685,928,832 bytes，多形状/INT8验收为2,686,190,976 bytes。六份部署文件哈希一致，上一轮8189恢复cold66.725s、缓存14.751s、检查队列为空。[部署证据](evidence/moss-bf16-next-2026-11-11.json)
- 下一阶段建议验证DiT原生FP8 Linear的真实shape、量化总成本和多timestep质量。BF16已有大Linear接近同shape可用速度，attention数据流和文本提交保留按证据启动的研究机会。详细决策见复盘。
- 官方/外部实现依据分别见 [INT8/FP8调查](GAUDI2_INT8_FP8_KERNEL_SURVEY_2026-11-10.md)、[JChen核查](JCHEN_GAUDI2_KERNEL_REVIEW_2026-11-10.md) 和 [硬件下界/5090](MOSS_HARDWARE_LIMIT_AND_5090_2026-11-10.md)。
