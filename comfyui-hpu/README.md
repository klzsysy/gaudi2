# Gaudi2 HPU 适配交接入口

更新时间：2026-11-10（Asia/Shanghai）。本文件只保留当前状态和交接要点；验证细节见 `PROGRESS.md`，最新结果见 [29.925 秒 KV/attention 优化](MOSS_KV_OPTIMIZATION_2026-11-10.md)。

**文档归属：**所有过程、失败/成功证据和个人环境信息只保存在本调查仓库，不进入 `/home/sonny/ComfyUI` 或 `ComfyUI-Gaudi` 产品仓库。

## 目标与边界

- 单卡 Gaudi2 96 GiB，Habana 1.24.2，Python 3.10，Habana PyTorch `2.13.0a0+git2469d18`，实际验证固定为 eager（`PT_HPU_LAZY_MODE=0`）+ bf16。
- 服务器独立 checkout：`/home/sonny/ComfyUI-Gaudi`；原始 `/home/sonny/ComfyUI` 只读复用模型，禁止修改其代码、环境、配置、输出或模型。
- 服务器使用共享模型目录 `/home/sonny/ComfyUI-Models/models`；下载只使用服务器已有 `HF_ENDPOINT`、`HF_TOKEN`，不记录凭据。
- 调查结论以服务器实测为准；启动成功、设备可见或单个算子通过不等于完整模型验证成功。

## 已验证状态

- 阶段一设备后端：`8326476f`。HPU 探测、设备/内存/同步、bf16 策略、attention、ModelPatcher 加载卸载、HTTP `/system_stats` 和 CPU 启动 smoke 通过；启动时缺少 `GC_KERNEL_PATH`、HPU 设备被服务占用等失败路径已记录并修正。
- 阶段二 Krea2：`fc1c891f`。HPU 文本/DiT/VAE bf16，512/1024、卸载重载通过。初始 FP8 E4M3 解码产生 `inf/nan`，完整 256 值解码表修复后 10 项回归通过；不宣称原生 INT8/FP8 GEMM。
- 阶段三 Qwen-Image-2.1 Turbo：`6db162dd`。首次异常纹理由 HPU VAE 的带 layout metadata 原地残差累加触发；最小 HPU 非原地累加修复后，512/1024、长短提示词和卸载重载通过，14 项回归通过。
- 阶段四 MOSS-Image SFT：`21e98390`。复用现有 Flux2 VAE，512/1024、25 步 `er_sde/simple`、长短提示词和卸载重载通过；文本/DiT/VAE 均在 HPU，VAE roundtrip 通过。
- MOSS TextGenerate 采样修复：`0139b1da`。HPU `gather_elements_fwd_i32` 拒绝 int64 索引；仅将 top-k 索引临时转 int32 后 gather，完整用户扩写图通过，采样回归 **26 passed**。
- MOSS 输出头回退修复：`0690d18f`。普通/MTP 路径改为调用量化 lm_head 层自身，避免 `torch._int_mm` CPU fallback；解码回归 **42 passed, 1 skipped**，同一完整请求约 `337.013 s → 134.586 s`。这不是全面性能优化完成的声明。

## 当前性能结论

- **最新节点测量（当前 UI 文本 seed=1）**：各五次 warm，完整重算 **30.786 s**、文本缓存 **19.900 s**；完整代表请求扩写 9.093、正负编码 1.817、DiT/采样 19.611、VAE/图片输出 0.246 s。单独诊断中，DiT 42 层主干约占采样节点 89.6%，RoPE/text_embedder 重复 50 次合计 1.34 s。下一步优先 DiT 的执行内条件预计算及主干归因；详见 [pipeline 节点统计](MOSS_PIPELINE_TIMING_2026-11-10.md)。本轮只测量，日常服务与原工作流缓存已恢复。
- **当前部署 `441d459d`** 在 `3fec1143` 性能优化基础上修复浏览器断连触发的 SIGPIPE/Synapse fatal。代码本来已同步；`Host malloc failed` 是运行时失效后的连带报错。恢复 Python SIGPIPE 策略后，8 次中断下载、显式信号注入和用户原 1024² pending 图重放通过；详见 [故障恢复记录](HPU_SIGPIPE_RECOVERY_2026-11-10.md)。
- **最新产品 `3fec1143`**：HPU KV attention/tree merge 区域编译 + 普通 BF16 Qwen3.5 短步完整 attention 编译；另修复 HPU 在 MTP 分支提交时同 storage 视图复制可能不更新目标行的问题。完整用户图 1 cold+5 warm，median **29.925 s**，相对此前 36.648 s 约少 18.34%；自然结束 tokens 不同，不是同轨迹 A/B。正式产品固定 32-token teacher 对照 **4.312 → 3.002 s**，top-1 32/32、平均 KL 0.0002996。详见 [本轮记录](MOSS_KV_OPTIMIZATION_2026-11-10.md)。
- 最终 33 项 KV/attention 专项、42 项 Qwen decode 回归通过，1 项 CUDA 专属跳过；512/768、MTP auto/off、卸载重载、原 INT8 图通过，fallback 为空。卸载后 allocator InUse 约 **93.98 MiB**。当前全图/文本/DiT AIP 平均 **45.22%/22.20%/57.04%**，仍未达到 80%。新 median 请求中 DiT/采样 **19.665 s**、文本 **8.165 s**，下一重点转向 DiT 剩余开销，文本侧继续查 prefill/GDN/调度。
- 对讨论组提出的 FP8 attention、分页 KV 镜像、8K 分片和前缀缓存进行了 [逐项评估与厂商算子实测](ATTENTION_IDEAS_REVIEW_2026-11-10.md)。当前 8-token 联合计算/连续 KV 已存在，FP8 未证明优于 BF16；没有引入 FP8、自写 kernel、跨请求模型 tensor cache 或修改生成上限。
- **上一版本 36.648 秒请求拆解**：扩写 14.950 s（40.79%）、正负编码 1.754 s、DiT/采样 19.667 s（53.66%）、VAE/图片输出 0.256 s、其它 0.021 s。使用同一次请求，可相加；与各阶段独立中位数区分。该分析确定了本轮 KV attention 的优化优先级，详见 [阶段拆解与计划](MOSS_WORKFLOW_BREAKDOWN_2026-11-10.md)。
- **上一轮执行优化 `05d2a0a5`**：MOSS 投影/FFN 分区编译 + 厂商静态 recipe，固定 caption 的 DiT median **28.225 → 19.456 s（-31.07%）**；普通 BF16 Qwen3.5 短步 MLP 编译，固定 32-token teacher 轨迹 median **4.876 → 4.223 s（-13.38%）**。完整 BF16 用户图五次 warm median **52.591 → 36.648 s**，自然结束 token 数不同，不能当作同轨迹 A/B。详见 [block/文本执行优化记录](MOSS_BLOCK_OPTIMIZATION_2026-11-09.md)。
- 九组真实 block 的 FP64 对照未见系统性精度恶化；固定 seed 的画面和扩写细节仍会改变。量化/自定义 operations 不启用新增 MLP 编译，原 INT8 图通过；forward hooks、CPU、训练、禁用编译及模型重载路径保留。最终 AIP 平均：整图 **37.93%**、扩写 **16.79%**、DiT **57.23%**，峰值 68%，仍未达到 80%。
- **BF16 DiT attention 优化已落地 `f739cb93`**：固定 caption、1024²/25 步、各 1 cold + 5 warm，DiT median **37.916 → 28.217 s（-25.58%）**，扩写关闭的整图 **40.093 → 30.258 s**，allocator 峰值 **44.35 → 40.11 GiB**；图像 MAE 0.12055/255，与基线重复波动相当。完整 BF16 扩写图 warm median **52.591 s**，自然结束 token 数变化，不与旧 60.063 s 当作严格同轨迹 A/B。详见 [DiT 优化记录](MOSS_DIT_OPTIMIZATION_2026-11-09.md)。
- 本轮仅编译 MOSS 既有 attention 区域，不自写 kernel。7 项专项通过；512/768/1024、加权提示词、卸载重载与原 INT8 两次完整图均通过、fallback 为空。共享动态 SDPA 和整块编译候选未合入。完整用户图 AIP 平均 **35.85%**、文本 **15.20%**、DiT **50.33%**，尚未达到 80% 目标。
- DiT 数值定位/分区、动态 shape、文本 MLP 与 KV/full attention 已推进；暂不自写 kernel，FP8 继续排在 BF16 执行优化之后。硬件和路线依据见 [多维路线评估](MOSS_OPTIMIZATION_STRATEGY_2026-11-09.md)，各阶段占比以本轮记录为准。
- **GDN/conv HPU 区域编译有效**：同一 BF16 文本模型、固定用户图（1024²/25 步），各 1 cold + 5 warm；端到端 median **131.031 → 60.063 s**，TextGenerate **96.740 → 20.156 s**。自然结束输出为原始 375 tokens、编译 303–333 tokens；吞吐约 **3.88 → 15.03 token/s**，非逐 token 等价声明。59 项回归通过、1 项 CUDA 专属跳过；状态/权重复用增强后 5 项专项复验通过。详见 [区域编译记录](MOSS_GDN_COMPILE_2026-11-09.md)。
- 产品提交/服务器部署为 **`441d459d`**，本地分支 `codex/gaudi2-hpu` 未 push。原 INT8 模型通过完整图和 latent/VAE 浮点 finite 检查，可使用 DiT 与 KV 核心编译；完整 attention 投影和 MLP 编译仅针对普通 BF16 文本模型。`--disable-comfy-compiler` 可关闭区域编译。
- 严格 P1 首轮 B0/B1：1024²、25 步、固定用户图，5 次 warm median 分别为 **137.282 s** 和 **40.183 s**；所有请求 fallback 为空，327-token 输出 ID 一致，图片和 latent finite 检查通过。
- FP8 E4M3/E5M2 `torch.mm` 在当前 eager 栈原生 HPU 通过，约 525.88/519.63 TFLOP/s；BF16 同 shape 275.76 TFLOP/s。INT8 `torch._int_mm` 发生 fallback，不作为 HPU 峰值或生产优化依据。
- lm_head 解量化缓存、强制 embedding 留在 HPU、全模型/局部 FP8 文本编码器均未通过重复收益门槛，均未进入产品代码；当前主要线索是 `_copy_from`、小批量自回归和 host/device 控制同步。
- 服务器已有 `qwen3.5_9b_w4a8_gptq.safetensors`，但当前 HPU 只注册 comfy-kitchen eager W4A8 backend；MOSS 默认仍逐层解量化到 BF16，强制 native 路径会回到 `torch._int_mm` fallback。下一候选是从已验证 INT8 ConvRot 生成独立普通 BF16 文本模型，做空间换时间 A/B；详见性能记录。
- BF16 常驻已验证：由现有 INT8 ConvRot 转换的普通 BF16 文本模型约 19.31 GB，MTP auto 32-token 5 次 warm median **9.407 s**，相对 INT8 路径 **10.097 s** 约快 6.8%，fallback 为空；完整 1024² MOSS 工作流成功出图。JChen 分块 top-k 移植因 tie policy 改变采样 token 且收益仅约 0.6% 已回滚。
- JChen 的 BF16 `lm_head`→`fp8_gemm_v2` probe 单次为 9.580 s/32-token，未证明收益、未进入产品。源码复核纠正：MTP tree 存在，但 GDN 在 HPU 上走 torch 参考实现；异步预取与 Comfy 图重放被设备门控关闭，不能视为已应用 JChen 的 HPU 优化。BF16 6.8% 也是跨进程、不同末尾 token 的描述性差异，严格端到端收益尚未建立。
- B2/B4 完整控制矩阵、跨进程 cold 和更细的设备归因仍未完成；厂商高层 profiler 摘要存在 MME 数据兼容性限制，当前用原始 HPU trace + Perfetto SQL。

## 驱动与模型存储

- 服务器重启后 `5.15.0-202`/`5.15.0-186` 双内核 DKMS、`hl-smi` Operational、BF16 HPU matmul 和 `/system_stats` 均通过。驱动安装脚本已改为检查全部可启动内核；19 项模拟回归及实际双内核只读检查通过，补丁在 `patches/install-gaudi2-all-boot-kernels.patch`。
- 两个 checkout 的模型已合并到共享目录，分类 symlink 和 `extra_model_paths.yaml` 验证通过；Gaudi checkout 实际加载 Qwen 2.1 Turbo 并出图，原始 ComfyUI 未启动。

## 已知边界与接续工作

- 验证范围不包括编辑/控制 LoRA、SLG、多卡、其它 VAE、其它 custom nodes、lazy/HPU Graph 或超大分辨率。区域编译已验证 Qwen3.5 deferred conv/GDN、普通 BF16 短步 MLP 与 MOSS attention/投影/FFN；batch=2 只做算子回归，没有完整批量工作流验收。
- tokenization、采样控制和已有 Flux RoPE float64 频率表仍可能走 CPU；“模型在 HPU”不等于没有任何 CPU 算子。
- 若继续性能调查，应补齐 B2/B4、更多提示词质量和区域编译之外的 host/device 归因；不要将上述单工作流结果外推到所有模型。

## 接续工作流程

1. 读取本仓库 `AGENTS.md`、本文件、`PROGRESS.md`，再检查产品 checkout 的状态。
2. 检查服务器进程、`hl-smi`、磁盘和队列；HPU 单卡任务必须串行。
3. 只同步已审查的产品代码到 `/home/sonny/ComfyUI-Gaudi`，不触碰原始 `/home/sonny/ComfyUI`。
4. 失败先记录最小复现、版本、证据和结论；未验证的猜测明确标注。
