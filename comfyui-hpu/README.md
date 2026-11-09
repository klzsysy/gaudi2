# Gaudi2 HPU 适配交接入口

更新时间：2026-11-09（Asia/Shanghai）。本文件只保留当前状态和交接要点；验证细节见 `PROGRESS.md`，性能调查见 `MOSS_HPU_PERFORMANCE_PLAN_2026-11-06.md` 与 `MOSS_HPU_PERFORMANCE_RUN_2026-11-08.md`。

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

- 严格 P1 首轮 B0/B1：1024²、25 步、固定用户图，5 次 warm median 分别为 **137.282 s** 和 **40.183 s**；所有请求 fallback 为空，327-token 输出 ID 一致，图片和 latent finite 检查通过。
- FP8 E4M3/E5M2 `torch.mm` 在当前 eager 栈原生 HPU 通过，约 525.88/519.63 TFLOP/s；BF16 同 shape 275.76 TFLOP/s。INT8 `torch._int_mm` 发生 fallback，不作为 HPU 峰值或生产优化依据。
- lm_head 解量化缓存、强制 embedding 留在 HPU、全模型/局部 FP8 文本编码器均未通过重复收益门槛，均未进入产品代码；当前主要线索是 `_copy_from`、小批量自回归和 host/device 控制同步。
- B2/B4 完整控制矩阵、跨进程 cold 和更细的设备归因仍未完成；厂商高层 profiler 摘要存在 MME 数据兼容性限制，当前用原始 HPU trace + Perfetto SQL。

## 驱动与模型存储

- 服务器重启后 `5.15.0-202`/`5.15.0-186` 双内核 DKMS、`hl-smi` Operational、BF16 HPU matmul 和 `/system_stats` 均通过。驱动安装脚本已改为检查全部可启动内核；19 项模拟回归及实际双内核只读检查通过，补丁在 `patches/install-gaudi2-all-boot-kernels.patch`。
- 两个 checkout 的模型已合并到共享目录，分类 symlink 和 `extra_model_paths.yaml` 验证通过；Gaudi checkout 实际加载 Qwen 2.1 Turbo 并出图，原始 ComfyUI 未启动。

## 已知边界与接续工作

- 验证范围不包括编辑/控制 LoRA、SLG、批量/多卡、其它 VAE、custom nodes、lazy/compile/HPU Graph 或超大分辨率。
- tokenization、采样控制和已有 Flux RoPE float64 频率表仍可能走 CPU；“模型在 HPU”不等于没有任何 CPU 算子。
- 若继续性能调查，应先完成 B2/B4，并把 `_copy_from` 按采样回传、解量化和缓存/预取归因，再评估单变量改动。

## 接续工作流程

1. 读取本仓库 `AGENTS.md`、本文件、`PROGRESS.md`，再检查产品 checkout 的状态。
2. 检查服务器进程、`hl-smi`、磁盘和队列；HPU 单卡任务必须串行。
3. 只同步已审查的产品代码到 `/home/sonny/ComfyUI-Gaudi`，不触碰原始 `/home/sonny/ComfyUI`。
4. 失败先记录最小复现、版本、证据和结论；未验证的猜测明确标注。
