# Gaudi2 HPU 适配交接入口

更新时间为2026-11-12（Asia/Shanghai）。本文件保留当前状态和交接要点，验证细节见 `PROGRESS.md`，优化复盘与路线判断见 [BF16与FP8](MOSS_OPTIMIZATION_STRATEGY_2026-11-09.md)。

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

## 当前性能与路线

- Qwen3.5 FP8文本已转换并适配普通CLIPLoader，文件`qwen3.5_9b_fp8_from_int8.safetensors`为11.377GB，来自现有INT8解码BF16。固定298-token完整流程18.954→18.933s，当前吞吐基本持平，allocator峰值34.777→27.267GiB。251项回归、自然扩写与重载通过。耗时分布、失败候选及硬件余量分析见 [全流程FP8](MOSS_FP8_SYSTEM_OPTIMIZATION_2026-11-12.md)。

- Moody MOSS E4M3主模型原生FP8适配已完成实际流程验证。同文件常驻BF16参考/原生FP8固定caption为13.993/12.810s（5 warm中位数），原用户含扩写warm中位数20.666s，初始222.10s慢路径已消除。修复重复量化元数据warning，文本沿用现有BF16配置。测试条件、质量变化和显存结果见 [FP8适配](MOSS_FP8_ADAPTATION_2026-11-11.md)。

- BF16对照基准为`a8f42df2`，DiT FP8基准为`2227f7d3`，当前部署为`148b252a`，本地产品分支`codex/gaudi2-hpu`。固定工作量5 warm中位数，独立298-token文本4.867s、完整MTP工作流19.978s、固定caption缓存14.386s、DiT14.134s。完整warm19.906–20.122s，cold64.665s，缓存14s目标仍差0.386s。
- 最终194 passed、1 CUDA skip，MTP off/3/tree3s1、512/768/1024、重载与原INT8通过，fallback为空。自然扩写与同seed图像细节会变化，质量范围限于已验收集合。完整证据见 [最新验收](MOSS_BF16_ATTENTION_AND_SUBMISSION_2026-11-11.md)。
- 各轮测试要点、收益、失败候选和下一步判断集中在 [BF16复盘与FP8路线](MOSS_OPTIMIZATION_STRATEGY_2026-11-09.md)。FP8全流程已补齐实测，接续优先级为attention数据流、量化/逐元素融合与文本提交粒度。
- 大Linear轮换374–404TF/s，旧六段DiT预算为MME9.225s、softmax2.398s、其它1.805s、间隙0.752s。最新文本改写后两步trace的Linear调用154→10、launch964→820。AIP50–70%需结合各阶段预算解释，见 [预算与利用率](MOSS_BF16_PLAN_COMPLETION_2026-11-11.md)。
- 原生FP8 GEMM已有本机算子验证，本轮已补充Moody主模型真实shape与同文件BF16对照，见FP8适配报告。INT8 ConvRot/W4A8当前走解量化计算，`_int_mm`出现CPU fallback。可复用接口与外部实现见 [算子调查](GAUDI2_INT8_FP8_KERNEL_SURVEY_2026-11-10.md) 和 [JChen源码核查](JCHEN_GAUDI2_KERNEL_REVIEW_2026-11-10.md)。
- BF16扩写激活与FP32普通编码分别验证，GDN状态FP32。`--disable-comfy-compiler`关闭区域编译。完整pipeline卸载稳定约2.50GiB，文本单独卸载约94MiB，操作约束见 `ENVIRONMENT.md`。
- 硬件算术下界和公开5090场景保留在 [对标记录](MOSS_HARDWARE_LIMIT_AND_5090_2026-11-10.md)。单次公开成绩与本机固定工作量基准各按其原条件解释。

## 驱动与模型存储

- 服务器重启后 `5.15.0-202`/`5.15.0-186` 双内核 DKMS、`hl-smi` Operational、BF16 HPU matmul 和 `/system_stats` 均通过。驱动安装脚本已改为检查全部可启动内核；19 项模拟回归及实际双内核只读检查通过，补丁在 `patches/install-gaudi2-all-boot-kernels.patch`。
- 两个 checkout 的模型已合并到共享目录，分类 symlink 和 `extra_model_paths.yaml` 验证通过；Gaudi checkout 实际加载 Qwen 2.1 Turbo 并出图，原始 ComfyUI 未启动。

## 已知边界与接续工作

- 验证范围不包括编辑/控制 LoRA、SLG、多卡、其它 VAE、其它 custom nodes、lazy/HPU Graph 或超大分辨率。区域编译已验证 Qwen3.5 deferred conv/GDN、普通 BF16 短步 MLP 与 MOSS attention/投影/FFN；batch=2 只做算子回归，没有完整批量工作流验收。
- tokenization、采样控制和已有 Flux RoPE float64 频率表仍可能走 CPU；“模型在 HPU”不等于没有任何 CPU 算子。
- 后续试点沿用固定caption/MTP程序、真实输入和多提示词质量集合。批量工作流、其它模型及未测功能的结论需独立验证。

## 接续工作流程

1. 读取本仓库 `AGENTS.md`、本文件、`PROGRESS.md`，再检查产品 checkout 的状态。
2. 检查服务器进程、`hl-smi`、磁盘和队列；HPU 单卡任务必须串行。
3. 只同步已审查的产品代码到 `/home/sonny/ComfyUI-Gaudi`，不触碰原始 `/home/sonny/ComfyUI`。
4. 失败先记录最小复现、版本、证据和结论；未验证的猜测明确标注。
