# Gaudi2 HPU 适配交接入口

更新时间：2026-11-08（Asia/Shanghai）。每次结束工作前更新本文件和 `PROGRESS.md`；阶段通过后分别提交适配仓库的代码/测试及本调查仓库的文档。不要把启动成功写成模型验证成功。

**文档归属：用户明确要求所有过程和个人环境信息仅保存在 `/Users/sonny/work/github/klzsysy/gaudi2`，不得进入 ComfyUI-Gaudi 仓库或其提交历史。**

## 目标与基线

- 分支：`codex/gaudi2-hpu`，起点 `fb2b26fd`（用户 fork 的 master，已含 Krea2 / QwenImage21）。
- 本地：`/Users/sonny/work/github/klzsysy/ComfyUI-Gaudi`。
- 服务器：`ssh sonny@192.168.10.21`；独立目录 `/home/sonny/ComfyUI-Gaudi`。
- **禁止修改服务器 `/home/sonny/ComfyUI`，包括 venv、配置、输出和模型。** 可只读复用模型，通过独立 `extra_model_paths.yaml` 配置。
- Habana 基础运行时：`/opt/gaudi-install/venv-1.24.2-481-ubuntu2204`；禁止用 pip 替换其 torch，测试依赖另建环境。
- 单卡 Gaudi2 96 GiB，驱动 1.24.2，Python 3.10，torch `2.13.0a0+git2469d18`；实际 bridge 默认 eager（`PT_HPU_LAZY_MODE=0`）。
- 既有调查在相邻 `gaudi2/GAUDI2_COMFYUI_HPU_ASSESSMENT.md`、`GAUDI2_COMFYUI_OPENSOURCE_SURVEY.md`。调查是线索，API 与算子能力以服务器实测为准。

## 当前状态

- 2026-11-08 服务器恢复，驱动/DKMS/Operational、共享模型和独立服务版本核对通过。补装 sysstat 与当前内核 linux-tools；TensorBoard/HPU 插件、Perfetto Python 原已存在，补齐原生 trace_processor；HPU profiler smoke 和 Perfetto SQL 通过。计划 review 修正了节点缓存、seed、同步扰动与 HBM 字节估算，开始 P1 严格基线。详见 [本轮记录](MOSS_HPU_PERFORMANCE_RUN_2026-11-08.md)。
- P1 首轮 B0/B1 各 5 次真实 warm 计算完成：median 137.282 / 40.183 s；所有请求 fallback 为空，327-token 扩写 ID 一致，图像检查通过。同步诊断覆盖 99.963% 请求时间；真实 MOSS HPU trace、原生 .hltv 和离线 parser 通过。厂商高层自动摘要仍有 MME 数据兼容性限制，当前用 Perfetto SQL。B2/B4 完整控制矩阵待继续，未实施新的推理优化；正常用户服务已恢复（8189，版本 `0690d18f`）。
- 低精度能力探针已补测：FP8 E4M3/E5M2 原生 HPU `torch.mm` 约 525.88/519.63 TFLOP/s，BF16 同 shape 275.76 TFLOP/s；对应公开 MME 理论值 865/865/432 TFLOP/s。INT8 `torch._int_mm` 发生 `aten::_int_mm` fallback，3.17 TFLOP/s 不作为 HPU 峰值；Gaudi2 公开资料没有可套用的 INT8 MME 峰值。详见 [本轮记录](MOSS_HPU_PERFORMANCE_RUN_2026-11-08.md)。
- 已正式开始 P2/P3。lm_head 解量化缓存和强制 HPU embedding 两个候选均未通过 MTP auto 的重复收益门槛，已停止本轮性能改动；trace 线索集中在 `aten::_copy_from`，需新调用归因后再继续。产品代码、部署配置和模型未改，服务已恢复（PID `109952`）。
- 复核 CLIP 日志：`dtype: torch.bfloat16` 是量化文本编码器的计算 dtype，权重仍为 INT8 `QuantizedTensor`；强制 native INT8 会命中 `torch._int_mm` CPU fallback。MTP auto 32-token 旁采样 HPU AIP 平均 12.65%，低负载主要来自小批量自回归和 host/device 控制同步，详见本轮记录。
- 2026-11-09 FP8 文本编码器转换验证：从现有 INT8 权重解量化生成独立 E4M3FN 文件；全模型和仅 lm_head FP8 都比 INT8→BF16 慢，未进入生产。当前 Habana/comfy-kitchen 栈不支持该路径作为优化。
- 2026-11-06 重启故障已修复并完成真实重启验证：新内核 `5.15.0-202` 的 Habana DKMS 模块与旧内核 `5.15.0-186` 均通过检查；重启后 `hl-smi` / Operational / BF16 matmul 通过，ComfyUI-Gaudi 已恢复并由 `/system_stats` 确认 `hpu:0 GAUDI2`。详见 [驱动恢复记录](DRIVER_RECOVERY_2026-11-06.md) 和 [重启验证记录](PROGRESS.md)。
- 已修复服务器驱动安装脚本只覆盖 `uname -r` 的遗漏：检查并处理全部可启动内核，19 项模拟回归及实际双内核只读检查通过。原文件已备份；驱动仓库提交 `8d291ee`（含 H13SSL-N 硬件与安装前验证文档），未 push。补丁及回退位置见上述恢复记录。
- 用户试用部署已修复 MOSS 扩写及输出头 CPU 回退：`0690d18f`，访问 http://192.168.10.21:8189 ，根目录 `start-hpu.sh` / `stop-hpu.sh` 管理服务；详见 ENVIRONMENT.md。

- 阶段一完成：`8326476f`。
- 阶段二完成：`fc1c891f`；HPU/bf16 完整生图，512/1024 和卸载重载已验证。
- 阶段三完成：`6db162dd`；Turbo 固定 8 步，512/1024、长短提示词、卸载重载已验证。
- 第四阶段完成：`21e98390`；MOSS-Image SFT bf16 + Qwen3.5 9B int8 ConvRot，复用已有 Flux2 VAE，512/1024 和重载验证通过，基础出图无新增核心补丁；用户试用后补测并修复 TextGenerate 的 HPU token 采样，详见 PROGRESS.md。
- MOSS 扩写性能修复：输出头遵守层自身精度策略，实测 HPU 不再回退 INT8 matmul 到 CPU；同一完整请求从 337 秒降至 135 秒。42 项解码回归通过（另 1 项 CUDA 测试跳过）。这不是全面性能优化完成的声明。
- 已建立 MOSS-Image HPU 性能调查计划：先冻结 1024² 用户基线 143.34 s，分离 TextGenerate/DiT/VAE，校准 HBM 与 GEMM 上限，再用重复实验和 roofline 指标归因；本轮只完成环境采集，没有启动新的性能测试。详见 [MOSS HPU 性能优化计划](MOSS_HPU_PERFORMANCE_PLAN_2026-11-06.md)。
- 已补充 Gaudi2 架构到 MOSS 的硬件映射和观测工具盘点：`hl-smi`、HPU profiler、`synprof_parser`、`habana_perf_tool`、`torch.profiler` HPU activity 和 fallback 计数器已具备或有运行证据；完整硬件时间线和可视化端仍待空卡验证，详见同一计划文档。
- 2026-11-06 服务器断电导致计划暂停；11-08 已恢复并按先工具、再 review、再 P1 的顺序继续。
- 详细数值错误、修复与证据见 `PROGRESS.md`。
- 验证为 eager + bf16 正确性基线，不是性能优化或所有模型/自定义节点支持声明。26 项回归通过，三个目标模型的文本/DiT/VAE 都使用 HPU；RoPE 频率表沿用已有 CPU float64 路径，不能宣称无任何 CPU 计算。
- 代码分支工作树干净，四个阶段均本地 commit，未 push。
- 成功图片：`results/krea2-hpu-1024.png`、`results/qwen21-turbo-1024.png`、`results/moss-sft-1024.png`；`*-failed`、`*-first` 及故障排查图片保留为失败证据。**`comfyui-hpu/results/` 已不在 git 中跟踪**（历史提交里的这些产物已从 git 历史清除，见 `.gitignore`），产物只保存在服务器部署目录与本地备份。
- 模型集中存储已完成：原两套模型已合并到 `/home/sonny/ComfyUI-Models/models`，两份 `extra_model_paths.yaml` 共同引用，checkout 内保留分类 symlink。Gaudi 版已实际加载 Qwen 2.1 Turbo 并出图；原版未启动。迁移、备份和验证证据见 [模型集中存储方案](MODEL_STORAGE_PLAN_2026-11-06.md)。
- ComfyUI-Manager 调查：`--enable-manager` 实际加载成功；启动日志的 `PyTorch is not installed` 是缺少 torchaudio 的误报，禁止为此安装 torchaudio；服务器的参数改动尚未进入代码仓库。
## 执行计划与验收

1. **阶段一：设备后端。** 原生 HPU 探测、设备枚举/名称、内存、同步、bf16 精度、安全关闭 CUDA 专用机制；独立环境启动 HTTP 服务，`/system_stats` 返回 HPU；通过真实 HPU matmul、attention、模型加载/卸载 smoke test；检查 CPU 路径。通过后 commit。
2. **阶段二：Krea2。** 核对权重和官方模板，优先已有权重；验证文本编码→采样→VAE→保存完整原生工作流。记录 seed、尺寸、步数、耗时、内存、图片、数值检查及已知 CPU fallback；至少重复一次并验证卸载。通过后 commit。
3. **阶段三：Qwen-Image-2.1 Turbo。** 获取匹配的原生模型、文本编码器、VAE；先小尺寸验证，再 Turbo 8 步完整生图；记录与阶段二相同的证据。通过后 commit。

4. **阶段四：MOSS-Image SFT。** 使用用户选定组件、复用现有 VAE；对照官方模板验证 512/1024、长短提示词和卸载重载，记录数值、图片及版本，通过后 commit。

先建立 eager + bf16 正确性基线，再考虑 fused attention、量化、compile/HPU Graph 性能。保持现有节点和模型加载接口；不通过全局 CUDA monkeypatch 隐藏设备类型。

## 接续工作流程

1. 读本调查仓库 `AGENTS.md`、本文件、`PROGRESS.md`，再读代码仓库 `AGENTS.md` 并检查两个仓库 git status/log。
2. 检查服务器进程、`hl-smi`、剩余磁盘，再启动任务；不要争用已有 HPU 作业。
3. 仅同步此 fork 到独立目录，不对原 `~/ComfyUI` 执行 rsync/install/git/输出操作。
4. 失败先记录完整日志位置、最小复现、版本和结论，再做最小修改。未验证的猜测明确标注。
5. shell 中有 `HF_ENDPOINT`、`HF_TOKEN` 和 `http_proxy_enable`；仅在需要下载时使用，禁止把令牌/代理凭据写入文档或输出。

官方 API 参考：https://docs.habana.ai/en/latest/PyTorch/Reference/Python_Packages.html
