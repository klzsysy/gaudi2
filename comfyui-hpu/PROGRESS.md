# 执行记录

## 2026-11-08：恢复性能计划，补齐可观测性并建立首轮严格基线

- 恢复后健康检查通过：`5.15.0-202-generic`、双内核 Habana DKMS installed、Operational、共享模型文件与独立服务 `0690d18f` 正常。队列为空后停止独立服务，串行执行 HPU 调查；原始 `/home/sonny/ComfyUI` 未操作。
- 新增 sysstat 与当前内核 linux-tools（5 个包，无升级/移除）；`pidstat/iostat/sar/perf` 可用。perf 保留 paranoid=4、使用单次 sudo。基础环境已有 TensorBoard/HPU 插件/Perfetto Python；补齐 native trace_processor，初次直连约 130 s 超时后使用已有代理成功。synprof_parser 修正进程级动态库路径；真实 .hltv 采集和离线转换通过。
- review 修正节点缓存造成假 warm、cold/warm 样本数、文本/扩散两个 seed、profiler/同步扰动、逻辑字节与 HBM 事务区分及原生 TextGenerate 计数来源。冻结保存的用户 API 图 SHA256；其 seed 与旧 143.34 s 记录不同，不计算旧值相对收益。
- B0 1 process cold + 5 warm、B1 1 transition + 5 warm、同步 B1 诊断均通过，共 13 请求；关键计算节点每次执行，warm 保留 loader。B0 warm median **137.282 s**（135.893–137.526），B1 **40.183 s**（39.690–40.314）。6 次 B0 均为 327 tokens 且 ID 一致，fallback 全为空，1024² 图片正常。
- 同步 B1 40.313 s，分段覆盖 **99.963%**；KSampler 38.006 s、步间隔 median 1.5212 s，VAE 0.136 s；latent/图像浮点 finite 检查通过。正常 B0 节点计时中位数为扩写 96.966 s、KSampler 37.960 s，后续优先归因扩写。
- BF16 大 GEMM 约 260.40 TFLOP/s，copy 读+写有效带宽 1949.35 GB/s；只是具体 eager 微基准，不是模型 MFU/MBU。真实用户 prompt 的 8-token 无 profiler/profile 为 4.1018/5.2150 s，ID 相同、fallback 空；930 MiB CPU+HPU trace 已在服务器导出并通过 Perfetto SQL。厂商高层摘要在带 step 的 MOSS trace 上仍报 MMEEvents 不可用，作为工具兼容性限制保留。
- 切换文本 probe 时曾过早获取设备，首次 acquire failed；保留日志并在旧进程退出、hl-smi 空卡后重试成功。调查服务与旁采样均已停止，正常服务恢复 PID **90386**、8189、`hpu:0 GAUDI2`、空队列。未改基础 torch、模型或推理代码。
- P1 尚未全部完成：B2/B4 完整控制矩阵、跨进程 cold 与更完整设备归因待继续。完整工具版本、路径、失败证据、结果和下一步见 [本轮记录](MOSS_HPU_PERFORMANCE_RUN_2026-11-08.md)；原始产物在服务器和本机忽略的 `results/performance-2026-11-08/`，不入 Git。
- 应用户建议补测低精度：独立空卡探针确认 FP8 E4M3/E5M2 `torch.mm` 原生 HPU，无 fallback，4096³ 分别 525.88/519.63 TFLOP/s；BF16 同探针 275.76 TFLOP/s。公开理论峰值 865/432 TFLOP/s，实测约 60.8/60.1/63.8%。INT8 `torch._int_mm` 仅 3.17 TFLOP/s 且 `aten::_int_mm` fallback 103 次，不能称 HPU INT8 性能；公开规格也没有可套用的 INT8 MME 峰值。E4M3FNUZ/E5M2FNUZ 不支持；FP8 custom op schema 存在但未在缺少 scale/layout contract 时硬测。服务已恢复 PID 92985。详见本轮记录低精度章节和 `probes/hpu_low_precision.py`。

## 2026-11-06：服务器断电，暂停性能计划

- 用户反馈服务器 `192.168.10.21` 已断电，本轮任务暂停；不再尝试 SSH、启动服务、运行微基准或 profiler。
- 断电前最后一次只读状态：内核 `5.15.0-202-generic`，Gaudi2 HL-225、驱动 `1.24.2-52a555d`，ComfyUI-Gaudi PID 5202，服务监听 `192.168.10.21:8189`，队列为空；`hl-smi` 约使用 22480/98304 MiB、功耗 83 W、温度 43–51°C，AIP 利用率 0%。
- 断电前确认的工具：`hl-smi`、`hl-prof-config`、`/opt/habanalabs/bin/synprof_configuration`、`synprof_parser`、`habana_perf_tool`、Habana `torch.profiler` HPU activity 和 `habana_frameworks.torch.utils.debug` fallback 计数接口；尚未在 MOSS 图上采集新的 HPU 硬件 trace。
- 断电前未执行任何新性能测试、未改服务器代码、未修改原始 `/home/sonny/ComfyUI`，也没有改变模型或运行配置。服务器恢复后必须先重新做驱动/设备/服务健康检查，再决定是否继续 P1。
- 计划与架构/工具信息见 [MOSS HPU 性能优化计划](MOSS_HPU_PERFORMANCE_PLAN_2026-11-06.md)。

## 2026-11-06：MOSS HPU 性能优化计划与环境冻结

- 按用户要求先不改代码、不启动新性能测试；只读采集独立 `/home/sonny/ComfyUI-Gaudi`、HL-225、驱动、PyTorch/ComfyUI 版本、当前进程参数、模型文件和用户目标工作流。
- 用户给出的目标基线为 MOSS 1024×1024、25 步、端到端 143.34 s。当前采集确认服务为 eager、队列为空，但 PID 5202 正持有单卡，因此没有启动会抢占 HPU 的微基准或 profiler。
- 已建立 B0 用户端到端、B1 无扩写、B2 文本单独、B3 DiT/VAE 分段、B4 尺寸/步数/缓存控制组，以及 HBM/GEMM 上限、HPU 状态、fallback、阶段耗时和质量验收契约。
- 计划文档：[MOSS HPU 性能优化计划](MOSS_HPU_PERFORMANCE_PLAN_2026-11-06.md)。其中明确区分 143.34 s 用户基线与此前 134.586 s 的非严格重放结果；后续先完成严格 warm/cold 重复基线，再进入单变量优化。
- 只读盘点补充了 Gaudi2 的 MME/TPC/HBM/SRAM/PCIe/RoCE 架构，以及本机工具状态：`hl-smi`、`hl-prof-config`/`synprof_parser`、`habana_perf_tool`、HPU `torch.profiler` activity 和 fallback 调试接口已具备；`torch-tb-profiler`/Perfetto 尚未确认，完整设备时间线尚未采集。

## 2026-11-06：重启后驱动与 ComfyUI-Gaudi 验证

- 按用户授权执行服务器重启。重启前确认 Gaudi 队列为空，运行 `--verify-only --skip-pytorch` 检查 `5.15.0-202`/`5.15.0-186` 两个内核，并优雅停止 PID 35778；没有中断生成任务。
- 服务器重启后约 2 分钟 SSH 恢复，运行内核为 `5.15.0-202-generic`。`hl-smi` 显示 HL-225 Operational、驱动 `1.24.2-52a555d`；重启后再次执行双内核 DKMS 检查通过。
- 独立 Habana Python 环境 BF16 HPU matmul 通过，最大误差 0；迁移后的 13 个工作流文件仍在新版目录。
- 使用 `~/ComfyUI-Gaudi/start-hpu.sh` 恢复服务，当前 PID 2995；`/system_stats` 返回 `hpu:0 GAUDI2`、总 HPU 内存 `101601996800` bytes。服务未配置开机自启，后续仍需手动启动。

## 2026-11-06：驱动 README 补充硬件验证并提交

- 按用户要求，在驱动仓库 README 的 Validated setup 新增 Supermicro H13SSL-N rev 1.01 / BIOS 3.8、AMD ES 100-000000897-03（32C/64T）、4×24 GB DDR5-4800、Gaudi2 HL-225 / Gen4 x16、Ubuntu 22.04.5、Habana 1.24.2 / Python 3.10 配置；保留原 GIGABYTE 平台结果。
- Before installation 新增本机已验证列：IOMMU identity、NIC 参数、两个内核 headers/DKMS、运行库与 Python 版本；未测量的供电/接线及未复测的完整重启明确标注。未再次运行模型。
- 驱动仓库提交 `8d291ee568e84291a7eee48b2515e345f076ebbb`：`Fix DKMS coverage for all bootable kernels`。包括本轮脚本修复、19 项回归测试与 README，共四个文件，未 push；原有未跟踪文件未提交。
- 最终补丁已同步到 `patches/install-gaudi2-all-boot-kernels.patch`。本调查仓库同时保存前两轮驱动恢复、修复与当前硬件核对记录；用户原有 `gaudi2-hardware.md` 保持未跟踪状态。

## 2026-11-06：修复驱动安装脚本的多内核遗漏

- 用户指定服务器驱动仓库 `scripts/install-gaudi2.sh`，基线 `ecbc579`。阶段 4/5 只使用 `uname -r`，健康驱动复用也绕过其它内核，因此脚本确实无法防止本次重启故障。
- 已在原路径修复：当前内核 + 有效 `/boot/vmlinuz-*` 镜像逐个检查/构建/安装/验证并刷新 initramfs；安装前先检查 headers；当前活动模块的复用与其它内核构建解耦。
- 新 helper 校验精确 DKMS 版本、五个非空模块、vermagic、DKMS 路径、Gaudi2 alias 和主模块补丁字符串，失败非零退出；不扫描已删除内核的残留 modules 目录。
- 19 项模拟命令回归通过。实际已部署入口执行 `--verify-only --skip-pytorch` 退出 0，`186`/`202` 两内核通过，Operational/运行库链接正常。语法检查和 diff 检查通过。没有重新运行完整安装或重启，没有加载模型。
- 原文件 tar 备份位于服务器 `/home/sonny/gaudi2/installer-backup.5PMyzd/`；可审查补丁为本仓库 `comfyui-hpu/patches/install-gaudi2-all-boot-kernels.patch`。代码尚未 commit/push；原有未跟踪的 Qwen 脚本/图片/文档未动。
- 详细设计边界、测试、回滚见 [驱动恢复记录](DRIVER_RECOVERY_2026-11-06.md)。原 ComfyUI 和独立 ComfyUI-Gaudi 产品代码、Python 环境均未改动。

## 2026-11-06：重启后驱动失效，已在线恢复

- 现象：PCI `1da3:1020` 可见，`hl-smi` 无设备。实际已启动 `5.15.0-202`，Habana DKMS 仅安装在旧 `5.15.0-186`，加载的内核自带驱动没有 Gaudi2 alias。
- 日志确认新内核于 11 月 5 日 09:21 安装，Habana 驱动同日 14:35 重装；厂商 postinst 的 dkms build/install 未指定目标内核，遗漏已安装但尚未启动的 `202`。
- 已为 `202` 成功编译/安装现有 `1.24.2-481`，卸载未绑定的内核自带模块，按 compat/cn/en/ib/主模块顺序加载，约 32 秒后固件和设备初始化完成。
- `hl-smi` 识别 HL-225 96 GiB，驱动 `1.24.2-52a555d`；sysfs 为 Operational；原 Habana Python 小型 BF16 HPU matmul 与 CPU 参考一致。未加载或验证生成模型。
- 备份并刷新当前 initramfs；原镜像没有 Habana .ko，并非旧模块藏在 initramfs 的问题。原有 modules-load、单卡 NIC 配置和 DKMS 自动安装 hook 保留。旧 `186` 驱动仍 installed。
- 无需重启；没有修改原 ComfyUI、venv、模型或 ComfyUI-Gaudi 产品代码；未启动 ComfyUI 服务，未新增自启或锁定内核。完整重启链路尚未实测。
- 详细证据、修复命令、备份位置和防复发步骤见 [DRIVER_RECOVERY_2026-11-06.md](DRIVER_RECOVERY_2026-11-06.md)。

## 2026-11-05：阶段一完成，阶段二待执行

- 已创建 `codex/gaudi2-hpu`，工作树起始干净。上游基线 `fb2b26fd`。
- SSH 成功；HPU 空闲；home 剩余约 1.1 TiB。原 `~/ComfyUI` 未修改。
- 实测 torch 在 import 时已注册 `torch.hpu`；可用设备为 `GAUDI2`。
- `memory_stats()` 提供 `Limit=101601996800`、`InUse=5767296`；当前版本也有 `mem_get_info`、`memory_reserved`、`memory_allocated`，**没有 `empty_cache`**。缓存清理使用同步，不调用不存在的 API。
- 当前 bridge 默认 eager，与旧调查中“默认 lazy”的概括不同；启动脚本显式固定 eager 以可复现。
- 基础环境 torchvision 0.16.2 的 torch 声明冲突已知；必须隔离依赖安装，不覆盖原运行时。

### 已实现与验证

- `comfy/model_management.py` 增加 HPU 探测、枚举、名称、内存、同步、设备设置、bf16 策略；关闭本 fork 尚未验证的 fp16/fp8/int8/fp64、pinned memory 和 async offload。原生 PyTorch attention 可用，无 CUDA migration monkeypatch。
- 独立 `.venv` 通过 `gaudi-base.pth` 引用 Habana 基础 site-packages，新依赖仅装入此 `.venv`。原 torch 和原 ComfyUI 未改。
- `scripts/gaudi/smoke.py` 实机 PASS：bf16 matmul 与 CPU 参考比对、ComfyUI attention 与 CPU 参考比对、ModelPatcher 加载到 HPU/卸载到 CPU、内存查询/同步。日志：服务器 `test-results/smoke.log`。
- `tests-unit/comfy_test/model_management_test.py`：7 passed（包括已有 NPU 测试和新增 HPU/CPU 策略测试）。日志 `test-results/unit.log`。
- HTTP `/`、`/object_info`、`/system_stats` 成功；设备 `hpu:0 GAUDI2`，总可用内存 101601996800 bytes。证据 `test-results/system-stats.json`、`server.log`。
- `main.py --cpu --disable-all-custom-nodes --quick-test-for-ci` 成功。CPU 启动不分配 HPU。
- 未验证其他实体加速器，未验证阶段二/三模型，阶段一不代表模型已跑通。

### 失败路径与修复（避免重试）

1. 非交互 SSH 不加载 `/etc/profile.d/habanalabs.sh`，缺少 `GC_KERNEL_PATH`；matmul 可成功，但 attention 图编译 `synStatus 26`。开启 `ENABLE_CONSOLE=true LOG_LEVEL_ALL=3` 才显示 `Environment variable GC_KERNEL_PATH is undefined`。显式设置 `/usr/lib/habanalabs/libtpc_kernels.so` 后 smoke 完整通过。**无需更改 attention 算法。**
2. 同时运行服务和独立 HPU Python 会报 `Device acquire failed`；当前单卡独占，测试串行执行，停止服务并等待退出后再跑 smoke。
3. 服务器未安装 rsync；使用 `COPYFILE_DISABLE=1 tar ... | ssh ... tar ...` 同步，或小文件用 scp，目标仅 `~/ComfyUI-Gaudi`。
4. 国内 pip 镜像缺最新模板依赖，resolver 将模板降至 0.11.63。使用 shell 的 `http_proxy_enable` + 官方 PyPI 固定安装 `comfyui-workflow-templates==0.11.102` 后解决。
5. `pip check` 剩一个继承冲突：torchvision 0.16.2 声明 torch==2.1.2，而 Habana torch 为 2.13.0a0；torchvision image 扩展不能加载。当前启动/矩阵乘/attention/加载测试通过；后续涉及 torchvision 扩展时需独立处理，不能因此覆盖 Habana torch。

下一步：通过原生工作流读取已有 Krea2 Turbo int8 ConvRot、Qwen3VL 4B fp8、Qwen VAE，验证是否需要 CPU 解量化/HPU 算子修复。

### 文档归属修正与阶段提交

- 用户明确要求过程文档仅在 `gaudi2`；已迁移到本目录，恢复 ComfyUI 的原始 AGENTS.md，并 amend 未推送的阶段一提交。
- 当前有效阶段一代码提交：`8326476f`（替代原 `0b15a286`），ComfyUI 分支历史不再含过程文档或个人服务器路径。未推送远端。
- 进程 SIGTERM 后需等待退出再重启；立即启动会因旧进程仍持卡而报 acquire failed。
- 阶段二：已开始 512×512、seed 42、8 步 Euler/simple 的 Krea2 原生 API 流程；输出独立目录 `ComfyUI-Gaudi/output/gaudi/`。

## 阶段二进行中：数值验证不能省略

- Krea2 首次完整流程成功（文本 CPU fp16，DiT/VAE HPU bf16）：512×512、8 步、seed42，61.36 s；红色茶壶图片符合提示。证据 `results/krea2-first.png` 和 history。
- 补 text_encoder_device/dtype 后，文本也在 HPU bf16，流程 17.41 s 返回 success，**但保存时出现 invalid value encountered in cast，图片数值异常，不能作为通过**。正在定位文本编码器 HPU 非有限值。
- 同时准备阶段三匹配权重，只下载到独立 models：Comfy-Org/Qwen-Image-2.1，revision `aacf31a6d48e707f352ecba714cee0680da3479b`。Turbo bf16 14230280616 bytes、Qwen3VL 8B int8 ConvRot 9350798360 bytes、VAE bf16 675509688 bytes。下载日志服务器 `test-results/qwen-download.log`。不需要 prompt enhancer 模型。

## 阶段二完成（2026-11-05）

- 数值根因已实测：`[240,256,288,320,384,448]` 存为 E4M3FN 后在 HPU 转 float 得到 `[240,inf,nan,nan,nan,nan]`。第一处非有限输出位于 Qwen3VL 第 0 层 q_proj。不是 prompt、采样或 VAE 问题。
- 修复在 `comfy/quant_ops.py::_TensorCoreFP8LayoutBase.dequantize`：HPU E4M3FN 通过完整 256 值解码表映射存储字节，保持原始权重/scale/量化状态，返回正常计算精度；其余设备走现有 kitchen 路径。表每次临时创建，不引入持久显存缓存。代价为解码时 int64 索引临时内存；当前正确性优先，后续可优化。
- `text_encoder_device` 允许 normal/high VRAM 的 HPU，默认 bf16；low VRAM 仍走原 CPU 策略。
- 单测 **10 passed**；包括全部 256 FP8 字节在 bf16/fp32 下与 CPU **逐值精确一致**（保留合法 NaN 编码），以及 HPU/CPU/NPU 策略回归。
- 原生已有权重无需转换文件：Krea2 Turbo int8 ConvRot + Qwen3VL 4B fp8 scaled + Qwen VAE。文本编码器、DiT、VAE 均日志确认 HPU/bf16；int8/FP8 存储通过解量化计算，不宣称原生量化 GEMM 加速。

| 测试 | 配置 | 服务端耗时 | 结果 |
|---|---|---|---|
| 修复后首次 | 512², seed42, 8 steps, Euler/simple, cfg1 | 15.02 s | 红茶壶正常，无 NaN warning |
| 第二分辨率 | 1024², seed43，同上 | 10.93 s | 红茶壶正常，图像标准差 75.36 |
| 卸载后重载 | 512², seed44，提示改蓝茶壶 | 13.75 s | 成功，触发文本/DiT/VAE 重新加载 |

- `/free` 后 HPU `InUse` 降至 807141504 bytes，再重载成功；不把采样 tqdm 时间当端到端耗时（HPU 异步执行）。上述非严格性能基准，后续比较应控制缓存和编译状态。
- 图片、history、完整服务日志见 `results/krea2-*`；额外 1024/reload 工作流见 `workflows/`。基础工作流和提交脚本位于代码仓库 `scripts/gaudi/`，不含个人环境信息。
- 阶段二代码提交：`fc1c891f`。阶段三正在下载匹配权重；官方 Turbo 使用 **ManualSigmas** 固定 8 步序列，不能直接沿用 Krea2 的 simple scheduler。

## 阶段三进行中：流程成功但图片异常

- 三个官方文件已完整下载至独立 models，revision/大小见前文。使用官方 ManualSigmas 8 步，CLIP/DiT/VAE 都在 HPU/bf16。
- 首次 512（14.41 s）和 1024（7.97 s）HTTP history success，但图片为异常纹理，**第三阶段未通过**。初始图片保留作失败证据，不作为结果展示。
- 禁用 Qwen prefix cache 仍异常；换 CPU 文本编码器仍异常，排除这两条简单修复路径。采样 latent 已保存于服务器 output/gaudi/qwen21-cache-off_00001_.latent，shape [1,64,32,32]，std 4.4621；这是 safetensors 文件，扩展名 .latent，须用 safetensors.torch.load_file，不能用按扩展名判断的 load_torch_file。
- probes/qwen_compare.py：用完整实际 DiT 权重、固定小尺寸随机输入对比 CPU/HPU，中间层均有限；第一层输出相对 RMS 差 0.020，最终输出相对 RMS 差 0.112。尚不能据此解释异常图，继续隔离 VAE。
- 浏览器经 SSH tunnel 验证 UI 和模板列表正常，console error 为空。隧道 127.0.0.1:8189；后续服务停止时页面断开属预期。

### Qwen VAE 根因与修复验证

- 同一 latent 经 **CPU fp32 VAE** 正确得到红茶壶，证明 HPU 文本与 DiT 已产生有效潜变量；对应 `results/qwen-cpu-vae.png`。
- HPU VAE fp32 仍错误；取消 strip 分块但保留原地 add_ 仍错误；给 add_ 的输入加 contiguous() 也仍错误。不要重复这些路径。
- CPU/HPU fp32 逐层对照：上采样主分支、shortcut 分支分别正确，但合并后突然相对 RMS 误差 >1；非原地相加后完整 512 VAE 每个采样检查点相对误差四舍五入到 5 位均为 0。
- 最小复现：HPU Conv2d 输出 unsqueeze 为 5D，与 DupUp3D 的非连续切片做 narrow(...).add_(...)；4 通道最大绝对错误 48，64 通道 993。单纯 ones 张量没有卷积 layout metadata，不复现。
- 尝试 dst.copy_(dst+y) 保持原地语义，会触发 HPU `_copy_from` schema 编译错误，也不可用。
- 最小修复：Wan 2.2 `strip_apply` 对 HPU 的 out 累加使用 `out + strip_apply(..., out=None)`；保留分块计算，返回新结果，避免对原卷积视图写入。仅新 HPU 分支，已有 CPU/CUDA 行为保留。代价为多一份残差输出临时内存。这个内部 helper 的 HPU out 参数不再被原地修改，调用者都使用其返回值。
- 增加真实 HPU/CPU、分块/不分块回归测试；正在完整工作流重测。

## 阶段三完成（2026-11-05）

代码提交：`6db162dd`，独立测试服务已运行此版本。修复仅 3 行 HPU VAE 残差分支，并附可复现工作流及回归测试。

| 测试 | 配置 | 服务端端到端耗时 | 结果 |
|---|---|---|---|
| 修复后首次 | 512², seed42, 红茶壶短提示词 | 12.91 s | 正确 RGBA 图片 |
| 大尺寸与长提示词 | 1024², seed43, 蓝茶壶/茶杯/点心/窗光 | 7.85 s | 构图与描述吻合 |
| 卸载后重新加载 | 512², seed44, 绿茶壶 | 11.47 s | 文本/DiT/VAE 重载并正常出图 |

- 三次均使用 HPU 文本编码、HPU DiT、HPU VAE，bf16；未设置 cpu-vae，未使用自定义节点。
- `/free` 后 `torch_vram_total`（HPU InUse）为 911596416 bytes，重载成功。结束时再次释放模型，服务保持空闲。
- 同一潜变量的 CPU fp32 / HPU bf16 VAE PNG 对照：MAE 0.2102/255，PSNR 54.49 dB（RGBA 包含 alpha，非整套生成器一致性基准）。
- 14 项测试通过，包括已有 NPU 回归、CPU策略、HPU FP8 全范围和 VAE 带卷积布局元数据的分块/不分块残差。
- 最终图片：`results/qwen21-turbo-512.png`、`qwen21-turbo-1024.png`、`qwen21-reload.png`；完整 history/日志：`qwen-fixed.json`、`qwen-1024-fixed.json`、`qwen-reload.json`、`qwen-fixed-server.log`。
- 原始 `~/ComfyUI` 只读取已有模型，未执行任何安装、改文件、配置、输出或 Git 操作。新增 Qwen 权重、venv、日志和输出全部在独立目录。

## 当前交接与后续范围

- 三个已授权阶段均完成；代码分支 `codex/gaudi2-hpu` 的顺序提交为 `8326476f` → `fc1c891f` → `6db162dd`；未 push。源码工作树干净。
- 本调查仓库原有暂存的 `GAUDI2_COMFYUI_OPENSOURCE_SURVEY.md` 保留原状，未夹带到阶段文档提交。
- 此次为正确性基线：eager 模式、bf16、串行 offload、禁用 Dynamic VRAM/pinned memory/custom nodes。INT8/FP8 权重可加载但使用解量化计算，并非原生量化加速。
- 已知 CPU 工作：tokenization、噪声/采样控制、已有 Flux RoPE float64 频率表回落；文档中的“模型在 HPU”不意味着没有任何 CPU 算子。
- 继承的 torchvision 0.16.2 版本声明冲突/图像扩展 warning 尚存，不影响已验证模型；不得为消除 warning 替换 Habana torch。
- 尚未验证：图像编辑、LoRA、批量/多卡、其他 VAE/模型、custom nodes、lazy/torch.compile/HPU Graph、超大分辨率、完整 CPU/CUDA/ROCm 等实体设备回归。不要扩大已验证范围。
- 后续如需优化：先固定输入并用同步后端到端计时记录吞吐与峰值显存，再考虑 fused attention、FP8 全范围解码临时内存、HPU Graph；保留本次正确输出做回归。

## 第四阶段进行中：MOSS-Image SFT

用户选定：`diffusion_models/moss_image_sft_bf16.safetensors`、`text_encoders/qwen3.5_9b_int8_convrot.safetensors`、`vae/flux2-vae.safetensors`。在原适配分支继续，完成后 commit。

- 官方资源仓库 `Comfy-Org/MOSS-Image`，revision `2929efe3cdd506c0a430bcfca70a9a8d805cc095`。新模型 10164655808 bytes，文本编码器 9748121532 bytes；完整 SHA256 存服务器 `test-results/moss-resources.json`。
- 已找到原目录 VAE，**复用而不下载**：`/home/sonny/ComfyUI/models/vae/flux2-vae.safetensors`，336213556 bytes，SHA256 `d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5`。其封装大小/哈希与 MOSS 仓库文件不同，需核对结构并通过实际 VAE 验证，不以文件名推断字节完全相同。
- 官方已安装模板：image_moss_image_t2i.json；原生 CLIPLoader 类型 moss_image，EmptyFlux2LatentImage，25 步 er_sde/simple，cfg2.5；使用用户选择的 bf16/int8/完整 VAE 替换模板默认量化/小 VAE，关闭可选 prompt enhancer 与 skip-layer guidance 建立基线。
- 用户再次明确：**下载必须优先使用服务器已有 HF_ENDPOINT、HF_TOKEN；不要默认覆盖 HF_ENDPOINT 为官方站点。** 初次下载误沿用上阶段官方站点+代理配置，现停止并按已有环境续传。HF_TOKEN 由 huggingface_hub 自动读取，不打印或记录值。只有明确失败证据才考虑备用来源。

### 第四阶段资源与预检完成

- 使用已有 HF_ENDPOINT/HF_TOKEN 的下载已完成，两个新权重 SHA256 均匹配官方 LFS 元数据。没有下载完整 VAE；只读取官方 safetensors header 做结构核对。
- 旧 VAE 是 Diffusers 命名；经原有 `convert_vae_state_dict` 和 quant_conv/post_quant_conv 前缀归一化后，251 个张量的名称/shape/dtype 与官方文件全部一致。完整字节哈希不同，不宣称权重 payload 已逐字节比较。
- 实际 HPU bf16 VAEEncode→VAEDecode 成功，0.45 s；图片 `results/moss-vae-roundtrip.png`，结构核对 `results/moss-vae-reuse.json`。
- Qwen3.5 `torch_chunk_gated_delta_rule` 在长度 17、65、128 与 CPU 参考对照通过，输出有限，最大绝对差分别 0.0004883、0.0009766、0.0009766；包括 recurrent state 比对。探针 `probes/moss_delta.py`，日志 `results/moss-delta.log`。
- 正在执行 MOSS SFT 512²、seed42、25步 er_sde/simple、cfg2.5 的完整原生工作流。此时尚未宣称阶段四通过。

## 第四阶段完成：MOSS-Image SFT（2026-11-05）

代码提交：`21e98390`（`codex/gaudi2-hpu`），新增通用原生 API 工作流 `scripts/gaudi/moss-image-sft-api.json`。既有 HPU 适配足够运行选定组合，本阶段**没有新增核心代码补丁**，没有重复运行与此次仅新增工作流无关的整套单测。

| 验证 | 配置 | 服务端端到端耗时 | 结果 |
|---|---|---|---|
| 首次生图 | 512², seed42, 红茶壶短提示词 | 23.70 s | 正常 RGB 图片，文本/DiT/VAE HPU bf16 |
| 大尺寸长提示词 | 1024², seed43, 蓝茶壶/茶杯/点心/窗光 | 34.11 s | 画面与提示吻合，无黑图/异常纹理 |
| 卸载后重载 | 512², seed44, 绿茶壶 | 24.28 s | 日志确认三个组件重新加载，正常生图 |

- 都使用 **25 步 er_sde、simple、cfg2.5**。首次生图前做过独立 VAE roundtrip，因此首次时间不是所有组件冷启动基准；1024 运行保留模型加载缓存。未宣称严格性能或跨设备等价性基准。
- 现有 Flux2 VAE 复用成功，roundtrip PSNR 43.23 dB（8-bit RGB）。未下载 VAE 权重 payload，原 `/home/sonny/ComfyUI` 没有改动。
- 512/1024/重载图片像素范围均 0–255，标准差分别 67.14、72.39、64.14。图片逐张目视检查正常，服务日志没有非有限值警告。
- 卸载后 InUse 从 22767054080 bytes 最终下降到 2686190976 bytes。释放异步进行，中途可看到约 11 GB。早期探针用“低于 2 GiB”作为成功阈值导致误报；该阈值不适用于本次暖机后的常驻占用。以释放前后差值、稳定后读数和下一次加载日志为依据。尚未对剩余约 2.50 GiB 的运行时占用做归因或长时间泄漏测试。
- 第一轮下载误覆盖了 HF_ENDPOINT；按用户纠正停止后，以已有 HF_ENDPOINT/HF_TOKEN 完成两个新权重并校验 SHA256。SDK 为不同下载路径创建了独立临时文件，不能声称所有旧分块都被续用；验证完成后已清理旧站点遗留的 4844421120 bytes 不完整临时文件。后续偏好已写入本仓库 AGENTS.md 和 ENVIRONMENT.md。
- 资源清单 `results/moss-resources.json`；复用判断 `moss-vae-reuse.json`；下载校验 `moss-download-configured.log`；测试 history `moss-first.json`、`moss-1024.json`、`moss-reload.json`；完整服务日志 `moss-server.log`；内存 `moss-unload-memory.json`。
- 成功图片：`results/moss-sft-512.png`、`results/moss-sft-1024.png`、`results/moss-sft-reload.png`。长提示词/重载/VAE 单独工作流在 `workflows/moss-*.json`，预检脚本在 `probes/moss_delta.py`。
- 当前四阶段代码均已本地提交，未 push；服务保持运行且已释放模型。验证范围为选定的 SFT 文生图组合，不包括 MOSS Base、编辑/控制 LoRA、SLG、prompt enhancer 或其它量化变体。

## 用户试用部署（2026-11-05）

- 用户授权部署当前验证版并提供启停脚本。部署核心为第四阶段 `21e98390`，管理脚本提交 `3bda1351`；源码没有新的推理变更。
- 独立部署位置仍为 `/home/sonny/ComfyUI-Gaudi`，原 `/home/sonny/ComfyUI` 未改动。先确认队列为空，再停止此前实验进程；以代码仓库 `git archive HEAD` 同步已提交版本，保留独立 venv、models、output 和 extra_model_paths.yaml。
- `start-hpu.sh` 后台启动、等待 HTTP 就绪、记录 PID 和日志；同目录/同 main.py 进程校验、flock 避免重复管理、端口占用预检。`stop-hpu.sh` 仅对匹配本目录的已记录 main.py 进程发送 SIGTERM，并等待退出；不会批量 pkill 或停止其他目录的服务。
- 服务器 `.gaudi.env` 设置 `COMFYUI_LISTEN=192.168.10.21`、`COMFYUI_PORT=8189`，因此用户可直接访问局域网 URL，不再需要 SSH tunnel。代码脚本默认仍只绑定 127.0.0.1；服务器地址仅保存在本调查仓库 deployment/gaudi.env 和服务器忽略的本地配置中。
- PID `.run/hpu.pid`，版本 `.run/deployed-version`，日志 `logs/hpu.log`，都不进入代码 Git。脚本继承当前 shell 环境，不覆盖 HF_ENDPOINT/HF_TOKEN，不把凭据写入配置。
- 已实测：后台就绪、从当前电脑访问 LAN /system_stats（HPU GAUDI2）、重复启动不产生第二进程、正常停止、重复停止、过期 PID 清理、伪造为其他 sleep 进程的 PID 不会误杀、端口占用在启动 HPU 进程前被拒绝。
- 最终以新脚本重新启动，供用户试用。代码与部署记录仅本地提交，未 push；未创建开机自启服务。
- 最终确认：新脚本启动 PID 289333，局域网首页 HTTP 200，/system_stats 返回 hpu:0 GAUDI2，UNETLoader 与 VAELoader 下拉资源均包含三个已验证模型组合；证据 results/deployment-system-stats.json。

## 用户试用问题：MOSS 提示词扩写（2026-11-05，已修复并部署）

- 用户在官方 MOSS 模板启用 TextGenerate 提示词扩写，`mtp=auto`、sampling on、top_k=64、top_p=0.95、min_p=0.05、temperature=0.7、max_length=4096。在 Qwen3.5 `_generate_mtp` 首个 `nt_buf[0].item()` 同步时触发 synStatus 26。
- 之前第四阶段只测了关闭 enhancer 的基础文生图。该结果真实但覆盖范围不足，不能推广为官方模板默认扩写路径可用；本轮补齐用户实际工作流，不能以关闭扩写作为修复验收。
- 已通过 `/history` 保存失败请求 `5007afb6-2c25-447d-b36d-5db621adf12d` 和相关 history：`results/moss-textgen-user-failure.json`；原始展开后的 API 图：`workflows/moss-user-textgen.json`。用户图含 ResolutionSelector、TextGenerate、PreviewAny、ComfySwitchNode，分辨率 1 MP 正方形，25 步 er_sde/simple、cfg2.5、SLG 关闭。
- 队列空闲后停止独立服务 PID 289333；原 ComfyUI 未动。纯随机 logits 即可在 `BaseGenerate.sample_token` 复现，无需权重、MTP 或完整 transformer。
- 开启临时诊断 `ENABLE_CONSOLE=true LOG_LEVEL_ALL=3` 得到底层错误：`gather_elements_fwd_i32` 收到 int64 数据，`GLUE_INCOMPATIBLE_DATA_TYPE`。触发点是 top-k 采样后的 `top_indices.gather(1, next_token)`，不是 `.item()` 本身，也不是缺 GC_KERNEL_PATH。
- 最小候选修复位于 `comfy/text_encoders/llama.py::BaseGenerate.sample_token`：仅 HPU 将词表索引转 int32 后 gather，再转回 torch.long；采样、模型计算、返回值仍在 HPU，不关闭 MTP、不移动文本模型到 CPU。词表索引 248320 范围可由 int32 精确表示。
- 独立采样/索引探针证据 `results/moss-sampling-probe.log`（旧路径失败）、`results/moss-gather-probe.log`（候选路径成功）。随后按真实用户完整工作流重跑，验收见下节。

### 原始工作流通过与补充测试

- 代码修复提交 `0139b1da`，只修改 shared `BaseGenerate.sample_token` 的 HPU top-k 索引读取并新增测试。API、工作流参数和模型权重均未改变。采样随机数仍使用原 HPU generator；未引入 CPU 采样。
- 原样重放用户失败 API 图（包括完整 system prompt、实际 prompt、seed、max_length=4096、mtp=auto、sampling on、三个模型文件、25 步与 1024² 分辨率），请求 `87a58349-b722-40c5-b8ba-ba14afe7a4f9` 成功。扩写约 251 tokens 后正常结束，得到可读、符合原提示的英文段落，随后正常完成出图。
- 端到端 350.39 s（服务日志）；其中文本生成进度条约 305 s。此轮开启诊断日志，仅用于正确性验证，不作为性能基准；不能宣传为扩写高性能支持。尚未做专门的自回归性能优化。
- 图片 `results/moss-user-textgen-fixed.png`：1024×1024，白色气球袖大衣、夜间剧院、MOSS 灯牌、湿地反光正确，已目视检查。完整 history `results/moss-user-textgen-fix1.json` 含扩写文本和图片路径；`results/moss-textgen-fixed-server.log` 是诊断服务日志中 ComfyUI 事件和 token 进度的筛选摘录；完整原始日志仍在服务器 logs/hpu.log。
- 单卡停止服务后运行回归：**26 passed**。原有 14 项，加 CPU/HPU × bf16/fp32 × top_k=0/1/64 的 12 项采样测试；检查高位词表编号、batch 维度、设备/torch.long 返回类型、相同设备固定 seed 可重现及 top_k=1 精确结果。不要求 CPU/HPU 随机序列相同。
- 已将已提交代码通过 git archive 部署，恢复不带诊断环境变量的根目录启动脚本，进程 PID 302801。重启后的短提示词扩写与完整出图也已通过，mtp=off，覆盖另一生成入口。

- 重启后的补测：提示为红茶壶，短 system prompt，max_length=256、mtp=off、sampling on、seed42，512²、25 步。生成约 80 tokens 的完整英文段落（进度条最后显示 79/256），随后正常出图，服务端端到端 125.09 s；说明关闭诊断日志后自回归生成仍偏慢，后续性能优化应另设基准。证据 `results/moss-textgen-reload.json`、`moss-textgen-reload.png`，可复现图 `workflows/moss-textgen-reload.json`。
- 最终状态：PID 302801、http://192.168.10.21:8189 在线且队列为空；`PT_HPU_LAZY_MODE=0`，临时 `ENABLE_CONSOLE` / `LOG_LEVEL_ALL` 均未设置；版本文件为 `0139b1da33fa72e78f07bd1e951a254e7bec0262`。系统状态保存在 `results/moss-textgen-deployment.json`。没有强制释放用户服务模型，当前暖机占用约 19.3 GiB。
- 原始 `~/ComfyUI` 未修改；代码仓库只提交产品修复与测试，所有复现图、日志和调查记录均只在 gaudi2；未 push。本仓库用户原有暂存的调查文档保持不动。

## MOSS 扩写性能问题（2026-11-06，主要 CPU 回退问题已修复并部署）

- 用户反馈约 1.28 秒/token，hl-smi 利用率为 0%。连续 12 次实际采样显示 0%～26% 交替，功耗约 83～159W；模型不是完全没有在 HPU 运行，但存在很长空闲间隙。证据 `results/moss-performance-live-samples.json`。等待用户任务完成、队列空闲后停止 PID 302801 做独立诊断，没有中断其正在生成的结果。
- 新建 `probes/moss_performance.py`，加载实际 Qwen3.5 9B INT8 ConvRot，固定短提示词/seed42/sampling on/mtp off，2-token 暖机后采样 8 tokens。输出头是 MixedPrecisionOps.Linear，`_full_precision_mm=True`，权重已在 HPU。
- 原版 8 tokens 10.19 s；CPU profiler 中 `aten::_int_mm` 7 次合计约 7.2 s，包含大量 `_copy_from`。Habana `_get_fallback_op_count()` 确认 `aten::_int_mm` CPU fallback 共 8 次（含暖机）。调用栈定位 `BaseGenerate.logits` 直接 `F.linear`，绕过 Linear.forward 的 `_full_precision_mm`；涉及 [1,4096] × [4096,248320]，约 1 GB 的 INT8 词表矩阵每步走 CPU fallback。MTP 的 verify_logits 也直接使用 F.linear，存在同类问题。
- 候选修复：存在独立 lm_head 时调用该层本身，复用原有量化/精度/权重 patch 策略；普通 logits 和 Qwen3.5 MTP verify 两处改动。共享 embedding 的原逻辑保持原状。没有开启不支持的 INT8 compute，也没有引入跨执行权重缓存。
- 修复后同样 8 tokens 4.47 s，token ID 逐个相同，CPU fallback 计数为空。剩余 `_copy_from` 大头是单个 token 标量回读时等待设备完成；不能把 profiler 所有 copy 时间都解释为主机数据搬运。ConvRot 权重仍逐次解量化，后续可进一步优化，但不应在定位前盲目缓存或更改模型数学。
- CPU Qwen3.5 解码回归 **42 passed, 1 skipped**：新增 3 个量化 lm_head 测试，覆盖普通/auto MTP/tree2s1，禁用 INT8 matmul 后应与显式反量化权重输出一致；原有 greedy、sampled、tree、KV 等测试均通过。唯一 skip 要求实体 CUDA DynamicVRAM。随后验证了真实 HPU MTP 的 fallback 计数、完整默认工作流和利用率，结果如下。
- Profiler 导出接口有继承不兼容：外层 `prof.export_chrome_trace()` 向 Habana wrapper 多传一个参数，TypeError；改用 `prof.profiler.export_chrome_trace(path)` 已导出。原始 trace 各约 146MB 留在独立服务器 test-results，不复制到 Git；性能表和摘要进入本仓库。

### 性能修复验收与最终部署

- 代码提交 `0690d18f`，仅 llama.py、qwen35.py 的 lm_head 调用及 qwen35_decode_test.py 的 3 项回归测试；复用层本身处理精度/量化/patch，不引入 HPU 私有缓存或额外模型副本。共享 embedding 作为输出头的其它模型仍保持原路径，未宣称这次修复覆盖所有模型。
- 真实 HPU MTP auto：2-token 暖机后固定生成 32 tokens，10.412 s（约 3.07 token/s），`fallback_counts {}`。日志 `results/moss-head-fixed-mtp.log`；前一普通生成 profile 的 8 tokens 从 10.1917 s → 4.4680 s（2.28 倍），8 个 token ID 完全一致。前后均是相同模型、输入、seed 与脚本参数；CPU profiler 有开销，不能当纯生成吞吐。
- 用户本轮已完成的请求 `1c14ab46-05a1-47ab-9e89-53348da828b3` 保存于 `results/moss-performance-user-history.json`，耗时 337.013 s。使用其完全相同的展开 API 图（包括 seed=133401810542576）重放，见 `workflows/moss-performance-full.json`；修复版完成 **134.586 s**（约 2.50 倍端到端提速）。旧请求已有模型缓存，新服务为重启后的加载；不是多轮严格性能基准。
- 新完整扩写最后进度为 307/4096、约 89 s、3.44 token/s，较旧输出更长；max_length=4096 是上限，实际遇到结束 token 即停。新输出仍是语义正确的完整英文段落，1024² 图片中的白大衣/MOSS 剧院/湿地反光正常，目视检查通过。结果 `moss-performance-fixed-full.json`、`moss-performance-fixed.png`；服务日志 `moss-performance-fixed-server.log`。
- 连续采样 90 次，其中有本次扩写进度的 84 次用于统计：HPU 利用率 min 0%、max 57%、mean 21.96%，9 次为 0%。证据 `moss-performance-fixed-utilization.json`。与修复前 12 次采样的 0～26% 比较只能说明活跃程度改善，不能当严格配对的利用率基准；也不能宣称利用率不再为 0% 或已充分利用 HPU。
- 量化权重仍按现有逻辑临时解量化为计算精度，eager Python 调度也仍存在；进一步提升吞吐需要独立的设备时间线/算子优化评测。本轮仅完成已实证的输出头 CPU fallback 修复，未实施盲目切 lazy、全模型 BF16 常驻缓存或改变 ConvRot 数学的方案。
- 已通过 git archive 部署已提交版本并用正常根目录脚本启动，PID 325542，版本文件为 `0690d18f` 对应完整 SHA；最终队列为空，http://192.168.10.21:8189 可用。系统状态 `moss-performance-deployment.json`，汇总 `moss-performance-summary.json`。
- 代码、文档分别本地提交，未 push；原始 ~/ComfyUI 未改动，用户原有暂存的 GAUDI2_COMFYUI_OPENSOURCE_SURVEY.md 未夹带。
