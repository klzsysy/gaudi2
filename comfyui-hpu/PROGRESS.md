# 执行记录

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
