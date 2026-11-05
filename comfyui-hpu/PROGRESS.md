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
