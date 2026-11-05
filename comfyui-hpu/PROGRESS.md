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
