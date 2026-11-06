# 独立环境与复现

## 内核与驱动（2026-11-06 恢复）

当前内核为 `5.15.0-202-generic`；Habana `1.24.2-481` 已为 `186` 和 `202` 两个内核安装。重启前必须检查目标内核对应的 DKMS installed 状态，不能仅凭 `lspci` 或 `lsmod` 判断驱动可用。驱动重装时可能只编译正在运行的内核，遗漏已经安装但尚未启动的新内核。修复和检查命令见 [驱动恢复记录](DRIVER_RECOVERY_2026-11-06.md)。

本轮已完成真实重启验证：重启后 `hl-smi`、两个内核的 DKMS 检查和 BF16 HPU matmul 均通过；随后已运行下文的 `./start-hpu.sh` 恢复 ComfyUI-Gaudi。仍未新增开机自启。

已修复服务器 `/home/sonny/gaudi2/gaudi2-single-card-driver/scripts/install-gaudi2.sh`：安装模式覆盖当前内核和 `/boot` 中的全部内核镜像，缺 headers 或某个内核驱动校验失败时拒绝报成功。以后重启前可运行 `sudo bash /home/sonny/gaudi2/gaudi2-single-card-driver/scripts/install-gaudi2.sh --verify-only --skip-pytorch`；这项检查不重建驱动、不运行 HPU 计算。脚本依赖同仓库新增的 `scripts/lib/gaudi-dkms.sh`，同步时需一起携带。细节和备份见恢复记录。

## Python 环境与基础检查

以下在服务器独立 checkout `~/ComfyUI-Gaudi` 中执行；不要在原 `~/ComfyUI` 中运行安装命令。

```bash
/opt/gaudi-install/venv-1.24.2-481-ubuntu2204/bin/python -m venv .venv
printf '%s\n' /opt/gaudi-install/venv-1.24.2-481-ubuntu2204/lib/python3.10/site-packages > .venv/lib/python3.10/site-packages/gaudi-base.pth
.venv/bin/python -m pip install --no-deps -r requirements.txt
.venv/bin/python -m pip install multidict propcache trampoline greenlet python-dotenv kornia-rs mako aiohappyeyeballs aiosignal async-timeout attrs frozenlist 'setuptools>=77.0.3' pytest
# shell 中启用已有代理；不要打印 HF_TOKEN 或代理凭据
http_proxy_enable
.venv/bin/python -m pip install --index-url https://pypi.org/simple comfyui-workflow-templates==0.11.102
.venv/bin/python -m pip check
```

最后一项预期存在继承的 torchvision 0.16.2 / Habana torch 版本声明冲突，详见 PROGRESS.md。不要执行无约束的普通 `pip install -r requirements.txt`。

```bash
export PT_HPU_LAZY_MODE=0
export GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so
.venv/bin/python -m scripts.gaudi.smoke
.venv/bin/python -m pytest tests-unit/comfy_test/model_management_test.py -q
bash scripts/gaudi/start.sh
```

服务默认仅绑定服务器 `127.0.0.1:8189`，本地访问使用：

```bash
ssh -N -L 8189:127.0.0.1:8189 sonny@192.168.10.21
```

浏览器打开 `http://127.0.0.1:8189`。独立 smoke 与服务必须串行占用 HPU。

## 用户试用启停（2026-11-05 部署）

当前部署提交 `0690d18f`（修复 MOSS TextGenerate 的采样错误与输出头 CPU 回退；启停脚本仍为 `3bda1351`），监听服务器局域网地址，可直接打开 http://192.168.10.21:8189 。

```bash
ssh sonny@192.168.10.21
cd ~/ComfyUI-Gaudi
./start-hpu.sh   # 后台启动，等待页面就绪；已启动时直接返回
./stop-hpu.sh    # 停止该目录管理的服务，等待 HPU 进程退出
```

```bash
tail -f ~/ComfyUI-Gaudi/logs/hpu.log
cat ~/ComfyUI-Gaudi/.run/deployed-version
```

`start-hpu.sh` 读取服务器 `.gaudi.env` 的监听地址/端口；该文件不进入代码仓库。配置副本在本调查仓库 `deployment/gaudi.env`。`scripts/gaudi/start.sh` 仍是原有前台开发入口；用户日常使用根目录脚本即可，避免绕过 PID 管理再启动第二个 HPU 进程。没有配置开机自启。停止时会终止该服务正在进行的任务。


## 模型下载优先级

模型下载先使用服务器 shell 已有的 `HF_ENDPOINT` 和 `HF_TOKEN`，不要显式覆盖端点。`huggingface_hub` 可自动读取；需要显式传参时从 `os.environ` 取值而不是写入脚本。先检查已有文件，尤其是 VAE，避免重复下载。代理或备用站点仅在必要时使用并记录失败证据。

## 模型路径与工作流

服务器两个 checkout 的 `extra_model_paths.yaml` 都引用同一共享模型仓库：

```yaml
shared_models:
  base_path: /home/sonny/ComfyUI-Models
  checkpoints: models/checkpoints
  configs: models/configs
  clip: models/clip
  diffusion_models: |
    models/diffusion_models
    models/unet
  loras: models/loras
  text_encoders: |
    models/text_encoders
    models/clip
  vae: models/vae
```

共享目录按原有一级分类保存全部模型；checkout 的 `models/<分类>` 是兼容 symlink，旧式直接路径仍可用。Qwen 2.1 Turbo、MOSS、Qwen3.5/Qwen3VL 和 VAE 都已在共享目录中，文件和固定 revision 见 PROGRESS.md。加载不需要 HF 访问或令牌；只有重新下载才需要已有 shell 代理/环境。迁移和原配置备份见 MODEL_STORAGE_PLAN_2026-11-06.md。

服务运行后，在服务器独立 checkout 执行：

```bash
.venv/bin/python -m scripts.gaudi.submit scripts/gaudi/krea2-api.json --url http://192.168.10.21:8189 --result test-results/krea2.json
.venv/bin/python -m scripts.gaudi.submit scripts/gaudi/qwen-image21-turbo-api.json --url http://192.168.10.21:8189 --result test-results/qwen21.json
.venv/bin/python -m scripts.gaudi.submit scripts/gaudi/moss-image-sft-api.json --url http://192.168.10.21:8189 --result test-results/moss.json
```

三份 JSON 是 API 工作流，使用原生节点，不需要自定义节点。额外 1024/重载变体保存在本调查仓库 `workflows/`。图片写入独立 `output/gaudi/`。

新增回归测试（停止服务后运行）：

```bash
.venv/bin/python -m pytest tests-unit/comfy_test/model_management_test.py tests-unit/comfy_test/hpu_quant_test.py tests-unit/comfy_test/hpu_vae_test.py tests-unit/comfy_test/hpu_textgen_test.py -q
```

服务器目录通过 tar/scp 同步源码，**没有 `.git` 元数据**；本地代码仓库才是版本来源。同步时仅从已审查代码提交打包，例如 `git archive codex/gaudi2-hpu | ssh sonny@192.168.10.21 'tar -xf - -C ~/ComfyUI-Gaudi'`，不要复制本调查仓库、过程文档、令牌或配置。修改后重启服务，先等待旧进程彻底退出。当前部署版本为 `0690d18f`。

MOSS SFT 与 Qwen3.5 9B 的权重、Flux2 VAE 和其它模型现在都位于共享 models；MOSS 固定验证参数是 25 步 er_sde/simple、cfg2.5；基础 API 工作流没有启用可选 prompt enhancer、SLG 或编辑 LoRA。用户试用后已补测官方模板的 TextGenerate 扩写（MTP auto、sampling on）：原始 1024² 工作流成功，证据 workflows/moss-user-textgen.json 和 results/moss-user-textgen-fix1.json。扩写仍明显慢于直接文本编码，详细耗时见 PROGRESS.md；SLG/编辑 LoRA 仍未验证。


## MOSS 文本生成性能修复（2026-11-06）

`0690d18f` 让普通生成和 MTP 校验调用 lm_head 层本身，遵守已设置的 BF16 计算策略。旧版直接 F.linear 跳过策略，约 1 GB INT8 词表矩阵每步触发 CPU fallback；修复后真实 HPU 普通/MTP 短序列的 fallback 计数均为空。默认 mtp=auto 可继续使用，不需要改工作流或换模型。

同一完整 1024² 请求的实际端到端从 337.013 s 降至 134.586 s；新扩写约 307 tokens / 89 s，末尾进度约 3.44 token/s。文本和图片会因输出头计算路径修正而与旧版有所不同。采样时 HPU 利用率仍有短暂 0%，84 个扩写阶段样本平均约 22%、最高 57%；当前没有达到硬件极限吞吐。

诊断脚本、结果和过程只在本调查仓库 probes/moss_performance.py、results/moss-performance-*。完整 CPU profiler trace 各约 146 MB 留在独立服务器 test-results/moss-{baseline,head-fixed}-trace.json，没有加入代码仓库或本调查 Git。
