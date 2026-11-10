# 独立环境与复现

本文件只保留可操作的环境约束、启动方式和模型路径。验证结论见 `PROGRESS.md`；性能调查的测量契约见 `MOSS_HPU_PERFORMANCE_PLAN_2026-11-06.md`。

## 服务器与安全边界

- 服务器：`sonny@192.168.10.21`；产品测试 checkout：`/home/sonny/ComfyUI-Gaudi`。
- **禁止修改 `/home/sonny/ComfyUI`**，包括 venv、代码、配置、输出和模型；只允许通过独立配置只读复用模型。
- 基础运行时：`/opt/gaudi-install/venv-1.24.2-481-ubuntu2204`；不要用普通 pip resolver 替换 Habana torch。测试依赖安装到独立 `.venv`。
- 当前验证环境：Gaudi2 HL-225、Habana 1.24.2、Python 3.10、Habana torch `2.13.0a0+git2469d18`、eager（`PT_HPU_LAZY_MODE=0`）。
- 模型下载先使用服务器已有 `HF_ENDPOINT`、`HF_TOKEN` 和 shell 代理；不把令牌或代理凭据写入命令、脚本、日志或文档。已有 VAE 优先复用。

## 内核、驱动和设备检查

当前内核为 `5.15.0-202-generic`；Habana DKMS 已覆盖 `5.15.0-186-generic` 和 `5.15.0-202-generic`。重启前检查目标内核的 DKMS，而不是只看 `lspci` 或 `lsmod`：

```bash
sudo bash /home/sonny/gaudi2/gaudi2-single-card-driver/scripts/install-gaudi2.sh \
  --verify-only --skip-pytorch
hl-smi
```

驱动脚本现在枚举当前内核和 `/boot` 中的有效内核镜像，校验五个 Habana 模块、vermagic、Gaudi2 PCI alias、DKMS 路径和补丁字符串；审查补丁保存在 `patches/install-gaudi2-all-boot-kernels.patch`。完整重启、BF16 matmul、双内核 verify 和 19 项模拟回归已通过。未配置开机自启。

## Python 环境

在 `/home/sonny/ComfyUI-Gaudi` 执行，禁止在原始 checkout 中安装：

```bash
/opt/gaudi-install/venv-1.24.2-481-ubuntu2204/bin/python -m venv .venv
printf '%s\n' /opt/gaudi-install/venv-1.24.2-481-ubuntu2204/lib/python3.10/site-packages \
  > .venv/lib/python3.10/site-packages/gaudi-base.pth
.venv/bin/python -m pip install --no-deps -r requirements.txt
.venv/bin/python -m pip install multidict propcache trampoline greenlet python-dotenv \
  kornia-rs mako aiohappyeyeballs aiosignal async-timeout attrs frozenlist \
  'setuptools>=77.0.3' pytest
http_proxy_enable
.venv/bin/python -m pip install --index-url https://pypi.org/simple \
  comfyui-workflow-templates==0.11.102
.venv/bin/python -m pip check
```

`pip check` 预期保留继承的 torchvision 0.16.2 / Habana torch 声明冲突；不要为消除该 warning 替换 Habana torch。

## Smoke、回归和服务

```bash
export PT_HPU_LAZY_MODE=0
export GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so
.venv/bin/python -m scripts.gaudi.smoke
.venv/bin/python -m pytest tests-unit/comfy_test/model_management_test.py -q
```

HPU 服务与独立 smoke/probe 必须串行占卡。用户试用服务：

```bash
ssh sonny@192.168.10.21
cd ~/ComfyUI-Gaudi
./start-hpu.sh
./stop-hpu.sh
tail -f logs/hpu.log
cat .run/deployed-version
```

当前部署提交为 `05d2a0a5`，监听 `http://192.168.10.21:8189`；根目录脚本负责 PID、端口和同目录进程校验，`scripts/gaudi/start.sh` 仍是前台开发入口。停止服务会终止其正在执行的任务。

Qwen3.5 HPU decode 默认对 deferred conv/GDN 使用区域编译；启动参数 `--disable-comfy-compiler` 可关闭。首次 shape 编译有开销，舍入差异可能改变采样 caption；验收细节见 `MOSS_GDN_COMPILE_2026-11-09.md`。已有 INT8 和派生 BF16 文本模型均通过完整工作流。

MOSS BF16 HPU attention 也默认使用区域编译，序列长度自动适配；同一 `--disable-comfy-compiler` 参数可关闭。CPU、训练及自定义 attention override 走原路径。只涉及 MOSS attention，没有全局替换 SDPA；不同分辨率、加权提示词和卸载重载已通过，详见 `MOSS_DIT_OPTIMIZATION_2026-11-09.md`。

后续已将 MOSS 编译范围扩至投影/FFN，保留调制与残差原执行方式，并在这两个区域按实际 shape 使用厂商静态 recipe。Qwen3.5 普通 BF16 文本模型在使用 KV cache、每步不超过 8 tokens 时编译 MLP；量化或自定义 operations 保留原路径。模块/子模块/全局 forward hooks 存在时保留 eager 调用。上述优化均可用 `--disable-comfy-compiler` 关闭，细节见 `MOSS_BLOCK_OPTIMIZATION_2026-11-09.md`。固定 seed 的文本和图片细节可能变化。

## 共享模型路径

两个 checkout 的 `extra_model_paths.yaml` 使用同一配置：

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

`models/<分类>` symlink 保留旧式直接路径兼容性。Qwen 2.1 Turbo、MOSS、Qwen3.5/Qwen3VL 和 VAE 已在共享目录；迁移验证和旧配置备份见 `MODEL_STORAGE_PLAN_2026-11-06.md`。

## API 工作流与验证边界

服务器独立 checkout 可提交：

```bash
.venv/bin/python -m scripts.gaudi.submit scripts/gaudi/krea2-api.json \
  --url http://192.168.10.21:8189 --result test-results/krea2.json
.venv/bin/python -m scripts.gaudi.submit scripts/gaudi/qwen-image21-turbo-api.json \
  --url http://192.168.10.21:8189 --result test-results/qwen21.json
.venv/bin/python -m scripts.gaudi.submit scripts/gaudi/moss-image-sft-api.json \
  --url http://192.168.10.21:8189 --result test-results/moss.json
```

已验证的 MOSS 基础工作流为 25 步 `er_sde/simple`、cfg2.5；基础图关闭 prompt enhancer、SLG 和编辑 LoRA。官方 TextGenerate 扩写图也已验证，但自回归仍明显慢于直接文本编码。编辑/控制 LoRA、SLG、完整批量/多卡、lazy/通用全模型编译/HPU Graph、超大分辨率和其它 custom nodes 未验证。
