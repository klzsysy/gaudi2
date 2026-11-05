# 在 Intel Gaudi2 上运行 Qwen-Image-2.1-Turbo

本文记录了在单卡 Intel Gaudi2 上运行 `Qwen/Qwen-Image-2.1-Turbo` 的可复用流程。参考验证环境为 Ubuntu 22.04、Gaudi 软件栈 `1.24.2-481`、Habana PyTorch `2.13.0a0+git2469d18`、Diffusers `0.41.0` 和 Transformers `5.20.2`。测试以 BF16 在 HPU 上生成了 512×512 图像；更高分辨率和其他版本组合需要单独验证。

## 结果与限制

模型和权重可以正常下载、加载到 HPU，但原版 Diffusers 在首次去噪时会因 HPU 不支持复数张量而失败：

```text
RuntimeError: Complex datatype is not supported on HPU device.
```

根因是 Qwen-Image-2.1 的旋转位置编码（RoPE）以复数频率表示，并通过 `view_as_complex` 对 attention 的 query/key 做旋转。随附补丁把频率拆成实数 cos/sin，在实数域完成等价旋转，并适配 `[batch, sequence, heads, dim]` 张量布局及 KV cache 分支。

上游模型卡中的示例使用 CUDA 设备。Gaudi 上需将 pipeline 放到 `hpu`，并使用本文补丁。此补丁直接修改 venv 中安装的 Diffusers 文件；升级或重装 Diffusers 会覆盖补丁。补丁针对 Diffusers 0.41.0 的源码上下文，版本不同时必须检查并重新验证。

## 安装 Habana PyTorch

先按 Gaudi 驱动仓库的硬件和驱动要求完成驱动部署。确认 Gaudi 设备处于 `Operational`，并按部署手册确认启动参数和 IOMMU 状态。若尚未获取驱动仓库：

```bash
git clone https://github.com/klzsysy/gaudi2-single-card-driver.git
cd gaudi2-single-card-driver
```

在驱动仓库根目录运行安装脚本：

```bash
chmod +x scripts/install-gaudi2.sh
./scripts/install-gaudi2.sh
```

脚本会创建独立的 venv 并安装匹配版本的 Habana PyTorch。记录安装脚本输出的 venv 路径，后续通过变量引用它：

```bash
export GAUDI_VENV=/path/to/installer-created/venv
source /etc/profile.d/habanalabs.sh
export LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc.so.4
"$GAUDI_VENV/bin/python" -c \
  'import torch; print(torch.__version__, torch.hpu.is_available(), torch.hpu.device_count())'
```

确认输出显示 HPU 可用且设备数大于零。安装脚本的基础矩阵乘、ReLU 和前后向检查也应通过。

## 安装模型依赖

以下版本组合在参考环境中通过模型加载和生成验证。不要让依赖安装器替换 Habana 提供的 PyTorch：

```bash
"$GAUDI_VENV/bin/pip" install \
  'diffusers==0.41.0' \
  'transformers==5.20.2' \
  'accelerate==1.15.0' \
  safetensors

# Transformers 的 Qwen3-VL processor 需要 torchvision。
# --no-deps 避免 pip 改动 Habana PyTorch；版本应按当前环境复核。
"$GAUDI_VENV/bin/pip" install --no-deps 'torchvision==0.16.2'
```

参考环境中 torchvision 导入时会提示可选的 `torchvision.io` 扩展不可用；该扩展未被本次文生图流程使用。

## 准备 Hugging Face 环境

下载模型前，在启动 Python 的同一个 shell 中配置 HF 环境变量。可以从受限权限的环境文件载入，也可以使用系统已经配置好的变量：

```bash
source /path/to/huggingface-environment.sh
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1

# 不要把 HF_TOKEN 写进仓库、命令历史或日志。
test -n "${HF_TOKEN:-}" || echo 'HF_TOKEN is not set (a token may be optional for public files)'
```

非交互 shell 不一定会加载交互式 shell 的启动文件。应显式载入安全的环境文件，或在运行命令的 shell 中直接继承变量。若使用镜像，按实际网络环境设置 `HF_ENDPOINT`。模型缓存可能占用数十 GB。

## 应用 Gaudi RoPE 补丁

确认安装路径和 Python 版本后，先备份目标文件并检查补丁上下文：

```bash
export SITE_PACKAGES="$GAUDI_VENV/lib/python3.10/site-packages"
export DIFFUSERS_FILE="$SITE_PACKAGES/diffusers/models/transformers/transformer_qwenimage21.py"
cp -a "$DIFFUSERS_FILE" "$DIFFUSERS_FILE.orig"

PATCH_FILE=/path/to/Qwen-Image-2.1-Turbo/qwen-image21-gaudi-hpu-rope.patch
patch --dry-run -p1 -d "$SITE_PACKAGES" < "$PATCH_FILE"
```

如果 dry-run 成功，再应用补丁：

```bash
patch -p1 -d "$SITE_PACKAGES" < "$PATCH_FILE"
```

补丁文件使用相对路径 `diffusers/models/transformers/transformer_qwenimage21.py`。Python 版本不是 3.10 时，将 `SITE_PACKAGES` 改为实际 site-packages 路径。

回滚时恢复备份：

```bash
cp -a "$DIFFUSERS_FILE.orig" "$DIFFUSERS_FILE"
```

## 运行文生图

在已载入 HF 和 Habana 环境的 shell 中运行以下示例：

```bash
"$GAUDI_VENV/bin/python" - <<'PY'
import torch
from diffusers import QwenImage21Pipeline

model_id = "Qwen/Qwen-Image-2.1-Turbo"
pipe = QwenImage21Pipeline.from_pretrained(model_id, dtype=torch.bfloat16)
pipe = pipe.to("hpu")

image = pipe(
    prompt="A red panda beneath a blooming cherry tree at sunrise, detailed illustration",
    width=512,
    height=512,
    num_inference_steps=8,
    use_kv_cache=True,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
image.save("qwen-image-2.1-turbo.png")
PY
```

验证成功应得到 PNG 文件，且 pipeline 完成 8 步推理。参考环境生成 512×512 图像成功；模型卡推荐的大尺寸未在该验证中测试。尺寸增大将显著增加显存需求和运行时间。

## 验证 RoPE 数值

补丁通过 cos/sin 实数公式实现与复数乘法相同的二维旋转。核心关系为：

```text
x_rot = x_real * cos - x_imag * sin
y_rot = x_imag * cos + x_real * sin
```

参考环境中，对 FP32 和 BF16 输入比较原复数实现与补丁后的 CPU/HPU 实现，最大绝对误差均为 `0.0`；端到端生成也通过。升级依赖、修改补丁或更换硬件软件栈后，应重新做数值比较和端到端生成验证。

## 参考

- [Qwen-Image-2.1-Turbo 模型卡](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)
- 同目录补丁：[qwen-image21-gaudi-hpu-rope.patch](qwen-image21-gaudi-hpu-rope.patch)
