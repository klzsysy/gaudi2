# Gaudi2 / ComfyUI HPU 兼容性调查结论

调查日期：2026-11-05  
目标主机：`sonny@192.168.10.21`（Intel Gaudi2 单卡）

## 结论

当前 `~/ComfyUI` 不能直接把 `venv` 替换成 `/opt/gaudi-install/venv-1.24.2-481-ubuntu2204/`。直接替换会在 ComfyUI 初始化阶段失败，不能进入可用的 HPU 推理环境。

升级到当前 ComfyUI 稳定版 `v0.44.0` 是后续工作的正确基线，因为它已经包含 Krea2 和 Qwen-Image-2.1/Turbo 的原生模型代码及工作流。但升级本身不能解决 Gaudi HPU：上游最新版的设备管理代码仍没有 `torch.hpu` 分支，未识别的设备会回退到 `torch.cuda.current_device()`；Gaudi Habana PyTorch 上 `torch.cuda.is_available()` 为 `False`，因此仍会失败。

建议先建立独立的 HPU fork/分支并补齐 HPU 后端，再适配模型。现有 NVIDIA ComfyUI 和它的 `venv` 应保持原样，通过独立启动脚本选择后端。

## 已检查的环境

| 项目 | 结果 |
|---|---|
| 硬件 | Habana Labs Gaudi2，`lspci` 显示 accelerator，HPU 处于可用状态 |
| HPU Python | Python 3.10.12 |
| HPU PyTorch | `2.13.0a0+git2469d18` |
| HPU 设备 | `torch.hpu.is_available() == True`，设备数为 1，名称 `GAUDI2` |
| 当前 ComfyUI | `/home/sonny/ComfyUI`，`ComfyUI v0.28.0`，detached HEAD |
| 当前 NVIDIA venv | PyTorch `2.12.0+cu130`，依赖完整，`pip check` 通过 |
| HPU venv | Habana PyTorch 运行时可用，但不是完整的 ComfyUI 依赖环境 |

## 为什么不能直接替换 venv

HPU venv 缺少当前 ComfyUI 启动所需的多个包，包括 `comfy_aimdo`、`SQLAlchemy`、`alembic`、`aiohttp`、`torchsde` 和 `sentencepiece` 等。它还存在已知的依赖声明冲突：`torchvision==0.16.2` 声明需要 `torch==2.1.2`，实际 Habana PyTorch 是 `2.13.0a0+git2469d18`。不能用普通的 `pip install -r requirements.txt` 覆盖 HPU venv，因为 `torch`、`torchvision` 和 `torchaudio` 会被解析成 CUDA/CPU 版本，破坏 Habana 运行时。

当前 ComfyUI 的 `comfy/model_management.py` 设备路径只处理 CUDA、XPU、NPU、MLU 等后端，默认分支调用：

```python
return torch.device(torch.cuda.current_device())
```

在 HPU venv 中实际结果是：

```text
torch.cuda.is_available() == False
AssertionError: Torch not compiled with CUDA enabled
```

即使补上最初的设备探测，后续还需要处理 HPU 设备名称、显存查询、空闲显存查询、同步、缓存清理、dtype 判断、异步 offload、pinned memory 和 CUDA 专用辅助路径。`comfy/pinned_memory.py` 和若干模型/扩展代码仍直接调用 `torch.cuda` API，不能只改一个 `if` 分支。

## 升级 ComfyUI 的调查结果

已对官方当前源码和稳定版进行核对：

- 当前稳定版为 `v0.44.0`。
- 官方最新版源码的 `get_torch_device()` 仍没有 `torch.hpu` 分支，`get_total_memory()`、`get_free_memory()`、`synchronize()` 和 `soft_empty_cache()` 也没有 HPU 分支。
- 最新版已经有 `QwenImage21`、`QwenImage21Transformer2DModel`、`QwenImage21` 文本编码器和检测逻辑。
- 最新版已经有 Krea2 原生模型实现。
- 官方工作流模板包含 Qwen-Image-2.1 Turbo 的文本生图和图像编辑工作流，Turbo 使用 8 步采样。

因此升级的作用是补齐模型架构、模型检测和官方工作流；它不会自动把 Gaudi 当成可用设备。

参考：

- [ComfyUI v0.44.0 发布页](https://github.com/Comfy-Org/ComfyUI/releases)
- [官方 model_management.py](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_management.py?plain=1)
- [Qwen-Image-2.1 官方 ComfyUI 文档](https://github.com/Comfy-Org/docs/blob/main/tutorials/image/qwen/qwen-image-2-1.mdx)
- [Qwen-Image-2.1 Turbo 官方说明](https://github.com/QwenLM/Qwen-Image-2.1)

## 服务器上的模型现状

### Krea2

现有模型目录已经包含 Krea2 权重，例如：

```text
models/diffusion_models/krea2/krea2_turbo_int8_convrot.safetensors
models/diffusion_models/krea2/moodyKrea2Mix_v50.safetensors
models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors
models/vae/qwen_image_vae.safetensors
```

当前 ComfyUI v0.28.0 已经包含 Krea2 的模型检测、模型类和文本编码器入口。因此 Krea2 的“模型架构识别”基础已经存在；在 Gaudi 上能否实际推理仍取决于 HPU 后端、量化算子和显存管理是否适配。调查按要求在此停止，没有执行模型推理验证。

### Qwen-Image-2.1 / Turbo

当前服务器没有发现 ComfyUI 官方格式的 Qwen-Image-2.1/Turbo 权重，例如：

```text
qwen_image_2.1_bf16.safetensors
qwen_image_2.1_turbo_bf16.safetensors
qwen_image_2.1_int8_convrot.safetensors
qwen_image_2.1_turbo_int8_convrot.safetensors
qwen3vl_8b_int8_convrot.safetensors
qwen_image_2.1_vae_bf16.safetensors
```

服务器现有的 `qwen3vl_8b_fp8_scaled.safetensors` 和 `qwen_image_vae.safetensors` 不能据此证明 Qwen-Image-2.1/Turbo 已经具备完整加载条件。应使用 ComfyUI 官方模型仓库提供的匹配权重和 VAE；不能把 Diffusers pipeline 权重文件直接当作 ComfyUI 原生权重使用。

官方工作流列出的 Qwen-Image-2.1 组件和目录结构见：[官方工作流模板](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_t2i.json)。

仓库中已有的 `Qwen-Image-2.1-Turbo/qwen-image21-gaudi-hpu-rope.patch` 属于 Diffusers 路径的 HPU RoPE 补丁，解决的是复数 RoPE 在 HPU 上不支持的问题；它不能替代 ComfyUI HPU 后端，也不能让旧版 ComfyUI 识别 Qwen-Image-2.1。

## 推荐的安全实施顺序

1. 以 ComfyUI `v0.44.0` 或其对应提交建立独立 `ComfyUI-hpu` checkout/fork，保留当前 `~/ComfyUI` 不变。
2. 继续使用 `/opt/gaudi-install/venv-1.24.2-481-ubuntu2204/` 作为 HPU PyTorch 基础，按 HPU 版本固定安装其余 Python 依赖，禁止依赖安装器替换 Habana PyTorch。
3. 在 ComfyUI 后端增加显式 HPU capability：设备选择、内存统计、同步、dtype、offload 和不支持的 CUDA 专用路径降级/关闭。
4. 为 HPU 单独配置启动参数和脚本；初期禁用 pinned memory、异步 offload、CUDA-only quantization/kernel 路径和不兼容的自定义节点。
5. 先做纯 HPU smoke test（设备、矩阵乘、attention、VAE、小模型加载），再做 Krea2，最后做 Qwen-Image-2.1/Turbo。
6. 每一步都保留可回滚提交；NVIDIA 继续使用现有 `~/ComfyUI/start.sh` 和现有 `venv`。

建议的目录关系：

```text
/home/sonny/ComfyUI                 # 现有 NVIDIA 环境，保持不变
/home/sonny/ComfyUI-hpu             # 独立 HPU fork/checkout
/opt/gaudi-install/venv-1.24.2-481-ubuntu2204/  # Habana PyTorch
/home/sonny/ComfyUI/models          # 可通过配置共享，不复制大模型
```

## 调查边界

本次已经完成硬件、Python 环境、依赖、设备探测、上游版本、模型代码和模型文件清单调查，并停止了隔离启动验证。没有修改现有 `~/ComfyUI`、现有 NVIDIA venv 或服务器模型文件，也没有继续执行 Krea2/Qwen-Image-2.1/Turbo 的模型推理验证。
