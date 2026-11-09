# Gaudi2 / ComfyUI HPU 初始兼容性评估

调查日期：2026-11-05  
目标主机：Intel Gaudi2 单卡

本文是适配开始前的基线记录，不替代当前的验证结果；后续实测以 comfyui-hpu/PROGRESS.md 为准。

## 初始结论

调查时，原始 /home/sonny/ComfyUI 不能直接把 NVIDIA venv 替换为 Habana venv。HPU venv 缺少 ComfyUI 依赖，且普通 pip 安装会尝试替换 Habana torch。上游 ComfyUI 当时没有 torch.hpu 分支，未识别设备会走 torch.cuda.current_device()，因此无法直接启动可用 HPU 环境。

安全路线是建立独立 HPU checkout、继承 Habana 基础 site-packages、只安装缺失依赖，并显式实现 HPU 设备/内存/同步/dtype/offload 边界。原始 NVIDIA 环境必须保持不变。

## 关键环境事实

| 项目 | 调查时结果 |
|---|---|
| HPU | Gaudi2，torch.hpu 可用，设备名 GAUDI2 |
| HPU Python | Python 3.10.12 |
| Habana PyTorch | 2.13.0a0+git2469d18 |
| 原始 ComfyUI | v0.28.0，detached HEAD |
| NVIDIA torch | 2.12.0+cu130 |
| HPU venv | 运行时可用，但不是完整 ComfyUI 依赖环境 |

HPU venv 的已知依赖缺口包括 comfy_aimdo、SQLAlchemy、alembic、aiohttp、torchsde 和 sentencepiece；torchvision 0.16.2 还声明依赖 torch 2.1.2。不能用无约束的 requirements 安装覆盖 Habana torch。

## 需要补齐的后端边界

除 get_torch_device 外，还需要处理：

- total/free memory、synchronize、soft_empty_cache、设备设置和 dtype 策略；
- pinned memory、async offload、Dynamic VRAM 等 CUDA 专用路径；
- 量化/attention/kernel 路径的能力降级；
- HPU 独立启动参数、模型加载/卸载和可回滚提交。

因此最初计划先做设备、matmul、attention、模型加载卸载和 HTTP smoke，再做 Krea2、Qwen-Image-2.1 Turbo 与 MOSS。后续四阶段和失败修复已记录在 comfyui-hpu/PROGRESS.md。

## 调查时的模型边界

调查当日服务器尚未确认 ComfyUI 原生格式的 Qwen-Image-2.1/Turbo 组件，不能把 Diffusers 权重或已有通用 VAE 当成匹配的原生组合。Qwen-Image-2.1-Turbo/qwen-image21-gaudi-hpu-rope.patch 只解决 Diffusers 路径的 HPU RoPE，不替代 ComfyUI HPU 后端。

## 边界

本次初始调查没有修改原始 ComfyUI、其 venv 或模型，也没有把未验证的模型兼容性写成结论。后续实际成功/失败路径、数值修复、图片验收和性能限制以 comfyui-hpu 目录为准。
