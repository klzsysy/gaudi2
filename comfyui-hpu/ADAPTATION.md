# HPU后端与模型正确性修复

本篇合并2026-11-05起的初始可行性调查和四类适配结果。性能专题从 [优化索引](README.md) 进入，操作环境见 [ENVIRONMENT](ENVIRONMENT.md)。

## 后端接入与初始选择

2026-11-05的上游ComfyUI和开源检索未找到可直接使用的HPU后端。Kitchen eager允许HPU，但专用加速后端与实际算子支持还需核对。Optimum Habana/Diffusers、旧A1111 HPU分支和vLLM Gaudi提供设备/精度/图执行经验，项目结构和模型执行契约各自不同。[ComfyUI NPU接入参考](https://github.com/Comfy-Org/ComfyUI/pull/16057) · [Kitchen](https://github.com/Comfy-Org/comfy-kitchen) · [Optimum Habana](https://github.com/huggingface/optimum-habana) · [A1111 HPU分支](https://github.com/CloudBrigade/stable-diffusion-webui/tree/hpu_devices)

采用独立checkout和继承Habana site-packages的venv。`8326476f`补齐设备探测、内存、同步、cache、BF16策略和CUDA专用offload/pinned-memory路径门控。BF16 matmul、Comfy attention、ModelPatcher加载/卸载、HTTP根路由/object_info/system_stats及CPU quick-test通过。

非交互SSH缺`GC_KERNEL_PATH`曾使attention编译报`synStatus26`，显式配置厂商TPC库后通过。服务持卡与独立probe需串行，依赖安装需保留厂商torch。故障定位与环境恢复详见 [运行时](optimization/RUNTIME.md)。

## E4M3文件解码兼容

Krea2初次history成功但图片异常，HPU直接将部分E4M3FN高编码转浮点得到inf/nan，首个非有限值位于Qwen3VL第0层q_proj。`fc1c891f`在HPU解量化采用完整256值CPU语义解码表，保留weight/scale/量化状态。

10项回归验证所有编码在BF16/FP32下与CPU逐值一致，合法NaN编码单列。512²、1024²及重载正常出图，文本/DiT/VAE均在HPU BF16。该修复处理文件读取兼容，后续原生FP8范围/scale处理见 [DiT FP8](optimization/DIT_FP8.md)。

## VAE卷积布局与原地残差

Qwen-Image-2.1 Turbo首次512²/1024²均返回success但有异常纹理。禁用prefix cache及CPU文本编码无效，同latent交给CPU FP32 VAE得到正确图像。HPU逐层定位到带卷积layout metadata的`strip_apply`原地残差合并。

取消分块、`contiguous()`、`copy_`均无效，4/64通道最小复现最大绝对误差48/993。`6db162dd`的HPU分支保留分块并返回新张量完成`out + residual`。14项回归、512²/1024²、长提示词和卸载重载通过。CPU FP32/HPU BF16 VAE PNG MAE0.2102（0–255）、PSNR54.49dB。跨设备latent隔离与逐层检查比只看finite更能缩小故障范围。

## MOSS模型组合

`21e98390`验证SFT BF16 DiT、Qwen3.5 9B INT8 ConvRot和已有Flux2 VAE。header/shape/dtype对照确认旧VAE与官方文件251个张量一致，复用已有组件。

GDN长度17/65/128相对CPU输出有限、最大绝对差0.0004883/0.0009766/0.0009766，VAE roundtrip通过。512²/1024²、长短提示词、25步er_sde/simple、CFG2.5及卸载重载通过。后续官方TextGenerate的int64 gather和lm_head fallback修复分别见 [文本decode](optimization/TEXT_DECODE.md) 与 [文本精度](optimization/TEXT_PRECISION.md)。

## 验证范围与经验

当前完整工作流覆盖Krea2、Qwen-Image-2.1 Turbo、MOSS SFT及Moody FP8的已记录组合。编辑/控制LoRA、SLG、多卡、完整批量、其它VAE/custom nodes及超大尺寸需要各自验收。batch2目前主要有算子回归。

设备可见、history成功、单算子finite、VAE对照和完整图像分别提供不同证据。先定位首个数值偏离，再选择最小设备分支，最后经过正常入口和资源生命周期复验。Diffusers的独立RoPE实验仍由 [Qwen说明](../Qwen-Image-2.1-Turbo/README.md) 提供复现，ComfyUI后端遵循这里的验证路径。
