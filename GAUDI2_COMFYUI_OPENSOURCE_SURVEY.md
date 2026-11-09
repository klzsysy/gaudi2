# ComfyUI × Intel Gaudi（HPU）开源项目调查

调查日期：2026-11-05  
调查问题：是否存在已经为 ComfyUI 增加 Intel HPU/Gaudi 后端的开源项目？

## 结论

调查时没有找到可直接使用的 ComfyUI HPU 后端：GitHub 仓库/README、issue/PR、上游源码、PyPI 和 Gitee 的关键词检索均无有效项目。唯一容易混淆的 Gaudi-Vision 使用的是建筑风格名称，实际运行在 RTX 2070 SUPER/cuda:0。

因此本项目需要在独立 checkout 中实现 HPU 后端；下列项目只能提供迁移经验，不能直接替换当前代码或模型路径。

## 检索证据

| 检索 | 结果 |
|---|---|
| GitHub 仓库：comfyui+gaudi/hpu/habana | 0 个有效项目 |
| GitHub README、issue、PR | 无 ComfyUI HPU 适配 |
| 上游 ComfyUI 源码 | 无 hpu/gaudi/habana 命中 |
| comfy-kitchen | eager 设备白名单含 hpu，但加速后端没有 HPU 实现 |
| PyPI：comfyui-hpu/gaudi/habana | 全部 404 |
| Gitee 同名关键词 | 0 结果 |

## 可复用资产与限制

- optimum-habana：最完整的 Diffusers Gaudi 参考，调查时已 EOL；其 QwenImage 路径证明算子层可行，但不是 ComfyUI 原生模型/图执行路径。
- Habana gpu_migration 与旧版 Model-References：可用来识别 CUDA 假设和精度/算子边界，不应作为全局 monkeypatch 直接带入产品。
- CloudBrigade stable-diffusion-webui 的 hpu_devices 分支：展示设备、bf16、RNG、HPUGraph 和 pinned-memory 等坑位，但基于旧 A1111，不能直接复制。
- vllm-gaudi：插件/patches 的工程组织方式可参考，模型和 kernel ABI 与 ComfyUI 不同。
- Intel llm-scaler 的 ComfyUI-OmniXPU、OPEA Gaudi 文生图服务和 GGUF-XPU 项目分别面向 XPU 或 TGI/optimum-habana，不是 ComfyUI HPU 后端。

## 对实施方案的影响

最小 HPU 后端需要显式处理设备探测、内存、同步、dtype、offload 和不支持的 CUDA 专用路径；初期应关闭 pinned memory、async offload/Dynamic VRAM、CUDA/HIP kernel、fp8/int8 未验证路径和 sage/flash attention。应参考上游 NPU 的小范围后端 patch 形状，而不是重写框架。

Gaudi2 的验证应以实际 Habana 版本和算子结果为准。生态变化、lazy/HPU Graph 的动态 shape、bf16 优先级和量化路径均需在服务器上实测，不能从其它项目的 benchmark 外推。

## 参考

- ComfyUI model_management.py：https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_management.py
- ComfyUI NPU PR #16057：https://github.com/Comfy-Org/ComfyUI/pull/16057
- comfy-kitchen：https://github.com/Comfy-Org/comfy-kitchen
- optimum-habana：https://github.com/huggingface/optimum-habana
- Habana Model-References：https://github.com/HabanaAI/Model-References
- CloudBrigade HPU 分支：https://github.com/CloudBrigade/stable-diffusion-webui/tree/hpu_devices
- vllm-gaudi：https://github.com/vllm-project/vllm-gaudi
