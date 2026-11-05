# ComfyUI × Intel Gaudi（HPU）现成开源项目调研

调查日期：2026-11-05  
调查对象：是否存在"已经给 ComfyUI 加上 Intel HPU / Gaudi 设备支持"的开源项目  
结论适用前提：承接 `GAUDI2_COMFYUI_HPU_ASSESSMENT.md`（单卡 Gaudi2，Gaudi SW 1.24.2，torch 2.13.0a0，ComfyUI 目标基线 v0.44.0）

## 结论

**没有。截至今天，不存在任何开源项目给 ComfyUI 添加过 Intel HPU / Gaudi 设备后端。**

不是"有一个但比较冷门"，而是四种独立检索路径都指向零结果：没有 fork、没有自定义节点插件、没有上游 PR/issue、没有 PyPI 包。要在 Gaudi2 上跑 ComfyUI，必须自己做，这点和上一份评估的结论一致，现在有了更硬的证据。

不过**"HPU 适配知识"不是零**。有四类现成资产可以搬：

| 资产 | 性质 | 对本项目的价值 |
|---|---|---|
| `huggingface/optimum-habana` | Diffusers 层的 Gaudi 支持（已 EOL） | 抄"哪些算子在 HPU 上能跑、用什么精度/attention" |
| Habana `gpu_migration` + Model-References | 官方迁移工具 + SD 参考实现 | `torch.cuda.*` 自动映射到 HPU，是降低补丁量的最大杠杆 |
| `CloudBrigade/stable-diffusion-webui`（`hpu_devices` 分支） | 唯一一个"diffusion WebUI 移植到 Gaudi"的先例 | 踩坑清单几乎 1:1 对应 ComfyUI 要改的地方 |
| `vllm-project/vllm-gaudi` | 硬件插件化范式 | 工程结构参考（plugin/patches 而非大 fork） |

## 检索过程与证据

| 检索 | 方法 | 结果 |
|---|---|---|
| GitHub 仓库搜索 | `comfyui+gaudi` / `comfyui+hpu` / `comfyui+habana` | `total_count = 0` |
| GitHub 仓库搜索（含 README） | `gaudi comfyui in:readme` 等 3 组 | 无相关项目（唯一同名项目是假阳性，见下） |
| GitHub 全站 issue/PR | `comfyui gaudi`、`comfyui hpu` | 无 ComfyUI 的 HPU 相关 issue/PR |
| 上游源码 | `git clone` ComfyUI master，`grep -ri "hpu\|gaudi\|habana"` 于 `*.py/*.md/*.txt` | **0 命中** |
| Comfy 官方 kernel 库 | `Comfy-Org/comfy-kitchen` grep `hpu` | 仅 1 处（见下） |
| PyPI | `comfyui-hpu` / `comfyui-gaudi` / `comfyui-habana` / `comfy-hpu` | 全部 404 |
| Gitee | 同名关键词 API 搜索 | 0 结果 |

假阳性：`Hyperillion/Gaudi-Vision`（"A Gaudi Stylized AI Image Re-drawing System Based on ComfyUI"）。这里的 Gaudi 是高迪建筑风格，不是 Intel Gaudi——其日志显示运行在 `RTX 2070 SUPER`、`cuda:0`。检索时容易被这条误导。

## 上游 ComfyUI 的实际状态：多后端，但没有 HPU

ComfyUI master 的设备抽象已经是"多加速器"形状，唯独没有 HPU：

- `comfy/model_management.py::get_torch_device()` 只认 `xpu` / `npu` / `mlu`，其余全部 fallback 到 `torch.device(torch.cuda.current_device())`。
- `get_total_memory()` / `get_free_memory()` 走 `torch.cuda.memory_stats()` + `mem_get_info()`，没有 HPU 分支。
- `synchronize()`、`soft_empty_cache()`、`get_offload_stream()` 同样没有 HPU 路径（offload stream 只实现了 cuda/xpu/npu）。
- `comfy/pinned_memory.py` 直接调 `torch.cuda.cudart().cudaHostRegister()`。
- `comfy-kitchen`（Comfy 官方 kernel 库）backend 只有 `cuda / hip / triton / eager / ascend`，全库 137 处 `torch.cuda` 或 `device.type == "cuda"` 早退；唯一的 HPU 痕迹是 eager 后端设备白名单里列了 `"hpu"`：
  `comfy_kitchen/backends/eager/__init__.py:145: frozenset({"cpu", "cuda", "mps", "xpu", "hpu", "meta", "*"})`
  这意味着 HPU 上所有 comfy-kitchen 加速路径都会退到 eager 后端，能跑但会慢，且 fp8/int8 量化路径基本不可用。
- 官方文档 `comfy-org/docs` 的 Intel 安装页只讲 Arc / XPU，没有 Gaudi。

对比参考：昇腾 NPU 是有上游支持的（`comfy/model_management.py` 里的 `npu` 分支 + [PR #16057](https://github.com/Comfy-Org/ComfyUI/pull/16057) 只改了 `model_management.py` 19 行 + 单测 81 行，就补上了 NPU 的 async offload streams），并配套 [昇腾官方 ComfyUI 接入文档](https://github.com/ascend/docs/blob/main/sources/comfyui/quick_start.md)。**这就是"HPU 上游化"该长什么样子的模板**：一个有边界的小 patch，而不是重写框架。

## 可复用的现成资产（按价值排序）

### 1. huggingface/optimum-habana —— 最完整的 Gaudi 扩散栈，但已 EOL

- 2026-10-31 起**官方停止支持**；最后版本 `v1.21.2`（2026-10-01），对应 Gaudi SW **1.24** + Transformers 4.55.x —— 正好覆盖你服务器的 1.24.2 环境。官方声明"没有推荐的替代品，可以 Apache 2.0 fork"。
- Diffusers 覆盖比预想全，`optimum/habana/diffusers/pipelines/` 下有：`stable_diffusion`、`stable_diffusion_xl`、`stable_diffusion_3`、`flux`、**`qwenimage`**、`wan`、`cogvideox`、`stable_video_diffusion`、`i2vgen_xl`、`controlnet`。
- 还有 `attention_processor.py`、`autoencoders`、`unet_2d_condition.py`、`wan_transformer_3d.py` 等"哪些层在 HPU 上要特殊处理"的现成答案。
- **对你的意义**：Qwen-Image 在 Gaudi 上已经有跑通的 diffusers 路径，说明算子层面可行；但要注意这是 diffusers 管线，不是 ComfyUI 的原生模型代码路径，不能直接复用权重加载/图执行。

### 2. Habana `gpu_migration` + Model-References —— 降低补丁量的最大杠杆

- `habana_frameworks.torch.gpu_migration`：把 `torch.cuda.*` 自动映射到 HPU 的兼容层，官方定位就是"快速迁移强烈依赖 CUDA 的模型"。
- `HabanaAI/Model-References` 的 `PyTorch/generative_models/stable-diffusion`（在 `1.22.0` 及更早 tag 中；`main` 已重组、该路径 404）含 SD 的完整 Gaudi 参考实现，带 `ops_bf16.txt` / `ops_fp32.txt` 这类"哪些算子必须在 fp32 跑"的清单，以及 `hpu_config_*.yaml`。
- **对你的意义**：先用 gpu_migration 把 ComfyUI 跑起来（能跑通再逐段去掉），比一上来手改 137 处 `torch.cuda` 现实得多。

### 3. CloudBrigade/stable-diffusion-webui @ `hpu_devices` —— 唯一先例

2023-10 的一个 A1111 fork，`hpu_devices` 分支 6 个 HPU commit，实际改动仅 11 个文件、约 120 行（排除自动生成的 `manifest.json`）。它改了：

`modules/devices.py`（`is_available()` 探测 → `get_hpu_device_string()`、`torch_gc()` 空操作、`autocast("hpu")`）、新增 `modules/habana.py`（**HPUGraph 捕获/回放 + `DeviceRunner` + `htrandom` 替换 RNG**）、`modules/processing.py`（权重整体转 bf16、`CUSTOM_SOFTMAX_FLAVOR=1`）、`modules/rng.py`、`modules/sd_models.py`、`requirements.txt`（加 `apex`）。

**对你的意义**：这份改动清单几乎可以 1:1 映射到 ComfyUI 的 `model_management.py` / `model_patcher` / `sampling` / `pinned_memory`。缺点是 2023-11 停更、基于旧 A1111，只能当"坑位地图"用，不能当代码用。

### 4. vllm-project/vllm-gaudi —— 工程结构范式

66 stars、仍在活跃维护（2026-11-05 还有提交），定位是"Community maintained hardware plugin for vLLM on Intel Gaudi"。它用 `vllm_gaudi/patches/` 这一层做 monkeypatch，而不是维护一个巨型 fork。

**对你的意义**：如果最终目标不只是"自己能用"，而是"让别人也能用"，可以参考这个形状做 `comfyui-hpu` 插件 + 少量 patch，而不是长期维护一个落后的 ComfyUI fork。

### 5. 看似相关但不可用

- `intel/llm-scaler` 的 `omni` 子项目：**ComfyUI on Intel XPU**（Arc B 系列 / BMG），有 `ComfyUI-OmniXPU`、`omni_xpu_kernel`、`patches` 一整套——架构参考价值很高，但目标是 XPU，不是 HPU。
- `opea-project/GenAIExamples` 的 `Text2Image/docker_compose/intel/hpu/gaudi`：Gaudi 上的文生图微服务，走 TGI/optimum-habana，不是 ComfyUI。
- `analytics-zoo/ComfyUI-GGUF-XPU`：XPU，非 HPU。

## 对实施方案的影响

1. **不用等别人，也不存在"直接拿来用"的选项**，但也不该从零发明：先按 CloudBrigade 的坑位清单 + 昇腾 NPU 的上游 patch 形状，在 `model_management.py` 加一个有边界的 `hpu` 分支。
2. **最小可行集**：`hpu` 设备探测、`get_total_memory`/`get_free_memory`（用 `memory_stats()` 的 `Limit`/`InUse`，HPU 文档没有 `mem_get_info`）、`synchronize`、`soft_empty_cache`、`get_offload_stream` 返回 `None`、`set_torch_device`、`should_use_fp16/bf16`、`supports_fp8/int8` 全部降级。
3. **必须主动关闭**：pinned memory（cudaHostRegister）、async offload / Dynamic VRAM（2026 年的新机制，HPU 上没有对应实现）、comfy-kitchen 的 CUDA/HIP 后端、fp8 量化、sage/flash attention。
4. **两条路线，建议先走 A**：
   - A. 真 fork ComfyUI，补 HPU 后端 → 目标是 Qwen-Image-2.1 / Krea2 原生工作流可用，工作量大但可控。
   - B. 用 optimum-habana 的 `qwenimage` pipeline 做生成后端，ComfyUI 侧只写一个自定义节点 → 见效快，但 optimum-habana 已 EOL，长期维护风险高。
5. **值得试探上游**：Comfy 已经接受 XPU/NPU/MLU/IXUCA 这类分支，接受"小 patch 增加一个后端"的先例很明确。先开 issue/discussion 试水成本很低，而且能顺便确认官方对 HPU 的态度（是否有 Intel 内部在做）。

## 风险提示

- **生态在收缩**：optimum-habana 2026-10-31 EOL，官方明确"没有推荐替代品"。Gaudi 上的 transformers/diffusers 上游支持会逐渐落后，意味着 ComfyUI 侧要自己维护更多模型代码。
- **Lazy 模式 + HPU Graph**：HPU 默认 lazy 执行，动态 shape（分辨率和步数变化）会触发反复重编译；Model-References 有专门的 dynamic shape 文档，这是 diffusion 工作流的主要性能障碍之一。
- **bf16 是默认，不是可选项**：Gaudi 上 bf16 才有性能，fp16 支持有限；ComfyUI 很多模型默认 fp16，需要显式改走 bf16 或 fp32（这也影响显存估算和 VAE 精度）。

## 参考链接

- [ComfyUI 官方 model_management.py](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_management.py)
- [ComfyUI PR #16057（NPU async offload streams）](https://github.com/Comfy-Org/ComfyUI/pull/16057)
- [昇腾官方 ComfyUI 接入文档](https://github.com/ascend/docs/blob/main/sources/comfyui/quick_start.md)
- [Comfy 官方 Intel 安装文档（仅 XPU）](https://github.com/comfy-org/docs/blob/main/snippets/intel-install.mdx)
- [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen)
- [optimum-habana](https://github.com/huggingface/optimum-habana)（v1.21.2 为最终版本）
- [HabanaAI/Model-References](https://github.com/HabanaAI/Model-References)（SD 参考实现见 `1.22.0` tag）
- [Intel Gaudi PyTorch Python API](https://docs.habana.ai/en/latest/PyTorch/Reference/Python_Packages.html)
- [CloudBrigade/stable-diffusion-webui @ hpu_devices](https://github.com/CloudBrigade/stable-diffusion-webui/tree/hpu_devices)
- [vllm-project/vllm-gaudi](https://github.com/vllm-project/vllm-gaudi)
- [intel/llm-scaler（ComfyUI on XPU）](https://github.com/intel/llm-scaler)
- [opea-project/GenAIExamples Text2Image on Gaudi](https://github.com/opea-project/GenAIExamples/tree/main/Text2Image)
