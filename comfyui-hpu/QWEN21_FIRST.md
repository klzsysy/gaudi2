# Qwen Image 2.1 Turbo 首版

2026-11-13（Asia/Shanghai）。基线产品 `34e2649e`，首版产品 `9f2ff96e`，单卡 Gaudi2、Habana 1.24.2、eager。执行只使用独立 ComfyUI-Gaudi checkout，调查资料保存在本仓库。

## 输入与工作流

用户指定的 [官方基础文生图模板](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_t2i.json) 使用 25 步基础模型。依照同仓库 [官方 Turbo 模板](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_turbo_t2i.json) 替换为 Turbo 模型和 8 步 ManualSigmas，保留基础模板的示例提示词、ResolutionSelector、增强器系统提示词、PreviewAny、QwenImage21Cache、PNG 保存和子图组织。

- 服务器主模型 `qwen_image_2.1_turbo_bf16.safetensors` 的 header 核实为 265 个 BF16 张量。
- 基础文本编码继续使用已有 `qwen3vl_8b_int8_convrot.safetensors`，复用已有 `qwen_image_2.1_vae_bf16.safetensors`。
- 新下载文生图增强器 `qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors`，9,471,072,252 bytes，SHA256 `9182abae56fe05459840a86d22abd21f972061c92fce032630af680c8c5178d3` 与官方一致。使用服务器既有 HF_ENDPOINT/HF_TOKEN 成功完成，含校验约 146 秒。
- 增强器含 310 组 INT8 ConvRot 权重，未包含 MTP 张量。MOSS 的 GDN、norm、KV、RoPE 与输出头路径可复用，MTP 加速取决于模型是否包含对应权重。
- 默认 1024²、CFG 1、Euler，调度 `1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0`。直接图和增强图分开计时。

## 基线验收

首轮现行产品直接图 cold 33.697 秒，三次 warm 8.223/8.273/8.262 秒，中位 8.262 秒。warm 文本编码约 0.51 秒、采样 6.88 秒、VAE 0.54 秒。fallback 计数为空，进程 allocator 峰值 42.92 GiB。每次强制重算编码、采样、解码，保留 loader 输出。官方长提示词样图已目视核对人物、墨镜、手袋、服装和绿色几何背景。

最终实现的验收结果见后文。

原始资料位于服务器 `/home/sonny/gaudi2/comfyui-hpu/performance-qwen21-2026-11-13`，本地忽略目录 `results/qwen21-first-2026-11-13`。可复现入口为 `probes/qwen21_workflows.py`、`probes/qwen21_run.py` 和 `probes/qwen21_server.py`。

## 候选取舍与兼容修复

Attention＋FFN 两区原型首次三次 warm 为 6.184/6.215/6.325 秒，但切换 prefix-cache callable 后触发 Dynamo `recompile_limit (8)`，另一轮真实流程回落至约 7.73 秒。首版采用 FFN 单区静态 recipe，Attention/cache callbacks 保持正常执行。五次 warm 为 7.748/7.726/7.718/7.746/7.714 秒，中位 7.726 秒，相对现行基线约减少 6.5%。适用普通 BF16 Linear、HPU 推理，CPU、梯度、训练、禁用编译、forward hooks 和自定义/量化 operations 走原入口。

官方增强器的 presence_penalty=1.5 触发 Habana 1.24.2 的 bool scalar `index_fill_` schema 错误。布尔 mask 使用整数 1 填充后正确，保留原惩罚和采样参数，同时覆盖普通生成和 Qwen3.5 MTP 路径。新增正常 generate 入口回归验证初始重复 token 与后续 token 都正确入 mask。

原 INT8 增强图成功，2548 输入、1124 输出 tokens，首轮 193.540 秒（期间有 CPU 权重转换，单列），无并行转换的第二轮 161.042 秒。该检查点没有 MTP 参数，吞吐约 7 tokens/s。

复用既有转换脚本，将同一 INT8 文件的 310 组权重解码为本地 `qwen3.5_9b_qwen_image_2.1_pe_t2i.bf16_from_int8.safetensors`。输出 18,819,722,496 bytes，转换 41.488 秒，来源 metadata 指向原文件。BF16 常驻避免反复解量化并使用已有 dense MLP 编译路径。默认 HPU 工作流的 PE_model 采用此文件，原 INT8 API 验收图另行提供。两种计算路径的自然采样文本可能不同。


## 首版运行结果

| 场景 | 首次进程/形状运行 | 后续运行 |
|---|---:|---:|
| 官方长提示词、1024²、8 步、直接图 | 19.015s | 五次中位 7.726s，7.714–7.748s |
| 新增强器 BF16 常驻、thinking 开启、1024²完整图 | 70.569s | 42.340 / 42.229s |
| 英文 RIDE 海报、512² | 2.770s | 2.581s |
| 清理执行缓存、卸载后重载 1024² | 19.251s | 成功 |

增强器 BF16 三轮均为 2548 输入、1316 输出 tokens，warm TextGenerate 34.370/34.259s，采样约 6.57s。原 INT8 输出 1124 tokens，两个文件的自然生成内容不同，这些总时延反映各自真实工作量。BF16 样图为蓝釉茶壶和一只匹配茶杯，INT8 样图为青花茶具。已检查壶嘴、把手、杯子数量和窗边光照，未进行广泛质量评估。

FFN 路径直接图进程 allocator 峰值 44.36 GiB，含增强器与尺寸切换的累计峰值约 63 GiB。成功流程 fallback 计数均为空。基线重复图逐像素一致，FFN 与基线 RGB MAE 为 12.692/255，人物构图、手袋与衣服局部细节发生变化。

工作流保留默认 refine_prompt 关闭，thinking_mode 预设开启，增强器上限为官方模板的 4096。用户可在同一子图启用增强。主模型使用服务器实际 BF16 权重，增强器是用户指定 INT8 文件的本地 BF16 解码版本。


## 数值、生命周期与回归

九组实际 block 输入覆盖层 0/16/31、步 0/4/7、4096 个图像 token。编译版相对 eager 的 block RRMSE 为 0.00180%–0.01142%，所有输出有限。每组 FFN 抽取八个实际输入位置，与完整 FP64 Linear→SwiGLU→Linear 参考比较，编译/原版误差比为 0.99988–1.00228。局部误差接近，完整图的细节变化已单列。

最终四个测试文件合计 80 passed、1 skipped。测试覆盖普通生成的 presence penalty、Qwen3.5 decode/MTP、Qwen prefix cache、融合/分离 FFN、变长与权重更新、CPU、梯度、训练、禁用编译、后加 hooks 和自定义 operations。

`/free`执行后较早的5–8秒观测仍包含清理中资源，15秒与25秒两次稳定值均为911,596,416 bytes（约0.849GiB）。本轮完整卸载重载出图成功。原始 8 秒观测保留，不能作为最终卸载占用。

已安装可直接打开的 `user/default/workflows/Qwen-Image-2.1-Turbo-HPU.json`。产品提交为 `0c764d59`（mask兼容）和 `9f2ff96e`（FFN编译），三个产品源码与两个测试文件的本地/服务器 SHA256 全部一致。首版未覆盖2K、批量或图像编辑。


普通 `start-hpu.sh` 服务最终验收直接图三次成功，17.389/7.712/7.733 秒，后两次改变 noise seed 并复用节点文本缓存。首次启用增强器完整图 69.834 秒，最终可见扩写文本与观测服务 BF16 warm 完全相同。服务保持在 `http://192.168.10.21:8189`，队列为空。通过 userdata API 读回已安装 UI 工作流，与本仓库交付文件逐字节一致。


## 外层 seed 与 PE 缓存核查

2026-11-13 用户截图对应的最近12次请求来自带 KSampler 的基础文生图子图，主模型改为 Turbo BF16、8步 Euler/simple，基础编码器后改为 Qwen3-VL BF16，PE使用BF16解码权重，thinking关闭。子图 KSampler 的 `control_after_generate` 为 fixed，外层 seed 同时连到 KSampler.seed 和 TextGenerate.sampling_mode.seed。0是可复现的固定种子，前端需在实际种子来源上启用 randomize 才会每次提交新值。

历史显示首次 seed=0 完整执行27.116秒，随后两次相同输入为0.005/0.004秒，TextGenerate和KSampler均在execution_cached内。最后 seed=1 完整执行21.288秒，两个节点均重新执行。切换基础编码器为BF16的那次，PE命中缓存、编码与采样重算。PE执行由其输入及依赖变化触发，模型精度选择沿节点依赖传播。

原始请求保存在忽略目录 `results/qwen21-seed-2026-11-13`，去除提示词后的摘要见 `evidence/qwen21-seed-cache-2026-11-13.json`。此处为用户当前图的核查，首版固定Turbo调度的性能数据仍按前文条件解释。
