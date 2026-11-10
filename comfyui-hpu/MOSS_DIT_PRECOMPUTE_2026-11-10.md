# MOSS DiT 静态条件预计算

日期：2026-11-10（Asia/Shanghai）。基线 `441d459d`，产品提交 **`b1163be0566fdd5138f9b68c76301476f133a313`**。本轮保持 BF16、1024²、25 步 er_sde/simple、CFG 2.5，复用现有算子，不自写 kernel、不修改模型权重。

## 落地结果

固定同一 caption、seed、分辨率和步数，关闭扩写，各 1 cold+5 warm；计算节点强制重算，保留 loader：

| 指标 | 基线 | 条件预计算 |
|---|---:|---:|
| KSampler/DiT median | 19.581401 s | **18.384111 s** |
| 无扩写整图 median | 21.641613 s | **20.524122 s** |
| 每请求 RoPE / text_embedder 次数 | 50 / 50 | **2 / 2** |
| DiT AIP 平均 | 56.79% | **60.97%** |
| allocator 峰值 | 39.385 GiB | 39.498 GiB |

DiT 减少 **1.197291 s（6.11%）**，峰值显存增加约 0.113 GiB。图片对基线 MAE **0.11504/255**、PSNR **57.50 dB**；基线自身重复 MAE 0.11488，候选自身重复 0.11558。差异与当前重复运行波动相当，不声称逐像素相等。

重新启用用户当前完整扩写图（文本 seed=1），1 cold+5 warm：

- warm `[29.672969,29.731213,28.924530,29.234977,29.957261]`，median **29.672969 s**；对应前轮同图为 **30.786376 s**。
- 五次全部 298 tokens，逐 token ID 与前轮代表请求一致。文本阶段 median 9.1621 s，DiT/采样 median 18.4112 s。
- AIP 全图/文本/DiT 平均 **46.02% / 21.21% / 61.10%**，峰值 67%。仍未达到 80%，这些数值不是 MME FLOPs 或 HBM 带宽利用率。
- fallback 全为空。完整工作流的收益约 1.11 秒，不能把 DiT 内部 6.11% 直接套到整图。

## 实现与边界

在 `BaseModel.MOSSImage.extra_conds` 中调用模型的 `preprocess_text`，将本次执行的 text_embedder 输出和 RoPE 放入 conditioning。模型不新增跨执行 tensor cache；保留原始 context，用于不适用预计算时的原路径。

RoPE 为 FP32，用 conditioning payload 保留其精度，避免 `BaseModel._apply_model` 把所有浮点附加 tensor 自动转成 BF16。普通单 latent 的 `latent_shapes` 实际是 `[[B,C,H,W]]`，不是 None；首轮代码已根据真实运行参数修正这一识别，最终每请求确实预计算两次。

优化仅针对 HPU BF16 的静态推理。训练、带梯度执行、hooks、区域、ControlNet、参考图、packed latent 及不支持的 model/transformer options 保留原路径。运行中形状、RoPE options、设备或 hooks 不匹配时重新计算；不缓存 timestep 相关调制。CPU/其它设备不启用新预计算。

验证包括：

- 新 preparation 专项 **19 passed**；原有 MOSS attention/block **21 passed**。先前 39 项与后续复验有重复，不累加。
- 同输入跨多个 timestep 的输出逐值一致；batch=1/2、FP32 RoPE、加权 mask、同长度条件合批、hooks 后加、改变尺寸/位置参数、参考图 fallback 均检查。
- conditioning 销毁后预计算 tensor 的 Python weakref 释放，未保存到 model 上。
- 真实 512/MTP auto、768/MTP off、1024 卸载重载、原 INT8 完整图两次通过。另补 768 加权花瓶海报，正负条件均使用预计算，latent/VAE finite、fallback 为空。
- 复核固定 caption 街景及加权海报，主体与 MOSS 大字正常。海报额外的无意义小字与既有模型表现一致，不把此 smoke 当作大规模文字质量评价。

## 主干候选的取舍

除预计算外，还实际测试了 CFG 合批和厂商 SDPA 执行模式，没有将未验证的建议当作交付收益。

| 候选 | 实测 | 结论 |
|---|---|---|
| 原始长度条件分开执行 | DiT 19.581 s | 基线 |
| 左 padding + mask 的 CFG 合批 | DiT 19.402 s，3 warm；峰值 41.241 GiB，图片 MAE 4.115/255 | 收益不到 1%，弱于预计算，未合入 |
| 合批 + 预计算，保留各条件原 RoPE | DiT 19.749 s，3 warm；峰值 41.114 GiB，图片 MAE 3.470/255 | 慢于 18.384 s 产品候选，未合入 |
| 厂商 FusedSDPA 默认模式 | attention 3.61 ms，block 9.40 ms | 不优于当前编译 math attention |
| 厂商 fast softmax | attention 2.33 ms，block 8.10 ms | 速度有收益，但未通过本轮数值门槛 |
| fast softmax 前中心化 K | block 8.11–8.15 ms | 未消除数值误差，未合入 |
| 厂商 FP32 softmax | attention 11.41 ms，block 17.20 ms | 更慢 |

SDPA 数据来自真实首层 N=4408 的捕获输入与实际 block，每种 5 组×10 次。当前编译 math attention 约 3.39 ms、block 9.19 ms。对相同 CPU FP64 block 参考，当前相对 RMS 误差 **0.3139%**；fast softmax 约 **0.7212%**，FP32 K 中心化后仍约 **0.6700%**。这不是完整生成质量结论，但不足以在默认路径用近似计算换速度。recompute 开关未改善该取舍。

另只读确认本机 Habana 1.24.2 的 `hpu/graphs.py` 会在 eager 下拒绝 `HPUGraph`，明确要求 lazy mode；本产品当前依赖 eager + torch.compile。没有直接切换全栈执行模式或套用 CUDA graph 开关。

## 卸载检查的修正

更严格的 `/free` 检查最初设了“低于 512 MiB”，没有达到。该阈值来自文本模型单独测试，不适用于完整 pipeline，因此不能据此宣称新预计算泄漏。

实际记录到 API 返回后仍在异步释放，初次观测约 32.32 GB，重试 flags 后逐步降到 **2,685,928,832 bytes**。随后恢复优化前的两份产品文件，在独立进程运行相同完整图并使用相同卸载流程，基线稳定值也为 **2,685,928,832 bytes**，与新版本完全一致。当前证据没有新增残留；没有把这 2.50 GiB 的既有残留误记为完全零占用，也没有未经归因就声称它全部属于某一种底层缓存。

最初严格阈值超时日志保留在 `restore.log`，基线完整采样/卸载记录在 `full-unload-control`。后续脚本按相同基线的稳定占用比较，重试异步卸载 flags，而非沿用错误的绝对阈值。

## 证据与交付

服务器目录 `/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-10-dit-next`：

- `baseline/product`、`precompute/product`：固定 caption 的真实工作流对照、准备次数、图片、HPU CSV。
- `cfg-batch`、`cfg-prepared`、`sdpa-modes.json`、`sdpa-centered.json`：未合入候选。
- `full-product`：完整用户图五次 warm 与主要验收；`weighted/product`：加权提示词诊断。
- `tests-preparation.log`、`tests-preparation-shapes.log`、`tests-final.log`：回归与复验。
- `product-source` / `baseline-source`：受控卸载比较的两份源码版本。

产品提交 `b1163be0`，本地未 push；只更新独立 `/home/sonny/ComfyUI-Gaudi`，没有修改原始 `/home/sonny/ComfyUI` 或模型文件。所有实验与失败记录保存在本调查仓库。

8189 日常服务已恢复，并以用户原图重新加载模型和节点缓存。首次恢复请求成功（cold 46.665 s），随后只改 KSampler seed 的文本缓存请求为 **18.764 s**，确认 TextGenerate 与正负编码均命中缓存。该一次日常请求仅作为使用验证，正式收益仍采用前面的五次 warm 对照。

本轮稳定收益是去除重复条件计算。主干约 18 秒仍占绝大多数；下一阶段应围绕真实 shape 的矩阵与 attention 设备时间，评估现有 BF16 路径的剩余机会及独立的执行模式/原生 FP8 试点，继续以完整延迟、精度和资源生命周期验收。
