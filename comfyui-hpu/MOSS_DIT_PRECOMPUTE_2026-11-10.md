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

RoPE通过conditioning payload保留FP32精度。普通单latent的`latent_shapes`为`[[B,C,H,W]]`，需按实际形状判定预计算条件。

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

## 卸载与测量经验

完整pipeline基线与候选卸载稳定值均为2,685,928,832 bytes。释放为异步过程，应等待稳定值并用相同工作流对照。文本单独卸载的约94MiB只适用于对应测试范围。

## 证据与交付

服务器目录 `/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-10-dit-next`：

- `baseline/product`、`precompute/product`：固定 caption 的真实工作流对照、准备次数、图片、HPU CSV。
- `cfg-batch`、`cfg-prepared`、`sdpa-modes.json`、`sdpa-centered.json`：未合入候选。
- `full-product`：完整用户图五次 warm 与主要验收；`weighted/product`：加权提示词诊断。
- `tests-preparation.log`、`tests-preparation-shapes.log`、`tests-final.log`：回归与复验。
- `product-source` / `baseline-source`：受控卸载比较的两份源码版本。

固定caption的缓存使用验证为18.764s，单次结果。后续路线见 [BF16复盘与FP8判断](MOSS_OPTIMIZATION_STRATEGY_2026-11-09.md)。
