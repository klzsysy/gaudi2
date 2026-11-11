# MOSS BF16 硬件下界与 RTX 5090 对标

日期：2026-11-10（Asia/Shanghai）。产品 `b1163be0`。本轮只读服务器既有测量、checkpoint header、CPU tokenizer，并检索公开一手实测；没有新跑 HPU benchmark、修改产品或中断日常服务。

## 判断

- 当前 1024²、25 步、CFG 2.5 的 **DiT 必要矩阵工作约 2.496 PFLOP**。全部达到 Gaudi2 BF16 峰值时，矩阵部分下界约 **5.78 s**；实际 DiT median **18.384 s**，约为此下界的 **3.18 倍**。这不是承诺整个采样节点能做到 5.78 s。
- 完整 298-token 扩写图 median **29.673 s**。把同输出轨迹的历史诊断权重读取和 prefill/编码矩阵加入，得到约 **6.95 s 的不完整乐观工作量下界**；尚未计入状态、非线性、VAE、运行时等成本。不能把它称为已证明可实现的整图极限。
- 接近当前设置的 RTX 5090 一手报告：**BF16 14.7 s、INT8 5.8 s**，均为 1024²/25 步、文本缓存后的整图。另一位作者报 INT8 采样约 **7 s**。当前我方文本缓存的单次使用验证是 **18.764 s**，尚未达标。
- 建议验收分两层：先将 BF16 文本缓存出图压到 **≤14 s**，再追 **10–12 s**；最终“达到 5090 量化路径的体验”应以 **约 6–7 s** 作为外部目标。后一目标已经逼近本卡纯 BF16 矩阵下界，不能保证仅靠 BF16 调度优化达成；现有原生 FP8 适配可能需要重新进入路线，但本轮不改精度、不实施 kernel。

## 硬件数字的正确口径

| 指标 | Gaudi2 | RTX 5090 |
|---|---:|---:|
| 稠密 BF16 输入、FP32 累加 | 432 TFLOP/s | 209.5 TFLOP/s |
| 设备内存 | 标称 96 GB HBM2e | 32 GB GDDR7 |
| 标称内存带宽 | 2.45–2.46 TB/s | 1.792 TB/s |

依据：[Intel Gaudi 产品对照，第 3 页](https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf)、[Gaudi2 白皮书，第 6 页](https://habana.ai/wp-content/uploads/pdf/2021/gaudi2-whitepaper.pdf)、[NVIDIA RTX Blackwell 白皮书，第 47–48 页](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)。NVIDIA 表中的 BF16 `209.5/419` 是稠密/稀疏口径；FP16 累加、FP4 AI TOPS、INT8 也不能混用为 BF16 峰值。Gaudi2 的矩阵规格约为 5090 的 2.06 倍、带宽约 1.37 倍；这不保证算子、向量处理及端到端性能同比领先。

本机历史 4096³ BF16 GEMM 实测为 **275.76 TFLOP/s**，约标称的 63.8%。它并非本轮真实 shape 的持续上限，只用于检查估算尺度。把所有 DiT 矩阵都假设为这个速率，会得到 **9.05 s 的矩阵预算**，仍需另外考虑不可重叠的非 GEMM 工作。

## DiT 工作量复算

从真实 BF16 checkpoint 和当前源代码确认：42 层、宽度 `d=3072`、FFN 宽度 `f=8192`、24 heads、head_dim=128；QKV 与 up/gate 已合并。1024² 图像的 latent 为 64×64，共 4096 image tokens。

对固定 caption 和实际负提示词，用服务器现有 `MOSSImageTokenizer` 在 CPU 上复算：正条件 **307 tokens**、负条件 **23 tokens**，因此两个 CFG 前向的序列长度分别为 **4403、4119**。不能拿扩写输出 298 tokens 直接当作 conditioning 长度，也不再假设正负条件一样长。

FMA 计 2 FLOPs。每层、单 CFG 分支长度为 N 时：

```text
QKV + output + gated FFN: 2N × (4d² + 3df)
QKᵀ + AV:               4N²d
整次主干:                25 × 42 × [F(4403) + F(4119)]
```

| 矩阵部分 | 工作量 | 按 432 TFLOP/s 的时间 |
|---|---:|---:|
| 主干 Linear / FFN | 2.026677 PFLOP | 4.6914 s |
| 主干 attention 两次矩阵乘 | 0.469035 PFLOP | 1.0857 s |
| 图像输入输出、时间调制、条件预计算矩阵 | 0.000491 PFLOP | 0.0011 s |
| 总计 | **2.496203 PFLOP** | **5.7782 s** |

没有把 softmax/exp、SiTU/tanh/sigmoid、norm、RoPE、残差、额外内存流量、同步/提交当作免费可实现的工程事实；上表只是去掉这些费用后必需矩阵工作的乐观下界。也不假设当前 math attention 的每一种内部精度/布局都能直接达到 BF16 MME 峰值。

按 18.384111 s 实测倒算，**有效矩阵吞吐约 135.78 TFLOP/s，约理论峰值的 31.43%**。这是公式工作量除以采样节点时间，不是硬件 FLOP counter。此前 AIP 约 61% 表示活动指标，不能据此声称已达到 61% 算力峰值，也不能线性换算“达到 AIP 80% 会节省多少秒”。

| 假定矩阵吞吐 | 单纯矩阵预算 | 含义 |
|---|---:|---|
| 标称峰值的 100% | 5.78 s | 极度乐观，非整节点交付目标 |
| 标称峰值的 80% | 7.22 s | 尚未计不可重叠的其它开销 |
| 标称峰值的 60% | 9.63 s | 可用来制定 10–12 s 的研究目标，非预测 |
| 本机旧 4096³ GEMM 的 275.76 TFLOP/s | 9.05 s | shape 不匹配的参考值 |

主干矩阵中 Linear 占约 81.2%。若未来只把它变成真正的 FP8 MME、attention 仍保持 BF16，按厂商峰值粗算矩阵下界约 **3.43 s**，但没有计量化/布局/scale，也没有证明质量；这只说明 6–7 s 外部目标可能需要精度路线，而不是证明换 FP8 就能达标。

## 完整 pipeline 的下界为何更不确定

当前完整图的一次中位数代表请求可相加拆解为：扩写 **9.261122 s**，正负编码 **1.693151 s**，KSampler **18.451141 s**，VAE **0.136487 s**，图片输出 **0.113426 s**，其余约 **0.017642 s**；总计 **29.672969 s**。单阶段独立 median 分别为扩写 9.162109、KSampler 18.411157，不能拿独立中位数强行相加。

历史单独同步诊断对同一 298-token 输出记录了 129 次主模型 verify、258 次 MTP head、388 次 lm_head，输入 prefill 长度 1970。最新无同步请求输出 IDs 与其一致，但没有新记录所有内部调用次数，因此以下是沿用该诊断轨迹的估算。

BF16 checkpoint 静态字节为：32 层主干 13.839 GB，lm_head 2.034 GB，MTP 0.487 GB。embedding 只读选中行，不按整个 2.034 GB 每次扫描；视觉塔本次不运行。假设每次调用各读一次权重、无额外费用：

```text
权重流量 ≈ 129×13.839 + 388×2.034 + 258×0.487 = 2700 GB
理想传输时间 ≈ 2.700 TB / 2.45 TB/s = 1.102 s
prefill + 正负编码的主干 Linear 计算下界 ≈ 0.074 s
DiT + 上述文本必要工作 ≈ 5.778 + 1.102 + 0.074 = 6.954 s
```

这里既没有包含 KV/GDN 状态流量和递归计算，也没有包含完整 prefill attention、采样控制、host 同步、VAE 和输出；小 M GEMM 也不能以 432 TFLOP/s 持续执行。权重驻留/跨调用复用、MTP 树或接受率改变又会改变工作量。因此 **约 7 s 只是限定轨迹和存储假设下的不完整乐观参照，不是整图可实现下界的完整证明**。29.67/6.95≈4.27 不能解读为已发现可兑现的 4.27 倍优化。

## 公开 RTX 5090 结果

检索日期 2026-11-10。以下均为作者自己机器上的公开报告，并非我们复现的严格 A/B。优先采用包含模型、尺寸、步数和计时范围的来源。

| 来源 | 设置 | 作者报告时间 | 与我方的可比性 |
|---|---|---:|---|
| [KOBO Lab](https://localai-kobo.com/moss-image-comfyui/)，测试 2026-10-31，更新 11-05 | SFT BF16＋Qwen BF16；1024²、25 步、CFG 2.5、er_sde/simple | **14.7 s/张** | 最接近 BF16 设置；同提示词只换 seed，文本编码命中缓存；未启用 TextGenerate |
| 同一 KOBO Lab 测试 | SFT INT8 ConvRot＋Qwen W4A8；其余同上 | **5.8 s/张** | 量化 DiT 的用户体验目标；同样不含重算文本；使用 flux2-vae_small，我方用已有完整 VAE |
| [DSLab](https://dslab.tw/moss-image-lora)，2026-10-20 | SFT INT8、1024²、25 步；Qwen W4A8 | **采样约 7 s（3.5 it/s）** | 采样段旁证；另有 9:16 全图 8.7 s，不把它当作方图或扩写同口径 |
| [Niklas Lenz](https://niklaslenz.com/image/moss-image-sft-int8-comfyui-5090/)，测于 2026-10-20 | SFT INT8、1024²、50 步、CFG 3、euler/sgm_uniform、负分支跳层 22 | **12.81 s**，扩写后的文本为 **12.99 s**；各 12 图 median | 步数/采样器/SLG 不同；不直接除二宣称是 25 步实测 |
| 同一 Niklas Lenz 测试 | Qwen 自带 prompt upsampler，单独测 12 文本 | **扩写 median 1.4 s** | 文本事先生成并复用，1.4 s 不包含在上面的 12.99 s；未获得与我方 1970→298 tokens 等量的证据 |

KOBO 计时从提交到图片保存，ComfyUI `b5a43215`、torch `2.14.0+cu130`、Windows 11、9950X3D/128 GB。首张含加载/编译为 INT8 135 s、BF16 263 s，不能混进 warm 比较。作者没有发布重复样本数/原始逐次计时，因此 5.8/14.7 是方向性外部基线。

Niklas 页面给了逐张时间、工作流及原始结果链接，计时范围区分比较清楚；原始 JSON 链接在本次浏览工具读取时返回 403，未据此声称已独立复算其 median。它支持“INT8 的约 6–7 s/25 步量级值得认真对标”，但不建立当前完整扩写图的跨机胜负。

另外检查但不采用为主基线：

- [Forge Neo 原生实现](https://github.com/mishrasiddharth08/project-invisible-moss-image)：INT8、1024²、50 步 39.96 s，包含加载/转移/编码等，且后端不同。
- [RTX 4090/5090 用户对比](https://www.reddit.com/r/StableDiffusion/comments/1xk3n70/i_unlocked_bf16_on_an_rtx_4090_and_compared_it_to/)：MOSS Base、1024×1536、50 步×5 次，报告 BF16 98.4 s、INT8 42.4 s；CFG、缓存、Generate 总计口径不全，不能按像素/步数线性缩放为当前工作流成绩。
- Turbo 4/8 步、CFG 1、Spectrum/跳步等改变模型或采样工作量的结果不用于本轮同工作量目标。

注意模型名称：我们此前的“BF16 vs INT8”主要是 **Qwen 文本编码器** 对比，DiT 始终用 `moss_image_sft_bf16`；上表 5.8–7 s 的 INT8 则指 **MOSS DiT 本体也使用 INT8 ConvRot**。切换一个 Qwen INT8 文件不会自动得到这一档采样性能。

## 差距及目标

相对于 KOBO 的缓存后出图口径，用我方最后一次日常缓存验证 18.764 s 作近似：

- 对 BF16 14.7 s，我方耗时约 **1.28 倍**；需要再减少 **4.06 s / 21.7%**。分辨率/步数/CFG 相同，但 caption、VAE、版本和测量样本数不同，不能称为严格跨卡差距。
- 对 INT8 5.8 s，我方耗时约 **3.24 倍**；需要再减少 **12.96 s / 69.1%**。以另一作者 7 s 采样为参照，我方 DiT 18.384/7≈**2.63 倍**。
- 不能用整图 29.673 s 除以不含扩写的 5.8 s 来报告 GPU 差距，也不能拿我方 9.16 s 扩写直接除以外部不同长度的 1.4 s。

下一轮建议将可验收目标固定为：

1. **BF16 基本门槛**：1024²/25 步/CFG 2.5、不减少工作量，文本缓存整图五次 warm median ≤14 s；随后挑战 10–12 s。这是工程目标，不是已测性能。
2. **完整图并行目标**：保留 1970-token 输入、298-token 基准扩写，完整重算先 ≤20 s；不能只靠缓存包装成达标。
3. **最终 5090 量化体验门槛**：文本缓存整图约 6–7 s，同时维持质量。由于 5.78 s 的 BF16 矩阵预算几乎耗尽目标，纯 BF16 达成的工程余量非常小；若真实 shape 的 GEMM/非 GEMM 归因显示 BF16 已接近可用吞吐，应转向厂商原生 FP8 的校准适配，而不是继续承诺 BF16 无限制提速。
4. **最终同机对照契约**：导出相同正负 caption、分辨率、步数、CFG、SLG 设置，分列 DiT/编码/扩写/保存时间；再用同 prompt 和生成 token 数测完整图。没有这一对照之前，将公开数字作为目标，不宣称已经追平或精确落后多少倍。

算式可复算：[estimate_moss_bf16_roofline.py](probes/estimate_moss_bf16_roofline.py)。我方原始实测见 [DiT 预计算记录](MOSS_DIT_PRECOMPUTE_2026-11-10.md)、[pipeline 诊断](MOSS_PIPELINE_TIMING_2026-11-10.md)；服务器证据根目录仍为 `performance-2026-11-10-dit-next` 和 `performance-2026-11-10-pipeline`。
