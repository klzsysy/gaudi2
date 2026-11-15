# ComfyUI Gaudi2优化与适配记录

更新于2026-11-15（Asia/Shanghai）。产品版本为`dfca7cd6`，单卡Gaudi2、Habana1.24.2、eager＋区域编译。当前验收与未决问题见 [状态](PROGRESS.md)，环境与复现命令见 [ENVIRONMENT](ENVIRONMENT.md)。

## 按问题查找

| 主题 | 内容 |
|---|---|
| [优化方向与重启条件](optimization/STRATEGY.md) | FP8/BF16投入顺序、剩余机会、何时值得重测失败候选 |
| [测量与质量方法](optimization/MEASUREMENT.md) | 固定caption/MTP、cold/warm、A/A、logits、生命周期和测量陷阱 |
| [性能预算与硬件计数](optimization/BOTTLENECKS.md) | 节点/设备费用、AIP功率公式、HBM/BMON、理论下界与5090快照 |
| [BF16 DiT](optimization/DIT_BF16.md) | 分区、静态recipe、条件预计算、norm/RoPE与共享扩区失败 |
| [FP8 DiT](optimization/DIT_FP8.md) | 原生Linear接入、范围/scale、block融合、V量化和SiTU候选 |
| [DiT attention](optimization/ATTENTION.md) | BF16/FP8精度、head布局、8192断点、长序列query批次、SRAM调查 |
| [文本权重与量化](optimization/TEXT_PRECISION.md) | INT8/ConvRot、BF16常驻、FP8文本、W4A8及输出头路径 |
| [原生INT8 MME验证](optimization/INT8_NATIVE_MME.md) | HAL与直接Synapse验证阴性，按用户条件停止开发，原计划归档 |
| [文本decode与KV](optimization/TEXT_DECODE.md) | 采样、MLP/norm、KV状态、短步RoPE、前缀缓存边界 |
| [GDN递推](optimization/TEXT_GDN.md) | prefill分桶、并行递推、八行扩区、四行clone及TPC候选撤回 |
| [运行时与故障](optimization/RUNTIME.md) | lazy/Graph、SIGPIPE、异步输入、段错误、驱动和依赖恢复 |
| [算子与外部源码](optimization/KERNEL_REFERENCES.md) | 官方bridge、TPC接口、vLLM/INC/1Cat/JChen固定版本核查 |
| [Qwen Image 2.1 全BF16深入优化](QWEN21_BF16.md) | PE/DiT分项、trace、舍入边界、完整图保值与性能验收 |
| [Qwen Image 2.1 最新2MP工作流](QWEN21_PERFORMANCE_2026-11-15.md) | 1784-token扩写、2MP/CFG1.2耗时、GDN/MLP提交优化与回退复测 |
| [单token BF16硬件利用率](optimization/QWEN21_BF16_SINGLE_TOKEN.md) | AXI返回beat校准、约33% HBM带宽利用率、整步重放与GDN融合数值边界 |
| [Qwen PE后续文本加速](optimization/QWEN21_TEXT_NEXT.md) | 现有MTP草稿头复用、固定窗口与自然扩写收益、目标验证数值边界 |
| [Qwen Image 2.1 Turbo 首版](QWEN21_FIRST.md) | 官方文生图、匹配增强器、BF16 常驻、FFN 编译与首版验收 |
| [后端与模型正确性](ADAPTATION.md) | HPU接入、Krea2 E4M3解码、Qwen VAE原地布局问题、MOSS组合 |
| [模型共享与生命周期](MODEL_STORAGE.md) | 共享目录、历史备份、组件复用及量化文件所有权 |

## 阅读与复用

每个主题集中保留实验条件、有效/无效候选、质量与资源成本、取舍理由、复测条件及证据入口。指标采用各轮同条件基线，跨轮自然文本、模型、caption和缓存状态需要核对。

- `evidence/`保存结构化验收、数值与计时，`kernel-audit-2026-11-10/`保存外部源码commit、哈希和永久链接。
- `probes/`保存复现实验，`patches/`保留失败实现及恢复补丁，`workflows/`保存工作流。
- 大型原始资料位于服务器`/home/sonny/gaudi2/comfyui-hpu/performance-*`，本地`results/`为忽略的资料副本，各主题给出具体目录。
- 整理前完整按日期记录可从本仓库提交`9e622ab`查阅，例如`git show 9e622ab:comfyui-hpu/MOSS_THREE_STAGE_EXECUTION_2026-11-12.md`。现在的主题文档采用后续实测修正后的结论。

## 仓库与执行边界

所有调查、计划、失败路径、服务器信息和优化经验进入本仓库。兄弟ComfyUI-Gaudi仓库存放产品代码、便携脚本和测试。

服务器测试checkout为`/home/sonny/ComfyUI-Gaudi`，原始`/home/sonny/ComfyUI`只读。单卡HPU任务串行，下载使用已有HF_ENDPOINT/HF_TOKEN，已有模型组件优先复用，凭据不进入记录。继续实机工作前按AGENTS读取本文件、状态及环境。
