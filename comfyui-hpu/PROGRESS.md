# 当前状态与未决问题

截至2026-11-13，产品分支`codex/gaudi2-hpu`，最终提交`ed9552e581acfa37c6be2ac6f1a86a629f21d542`。本文件提供交接快照，测试过程和候选取舍集中于 [主题索引](README.md)。

## Qwen Image 2.1 全BF16优化

当前组件为Turbo BF16、Qwen3-VL 8B BF16、匹配文生图PE的BF16解码文件和Qwen Image 2.1 VAE BF16。`17b4b231`接入原生RoPE与无mask attention编译，`ed9552e5`编译temperature=1.0的HPU采样概率。投影/RMSNorm保留原舍入边界，有mask前缀采用原数学执行。

用户832×1504、8步Euler/simple、CFG1、PE开启且thinking关闭，五次完整warm中位21.028→16.079s，PE11.416→10.593s，DiT8.089→4.006s。复用文本结果后的整图9.2185→5.130s，进程峰值76.645→60.335GiB。回退原数学路径复验为20.790s，收益可复现。

12组真实block逐元素一致，六组完整扩写token与RGBA PNG逐元素一致，含不同seed、512²/1024²、thinking开启与官方Turbo调度。159项回归通过、1项跳过。普通8189首次49.833s、换seed18.267s、重载42.926s成功，清理稳定占用约3.15GiB。当前验证范围为batch1文生图，2K及编辑另验。[深入记录](QWEN21_BF16.md) · [证据](evidence/qwen21-bf16-2026-11-13.json)

## Qwen Image 2.1 Turbo 首版

`0c764d59`修复官方增强器 presence penalty 的 HPU 布尔 mask 更新，`9f2ff96e`接入普通 BF16 FFN 区域编译。使用官方基础文生图结构与 Turbo 8 步调度，复用已有 BF16 主模型、Qwen3-VL 8B、Qwen Image 2.1 VAE。新文生图增强器已下载并校验，默认 HPU 图使用它的本地 BF16 解码文件。

1024²直接图五次 warm 中位 7.726s，基线8.262s。开启 thinking 增强器的完整图为42.340/42.229s，2548输入/1316输出tokens。原 INT8 同提示词自然生成1124 tokens，完整热运行161.042s。512²及卸载重载通过，清理稳定后约0.849GiB。80项回归通过、1项跳过。九组真实 block 相对 eager RRMSE 为0.00180%–0.01142%，九组 FFN 抽样对CPU FP64的误差比0.99988–1.00228。相同seed图像细节存在变化，本轮覆盖512²和1024²。普通8189最终直接图17.389/7.712/7.733s、首次增强图69.834s成功，工作流已安装，队列为空。[首版记录](QWEN21_FIRST.md) · [证据](evidence/qwen21-first-2026-11-13.json)

## 当前接受的实现

| 方向 | 现行实现与主要提交 | 详细记录 |
|---|---|---|
| 设备与模型 | HPU后端、E4M3解码表、VAE非原地残差 | [适配](ADAPTATION.md) |
| BF16 DiT | 两段编译/静态recipe、条件预计算、native norm、验证后BLOCKWISE | [BF16](optimization/DIT_BF16.md) |
| FP8 DiT | `2227f7d3`原生Linear、`0995638a`FP32 Q/K norm及block扩区 | [FP8](optimization/DIT_FP8.md) |
| Attention | `700bdb6d`单head、`f67e7f3f`混合AV、`69fe4a54`移除8192上限、`ef16c195`长序列分批 | [Attention](optimization/ATTENTION.md) |
| 文本精度 | INT8解码BF16、`148b252a`原生FP8文本 | [量化](optimization/TEXT_PRECISION.md) |
| 文本执行 | KV行clone、MLP/norm/attention编译、`34e2649e`短步RoPE | [decode](optimization/TEXT_DECODE.md) |
| GDN | `60da4e33`区域编译、prefill分桶、`41f01e68`并行递推、`a8f42df2`八行扩区 | [GDN](optimization/TEXT_GDN.md) |
| 运行时 | `441d459d`初始化后恢复SIGPIPE忽略策略，独立eager环境 | [运行时](optimization/RUNTIME.md) |

## 最近一轮 MOSS 验收

各项为自己的同轮五次warm中位数。

| 场景 | 基线 | 最终候选 |
|---|---:|---:|
| 1088×1936固定caption整图 | 29.645s | 28.892s |
| 1024² Moody FP8＋FP8文本固定MTP完整流程 | 16.092s | 15.902s |
| 上述TextGenerate | 5.227s | 4.985s |
| 1024² SFT BF16＋BF16文本固定MTP完整流程 | 19.521s | 19.396s |
| 上述TextGenerate | 5.314s | 5.156s |

18组真实block、两种文本固定logits逐元素一致，六组固定tree3s1自然扩写token相同。BF16整block共享候选保留舍入后无稳定收益。341项广泛回归通过、1项CUDA跳过，动态padding整理后最终37项专项通过，源码树为`4d9cc16d8fc372dc580511bed073bb0ce606672e`。临时原文本回退另过335项，测试范围存在重叠。

最后一次普通8189验收原始1088×1936工作流三次成功，cold100.467秒，文本缓存29.113/28.887秒，四份产品文件哈希核对一致、队列为空。该状态记录来自本轮优化验收。[结构化证据](evidence/moss-three-stage-2026-11-13.json)

## 质量与调查边界

- 默认Auto额外文字在优化版和原始文本回退版均出现，已修正最初的单项归因。固定树保值对照通过，两侧样本仍有BF16茶壶把手不可见、FP8海报低对比度等表现。
- TPC GDN候选因完整logits变化和双把手样本撤回。32MiB SRAM设置、额外小M FP8编译及BF16扩区未达到各自接入条件，原因保留在主题候选表。
- QK scores仍落DRAM，概率tile已在SRAM，实际HBM字节/BMON校准、约0.015秒kernel归属和文本空档因果归因尚未完成。
- 调查进程的若干段错误、profiler退出tcache异常、部分vendor异步输入问题仍缺底层根因。已验产品路径和各故障触发条件分别记录。
- 质量范围限于已有模型/样本/参数组合。完整批量、多卡、编辑/控制LoRA、SLG、其它VAE/custom nodes及超大尺寸需要独立验证。

后续投入依据见 [优化方向](optimization/STRATEGY.md)，测量边界见 [验证方法](optimization/MEASUREMENT.md)。
