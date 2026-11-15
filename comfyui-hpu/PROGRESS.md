# 当前状态与未决问题

截至2026-11-15，产品分支`codex/gaudi2-hpu`，最新提交`dfca7cd672240da975983214c11d3057ec5b74b6`。本文件提供交接快照，测试过程和候选取舍集中于 [主题索引](README.md)。

## 2026-11-15 Qwen Image 2.1 最新2MP工作流

按用户确认，产品`dfca7cd6`与上一提交`ed9552e5`各独立生成8张原提示词2MP图，两类seed逐组配对。16份history确认完整重算扩写和采样，8组扩写文字及RGBA像素均逐元素一致。原图、扩写、8张原尺寸并排图、概览和ZIP已保存，供用户人工检查。当前服务恢复`dfca7cd6`，源文件哈希一致、队列为空。[图组记录](optimization/QWEN21_PRODUCT_AB.md) · [证据](evidence/qwen21-product-ab-2026-11-15.json)

用户要求先推进无MTP的BF16路径。本轮以AXI返回beat完成HBM有效载荷校准，128/512/2048 MiB copy及GEMV读取回算误差约−0.43%至−0.26%。真实解码约17.168 GB读取和1.261 GB写入/token，16轮无profiler为0.36819s，约0.801 TB/s，即峰值32.69%，取整敏感性上沿33.38%。整步eager编译与持久HPUGraph将decoder核心20.871→约17.8ms，存在0.79%–0.93% logits差异。GDN单tokenTPC使固定427-token耗时10.220→9.567s，但完整logits仍有0.986%误差，两个舍入修订也未对齐。普通服务最终运行`dfca7cd6`，原2MP图100.598s成功且像素一致、队列为空。DMA校准首版的地址使用错误曾触发设备自动复位，已恢复并以独立Synapse控制完成补测。[完整校准与验证](optimization/QWEN21_BF16_SINGLE_TOKEN.md) · [证据](evidence/qwen21-bf16-single-token-2026-11-15.json)

后续隔离试验复用已有Qwen3.5的243M参数MTP头。固定512-token窗口三次warm，单步42.03→四行验证58.57 tokens/s，吞吐提高39.35%。完整自然扩写两次warm为1784 tokens/41.507秒→1911 tokens/31.279秒，约61.10 tokens/s。相同前缀首个根token的单行/四行验证logits仍有2.022% RRMSE，完整图片和正常loader接入尚待验收。候选只在调查进程运行，线上使用`dfca7cd6`。实验后原2MP图94.763秒成功且像素一致，队列为空。[候选与边界](optimization/QWEN21_TEXT_NEXT.md) · [证据](evidence/qwen21-text-draft-2026-11-15.json)

已冻结最后一次用户请求，实际1088×1920、8步官方Turbo调度、CFG1.2、thinking开启、匹配PE的BF16文件，2803输入/1784输出tokens。原请求66.274秒。三次完整warm初始基线中位62.250秒，回退原产品三次为61.563秒，最终三次为60.689秒。按回退基线，完整耗时减少1.42%，PE42.575→41.713秒，约41.9→42.8 tokens/s。DiT约16.59秒，文本缓存图约18.47秒。

接入单token普通BF16的conv/delta联合编译及MLP残差融合，GDN两端投影维持原执行边界。16-token cProfile中recipe提交2720→2336、eager add1024→512、Python调用389151→313781。固定427-token全logits逐元素一致，五组不同seed/尺寸/thinking/官方调度的token和RGBA图片逐元素一致。初始组合另有五次完整warm中位60.713秒。157项回归通过、1项跳过，最终含新增单token hooks的39项专项通过，两组测试有重叠。

普通8189冷启动95.048秒，新提示词512²为10.614秒，卸载后原图重载91.124秒。卸载40–60秒稳定约8.984GiB，该2MP尺寸的原版卸载对照未采集。MOSS BF16固定MTP三组新鲜原版/新版扩写文字一致，历史样本漂移在原版中也存在，尚未归因。最终恢复Qwen原2MP图95.721秒成功，cold/reload/final像素与基线一致，队列为空，四份部署文件SHA256与产品提交对应文件一致。[证据](evidence/qwen21-performance-2026-11-15.json)

扩大整GDN编译区仍有约0.98% logits误差，审计确认输入和输出投影均有舍入变化。2MP DiT分组最快局部约5%，尚有数值变化，本轮继续采用原DiT数学路径。单行commit无稳定净收益，RoPE预计算变慢，关闭eager流水线使固定文本10.259→32.620秒。后续HBM校准结果及新的单token候选见本节开头。[完整调查](QWEN21_PERFORMANCE_2026-11-15.md)

## 2026-11-14 原生INT8 MME验证与开发终止

用户明确要求有原生INT8 MME才继续，能力核验阴性后已停止并放弃通用INT8开发。服务器安装库`Gaudi2HalReader::getSupportedMmeTypes()`返回`0x7906`，INT8位为零。直接Synapse的默认、inference、inference＋quantization模式共9组INT8 GEMM均在节点创建阶段失败，3组BF16对照编译成功。

本轮验证范围为已安装HAL与直接SDK建图/编译，INT8在执行前被拒绝。验证前后普通服务PID相同、队列为空。[验证结果](optimization/INT8_NATIVE_MME.md) · [证据](evidence/int8-native-mme-2026-11-14.json) · [历史计划](optimization/INT8_DEVELOPMENT_PLAN.md)

## 2026-11-14 INT8扩写慢调查

已核对普通8189最后一次120.523秒请求，Qwen Image 2.1 PE选择原INT8 ConvRot文件，文本循环约102秒、进度737、平均7.23 tokens/s。此前同提示词/文本seed的同源BF16扩写进度456、平均39.12 tokens/s，中段约42 tokens/s。thinking关闭，两个文件均无MTP权重。

部署源码确认INT8使用`full_precision_mm=True`，每次量化Linear调用先反量化/ConvRot还原为BF16，同时跳过短步MLP及完整attention区域编译。生成长度增长进一步增加耗时。已存在的同源BF16文件提供预先解码及编译路径。本轮完成只读历史、源码和模型header核查，各算子独立耗时尚未profile。[调查](optimization/TEXT_PRECISION.md) · [证据](evidence/qwen21-int8-slow-2026-11-14.json)

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
- QK scores仍落DRAM，概率tile已在SRAM。文本HBM访问已用AXI返回beat校准，DiT尚待按新口径重测。约0.015秒kernel归属和文本空档因果归因仍需补齐。
- 调查进程的若干段错误、profiler退出tcache异常、部分vendor异步输入问题仍缺底层根因。已验产品路径和各故障触发条件分别记录。
- 质量范围限于已有模型/样本/参数组合。完整批量、多卡、编辑/控制LoRA、SLG、其它VAE/custom nodes及超大尺寸需要独立验证。

后续投入依据见 [优化方向](optimization/STRATEGY.md)，测量边界见 [验证方法](optimization/MEASUREMENT.md)。
