# MOSS BF16复盘与FP8路线判断

更新于2026-11-11，基准产品`a8f42df2`。本文汇总各轮实测及下一阶段建议，原文件名保留用于既有链接。各阶段明细保留测试条件、结果、失败原因与复现入口。

## 当前判断

建议将下一阶段主要实验投入转向**DiT的原生FP8 Linear**。BF16基线继续用于生产、数值参考和兼容路径。BF16专项按新的热点证据或实现变化触发，优先保留attention数据流与文本提交两个方向。

理由来自当前费用结构。大Linear已接近同形状可用速度，近期低成本开关大多只带来约1%变化。最新GDN改写和扩区仍让固定文本下降14.9%、完整图下降5.1%，说明算法/执行结构变化继续有效。后续同量级BF16收益更依赖新的attention或状态融合实现，需要更高的开发和质量验证投入。

## 测试口径

- 单卡Gaudi2，Habana1.24.2，eager＋`hpu_backend`，1024²、25步、CFG2.5、er_sde/simple，模型与采样设置记录在各轮证据中。
- 固定caption对照用于DiT。完整MTP对照使用1970输入/298输出，重算原采样后代入已记录的token/树/控制决策，验证调用次数。自然生成用于完整行为与质量检查。
- 原则上各1 cold＋5 warm，报告中位数/范围。同步诊断、局部重放、节点缓存命中分别标注。各阶段独立中位数按各自统计解释。
- BF16文本模型来自现有INT8 ConvRot转换，扩写激活BF16，普通正负编码激活FP32。GDN递归状态使用FP32。文件dtype与实际算子dtype需分别核实。

## 已兑现的主要收益

| 阶段 | 有效改动 | 主要结果 | 证据 |
|---|---|---|---|
| GDN/conv | 编译Torch deferred参考实现 | 同输入conv6.60→0.45ms、GDN19.13→0.93ms。自然完整图131.03→60.06s | [GDN](MOSS_GDN_COMPILE_2026-11-09.md) |
| DiT attention | 模型所属区域编译 | 固定caption DiT37.916→28.217s | [attention](MOSS_DIT_OPTIMIZATION_2026-11-09.md) |
| block/MLP | 保留敏感舍入边界，分区＋静态recipe | DiT28.225→19.456s。固定32-token teacher4.876→4.223s | [block](MOSS_BLOCK_OPTIMIZATION_2026-11-09.md) |
| KV/完整短步attention | 编译完整容量KV及投影，修复分支行搬移 | 固定32-token teacher4.312→3.002s | [KV](MOSS_KV_OPTIMIZATION_2026-11-10.md) |
| 条件预计算 | text_embedder/RoPE每请求50→2次 | 固定caption DiT19.581→18.384s | [预计算](MOSS_DIT_PRECOMPUTE_2026-11-10.md) |
| norm/RoPE与prefill | native norm、验证后BLOCKWISE、GDN分桶 | DiT18.397→16.887→14.364s。FP32编码1.804→0.364s | [深挖](MOSS_BF16_DEEP_EXECUTION_2026-11-11.md) |
| 文本norm | 原公式区域编译 | 固定MTP独立文本7.481→5.670s，完整图22.461→21.060s | [预算/norm](MOSS_BF16_PLAN_COMPLETION_2026-11-11.md) |
| 并行GDN/外围投影、单head attention | 改写短递推，再扩大八行GDN编译区 | 固定文本5.720→4.867s，完整图21.044→19.978s，缓存14.628→14.386s | [最新验收](MOSS_BF16_ATTENTION_AND_SUBMISSION_2026-11-11.md) |

每行有自己的基线，早期自然caption和后期固定caption口径也有区别。当前可复用的正式基准为缓存14.386117s、DiT14.134462s、固定MTP完整19.978488s、独立文本4.866763s。完整warm范围19.906–20.122s，cold64.665s。缓存14s目标仍差0.386s。

最终194 passed、1 CUDA skip，MTP off/3/tree3s1、原INT8、512/768/1024及卸载重载通过。固定caption attention图片MAE0.09832/255，自身重复约0.104；自然扩写的三组图片MAE30.918–69.111/255，文本、构图和细节变化明显。质量判断限于已检查样本。[最终指标与哈希](evidence/moss-bf16-next-2026-11-11.json)

## 剩余空间及BF16投入价值

完整六段DiT时间线来自`64155fe4`，比最新单head改动早一轮。其预算可用于量级判断，下一次改造应刷新候选覆盖区域的时间。

| 部分 | 既有25步预算 | 当前判断 |
|---|---:|---|
| 大Linear | 活动并集5.126s，真实权重轮换374–404TF/s | 约达432TF/s峰值的86.5–93.6%。全达峰值的算术差约0.434s |
| QK/AV | 并集4.106s | 真实形状/布局效率较低，值得新算子或数据流研究 |
| softmax独占 | 2.398s | N×N中间矩阵仍被物化，融合潜力明确，合格实现尚待验证 |
| 其它kernel独占 | 1.805s | norm/RoPE等已做多轮优化，敏感舍入边界限制扩区 |
| 设备事件间隙 | 0.752s | 全部消除的理想降幅约5.2%，实际可消除量更小或有依赖约束 |

MME总并集为9.225s，以上Linear与QK/AV有交叠。按互斥分类总跨度14.181s。2.496PFLOP÷432TF/s得到5.778s，覆盖必要矩阵算术，softmax、流量、状态与调度仍需时间。AIP50–70%提供粗活动信息，工程收益需由这些费用分别估计。[预算原始依据](MOSS_BF16_PLAN_COMPLETION_2026-11-11.md)

最新文本两步trace仍有44.1%设备间隙，且CPU Linear154→10、launch964→820。短profile受采样与工具扰动影响，不能按比例直接乘4.867s。文本继续优化可改善完整扩写，固定caption缓存场景的主要收益仍需来自DiT。

值得保留的BF16方向有两个。其一是新的精度合格attention融合/在线softmax实现，其二是有完整轨迹证据的文本提交或GDN融合。既有厂商SDPA模式、布局/padding、双层合并和Graph开关的重复扫描优先级较低。

## 失败记录中应保留的判断

| 路线 | 已知结果 | 再投入的条件 |
|---|---|---|
| fast softmax | block约9.19→8.10ms，FP64误差0.314%→0.721%，中心化仍0.670% | 新精度方案及完整质量证据 |
| CFG合批/attention分块或padding | 适配成本抵消局部收益，部分数值更差 | 形状或底层数据流发生变化 |
| modulation/residual继续融合 | 局部少量收益，部分深层参考误差扩大 | 能保留敏感边界或有更好的数值实现 |
| 双block/lazy/Graph | 双层无收益，逐请求Graph预算15.24s对eager14.14s，另有ABI错误 | 执行模式/SDK或capture成本条件改变 |
| pending专门化、conv索引、单独外围Linear | 无稳定叠加收益，部分约1% | 新trace证明成为主要费用 |
| GDN扩大编译区域 | 旧递推与norm组合失败，改写递推后成功 | 重测依赖具体算法及图结构 |
| 四行GDN、KV搬移 | 完整模型暴露HPU输入/别名问题，分别用边界clone和源行clone修复 | 算子回放、状态回滚与完整MTP入口一起验收 |

## FP8已有依据

本机4096³ eager `torch.mm`为BF16约275.76TF/s、FP8 E4M3/E5M2约525.88/519.63TF/s，均无fallback。该BF16控制低于当前编译大Linear的374–404TF/s，FP8真实形状必须重新对照当前产品。旧全模型FP8文件通过`full_precision_mm=True`解量化执行，2-token显著变慢。原生FP8 lm_head probe为9.580s，BF16为9.407s，样本与自然token轨迹不足以证明收益。现有失败限定了这些执行方式的价值。[本机早期记录](MOSS_HPU_PERFORMANCE_RUN_2026-11-08.md)

官方1.24.2文档提供Gaudi2 FP8静态/动态校准、scale及`fp8_gemm_v2`路径。量化转换和scale配置会影响执行成本，FP8的新recipe也增加warmup。此处建议复用厂商算子，先在现有eager编译栈验证选定Linear区域。[Intel FP8指南](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html)

## 建议的FP8试点顺序

1. **先DiT FFN，再QKV/输出投影。** 两种真实N下轮换42层权重，比较当前编译BF16与FP8输入转换＋GEMM＋scale/bias＋BF16输出的总成本。FFN两个大Linear已有约3.40s轮换预算，所有四类Linear约5.14s，具备可测量的秒级目标。小型重复热循环之外，再验证完整block与25步。
2. **按扩散过程校准。** 使用已有BF16 DiT，采集浅/中/深层、早/中/晚timestep、正负条件及多提示词/尺寸的激活范围。校准集与最终质量检查提示词分开，检查饱和、相对误差、文字、人物和构图。先量化选定Linear，attention/norm/RoPE/残差按已验收精度执行，敏感Linear逐项回退。
3. **沿用产品生命周期。** 权重转换跟随加载、补丁更新和卸载，记录常驻字节、cold、shape切换和fallback。量化operations可能改变现有编译分流，必须确认DiT分区继续生效。静态scale与动态scale都计入实际成本，按性能/质量取舍。
4. **再考虑文本。** DiT试点证明净收益后，针对M=1/2/4/8的原生FP8 Linear重新测定。GDN FP32 state与MTP语义继续单独验收，固定程序与自然生成都检查。现有文本权重已由INT8派生，后续FP8质量评估需记录量化来源。

以下只是沿用旧Linear预算5.126s的Amdahl情景。设所有可覆盖Linear部分实现给定净加速，转换开销已包含、其它费用固定，以最新缓存14.386s/完整19.978s作参照。

| 覆盖部分净加速 | 节省时间 | 缓存整图估算 | 完整图估算 |
|---|---:|---:|---:|
| 1.3倍 | 1.183s | 13.203s | 18.796s |
| 1.5倍 | 1.709s | 12.677s | 18.270s |
| 2.0倍 | 2.563s | 11.823s | 17.415s |

实际覆盖比例、量化误差与额外调度决定结果。只加速Linear的情景把收益集中在约1–2.6s，达到6–7s缓存体验还需要attention及其它阶段共同改善。建议先完成一次真实shape→block→完整图的FP8试点，以质量合格后的端到端净收益决定继续投入。
