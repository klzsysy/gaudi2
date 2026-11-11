# MOSS BF16预算与文本norm编译

2026-11-11，`1412e1f7`→`64155fe4`。证据根目录为`/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-11-completion`，哈希/指标见 [证据清单](evidence/moss-bf16-completion-2026-11-11.json)。后续GDN改写又取得收益，本文结论以本轮实现和候选范围为界。

## 真实DiT预算

固定caption正307/负23、1024²/25步。step0/12/24×正负共6条trace，18组输入和42层真实权重。设备事件按区间并集统计，以下为六段平均外推的25步预算。

| 互斥分类 | 秒 |
|---|---:|
| 任一MME活动 | 9.225 |
| softmax独占 | 2.398 |
| 其它kernel独占 | 1.805 |
| 设备事件间隙 | 0.752 |
| 合计 | 14.181 |

无profiler正式DiT为14.400s。MME内部Linear5.126s、QK2.646s、AV1.539s，存在交叠，QK/AV并集4.106s。`GEMM`事件也包含AV，应按`bmm/`及`bmm_1/`路径分类。正step0编译区域为attention189.321ms、FFN72.277ms、区外kernel17.200ms、间隙15.031ms，attention区域同时包含投影、norm/RoPE及QK/AV/softmax。

| 42层权重轮换 | 正307 | 负23 | 吞吐 |
|---|---:|---:|---:|
| QKV | 26.322ms | 25.290ms | 387–398TF/s |
| 输出投影 | 9.082ms | 8.734ms | 374–384TF/s |
| up/gate | 46.039ms | 43.885ms | 397–404TF/s |
| down | 23.682ms | 22.575ms | 386–393TF/s |
| 完整42层 | 293.252ms | 272.374ms | 含其它操作 |
| 同层重复42次 | 293.198ms | 272.280ms | 热循环差异很小 |
| 两层合并编译 | 293.365ms | 272.404ms | 输出相同，无净收益 |

Linear独立轮换×25为5.140s，主干设备活动5.126s。大Linear已接近本栈同shape可用速度。attention仍物化N×N中间矩阵。

## 文本固定工作量与筛选

固定1970输入/298输出，记录258次draft top-7、130次sample、4次width和130组控制表。重放先执行原运算，再代入记录决策，逐项校验形状、次数和token，固定MTP工作量。自然生成另行验收。

| 候选 | 5 warm秒 | 取舍 |
|---|---:|---|
| 本轮基线 | 7.480832 | 控制组 |
| host pending/parity专门化 | 7.473226 | 无收益 |
| conv＋DeltaNet专门化 | 7.381380 | 约1.33%，处于波动内 |
| 整段DeltaNet | 6.559241 | 单独有效 |
| 正式Qwen35 norm编译 | 5.669533 | 采用，范围5.551044–5.701124 |
| norm＋整段DeltaNet | 5.861017 | 叠加更慢 |
| 整Transformer block | 6.427815 | 更慢 |
| MTP head | 5.635892 | 落在原波动范围 |
| 采样概率区域 | 5.682126 | 落在原波动范围 |
| 关闭嵌套编译 | 5.698868 | logits与当前候选相同 |
| 厂商hpu.rms_norm | 5.651456 | 约0.32%，FP64误差最大增至1.711倍 |

prefill/编码同时扩大MLP与attention区域，1970-token首步0.299443→0.253837s，正负编码0.335590→0.299453s，总计约省0.082s，约完整图0.4%。完整流程CPU线程32→16后21.060→21.168s。

norm编译复用ComfyUI原公式，保留FP32归约及普通编码dtype。两步MTP trace跨度137.309→108.419ms，设备活动61.118→59.396ms，launch1620→964，主要减少提交。该窗口含profiler扰动。

## lazy/Graph边界

单block graph正/负6.880/6.326ms，lazy为7.092/6.482ms，单shape capture约0.27s。42层按每请求预热/capture/replay/destroy，三次请求中正负25次forward预算约15.24s，eager14.141s，首次35.54s。相对eager链输出误差正3.44%、负2.30%。本安装栈lazy＋`hpu_backend`另报缺失`habana::graph::GraphStorage::get()`符号。执行模式复测需有运行时或生命周期条件变化。

## 正式产品验收

| 工作量，各1 cold＋5 warm | 基线 | norm产品 |
|---|---:|---:|
| 固定MTP完整图 | 22.460727s | 21.059925s |
| TextGenerate节点 | 7.476233s | 6.137305s |
| 正负编码 | 0.357449s | 0.318514s |
| KSampler | 14.374370s | 14.355267s |
| cold完整图 | 57.247283s | 58.229669s |

固定caption缓存14.652972s，DiT14.400162s。独立文本与工作流节点的执行环境不同，各自报告。自然扩写21.984322s，296→352 tokens。

- 34组真实norm输入覆盖主干/MTP、prefill/decode、FP32编码及非连续Q/K。主干FP64误差比0.84450–1.00681，BF16各组为1，Q/K为1±3.5e-11。
- 固定MTP logits为1022/1033 top-1，平均KL0.000719705。基线自身重复相同，候选存在真实舍入变化。
- 六组图片覆盖街景、海报、人物、复杂构图、长短文本与加权提示词，MAE2.184–11.017/255，主体/大字保持。自然caption有材质描述矛盾，图像同seed细节变化，质量评价限于此集合。
- 125 passed、1 CUDA skip，512/768/1024、卸载重载和原INT8通过，fallback为空。两侧allocator峰值39.3734GiB，卸载2,686,190,976 bytes。

## 50–70% AIP利用率与BF16上限复核

AIP使用约1000ms粗活动窗口。算力判断应结合真实shape吞吐、设备时间线和端到端延迟。[Intel指标说明](https://docs.habana.ai/en/latest/Management_and_Monitoring/Embedded_System_Tools_Guide/System_Management_Interface_Tool.html)

大Linear为373.87–404.34TF/s，达到432TF/s矩阵峰值的86.5–93.6%。必要矩阵工作量2.496203PFLOP÷14.400162s为约173.35TF/s，反映包含非矩阵阶段的有效工作效率。乐观矩阵预算5.778s还需加softmax、数据搬运、状态与调度。Linear自身峰值预算4.691s，与5.126s活动只差约0.434s。QK/AV形状对应约88.6/152.4TF/s。[硬件估算](MOSS_HARDWARE_LIMIT_AND_5090_2026-11-10.md)

即使理想消除0.752s设备间隙，也只占本轮DiT约5.2%。较大空间集中在attention数据流与文本提交。SOC记录期恒1650MHz，温度42–61°C，既有采样未支持把主要瓶颈归为降频。

此前把本轮候选验收称为“BF16收敛”范围过宽。后续递推改变后，GDN扩区重新取得收益，应以具体实现、输入和运行时标记失败结论的适用条件。[后续结果](MOSS_BF16_ATTENTION_AND_SUBMISSION_2026-11-11.md)
