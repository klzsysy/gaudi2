# MOSS BF16／FP8全栈性能瓶颈复核

2026-11-12，Asia/Shanghai。基于本仓库工作记录、结构化证据、本地产品提交`0995638a`源码及当日复核的一手资料。本文属于分析与后续实验设计，性能数值来自既有服务器测量，新增算式在本地复算。

第10节证据已修正AIP口径、attention存储假设和profiler扰动，BMON到实际HBM字节数的校准仍待完成。用户随后明确要求开始第9节优化，候选以壁钟、设备时间线与数值对照推进。[证据补齐记录](MOSS_EVIDENCE_BOUNDARIES_2026-11-12.md)、[优化执行记录](MOSS_OPTIMIZATION_EXECUTION_2026-11-12.md)

后续分层复核支持将研发主线转向图编译器／HPU bridge与attention kernel的联合数据流。新增证据包括节点外耗时中位15.9ms、DRAM score的非persistent属性，以及低扰动文本窗口约39.7%的kernel间隙。前三层的空间按具体路径评价，第三层上限仍未闭合，详见[分层边界复核](MOSS_LAYER_BOUNDARY_AUDIT_2026-11-12.md)。

## 1. 当前判断

当前主要限制集中在三个位置。

1. **DiT attention的中间张量、矩阵形状和MME/TPC调度。** QK仍用BF16，分数矩阵落DRAM，softmax与FP8 AV之间的概率tile已共享SRAM。矩阵形状与分数读写仍约束有效吞吐。
2. **量化、归一化、调制、SiTU、RoPE及布局转换的组合实现。** 最新扩区已经省下约1秒其它kernel时间，剩余2.65秒值得进一步拆解和定向融合。
3. **文本短步执行的提交、同步和状态更新。** FP8文本节省显存明显，固定完整流程速度基本持平，表明大矩阵峰值提升对这一阶段的影响有限。

分层归因中，模型实现、设备kernel、图编译器与用户态运行时的证据最强。主机配置和内核驱动保留待测假设。已测大Linear接近本栈有效上限，整体仍有秒级空间。继续优化需要改变执行结构和数据流。

## 2. 用最新基线定位时间

以下保留`0995638a`首轮FP8验收基准。配置为Moody主模型＋Qwen3.5 FP8文本，1970输入／298输出token的固定MTP程序，正负conditioning为307／23 tokens，1024²、25步、CFG2.5、`er_sde/simple`，各1 cold＋5 warm。后续同配置完整流程复测约16.0秒。

| 指标 | 最新结果 | 解释 |
|---|---:|---|
| 完整工作流 | 15.925s | 5 warm中位数，范围15.734–16.094s |
| TextGenerate | 5.040s | 约占完整流程31.6% |
| CLIPTextEncode | 0.386s | 正负普通编码合计 |
| KSampler／DiT | 10.211s | 约占完整流程64.1% |
| VAE | 0.132s | 小于1% |
| PreviewImage | 0.108s | 图片输出 |
| 完整cold | 77.273s | 加载、首次执行与编译等成本共同存在 |
| 日常服务文本缓存出图 | 10.696s | 单次部署验收，与固定程序基准各按自身条件解释 |
| allocator高水位 | 31.285GiB | 包含cold分配历史 |
| 请求结束常驻 | 17.987GiB | 仍有较多容量可用于受控workspace和缓存 |

各节点中位数独立统计。客户端轮询还会带来约0.6秒观测延迟，正式基准采用服务请求计时。当前自然扩写的已验收请求约20–21秒，token数和shape变化会改变真实用户等待时间。[结构化证据](evidence/moss-fp8-fusion-2026-11-12.json)

最新DiT设备预算采用GC控制后的step0／12／24×正负六段trace，按25步外推，重叠区间按互斥优先级归类。

| 互斥类别 | 外推秒数 | 设备跨度占比 |
|---|---:|---:|
| FP8 Linear | 2.514 | 24.5% |
| FP8 AV | 0.531 | 5.2% |
| BF16 QK | 2.406 | 23.4% |
| 其它MME | 0.003 | 约0.03% |
| softmax | 1.340 | 13.1% |
| 其它kernel | 2.652 | 25.8% |
| kernel间隙 | 0.815 | 7.9% |
| 合计 | 10.261 | 100% |

其与无profiler DiT10.183秒相差约0.77%。其它kernel约99.4%已关联至编译后图，SiTU约1.01秒、量化独占约0.20秒。优化后的并发和归类会变化，逐项节省量需要重新测量。[六段预算与归因](MOSS_EVIDENCE_BOUNDARIES_2026-11-12.md)

## 3. 利用率和硬件上限

### 3.1 大Linear的空间已经较小

BF16真实42层权重轮换达到374–404TF/s，约为432TF/s峰值的86.5%–93.6%。FP8 Linear在前轮trace中的活动阶段估算约806TF/s，约为865TF/s的93%。这些结果限定于所测shape和版本。当前含AV的FP8类别若按主干FLOP除以活动时间粗算，约743TF/s，反映Linear与AV的混合效率。[BF16预算](MOSS_BF16_PLAN_COMPLETION_2026-11-11.md)、[FP8预算](MOSS_FP8_SYSTEM_OPTIMIZATION_2026-11-12.md)

Gaudi2公开规格包括2个MME、24个TPC、48MB片上SRAM、2.46TB/s HBM带宽。BF16矩阵峰值432TF/s，BF16向量峰值11TF/s。softmax的exp、SiTU的tanh/sigmoid、归约、转换和访存各有不同吞吐约束。硬件资源之间的比例决定了高矩阵吞吐与较慢非矩阵阶段可以同时出现。[Intel产品对照，第3页](https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf)

### 3.2 AIP约58%需要更细的解释

最新DiT AIP均值58.5%，文本21.4%。后续核查已加载驱动的源码和反汇编，并进行531组功率／utilization配对，确认本机按功率比例计算该指标。当前驱动使用550000mW最大功率及60000mW静态功率，531组配对全部符合公式区间±1个百分点。此前依据官方通用文档进行的活动窗口解释已由本机实测更新。[证据补齐记录](MOSS_EVIDENCE_BOUNDARIES_2026-11-12.md)

后续同时观察端到端时间、MME有效TF/s、TPC执行与访存、HBM事务、提交间隙和能耗。AIP提升本身只作为辅助观察。

### 3.3 更新混合精度算术参照

当前固定工作量的主干Linear约2.026677PFLOP，QK和AV各0.234518PFLOP。Linear和AV采用FP8，QK采用BF16，主干矩阵运算量的FP8覆盖约90.6%。按每种精度的峰值分别折算，纯矩阵乐观预算为

```text
(2.026677 + 0.234518) PFLOP / 865 TFLOP/s
  + 0.234518 PFLOP / 432 TFLOP/s
≈ 3.157 s
```

该算术参照省略非线性、量化、状态、额外流量及提交，也假定所有矩阵形状达到峰值。当前10.211秒包含这些成本。进一步把QK从BF16换成FP8，其峰值算术预算只减少约0.272秒。QK当前主要MME预算约2.409秒，对应约97TF/s的粗估速率，因此QK效率和数据流的研究价值高于仅看精度覆盖比例。[FLOP复算依据](MOSS_HARDWARE_LIMIT_AND_5090_2026-11-10.md)

## 4. 最有价值的DiT研究方向

### 4.1 分块attention、在线softmax与SRAM流水

当前源码逐head构造BF16 scores及FP8 probabilities，再执行AV。后续编译后图确认，完整BF16 scores位于DRAM，FP8 probabilities则分块位于SRAM，并直接供AV消费。正条件N=4403时，QK输出tile为`[4403,1536]`，softmax／AV概率tile为`[4403,2048]`。24个head的完整BF16 scores逻辑大小约930.55MB。[编译图证据](MOSS_EVIDENCE_BOUNDARIES_2026-11-12.md)

原先按两份中间结果都写回HBM推导的5.50TB情景已更新。当前已确认的QK分数一次写出及一次读取对应

```text
25 × 42 × 24 × (4403² + 4119²) × 4 bytes ≈ 3.664 TB
3.664 TB / 2.46 TB/s ≈ 1.49 s
```

这是限定读写次数下的逻辑流量模型。实际HBM事务受多次softmax归约、布局和编译器调度影响，上述时间与现有MME／softmax阶段存在重叠，BMON字节口径仍待校准。

值得实现的候选是query/key分块、稳定在线softmax、FP32行max／行sum与输出累加、QK→softmax→AV的片上数据传递，以及双buffer流水。其目标是降低N²临时张量和引擎切换成本。保留BF16 QK也能研究这些收益。FlashAttention论文提供了减少HBM与SRAM之间访问的算法依据，Gaudi2上的实现需要重新设计MME与TPC协作。[FlashAttention原论文](https://arxiv.org/abs/2205.14135)

已有厂商FP8 FusedSDPA在真实输入上约14.75ms，当前参考约3.23ms。此前按head分组、query分块及padding也多次缺少净收益。新实验应证明recipe结构、中间张量驻留或调度发生了实质变化。[FP8候选结果](MOSS_FP8_SYSTEM_OPTIMIZATION_2026-11-12.md)、[BF16候选结果](MOSS_BF16_DEEP_EXECUTION_2026-11-11.md)

直接MME/TPC路线值得作为独立研究分支。已有外部实现演示了私有SCAL流、MME描述符、TPC和SRAM中间张量协作，应用于H3长序列。移植到MOSS需要适配约4K序列、描述符、SDK、输出所有权、同步和精度，作者的H3成绩只能作为可行性线索。常规集成优先使用当前栈可维护的接口，直接描述符路线按独立实验验证。[固定版本实现](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/native/mme-direct-attention/README.md)、[本仓库源码审查](JCHEN_GAUDI2_KERNEL_REVIEW_2026-11-10.md)

### 4.2 针对2.65秒其它kernel做显式融合

上一轮其它kernel从3.668降至2.651秒，而FP8 MME时间约3.045秒，说明矩阵周围的执行已有可兑现空间。后续六段预算为2.652秒，约99.4%已关联生成图。新的TPC候选应针对已识别的真实读写、舍入边界与复合区域设计。

| 候选 | 预期减少的工作 | 需要验证的条件 |
|---|---|---|
| RMSNorm＋调制＋逐行amax＋FP8写出 | 中间BF16张量、重复读写与launch | FP32归约及原有BF16舍入位置 |
| SiTU＋逐行amax＋FP8写出 | tanh/sigmoid输出及乘法中间张量 | 完整SiTU语义、极值与敏感层误差 |
| Q/K RMSNorm＋RoPE＋最终布局 | FP32/BF16转换、中间张量与transpose | norm后的BF16舍入、RoPE系数和非连续输入 |
| V量化与最终AV布局共同生成 | 24个head的量化准备与布局成本 | 逐head scale及FP32缩放精度 |
| 残差＋下一阶段归一化 | 残差重复访存 | 残差所有权、编译边界和扩散误差累积 |

SiTU的具体公式为`4*tanh(gate/4)*sigmoid(gate) * 25*tanh(hidden/25)`，由此可得绝对值上界100。可以探索固定范围或层／timestep校准scale来减少amax，但统一最大范围会降低小激活的量化分辨率。逐行动态、校准静态和混合方案需在实际分布上比较。[SiTU源码](/Users/sonny/work/github/klzsysy/ComfyUI-Gaudi/comfy/ldm/moss_image/model.py:18)、[现有量化入口](/Users/sonny/work/github/klzsysy/ComfyUI-Gaudi/comfy/quant_ops.py:214)

当前编译器可能已经融合部分相邻操作，应以生成kernel的实际访存和时间决定是否手写TPC。已测静态scale局部有0.02–0.14ms收益，跨输入校准仍待完成。V批量量化曾得到更快完整流程，精度修正后收益被抵消，说明缩放与舍入实现本身值得定向优化。[FP8深挖](MOSS_FP8_DEEP_OPTIMIZATION_2026-11-12.md)、[V量化对照](MOSS_FP8_FUSION_OPTIMIZATION_2026-11-12.md)

### 4.3 对精度和编译分区共同优化

全FP8 QK在18组Moody输入中的敏感层误差曾达18.5%–48.1%。当前FP8概率路径的随机压力测试还发现厂商softmax额外误差。下一轮可先改进概率计算精度，再评估按层／head／timestep选择精度，减少敏感分布上的误差。当前产品保留的调制方法边界带来约0.47秒DiT收益，整块内联反而退化，编译区域大小与性能呈现具体实现依赖。[attention精度](MOSS_FP8_DEEP_OPTIMIZATION_2026-11-12.md)、[分区对照](MOSS_FP8_FUSION_OPTIMIZATION_2026-11-12.md)

质量预算应同时观察相同量化权重上的FP32计算参考、相同caption的完整图和自然扩写。当前FP32参考仍使用量化后解码权重，原始权重量化损失需要另行评价。

## 5. 文本约5秒的空间

Qwen3.5短步主要是M=1／2／4／8的小矩阵、GDN FP32状态、卷积历史、KV更新、词表投影和MTP控制。FP8文本的固定流程18.954→18.933秒，显存峰值下降约7.51GiB。量化后的访存节省被其它开销部分抵消，下一步应按短步关键路径定位。[全流程文本记录](MOSS_FP8_SYSTEM_OPTIMIZATION_2026-11-12.md)

1. **GDN与状态提交的专用融合。** 将slot选择、gate/norm、短递推、state/checkpoint读写和conv history尽量合并。既有并行递推及八行扩区已让独立文本5.720→4.867秒，证明结构改写有效。自定义TPC应与当前并行递推直接对照，保存MTP接受、拒绝和回滚语义。
2. **词表投影到采样的提交粒度。** 词表为248320、hidden为4096。研究短步GEMV／小M GEMM、局部top-k与后续全局合并、减少logits中间张量和同步。top-p、min-p、惩罚项、随机数顺序及MTP draft／verify概率各有数据依赖，候选应覆盖真实采样参数。
3. **减少每轮主机依赖。** 让设备侧执行更多确定性的索引、接受判定与状态commit，只回传主机真正需要的控制量。trace需要关联CPU等待与设备队列是否耗尽，再决定扩大编译或重组控制流。
4. **MTP效率按每秒有效输出衡量。** 固定程序约130轮控制产生298 tokens，平均约2.29输出token／控制轮。真实策略可研究树宽、draft深度和自适应选择，并分别记录草稿成本、接受率、verify成本及自然生成长度。

文本AIP约21.4%属于功率口径。新的GC控制16轮窗口中，设备跨度548.497ms、kernel活动330.621ms、间隙217.876ms，约39.7%，窗口profiler扰动约1.24%。空档来源和可消除比例仍待分离，整段收益采用无profiler测量。近期共享量化和额外小M编译没有稳定部署收益，重新实验需有新的提交或kernel证据。[分层复核](MOSS_LAYER_BOUNDARY_AUDIT_2026-11-12.md)

当前连续KV和多token联合验证已实现，相关长上下文优化的重复投入价值较低。约2K有效上下文下，FP8 KV仅能节省约107MiB payload，需先证明KV访存成为热点。共同系统前缀缓存值得研究，但其直接作用范围主要是prefill，之前1970-token首步约0.254秒。GDN缓存还需覆盖recurrent state、conv history和续接语义。[既有attention建议复核](ATTENTION_IDEAS_REVIEW_2026-11-10.md)、[prefill计时](MOSS_BF16_PLAN_COMPLETION_2026-11-11.md)

## 6. ComfyUI、编译器、运行时和驱动分别承担什么

| 层次 | 现有证据与判断 | 可区分原因的实验 |
|---|---|---|
| ComfyUI工作流调度／UI | 最新服务执行窗口的节点外耗时中位15.9ms，约0.1%，当前优先级低 | 保留相同计时边界，浏览器、排队及HTTP窗口外耗时单独测量 |
| ComfyUI模型实现 | 常规dtype、quant入口、dispatcher、状态、预计算和编译区域已多轮优化，近期普通重排收益收窄 | 文本设备侧控制及新执行结构仍待因果对照 |
| 设备kernel | 大Linear余量较小，attention与TPC复合kernel仍有未闭合空间 | 结合真实shape、舍入和生成图，评价复合kernel及吞吐下界 |
| torch.compile／HPU bridge／图编译器 | Attention已处在同一图，非persistent的QK分数仍分配DRAM，当前主线优先级最高 | 追踪混合精度lowering、布局约束、bundle与共同tile，验证SRAM驻留和净收益 |
| Synapse／SCAL用户态运行时 | 低扰动16轮文本设备跨度内间隙约39.7%，归因仍横跨模型控制、bridge和运行时 | 对齐CPU提交、队列状态、设备开始结束与等待事件，受控重放相同recipe |
| 内核驱动／固件 | 历史DKMS问题已有修复，当前大Linear吞吐与稳定执行支持基本路径可用 | 仅在trace显示异常队列等待、DMA或固件停顿后，建立最小复现并比较配套软件版本 |
| 主机CPU／NUMA／PCIe | 已确认单NUMA节点，32→16线程无收益，绑核及迁移影响尚未受控验证 | CPU绑核、内存本地性、线程迁移、PCIe链路及H2D/D2H事件，优先看文本和cold |
| 时钟／散热／功率 | 既有SOC记录1650MHz、温度42–61°C，缺少持续降频证据 | 同一任务记录SOC/TPC时钟、功率与热限速计数，比较warm工作区间 |
| HBM／SRAM | QK分数DRAM驻留与真实等待计数已有证据，有效copy带宽1.98–2.15TB/s | 继续校准实际模型HBM字节及饱和率，结合片上tile与溢出解释 |

当前官方推荐eager＋torch.compile，lazy处于legacy维护路线。已有逐请求capture/replay/destroy的Graph实验约15.24秒，eager约14.14秒，同时出现输出误差和ABI问题。若继续研究Graph，可改变为固定shape的长生命周期buffer、一次capture、多次replay，并按模型卸载／patch失效。文本动态控制和设备内原地view操作需要符合Graph支持范围。[官方执行模式](https://docs.habana.ai/en/latest/PyTorch/Reference/PyTorch_Gaudi_Theory_of_Operations.html)、[HPU Graph约束](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html)

将DiT全部现有间隙消除，按当前预算最多约省0.82秒。图变换若同时改善kernel融合与布局，可以产生额外收益，其归因需要落到设备计算。保留带profiler退出时的tcache异常、异步四行GDN输入问题、FP8动态编译重载问题和GraphStorage符号错误的最小复现，有助于与厂商核对具体层次。[运行时失败记录](MOSS_BF16_PLAN_COMPLETION_2026-11-11.md)、[FP8重载记录](MOSS_FP8_SYSTEM_OPTIMIZATION_2026-11-12.md)

## 7. 内存容量、批量、形状与冷启动

96GB容量可用于预分配workspace、持久recipe输入输出、模型常驻和适量微批。收益取决于实际访存／提交减少。当前请求结束常驻约18GiB，增加中间buffer容量本身仍会消耗带宽。

CFG正负分支独立，可研究两个分支的流水或分组调度。已有padding合批缺少收益。当前单head优化要求batch=1，FP8 AV还要求N在4096–8192且无mask。直接合批、补mask或改形状可能退出优化路径，需同时适配dispatcher和kernel。512／768等尺寸因此还有单独开发空间。[产品选择条件](/Users/sonny/work/github/klzsysy/ComfyUI-Gaudi/comfy/ldm/modules/attention.py:630)

多请求batch与连续调度有机会提高每秒图片／token吞吐，需另外记录单请求延迟和排队。主模型大Linear已高效，新增batch更可能帮助小M文本及提交摊销。两个请求争用HBM和TPC时也可能相互拖慢。

冷启动约77秒值得单独拆为文件读取、权重准备／转移、图编译和warmup。真实用户不同caption长度触发的首次shape成本也应报告。静态分桶、recipe缓存和常驻worker的收益需计入padding与mask造成的算子变化。服务长期运行能摊销首次成本，换模型／尺寸的体验则依赖缓存生命周期设计。

若接受改变模型或采样工作量，还可研究扩散跨步特征复用、减少步数、蒸馏以及更短扩写。这些方案需要新的速度／质量曲线，当前25步固定工作量基准继续用于衡量执行效率。

## 8. 与5090的差距重新表述

当日复核公开作者测量，KOBO的1024²、25步、CFG2.5为BF16缓存整图14.7秒、INT8缓存整图5.8秒，使用SFT及小VAE。DSLab另报INT8采样约7秒。当前本机BF16固定caption缓存14.386秒，已处在公开BF16结果的同一量级。FP8日常缓存10.696秒，对照5.8秒仍有约4.9秒差距，对照7秒量级仍有数秒空间。权重版本、caption、VAE、软件版本和统计方式不同，这些差距用于设定方向性目标。[KOBO原始报告](https://localai-kobo.com/moss-image-comfyui/)、[DSLab原始报告](https://dslab.tw/moss-image-lora)、[BF16本机验收](MOSS_BF16_ATTENTION_AND_SUBMISSION_2026-11-11.md)

5090路径的优势需继续用相同模型与同工作量trace拆解。CUDA量化、attention、融合与提交的实现差异是合理候选原因。Gaudi2当前已验证的原生FP8路线能够继续推进，现有INT8文件主要走解量化计算，`_int_mm`在本机曾CPU fallback。继续深挖FP8数据流有更直接的已有基础。[量化算子调查](GAUDI2_INT8_FP8_KERNEL_SURVEY_2026-11-10.md)

## 9. 可计算的收益情景与投入顺序

用户已明确授权开始本节优化。下表保留为敏感性情景，实际取舍以候选实测为依据。已知SiTU约1.01秒且已融合，量化独占约0.20秒，softmax→AV已实现SRAM复用，这些测量约束候选的可兑现收益。BMON字节校准继续作为独立调查项。

以下敏感性分析把六段trace预算按10.183／10.26138归一到无profiler DiT壁钟。把其它部分固定，只缩短指定互斥时间。算法改变可能重新分配重叠、内存和量化成本。

| 假设 | DiT节省 | DiT估算 |
|---|---:|---:|
| 仅完全消除现有间隙 | 0.81s | 9.37s |
| QK主导MME＋softmax减少35%，其它kernel减少30%，间隙减少40% | 2.42s | 7.77s |
| 上述三组分别减少55%、45%、60% | 3.72s | 6.47s |

使用完整流程基线16.036秒，第二行若文本减少1秒，完整流程约12.62秒。第三行若文本减少2秒，约10.32秒。文本假设均相对已恢复的基线计算。缓存整图还需加入VAE、输出和工作流成本，前轮约0.51秒。6–7秒缓存体验需要attention和其它kernel同时取得较大改进。

首轮GDN候选曾使完整流程16.036→15.751秒，因logits及自然图片质量变化，已按用户要求撤回。SiTU原生实现及query分块候选耗时均上升。后续以质量保值为优先，见[当前执行](MOSS_QUALITY_PRESERVING_OPTIMIZATION_2026-11-12.md)及[历史记录](MOSS_OPTIMIZATION_EXECUTION_2026-11-12.md)。

质量保值轮完成9种布局及8组有效编译设置筛选。32MiB SRAM候选在90组输入和固定文本logits上逐元素一致，但完整流程中位数仅差0.23%，均值方向相反。编译后QK分数仍在DRAM，后续重点仍是联合分块与实际SRAM驻留。当前产品维持`0995638a`。

当前按分层复核结果调整投入顺序。

1. **主攻第四层与attention kernel的联合数据流。** 核查混合精度链的bridge lowering、布局约束和Synapse bundling，以N=4119／4403、D=128的最小子图验证QK分数SRAM驻留和实际收益。
2. **追踪文本主机控制与提交。** 将16轮窗口的CPU控制、recipe入队和设备间隙关联，按归因推进设备侧控制、较大执行单元或持久buffer。
3. **TPC复合kernel按预算配套推进。** 使用SiTU、norm／调制／量化及布局的已拆分预算，与当前生成代码直接比较，并保留精度与生命周期约束。
4. **维护质量与测量对照。** 使用90组真实输入、固定程序、完整计时及新鲜A/A图像对照。BMON实际字节校准作为持续调查项。
5. **按证据处理运行时与硬件问题。** CPU／NUMA、SDK／固件及驱动A/B围绕已观察等待或故障展开，提供可复现输入、版本和事件关联。

每个候选延续同工作量5 warm、完整入口、多尺寸、自然质量、fallback、状态回滚及卸载重载检查。最新融合的精度变化与显存增量说明，性能、数值和生命周期需要在同一候选上联合验收。

## 10. 当前最重要的证据边界

- 六段干净trace已完成，外推10.261秒，对照无profiler DiT10.183秒。
- 其它kernel约99.4%已通过编译后图关联，约0.015秒语义归属仍待补充。
- HBM有效带宽已校准至1.98–2.15TB/s，真实block的HBM命令及MME／TPC stall计数已取得。实际HBM字节数、BMON积分和带宽饱和率仍待校准。
- AIP已确认采用本机驱动的功率公式。GC对DiT诊断的影响及文本2／16轮profile扰动已量化。
- 新增72组输入及4组图像，当前FP32计算参考误差0.37%–4.20%，敏感层仍集中在末层。
- 收益情景继续作为假设；首次warm段错误保留为稳定性未决项。详细数据、失败路径与接续条件见[证据补齐记录](MOSS_EVIDENCE_BOUNDARIES_2026-11-12.md)。
