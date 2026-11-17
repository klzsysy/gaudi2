# Qwen Image 2.1硬件潜力深度优化计划

2026-11-16，Asia/Shanghai。基线为产品`07fedda`及用户13:54:50的FP8工作流。本计划以单卡Gaudi2、batch1交互式出图为主场景，将主导硬件资源及工作负载硬件上限的利用效率≥80%设为最终研发验收目标。当前资料支持明确的优化方向，80%的最终可达性仍需通过实现和测量证明。

## 1. 工作量与80%的定义

性能主线使用1088×1920、八步Euler/官方Turbo调度、CFG1.2、thinking开启、FP32来源FP8 PE、原FP8 DiT/CLIP及BF16 VAE。PE性能对照使用同一2803输入、1909输出token轨迹，每步真实执行模型和采样。完整自然生成另行记录输出长度、文本、图片和耗时。DiT使用相同正负caption、latent seed及KV auto设置。

硬件分母沿用本仓库的HBM2.45TB/s、FP8 MME865TFLOP/s、BF16 MME432TFLOP/s。80%分别为1.96TB/s、692TFLOP/s和345.6TFLOP/s。Intel架构页给出2.45TB/s，产品对照表四舍五入为2.46TB/s，本计划统一采用前者。芯片共有48MB SRAM，实际可分配容量由编译后存储计划核验。[Intel架构](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html) · [Intel产品规格表，第3页](https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf)

### 必须同时报告的指标

| 指标 | 计算与用途 | 最终目标 |
|---|---|---|
| 理论资源利用率 | 有效HBM字节/时间/2.45TB/s，或有用矩阵FLOPs/时间/对应dtype峰值 | 主导资源≥80% |
| 同shape可持续效率 | 实际组件吞吐/同精度、真实布局、权重轮换的最佳独立对照吞吐 | ≥80%，辅助定位实现差距 |
| 阶段硬件效率 | 经审计的阶段硬件时间下界/真实阶段wall time | PE decode及DiT主干分别≥80% |
| 完整用户体验 | 完整重算wall、自然输出token数、cold、跨shape、p50及范围 | 同轮基线持续改善 |

同shape微基准结果和规格峰值分别保留。hl-smi功率派生AIP、kernel活动覆盖、填充行及重复计算分别记录其自身含义。矩阵有用FLOPs采用真实token/像素行，MTP另计已接受tokens、草稿和被拒绝验证的成本。

阶段下界按计算依赖图和资源约束构建。每段记录必要HBM传输、FP8/BF16矩阵工作、TPC归约/特殊函数及必须串行的依赖。FP8和BF16共用MME，矩阵预算为各dtype时间之和。具有独立流水关系的MME/TPC/HBM取可重叠下界，存在数据依赖的段按关键路径组合。TPC下界使用可核验的指令吞吐和依赖约束，最小原语测量单列为经验参照。TPC约束无法完整审计时，该部分标为未闭合，厂商内部调度限制作为实现差距继续追踪。

P0须交付下界的公式、输入、假设、适用shape和不确定区间。算法融合后重新审核必要流量，持续输出原工作量对应的tokens/s、images/s和总时间。80%的最终判定取保守效率估计，缺少可验证下界的阶段标为待证明。

### 单token的物理约束

当前PE每token约15.859GFLOP的FP8 Linear工作和8.730GB的HBM流量，算术强度约1.817 FLOP/byte。按当前数据流，峰值HBM对应的矩阵速度约4.451TFLOP/s，即FP8规格峰值的0.515%。因此PE主线的80%目标落在带宽及完整解码关键路径上。

若要求PE持续达到692TFLOP/s，按2.45TB/s计算需要约282.4 FLOP/byte，约为当前算术强度的155倍。大量独立请求合批或多token验证才有机会提升权重复用到这一量级，实际KV、激活和验证成本还会提高要求。后文保留吞吐分支，并分别验收单请求延迟。

## 2. 基线与硬件预算

| 阶段 | 当前实测 | 当前流量保持时的80%资源预算 |
|---|---:|---:|
| PE后段单token | 13.854ms，HBM0.630TB/s、25.72%峰值 | 4.454ms，约224.5 tokens/s |
| PE完整1909 tokens | 27.267s，首token约0.433s | 约8.93s，按1908个decode间隔外推 |
| DiT负条件forward | 0.912191s，1.386TB，1.520TB/s | ≤0.707369s，HBM1.96TB/s |
| DiT完整八步/CFG1.2 | 15.144s，16次forward | 约11–12s的第一阶段预算，正负条件及首步需逐段核算 |
| CLIP/VAE/保存/其它 | 约2.36s | 先计入端到端预算 |
| 整图warm | 44.773s | 第一阶段工程预算约22–24s |

这些80%时间由当前采集流量推导，属于现有数据流的资源约束。PE计数采集扰动为71.7%，其0.630TB/s为用无profiler时间归一的估算，P0将补低扰动证据。DiT采集扰动0.18%，torch trace扰动约2.82%。[本轮完整调查](../QWEN21_LATEST_BOTTLENECKS_2026-11-16.md)

DiT负条件目标序列8160、KV序列8607、32层、hidden4096。FP8 Linear为113.903TFLOP，BF16 QK/AV按`4 × 32 × 8160 × 8607 × 4096`估算为36.822TFLOP。相应MME规格时间下界约216.916ms，80%矩阵效率对应271.145ms。当前HBM下界为565.895ms，attention融合具有显著降低资源下界的空间。271.145ms覆盖矩阵工作，完整forward还包含TPC、状态、布局及依赖成本。

数据流重构后的探索预算为PE8.5–10s、DiT6–8s、其余约2.36s，整图约17–20s。这是研究预算，最终80%达标还须按新的必要流量和关键路径逐项证明。各候选组合后统一重测完整流程。

## 3. 历史证据决定投入顺序

| 既有结果 | 本轮的具体推进点 |
|---|---|
| BF16大MLP及输出头曾达到2.005/2.200TB/s，即峰值81.8%/89.8% | 已有单卡80%带宽原语证据，重点补齐小矩阵、层间状态及执行链 |
| GDN直接更新＋整步编译将BF16 PE41.69→25.18s，recipe306→84/token | 对当前FP8重新统计recipe，进一步扩展图外尾部与层内融合 |
| 当前FP8已具备原生Linear、GDN单步直接更新、整步decoder和DiT FFN编译 | 接下来审查真实生成图、重复量化与实际kernel边界 |
| FP8 PE权重流量明显下降，完整解码速度与BF16接近 | 单token延迟中的非权重传输成本成为核心实验对象 |
| 历史GDN TPC局部较快，但整段logits漂移、自然图异常；新版直接更新还曾优于旧TPC | 基于当前`_decode_single`重新设计状态读取和归约，复用工具链及失败样本 |
| Qwen真实attention约18.52ms，mixed AV约12.35ms、FusedSDPA fast约11.86ms，后两者抽样误差约5% | 继续建立速度/精度曲线，并定位可用的精度组合、tile和层级选择 |
| MOSS的32MiB SRAM设置、布局切换、早期query分块收益不稳定 | 以实际score驻留、MME供数和HBM流量变化作为复测依据 |
| 同源MTP草稿实验旧BF16吞吐提升约39%–42%，单行/多行logits有差异 | 作为后续权重复用分支，以当前FP8版本重新建立基线 |
| 原生INT8 MME验证阴性，用户要求停止该开发 | 投入当前可用的FP8/BF16 MME和TPC融合 |

MOSS的42层、24头数据用于复用方法和失败条件。当前Qwen DiT为32层、32头，使用本轮真实输入重新构建样本集。[PE根因](QWEN21_DECODE_ROOT_CAUSE.md) · [执行优化](QWEN21_EXECUTION_VALIDATION.md) · [FP8优化](../QWEN21_FP8_OPTIMIZATION_2026-11-16.md) · [Attention历史](ATTENTION.md)

## 4. P0——建立低扰动测量与硬件上限模型

**交付**为当前FP8版本的完整耗时账本、真实shape上限表、可复算的阶段下界和首批候选排序。

1. 在独立进程预热固定实现和全部诊断入口，捕获时直接观察已编译图。使用当前失败trace的最小复现定位全局分支/record_function导致的重编译。保存重编译计数、编译日志与probe开销。
2. 将API、MME/TPC执行事件、BMON/SPMU分为独立采集轮。PE位置取32、512、1536及临近EOS，分别比较16/64/128-token窗口，窗口扩大时同步扩大或减少trace缓冲需求。降低采样频率后重新执行copy、GEMV和idle控制。
3. 以≤5%采集扰动作为测量设计目标。高扰动计数保留访问量信息，关键路径归因使用低扰动时间线和独立重放交叉验证。前后plain窗口交错测量。
4. 捕获当前FP8真实输入、权重及状态。按24个GDN、8个完整attention、32个MLP、输出头、采样和控制边界做重放与消融。权重轮换集合超过SRAM，记录每个区域的读写、时间、recipe、host回读及MME/TPC等待。
5. 小M矩阵扫描M=1/2/4/8/16/32，N/K使用真实层形状。DiT覆盖第1/4/8步、正负条件，以及首步建立KV和后续读取KV。同时记录MME starvation、TPC等待load、向量指令活动、SRAM占用与HBM带宽。
6. 针对native profiler结束后的段错误建立最小退出复现，记录所有子进程returncode、线程栈和采集完整性。普通服务恢复检查继续覆盖出图和队列。

当前BF16时代的84 recipes/token和主线程17.3ms属于历史测量，P0产出当前FP8对应数值后才分配新的host/device预算。初始性能账本要能解释至少95%的阶段墙钟区间，其余区间保留为未归因费用。

## 5. P1——闭合PE的完整token执行链

源码确认`_compiled_hpu_decode`覆盖32层和最终norm。输出头、采样、penalty更新、embedding、cache prepare/advance、位置准备以及每token的`.item()`仍由图外控制。当前代码位于产品`comfy/text_encoders/qwen35.py`、`llama.py`和`hpu_kv_attention.py`。

### 实验次序

1. **一个token的完整计算图。** 将最终norm、FP8输出头量化/GEMM、penalty、top-k/top-p和token选择纳入可编译区域。采样对全词表先应用penalty，再完成top_k20和top_p0.95，保留调用顺序和概率语义。
2. **设备持有状态。** 预分配token、position、seqlen、penalty、KV/GDN状态及输出环形buffer。逐次更新设备值，维持张量地址与图形状稳定。graph执行对象按权重版本、dtype/device及容量建立缓存，mutable状态由每个generate调用持有。
3. **2/4/8-token静态展开。** 下一token直接作为设备侧embedding索引，各步按真实采样结果顺序执行，主机按块接收token和完成状态。以块间中断检查控制取消响应时间。EOS通过设备active标记控制有效状态、penalty与输出长度，RNG按请求隔离并验证随机数消耗顺序。
4. **持久重放。** 先比较现有eager编译的完整token/小块图。对仍被host dispatch限制的部分，使用独立、配套lazy运行时验证长期HPUGraph，其capture成本按模型/容量生命周期摊销。合适路径以普通节点入口的完整净收益选择。

RNG capture能力、静态地址和原地view更新是具体技术验证项。官方Graph要求静态图及稳定地址，本机还存在eager/lazy模式边界。既有Graph核心17.804ms与eager整步17.825ms接近，新试验的增量来自图外尾部及完整token生命周期。[既有Graph试验](QWEN21_BF16_SINGLE_TOKEN.md) · [官方HPU Graph约束](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html)

**阶段预算**为13.85→约10ms/token，用于筛选这一轮的收益规模。若减少主机回读后耗时平台化，按新时间线将资源转向P2。每项记录真实recipe和设备launch数量，编译函数数量单列。

## 6. P2——PE小矩阵供数与GDN专用融合

### 6.1 共享激活量化及矩阵组合

当前`quant_ops.hpu_fp8_linear`逐调用计算行amax、scale和FP8 cast。Q/K/V及GDN QKV/Z等共享输入的投影可共用一次量化结果。首先核对编译器是否已完成公共子表达式消除，再对仍重复的区域接入显式共享。

模型已有`CombinedLinear`对象，其组合权重依赖VBAR residency。当前服务关闭dynamic VRAM，P0要确认普通HPU常驻路径的实际命中情况。新增路径优先复用现有组合表示，覆盖普通常驻FP8权重。分别比较共享quant后多个GEMM、分组recipe和拼接输出维的GEMM。

每组FP8权重保留自己的scale。先验证后端逐输出通道scale能力与精度，支持时可拼接；需要统一scale的实现单独记录额外量化误差。低精度输出后补scale引入的舍入也纳入比较。候选计入权重准备、bias、hooks、LoRA、换权重和卸载重载。

### 6.2 基于当前单步公式的GDN TPC

围绕`GatedDeltaNet._decode_single`的实际状态流设计两个候选边界。

- conv history读取/更新、FP32卷积累加、BF16舍入及SiLU、Q/K归一化形成前段融合。
- recurrent state按value列分块驻留，复用同一state tile计算key与query投影，执行rank-1更新并回写，随后融合gated RMSNorm及输出准备。

按4/8/16个value列、head分配及vector宽度扫描，记录每个TPC的工作量、寄存器/VLM使用和spill。保持FP32 state，显式处理现行卷积、输出、norm的BF16舍入位置。历史TPC构建的16KiB VLM经验和全芯片48MB SRAM分别建账。

该实现优先消除重复state读取、短kernel依赖和布局转换。以当前直接更新路径为同轮对照，完整1909-token状态/输出验证覆盖误差累积。[GDN历史与工具链](TEXT_GDN.md) · [当前单token数值经验](QWEN21_BF16_SINGLE_TOKEN.md)

### 6.3 FP8小M布局

对实际N/K权重轮换测量MME利用、两个MME的负载、供数stall和带宽。比较转置布局、输出维组合及可用的静态权重准备。权重准备在load/版本变化时完成，摊销成本随cold和重载报告。FP8权重无需反复解码即可进入现有原生GEMM。

**阶段预算**为约10→6–7ms/token，整段PE约12–14s。最难的后续目标仍是完整解码≤4.45ms/token及经审核的阶段效率≥80%。这些目标要求内核、调度与状态流一起改善。

## 7. P3——PE完整attention与剩余控制链

当前`single_decode`已使用BF16 Q/K/V、FP32 softmax的FusedSDPA，按完整KV容量构造mask。本工作流容量为6899，有效长度随生成从约2803增长到4712。

1. 比较256/512/1024粒度的有效长度分桶和预分配mask，量化更少KV读取与新shape编译/布局费用的净差。
2. 验证专用GQA decode kernel，按KV head复用历史K/V，在KV tile内执行QK、FP32 softmax统计和AV，局部输出按log-sum-exp合并。保留16个query heads、4个KV heads和D256的真实形状。
3. 审查动态valid length接口的可见性与query位置对齐。官方`valid_seq_len`具有causal模式约束，单token必须能访问全部合法历史，采用与现有mask等价的实现。
4. 依据P0独占时间处理RoPE、cache准备、采样及penalty更新，计入融合边界造成的舍入、额外编译和缓冲复制。

历史单独分桶只有约1.09%的完整短文本收益，P3的收益需要在P1/P2后的新瓶颈上确认。[历史逐项结果](QWEN21_EXECUTION_VALIDATION.md) · [FusedSDPA参数](https://docs.habana.ai/en/latest/PyTorch/Reference/Python_Packages.html#hpex-kernels-fusedsdpa)

## 8. D1——Qwen DiT attention的精度与执行路径

当前DiT kernel活动覆盖率已经96.81%。负条件trace中BatchGemm约386.5ms、softmax独占156.6ms、其它kernel独占186.7ms，设备跨度916.5ms。因此DiT主线投入集中于attention和中间量数据流。

1. 捕获32层中0/15/31层，第1/4/8步正负条件及异常误差层。分别记录首次前缀构建、后续rectangular Q/KV及缓存读写。
2. 在真实Q=8160、实际正负KV长度上重新筛选厂商FusedSDPA、默认/fast/FP32 softmax、BF16 QK＋FP8 AV，复用之前32/8/1 head分组结果。新增实验必须绑定精度方案、tile或新真实shape变化。
3. 将归约精度、概率量化scale、V量化及敏感层策略独立开关化，记录对FP32/FP64参考的误差分布、完整采样轨迹和图像变化。
4. 检查prefix K/V的重复cat、layout转换及生命周期。使用稳定buffer或接受分段KV的算子验证减少物化成本的收益。

厂商FusedSDPA具备降低attention存储需求的实现，fast及FP32 softmax的速度/精度需按本机结果选择。编译模式使用显式`recompute_mode`参数，比较每种模式的实际分配和图结构。[官方FusedSDPA说明](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_PyTorch_Models.html#using-fused-scaled-dot-product-attention-fusedsdpa)

**阶段预算**为DiT15.14→约11–12s，同时验证固定流量负条件forward≤0.707s的80%带宽目标。精度调整作为独立候选接受完整质量评价。

2026-11-17执行结果采用当前已验2094-token caption。五轮固定caption DiT15.243→10.995秒，完整重算38.508→34.138秒。负条件流量降至0.873133TB、时间0.662545秒，有效HBM带宽峰值占比53.79%。按新流量重算，80%预算为≤0.445476秒。D1时间预算已达成，阶段资源效率继续沿D2验证。数值变化、283项回归和交付状态见[D1执行记录](QWEN_D1_ATTENTION_2026-11-17.md)。

## 9. D2——MME/TPC片上attention流水

**核心目标**是让QK tile、softmax统计、概率及AV累加在受控片上数据流中衔接，降低完整score/probability矩阵的HBM往返。

### 9.1 建图和驻留证据

导出当前Qwen FX、Synapse节点、tensor allocation和bundle。核对QK/softmax/AV的精度、tile、消费者、alias、layout约束及实际SRAM/DRAM位置。MOSS曾观察到scores落DRAM、FP8概率tile位于SRAM，本轮Qwen以自己的生成图为准。

### 9.2 分块算法

以query tile128/256、KV tile256/512/1024和head组1/2/4/8建立有限搜索空间。MME计算QK和AV，TPC执行在线softmax的max/sum、旧累加量重标定及融合变换。首轮保留BF16 Q/K和FP32归约，再独立评估概率/V的FP8化。

SRAM预算逐项列入Q/K/V tile、scores、概率、FP32输出累加、双缓冲及运行时保留区。双缓冲和MME/TPC重叠是否发生，以实际时间线、allocation和stall计数确认。首个工程目标为负条件HBM流量降低≥40%，完整forward同时变快；随后按新的必要流量重算阶段80%目标。

### 9.3 实现层级

1. 厂商FusedSDPA和现有bridge lowering优先完成可用路径。
2. 基于固定bridge1.24.2源码构造最小子图，调整表达方式、分区、bundle与tensor lifetime，观察scores驻留变化。
3. 若受编译器接口约束，使用独立构建产物验证专用Synapse图/运行时模块，处理allocator、stream/event、cache失效及输出所有权。已有直接MME/TPC源码提供设计参考，按本机SDK重新验证支持边界。

公开CustomOp接口的单个descriptor对应一个TPC kernel。跨MME/TPC流水由组合建图或更底层运行时承担。独立构建与原服务依赖隔离，ABI、同步和卸载通过后才进入普通产品入口。[官方CustomOp边界](https://docs.habana.ai/en/latest/PyTorch/Reference/PyTorch_CustomOp_API/page_index.html#api-limitations) · [既有外部源码审计](KERNEL_REFERENCES.md)

### 9.4 补齐其余约20%的kernel费用

随后按独占时间排序处理调制norm、Q/K norm/RoPE、残差、FP8量化与scale写回。FFN已经区域编译，新增融合以实际tensor消失和字节/launch减少验证。对连续2/4层区域比较中间buffer、recipe分裂、编译时间和质量，选择完整DiT净收益最高的边界。

**阶段探索预算**为DiT6–8s。达到该时间后继续核验资源/阶段80%条件，余量不足则按新瓶颈继续迭代。历史32MiB SRAM设置和整block扩区只有在实际数据流改变后重测。

## 10. 权重复用与吞吐后续分支

PE主线从当前无MTP入口推进。此前暂缓的MTP保留为后续候选，复用已有243M草稿头和状态验证工具。

- 多token验证重新比较2/4/8行，并记录每轮已接受tokens、草稿时间、目标验证时间和state commit/rollback。性能模型采用`(draft + verify + commit) / accepted_tokens`。旧BF16收益以当前FP8约70 tokens/s重新对照。
- 对同一前缀比较单行和多行目标logits，检验数值变化及拒绝采样的概率语义，完成自然文本和配对图质量评估。
- 独立请求吞吐实验按2/4/8/16批次逐级扩展，统计token/image吞吐、p50/p95延迟、排队、每请求KV/GDN和峰值显存。内存容量从真实buffer清单和峰值推导。
- 若目标进一步要求持续≥692TFLOP/s FP8矩阵计算，单独验收足够大M的工作负载及真实有用吞吐。该结果同时标注批量、并发、内存和响应延迟。

[既有草稿头结果与数值边界](QWEN21_TEXT_NEXT.md)

## 11. 实验、质量与交付规则

每个候选记录基线commit/源码哈希、真实入口和shape、固定工作量、采集配置、plain时间、计数扰动、CPU/HPU内存、HBM字节、矩阵工作、TPC费用及输出质量。先单项，再组合交错A/B/A，最后使用普通ComfyUI节点运行。

正式性能至少1 cold＋5 warm。小于约3%的候选增加交错和独立进程复验，单独报告缓存命中及跨shape成本。新增输出长度、算法和precision时补完整自然请求，与固定1909-token结果共同展示。

质量验证覆盖当前Qwen真实层、不同文本/latent seed、512²/1024²/2MP、thinking、KV配置及受影响的hooks/LoRA/卸载路径。保留logits RRMSE、top-1、KL、state误差和FP32/FP64参考，逐元素一致作为强证据，数值有变化的候选进入人工配对图验收。TPC递推和多token路径重点检查长序列误差累积、EOS、取消、重复请求和状态隔离。

每个有效产品里程碑按仓库既有流程实施、自动验证、暂存两个仓库相关改动、部署独立`/home/sonny/ComfyUI-Gaudi`、验证普通服务，交付人工验收。部署身份包含基线commit、暂存树/patch及源文件哈希。用户明确确认验收后提交。计划、失败候选、服务器信息和证据归入本仓库，产品仓库存放产品代码及便携测试/脚本。[既有交付方法](MEASUREMENT.md)

## 12. 执行顺序与完成条件

| 顺序 | 交付结果 | 进入下一轮的依据 |
|---|---|---|
| P0 | 当前FP8低扰动账本、上限表、故障最小复现 | 明确host、MME供数、TPC依赖的实际预算 |
| P1 | 完整token图、设备状态、小块执行对照 | 证明图外边界成本可消除的份额 |
| P2 | 共享FP8量化/投影、GDN专用融合、真实小M布局 | 用完整文本确认局部融合收益 |
| P3 | 有效长度attention及控制链收敛 | PE decode向4.45ms/token和阶段80%逼近 |
| D1 | 精度可解释的Qwen attention候选 | DiT达到约11–12s并核验80%资源目标 |
| D2 | 片上attention流水和邻接融合 | 流量显著下降，重新证明阶段硬件效率 |
| 整合 | 固定/自然整图、cold、跨shape和生命周期报告 | PE、DiT分别达成80%证明及人工质量验收 |

单卡执行串行。P0之后可交替推进PE与DiT实验以复用已有捕获，但各候选使用独立进程和清晰基线。优先实现有完整阶段收益预算的路径。

最终交付包括各主导资源80%的实测证据、PE与DiT阶段上限模型、完整工作流收益和质量/生命周期验收。17–20秒作为深度优化探索预算保留，最终达标时间由经审核的硬件下界决定。若某阶段受当前SDK、数值要求或不可隐藏依赖限制，报告具体约束、剩余差距及下一条可验证实现路径，持续保留80%作为目标。
