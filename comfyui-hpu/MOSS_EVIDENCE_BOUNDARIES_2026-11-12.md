# MOSS性能证据边界补齐

2026-11-12，Asia/Shanghai。按用户要求，先完成全栈分析第10节的证据调查，再据结果进入第9节的实现投入。基线产品为`0995638a`。

## 交付状态

| 项目 | 已完成证据 | 当前边界 |
|---|---|---|
| DiT代表性 | step0／12／24×正负六段干净trace，两组独立5 warm | 当前1024²固定工作量下已核对 |
| 其它kernel | kernel哈希与编译后图、融合子图、tensor allocation关联 | 约99.4%已关联语义，约0.015秒保留未决归属 |
| HBM与引擎限制 | 带宽校准，真实block的HBM／MME／TPC SPMU及BMON原始事件 | 实际HBM字节数和BMON换算仍待校准 |
| AIP及文本扰动 | 已加载驱动反汇编、531组功率配对、GC控制及两种文本窗口 | 口径已确认，短profile扰动已量化 |
| 精度与质量覆盖 | 新增72组真实输入、4组新提示词图像及FP32参考 | 参考沿用解码后的量化权重，覆盖当前采样集合 |
| 收益估算可靠性 | 修正AIP、SRAM概率驻留和profiler间隙解释 | 用户已授权进入第9节，候选按已验证计时和质量验收 |

## 环境检查

- 实际运行内核为`5.15.0-203-generic`，驱动`1.24.2-52a555d`，设备Operational。与此前环境文档的202内核记录存在时间差。
- 当前服务绑定`192.168.10.21:8189`，队列空闲。主机为单socket、32核64线程、单NUMA节点。
- 新大型证据保存于服务器`/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-12-evidence`，摘要和调查脚本保存于本仓库。
- 首次检查127.0.0.1:8189连接被拒，随后通过进程参数和监听地址确认服务正常。服务器产品目录按部署文件管理，本地Git提交和源码哈希用于核对版本。

## 1. AIP的实际实现已确认

本机加载的驱动采用功率公式计算device utilization。源码链为`hlthunk_get_device_utilization`→`HL_INFO_DEVICE_UTILIZATION`→`hl_device_utilization`。实际驱动模块的srcversion与磁盘模块一致，反汇编确认了减法、乘100和整数除法。当前驱动读取的`max_power`为550000mW，Gaudi2的`dc_power_default`为60000mW。

```text
utilization = floor(100 × (clamp(curr_power, 60000, 550000) - 60000) / 490000)
```

通过同一hl-thunk接口，在utilization读取前后各读一次功率，共531组。516组的utilization落在两个功率读数推导的整数区间内，531组全部落在该区间±1个百分点内。例如242743／243437mW对应两侧计算结果均37%，实读37%。功率读取和利用率读取为顺序调用，瞬时变化解释了小幅偏差。

因此当前AIP适合作为这个驱动的功率派生观察量。此前从官方通用说明引出的活动窗口解释需要按本机实现修正。`hl-smi`展示的整板功率、query功率和驱动这一功率入口也应分别记录。

模块SHA256为`21fa94a804e1a4a3745dd256cdcb4381bf8762113bd664d843ad5cc304a7613f`，srcversion为`0A86AA51441E652BE8CD384`。服务器证据为`inventory/loaded-driver-utilization.asm`、`power-control/driver.jsonl`及本机`/usr/src/habanalabs-1.24.2-481/drivers/accel/habanalabs/common/device.c`第1616行。

## 2. 六段DiT预算和诊断扰动

使用同一Moody／Qwen FP8文件、1970输入／298输出固定MTP程序、正负条件307／23、1024²、25步、CFG2.5。两次独立进程的完整warm中位数为16.009和15.995秒，DiT为10.190和10.183秒，与前轮基准接近。

首次六段采集中，负条件三个窗口都出现约2秒`TorchDynamo Cache Lookup`停顿。控制实验将Python GC移出profiler窗口后，负条件第0步的设备跨度从2275ms回到197ms，最长CPU编译区域约0.88ms。模型运算沿用当前产品。新的六段预算采用GC控制后的trace，旧窗口保留在`dit-six-retry`供审计。

| 互斥类别 | 六段外推到25步 |
|---|---:|
| FP8 Linear | 2.513971s |
| FP8 AV | 0.530822s |
| BF16 QK | 2.405666s |
| 其它MME | 0.003307s |
| softmax | 1.340238s |
| 其它kernel | 2.652100s |
| kernel间隙 | 0.815277s |
| 合计 | **10.261380s** |

外推总量较无profiler DiT10.182960秒高约0.77%。同条件各timestep的矩阵和softmax预算接近，原step12计算预算获得支持。间隙仍以完整无profiler计时交叉约束。kernel与runtime事件按并集及固定优先级归类，各硬件lane的duration保持独立含义。

## 3. 其它kernel已关联到生成代码

从当前产品block导出正负条件的编译后图，图中包含融合子图、原始算子、dtype、shape、stride、SRAM／DRAM allocation。按融合kernel哈希关联六段trace，约99.4%的其它kernel互斥时间已获得语义归属，剩余约0.015秒列为未决。

| 主要独占成本 | 约秒数 |
|---|---:|
| SiTU | 1.01 |
| Q/K FP32归一化 | 0.27 |
| 调制归一化 | 0.26 |
| 量化准备与转换 | 0.20 |
| GEMM输出scale | 0.19 |
| RoPE及cast | 0.18 |
| 残差 | 0.15 |
| GEMM输出scale与布局copy共同活动 | 0.13 |
| Q/K归一化与布局copy共同活动 | 0.10 |

表中列出主要项，完整互斥组合保存在`graph-attribution.json`。共同活动项单列，避免把重叠费用重复相加。

SiTU的两个tanh、一个sigmoid、除法和乘法已经形成单个TPC融合kernel，正条件哈希`0x39BA8699`、负条件`0xD234406D`。后续如研究SiTU，应围绕实际执行吞吐、精度与下游量化衔接设定假设。当前量化准备的独占预算比此前把2.65秒整体视作量化／逐元素空间更具体。

## 4. Attention的真实存储布局

正条件编译后图明确显示以下数据流。

1. QK使用SRAM中的Q/K切片，输出BF16分数切片`[4403,1536]`，alias到DRAM中的完整`[4403,4403]`分数张量。
2. softmax读取DRAM分数切片`[4403,2048]`，直接生成SRAM中的FP8概率切片。
3. FP8 AV消费同一份SRAM概率切片，V和AV输出tile也位于SRAM。

原先假定scores和probabilities都完整写回HBM的5.50TB情景需要修正。编译器已经实现softmax→FP8 AV的分块片上复用。当前确认的完整分数写出／读回逻辑量为

```text
25 × 42 × 24 × (4403² + 4119²) × (2 + 2) bytes ≈ 3.664 TB
```

这个数表示QK分数的一次写出及一次读取，实际DRAM事务仍受到softmax读取、布局、cache和执行调度影响。后续可兑现的attention收益应围绕剩余QK分数往返和矩阵执行效率展开，已完成的概率驻留纳入当前基线。

## 5. HBM校准和真实硬件计数

最初报告中缺少可引用的带宽结果。本轮在服务器旧原始目录找到2026-11-08的256MiB缓冲区copy读写合计1.949TB/s，并重新测试更大的工作集。

| 每份缓冲区 | 读写合计有效带宽 |
|---|---:|
| 128MiB | 1.982TB/s |
| 512MiB | 2.096TB/s |
| 2048MiB | 2.148TB/s |

每组5次重复、每次50轮，显式同步，源和目标均超过片上SRAM容量。这里的逻辑字节与完整调用耗时给出有效带宽标尺。

本轮通过安装版本的`hl-prof-config`及Synapse profiler C API，导出真实block的HBM、MME、TPC SPMU与BMON事件。配置计数器后还需显式开启对应硬件单元，并开启周期采样。初始只有TPC结果的采集已被后续完整配置替代。

以下是正条件第12步第0层连续执行3次的累计计数，包含每次输入clone。

| 硬件计数 | 值 |
|---|---:|
| MME Operand A Data starvation | 20,064,585 |
| MME Operand B Data starvation | 20,070,883 |
| MME CTRL2EU Data propagation | 32,876,835 |
| TPC内存反压 | 42,847,212 |
| TPC VPU等待ld_tnsr | 48,316,602 |
| TPC lookup table miss等待 | 7,141,794 |
| TPC general vector stall | 73,445,133 |
| TPC vector pipe active | 297,386,679 |
| HBM读事务ACT命令 | 6,149,765 |
| HBM写事务ACT命令 | 4,470,257 |

负条件也取得完整计数。矩阵、copy和sigmoid控制负载验证了计数器会随实际工作变化，例如纯copy的MME数据传播为0，128→512MiB的copy使TPC active约3.99倍、HBM读ACT约3.95倍。由此可以确认真实模型存在MME供数等待及TPC内存／加载等待。

**尚待闭合的是实际HBM字节口径。** ACT记录内存命令行为，其与字节数的关系受访问模式影响。BMON已导出原始带宽事件，但目前对65535和4096两种采样窗口、128／512MiB已知copy的回算未取得一致字节数。因而实际模型HBM TB/s、总字节数和带宽饱和百分比继续保留为未决项。SPMU各项也有重叠，当前按原始计数报告。

接续需要核实本机1.24.2 BMON的counter值单位、采样窗口、事件压缩及有效字段语义，再用已知字节的读／写控制分别验证。另以自动enqueue采集产生厂商原生HLTV和JSON，原生JSON也出现同类读写带宽值及Avg高于Max的记录。该结果为TEF读取方式提供交叉核对，字节积分仍未通过控制工作量校准。当前已有原始事件、配置、C API状态与控制工作量，可直接用于这一核对。

## 6. 文本profile扰动已定量

固定MTP的全部draft、采样及130轮控制继续执行，输出298 tokens，各轮计数断言通过，fallback为空。普通独立文本约4.70秒。

| 采样窗口 | 无profiler、窗口首尾同步 | profiler窗口 | 观察 |
|---|---:|---:|---|
| 两轮，原采集3次中位数 | 67.34ms | 98.16ms | 窗口扰动约45.8% |
| 十六轮，原采集3次中位数 | 547.97ms | 602.70ms | 窗口扰动约10.0% |
| 两轮，GC控制复核 | 70.39ms | 92.24ms | 扰动仍约31.0% |
| 十六轮，GC控制复核 | 553.00ms | 559.85ms | 扰动约1.24% |

profiler停止及解析还会显著增加整次诊断时间，GC控制后的16轮profile请求约100秒，窗口本身约0.560秒。正式性能继续采用完整无profiler结果。后续短步归因优先采用较长窗口，并保留同窗口控制。

## 7. 扩展精度与图像证据

新增4组固定caption覆盖极短提示、四物体计数与文字、低对比细纹理、复杂书店构图。每组捕获正负条件、step0／12／24、layer0／20／41，共72组输入，正条件长度20／58／77／272。完整图、latent／VAE有限性及fallback检查通过。

| 与相同解码量化权重的FP32计算参考比较 | 中位RRMSE | 最大RRMSE |
|---|---:|---:|
| 全部72组 | 1.267% | 4.200% |
| layer0 | 1.267% | 2.385% |
| layer20 | 0.441% | 0.585% |
| layer41 | 3.125% | 4.200% |

全部输出有限，最大误差位于复杂书店正条件第12步第41层。图像检查中，单梨、四种颜色形状及FOUR SHAPES、针织／蕾丝／金属纹理、书店人物与猫及NIGHT BOOKS均保留。该集合拓展了已有18组单提示词层输入，后续敏感精度试验可复用全部90组。

## 8. 失败路径与恢复

- `diT-six`使用较早system wrapper，在cold77.035秒成功后，第一个warm进入DiT时段错误，profiler尚未启动。改用本轮已验收deep wrapper后，两组各5 warm、六段trace、质量、精度和文本对照完成。根因仍未确认，保留两个launcher版本与日志。
- 旧Python profiler API引用的模块在当前安装中缺失，改用当前torch.profiler及已安装头文件定义的Synapse C API。
- 自定义profiler配置通过`HABANA_PROF_CONFIG`加载，`HABANA_PROFILE=1`启用。最初错误配置路径导致无设备条目，该结果排除。torch.profiler的常规导出也未包含所需计数，硬件采集改由Synapse API完成。[官方配置说明](https://docs.habana.ai/en/latest/Profiling/Intel_Gaudi_Profiling/Configuration.html)
- 多线程同时写单一pre-graph文件导致rename失败，后续采用按图分别保存的post-graph目录，完成正负条件导出。
- 计数器导出直接依据本机`synTraceEvent`头文件读取TEF buffer，保留原始计数和未解释字段。所有校准失败结果均保留在各自目录。

产品9份源码／测试文件与`0995638a`验收哈希一致。首轮正常入口cold79.711秒、文本缓存10.709秒、768²／MTP off14.192秒。最后一次自动计数器控制结束后，8189服务重新启动并复验cold79.834秒、缓存10.693秒、768²／MTP off14.071秒，均success，队列为空。大型trace和原始事件继续保存在服务器。

## 9. 接续条件

本轮足以修正AIP解释、attention流量假设、kernel优先级和短profile方法。剩余重点为BMON字节口径及约0.015秒kernel语义归属，首次warm段错误另列稳定性调查。用户随后明确授权进入第9节，优化使用已验证计时和精度数据，BMON持续调查。[优化执行记录](MOSS_OPTIMIZATION_EXECUTION_2026-11-12.md)

[结构化证据](evidence/moss-evidence-boundaries-2026-11-12.json)保存六段预算、映射、计数、质量、扰动、版本和服务检查。复现入口为`probes/moss_evidence_jobs.py`，阶段1–5在独立checkout串行执行；各阶段的失败和修正见上文及`jobs.jsonl`。
