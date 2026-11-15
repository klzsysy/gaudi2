# 性能预算、硬件计数与外部对标

以下按实验版本保留可复算预算。最新1024²设备细分来自`0995638a`，长序列和文本后续收益见各主题。全部外部硬件/5090数据沿用2026-11-10至11-12的来源快照。

## 从工作流定位费用

最早严格基线`0690d18f`在原INT8 ConvRot文本下各做1 cold＋5 warm，完整自然327-token为137.282秒，固定caption整图40.183秒，TextGenerate96.966秒、KSampler37.960秒，全部fallback为空。这一基线表明清理CPU fallback后，HPU参考算子与提交仍有大量费用。用户更早143.34秒请求的seed和生命周期不同，收益比较使用本轮冻结工作流。

早期`05d2a0a5`的36.647989秒代表请求由扩写14.949898、正负编码1.754006、KSampler19.666598、VAE0.132846、图片输出0.123620及其它0.021022秒组成。该请求输出277 tokens，同组自然输出267–362 tokens。节点内`CLIP.generate`与TextGenerate仅差约7ms，推动后续KV/解码优化。

`441d459d`的完整自然298-token五轮中位30.786秒，文本缓存19.900秒。单独同步诊断32.879秒中的DiT主干18.150秒、text_embedder0.797秒、RoPE0.541秒，后两项各执行50次，由此启动条件预计算。文本诊断10.489秒中verify5.239、prefill2.035、MTP head0.971、lm_head0.573、采样0.609、其它1.063秒。诊断与无同步计时分别解释。

`0995638a`固定1970输入/298输出、正负307/23、1024²/25步的完整五轮中位15.925秒，TextGenerate5.040、普通编码0.386、DiT10.211、VAE0.132、图片输出0.108秒。节点中位数独立统计。后续两组完整复测16.009/15.995秒、DiT10.190/10.183秒。

最新五轮节点区间并集外只剩15.50–16.79ms，中位15.90ms，约服务执行窗口0.1%。文本Python控制、bridge提交和同步位于节点内，浏览器、排队及HTTP窗口另有计时范围。

## BF16与FP8设备预算

均以step0/12/24×正负六段trace外推25步，硬件lane用区间并集按互斥优先级归类。

| 类别 | BF16 `64155fe4` | 混合FP8 `0995638a` |
|---|---:|---:|
| MME总活动 | 9.225s | 下列分项 |
| FP8 Linear | — | 2.513971s |
| FP8 AV | — | 0.530822s |
| BF16 QK | 包含于MME | 2.405666s |
| 其它MME | 包含于MME | 0.003307s |
| softmax独占 | 2.398s | 1.340238s |
| 其它kernel独占 | 1.805s | 2.652100s |
| kernel间隙 | 0.752s | 0.815277s |
| 合计 | 14.181s | 10.261380s |
| 对应无profiler DiT | 14.400s | 10.182960s |

BF16 MME内部Linear5.126秒，QK2.646、AV1.539秒，QK/AV并集4.106秒，内部有重叠。真实42层权重轮换Linear外推5.140秒，吞吐374–404TF/s，热重复同层与真实轮换差异很小。FP8 Linear活动阶段约806TF/s，大矩阵已接近本栈同shape有效速度。

FP8的其它kernel约99.4%已关联生成图，约0.015秒语义待补齐。主要独占项为SiTU1.01、Q/K FP32 norm0.27、调制norm0.26、量化准备0.20、GEMM输出scale0.19、RoPE/cast0.18、残差0.15秒，另有scale/copy共同活动0.13、Q/K norm/copy0.10秒。SiTU已是单个融合TPC kernel，两个条件哈希为`0x39BA8699`和`0xD234406D`。

## AIP口径修正

本机`1.24.2-52a555d`加载模块的源码、srcversion及反汇编确认，`hlthunk_get_device_utilization`经`HL_INFO_DEVICE_UTILIZATION`进入功率公式。

```text
AIP = floor(100 × (clamp(curr_power, 60000, 550000) - 60000) / 490000)
```

单位mW。531组utilization前后功率配对中516组在两端公式整数区间内，全部在区间±1个百分点内。模块SHA256为`21fa94a804e1a4a3745dd256cdcb4381bf8762113bd664d843ad5cc304a7613f`，srcversion为`0A86AA51441E652BE8CD384`。这修正了早期按通用官方说明解释为活动窗口的结论。

因此本机AIP约58%可作功率派生观察量，算术效率由FLOP模型、实际shape吞吐和设备时间线判断。`utilization.memory`描述内存占用。已有SOC1650MHz、温度42–61°C数据未显示持续降频，主机单socket、32核64线程、单NUMA，32→16线程完整流程21.060→21.168秒，无收益。

## HBM、SRAM与计数器边界

2026-11-15后续完成计算路径的AXI返回beat校准，大块读写与GEMV读取回算误差约−0.44%至−0.26%。Qwen PE单token实测约18.43 GB读写、0.80 TB/s、峰值约33%。旧请求侧`BMON BW`及大窗口低计问题、取整误差和独立PDMA边界见[新校准](QWEN21_BF16_SINGLE_TOKEN.md)。以下保留2026-11-12的历史采集条件与当时未决问题。

scores落DRAM、FP8概率tile驻SRAM的实际图见 [Attention](ATTENTION.md)。完整scores一次写出/读取逻辑量为

```text
25 × 42 × 24 × (4403² + 4119²) × 4 bytes ≈ 3.664 TB
```

原先把scores和probabilities都完整写回HBM的5.50TB假设已被图证据修正。逻辑分数读写以本机有效copy带宽换算约1.7–1.85秒，其费用已经包含在QK/softmax预算中。

128/512/2048MiB每份buffer、各五组50轮同步copy，读写合计有效带宽1.982/2.096/2.148TB/s。真实block取得HBM/MME/TPC SPMU及BMON原始事件。正条件连续三次含输入clone，MME A/B starvation约20.06/20.07百万，TPC内存反压42.85百万、等待ld_tnsr48.32百万、vector active297.39百万，HBM读/写ACT为6.15/4.47百万。copy128→512MiB使TPC active约3.99倍、读ACT约3.95倍，支持计数响应真实负载。

BMON的65535/4096窗口对已知copy回算字节尚不一致，厂商原生JSON也出现Avg高于Max。ACT、stall有访问模式和重叠含义，实际模型HBM总字节、TB/s及饱和率仍待校准。后续核对本机counter单位、窗口、压缩字段，并用独立读/写控制验证。

硬件采集需`HABANA_PROF_CONFIG`、`HABANA_PROFILE=1`、对应单元和周期采样，普通torch.profiler导出未包含这些计数。使用安装头文件的`synTraceEvent`及Synapse C API，保留原字段。按图保存post-graph目录可避免多线程写同一文件的rename冲突。

## 文本间隙与profiler扰动

GC控制后的16轮窗口无profiler553.001ms、profiler559.853ms，扰动1.24%。设备跨度548.497ms，活动并集330.621ms，间隙217.876ms即39.7%，最大单次间隙0.966ms，`launch_recipe`事件2288次。两轮窗口即使GC控制仍约31%扰动，优先用较长窗口归因。

空档需继续区分模型主机依赖、bridge入队、同步和队列耗尽。160次Kitchen RoPE CPU区间累计约61ms、1744次Dynamo查找约75ms，存在嵌套。后续RoPE编译取得整段收益，但39.7%仍不能直接乘到整段文本时间。

## 算术下界与可兑现规模

MOSS为42层、d3072、FFN8192、24×128 heads，FMA计2 FLOPs。每层长度N的Linear为`2N × (4d² + 3df)`，QK/AV为`4N²d`。两条件4403/4119、25步总矩阵2.496203PFLOP，其中主干Linear2.026677、QK/AV各0.234518PFLOP。

以BF16峰值432TF/s计算必要矩阵预算5.778秒，当前Linear/AV FP8、QK BF16的对应预算3.157秒。两者省略非线性、量化、访存、状态及调度。继续把QK改FP8的峰值算术差仅约0.272秒，而真实QK约2.406秒，形状与供数效率值得优先研究。

历史文本129 verify、388 lm_head、258 MTP调用的一次权重读取估算约2.700TB，理想传输约1.102秒，prefill/编码矩阵约0.074秒。连同BF16 DiT得到6.954秒的不完整乐观参照，KV/GDN、完整prefill attention、控制和VAE仍有费用。小M吞吐需独立测量。

| 独立假设，参照约16.1秒完整流程 | 理想节省 |
|---|---:|
| FP8 Linear全达865TF/s | 约0.171s |
| SiTU减半 | 约0.505s |
| 量化独占降至0 | 约0.200s |
| 其它kernel整体减半 | 约1.326s |
| DiT间隙全部消除 | 约0.815s |
| QK＋softmax减少35% | 约1.311s |
| QK＋softmax＋AV减半 | 约2.138s |

这些情景有包含关系。按壁钟归一，QK/softmax、其它kernel、间隙分别减35%/30%/40%对应DiT约7.77秒，减55%/45%/60%对应约6.47秒，均属于实现前假设。新增同步、精度修正和引擎重叠会改变净收益。

## RTX 5090历史对标

| 来源与日期快照 | 作者配置/时间 | 使用边界 |
|---|---|---|
| KOBO，2026-10-31测试、11-05更新 | SFT BF16＋Qwen BF16，1024²/25步/CFG2.5，缓存整图14.7s；INT8 DiT＋W4A8文本5.8s | 同prompt换seed、不重算扩写，小VAE，缺原始重复样本数 |
| DSLab，2026-10-20 | INT8 DiT、1024²/25步，采样约7s | 另报9:16全图8.7s，尺寸口径单列 |
| Niklas Lenz，2026-10-20 | INT8、1024²/50步、CFG3、euler/sgm_uniform、SLG22，12图中位12.81/12.99s | 文本事先生成，扩写另测12文本中位1.4s，原始JSON当时403 |

来源保留为作者的一手报告快照。[KOBO](https://localai-kobo.com/moss-image-comfyui/) · [DSLab](https://dslab.tw/moss-image-lora) · [Niklas Lenz](https://niklaslenz.com/image/moss-image-sft-int8-comfyui-5090/)

本机BF16缓存五轮14.386秒、FP8当轮普通缓存单次10.696秒，处于不同权重、caption、VAE和软件条件。5090量化DiT的约6–7秒可作方向性体验目标，完整扩写应另建相同1970→298轨迹对照。历史BF16缓存≤14秒/完整≤20秒门槛分别得到14.386/19.978秒，10–12秒BF16缓存目标尚无交付结果。

硬件规格用稠密BF16/FP32累加口径，Gaudi2为432TF/s、96GB、2.45–2.46TB/s，5090为209.5TF/s、32GB、1.792TB/s。峰值与向量/矩阵资源比例只能提供尺度，跨机归因需要同模型trace。[Intel产品表](https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf) · [NVIDIA Blackwell白皮书](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)

## 证据与复现

原始目录为`performance-2026-11-10-pipeline`、`performance-2026-11-11-completion`及`performance-2026-11-12-evidence`。入口为`moss_pipeline_measure.py`、`summarize_moss_pipeline.py`、`estimate_moss_bf16_roofline.py`、`moss_evidence_jobs.py`及`moss_fp8_system_summary.py`。[BF16预算](../evidence/moss-bf16-completion-2026-11-11.json) · [计数与六段预算](../evidence/moss-evidence-boundaries-2026-11-12.json) · [节点外耗时与文本间隙](../evidence/moss-layer-boundary-audit-2026-11-12.json)
