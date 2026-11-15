# Qwen PE单token BF16的HBM校准与执行优化

2026-11-15。按用户指定顺序完成HBM计数校准，再验证整步执行复用和GDN融合。实验使用已验`dfca7cd6`、匹配PE的BF16文件、单卡Gaudi2和Habana1.24.2，目标模型采用现行单token PE，MTP调用为0。所有候选在独立调查进程运行。

## HBM计数校准

旧`BMON BW`字段不能直接积分成有效字节。大bandwidth window严重低计，缩到255后，HMMU请求侧读计数在copy/sum中约为已知量的1.5倍，在GEMV中约为1.25倍。帮助列出的HMMU物理地址b12/b13在安装库schema中缺失，可用的是b10/b11。这里继续保留原始字段及失败窗口，尚未把低计归因到特定硬件字段的溢出。

启用`bmon_AXI_Completion_Bandwidth`后，`BMON`类别返回每个bandwidth window的AXI beat计数。使用16个DDR MMU的Input Read/Write Response，关闭压缩，以已知大块copy、只读sum、只写fill、GEMV交叉校准。稳定设置为bandwidth_window=255、capture_window=4095或16383。回算公式为

```text
bytes = Σ Avg_beats × (capture_window + 1) / (bandwidth_window + 1) × 128
```

128 bytes/beat及窗口加一的解释经过以下独立负载验证。每项20轮，buffers远大于片上缓存。这里统计计算路径的HBM有效载荷，HBM协议/ECC/刷新开销另有口径。独立Synapse PDMA控制绕过这些HMMU计数，文本的小量host控制传输也不包含在本统计中。

| 对照 | 读取回算/已知量 | 写入回算/已知量 |
|---|---:|---:|
| copy 128 MiB | 99.711% | 99.625% |
| copy 512 MiB | 99.636% | 99.614% |
| copy 2048 MiB | 99.620% | 99.614% |
| sum 512 MiB | 99.664% | 包含归约中间写入 |
| fill 512 MiB | 少量运行时读取 | 99.560% |
| GEMV 128 MiB | 99.572% | 小输出受采样取整影响 |
| GEMV 512 MiB | 99.716% | 小输出受采样取整影响 |
| GEMV 2048 MiB | 99.744% | 小输出受采样取整影响 |

读取有效对照误差为−0.43%至−0.26%，大块写入为−0.44%至−0.37%。idle的Avg积分为0。Avg以整数beat报告，低频小写入会被量化，不能用GEMV的小输出单独校准写带宽。每个非零Max窗口补一个Avg beat给出取整敏感性范围，这个范围依赖Avg截断假设。

## 真实单token访问量与利用率

冻结用户最新1088×1920图的提示词，2803输入tokens、thinking开启、4096输出上限，因此KV容量与原工作流一致。分别采集第33次forward后的4轮以及第512次forward后的16轮，均包含单token主干、输出头、采样、embedding和状态准备。

| 指标 | 4轮窗口 | 16轮窗口 |
|---|---:|---:|
| 读取GB/token | 17.1668 | 17.1680 |
| 写入GB/token | 1.2595 | 1.2609 |
| 合计GB/token | 18.4263 | 18.4289 |
| profiler窗口秒 | 0.11197 | 0.39613 |
| 同进程无profiler窗口秒 | 未单列 | 0.36819 |

两个位置的访问量基本一致。16轮窗口的无profiler速度为43.46 tokens/s，按校准流量折算约0.801 TB/s，为2.45 TB/s峰值的32.69%。取整敏感性上沿约0.818 TB/s、33.38%。按此前包含prefill及节点费用的42.77 tokens/s归一，约为32.17%–32.85%。因此当前单token BF16平均HBM利用率可概括为约33%。16轮profile本身使窗口增加7.59%，利用率分母采用无profiler时间。

此前0.679 TB/s只包含每token15.873 GB主要权重。当前计数补入了KV、FP32状态、临时量和写入。纯HBM传输预算约7.5–7.7ms/token，属于省略计算、依赖与提交的乐观下界。

本机独立GEMV的有效流量吞吐随矩阵大小明显变化，128/512/2048 MiB权重分别为1.642/2.207/2.232 TB/s。小矩阵供数与固定成本、GDN递推和大量recipe之间的依赖共同限制整段速度。Qwen完整attention的Kitchen参考路径还会将K/V转为FP32。其独立费用及全部写入归属仍需要按算子进一步核算。Gaudi2官方HBM峰值口径见[架构说明](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html)。

## 整步编译与持久HPUGraph

从普通生成入口捕获真实单token输入、完整KV/GDN状态及目标logits。保持32层及输出头，状态prepare和RoPE在测量循环外，输入buffer复制计入循环。每次16轮、六组，去掉第一组后取中位。各次重放使用固定位置和既有buffer，因此本表覆盖decoder核心。

| 路径 | 核心ms/次 | 首次费用 | 数值结果 |
|---|---:|---:|---|
| 当前eager区域编译 | 20.871 | 2.417s | logits逐元素一致 |
| eager整步torch.compile | 17.825 | 142.872s | logits RRMSE 0.7883% |
| lazy逐步提交 | 174.388 | 1.087s | logits RRMSE 0.7883% |
| lazy持久HPUGraph | 17.804 | 首次1.148s，capture 0.166s | 两个真实位置RRMSE 0.7883%/0.9312% |

整步编译核心减少14.59%，Graph复核减少14.70%。Graph已验证更新输入、位置、KV/GDN缓冲后重放两个真实位置。GDN已提交state在这些整步试验中逐元素一致，logits仍有变化。该结果把历史逐请求DiT Graph实验推进到实际Qwen decoder长期buffer重放，尚未覆盖完整自然生成、模型patch、正常loader/卸载生命周期及整图净收益。

第一次lazy进程因未显式导入Habana core而走到CUDA探测，修正初始化后两条lazy路径完成。隔离试验将已有区域编译入口换成对应tensor函数，避免lazy内调用`hpu_backend`的旧ABI失败。普通eager进程的HPUGraph限制仍存在。eager整步编译提供同等级局部速度，后续可继续沿该入口解决数值边界及调用期状态所有权。[HPU Graph文档](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html)

## GDN单token融合

复用已有TPC工具链，针对单token实现已提交state更新和当前输出的融合核心。维持FP32 state、现行并行代数以及外部projection/conv/gate/norm协议。核内按四个value列处理128维state，保留pending0/1语义。该候选只用于单token实验。

同一真实step的完整decoder从20.871降到19.092ms，减少8.52%，该step的logits逐元素一致。24层已提交state的最大RRMSE为6.11e−8，存在细小累积舍入差异。

进一步使用既有427-token提示词及固定teacher轨迹，包含prefill，每项一次warm-up、三次无profiler warm及一次完整logits采集。基线中位10.2201s，候选9.5667s，耗时减少6.39%，吞吐提高6.83%。该用例与用户1784-token长扩写分别记录。

| 数值候选 | 完整logits RRMSE | top-1相同/427 | logits逐元素相同的行数 |
|---|---:|---:|---:|
| 原融合核心 | 0.9861% | 421 | 1 |
| 显式调整FMA累加顺序 | 0.9914% | 422 | 1 |
| 上述顺序加显式BF16核输出 | 1.0094% | 419 | 1 |

单step一致与state局部误差不足以证明长生成保值。三个候选的完整logits仍未对齐，最终图片质量也未作为候选完成验收。原融合核心保留在`probes/native_gdn_single`，后两项通过`patches/qwen-gdn-single-*.patch`复现。

## 运行事故与最终服务验收

直接DMA对照的首版误将PyTorch的`data_ptr()`传入Synapse，出现PDMA读写响应错误和驱动自动复位。停止该批次后核验设备恢复。随后尝试在PyTorch进程内直接申请Synapse内存，因现有设备内存池分配失败，最终用独立Synapse进程完成128/512/2048 MiB DMA控制。该控制的HMMU计数为0，确认其路径与本次计算端计数的区别。实验runner已经加入异常退出后停止后续任务的处理。

普通8189服务最终运行`dfca7cd6`，原1088×1920工作流100.598s成功，RGBA与上一轮已验输出逐元素一致，队列为空。原始`/home/sonny/ComfyUI`遵循只读约束，产品源码树干净。

## 证据与复现

- 紧凑指标见[evidence](../evidence/qwen21-bf16-single-token-2026-11-15.json)。服务器原始目录为`performance-qwen-hbm-2026-11-15`、`performance-qwen-decode-2026-11-15`和`performance-qwen-gdn-single-2026-11-15`，均位于`/home/sonny/gaudi2/comfyui-hpu`。
- 计数入口为`qwen_hbm_jobs.py --batch beats`及`--batch confirm`，分别验证512 MiB读写控制和128/2048 MiB、晚位置16轮文本。脚本保存独立profiler配置，`qwen_hbm_summary.py`回算请求侧和返回beat两套字段。单卡实验前确认队列为空并停止普通服务，runner在finally恢复服务。
- `qwen_single_decode.py`负责真实状态捕获及eager/compile/lazy/graph重放，`qwen_single_jobs.py`串行管理进程。`qwen_gdn_single_jobs.py`比较固定teacher完整文本。普通入口恢复由`qwen_hbm_restore.py`验证。
- 本轮后续优先定位GDN归约/舍入和整步编译的数值边界，再评价正常生成入口的收益。用户要求先推进无MTP路径，MTP集成暂缓。
