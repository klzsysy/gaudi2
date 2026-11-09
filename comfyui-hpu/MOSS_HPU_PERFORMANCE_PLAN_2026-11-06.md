# MOSS-Image HPU 性能优化计划

更新时间：2026-11-09（Asia/Shanghai）

本文定义单卡 Gaudi2 上 MOSS-Image 的性能调查方法和验收门槛。执行结果见 MOSS_HPU_PERFORMANCE_RUN_2026-11-08.md；产品代码、测试和环境记录分别留在各自仓库/文档中。

## 1. 目标与边界

目标工作流是用户提供的 MOSS-Image 文生图图：1024×1024、batch=1、25 步、er_sde/simple、cfg2.5，文本扩写开启。用户历史基线为 143.34 s，另有 134.586 s 的输出头修复重放；两者 seed、服务生命周期和缓存状态不同，不能直接计算收益。

首轮只覆盖：

- 单卡 Gaudi2、eager、batch=1、MOSS-Image SFT bf16 DiT；
- Qwen3.5 9B INT8 ConvRot 文本模型和已有 Flux2 VAE；
- TextGenerate、文本编码、DiT 采样、VAE 解码和保存的分段归因；
- 正确性、fallback、设备活动、阶段耗时和重复实验。

不把 MOSS Base、编辑/控制 LoRA、SLG、并发、多卡、其它量化版本、lazy/compile/HPU Graph 或跨设备比较纳入本轮结论。

## 2. 测量规则

- 相同 /prompt 可能命中节点缓存；warm 测试必须改变计算节点 fingerprint、保留 loader 结果，并断言目标节点确实执行。全图重算另列控制组。
- 每组至少 1 次进程 cold + 5 次真实 warm；cold、模型加载、编译和页缓存状态单独标记。5 次分位数只作描述，不作尾延迟置信估计。
- 同时冻结 KSampler seed 和 TextGenerate seed，并保存 API JSON、SHA256、实际 token ID、caption、history 和 PNG；不同请求的 seed 不能混算收益。
- 正常结果使用无 profiler wall time；同步分段、逐步同步、shape/memory profile 和硬件 trace 必须标为诊断请求，并报告扰动。
- 静态逻辑 FLOPs/字节与真实 HBM 事务分开；utilization.memory 不可积分成带宽，CPU copy 等待也不能直接解释为 PCIe 搬运。
- 低精度能力探针只说明算子可用性，不改变生产 dtype/量化策略。实测微基准与公开理论值同时记录，不能把未经校准的 MFU/MBU 当结论。
- fallback 必须用 Habana 计数器和 CPU profile 交叉检查；图像需 finite、可读、无异常纹理，文本需可读且结束条件正常。

## 3. 基线矩阵

### B0：用户端到端

固定用户图、1024²、25 步、er_sde/simple、cfg2.5、已有三个模型、prompt enhancer mtp=auto/sampling on、temperature 0.7、top_k 64、top_p 0.95、min_p 0.05、max_length 4096、thinking off。记录端到端、阶段耗时、输入/输出 token、token ID、fallback、显存、功耗、AIP、图片质量和版本。

### B1：无扩写图像路径

关闭 TextGenerate，使用同一最终英文 caption；保持 DiT/VAE、seed、尺寸和步数完全相同。B1 用来分离扩写成本，不能作为 B0 提速结论。

### B2：文本单独

固定 system/user prompt、采样参数和文本 seed，覆盖：

- mtp=off + greedy；
- mtp=off + sampling；
- mtp=auto + sampling；
- 固定短 token 数和自然结束序列。

记录输入/输出 token、token/s、MTP 草稿/接受/验证（若原生计数可得）、fallback 和 host/device 同步。

### B3：DiT/VAE 分段

固定最终 caption，绕过 TextGenerate，分别测文本编码、25 步 KSampler、VAE 和保存。KSampler 至少保留首步、稳态步、末步，并在异步边界显式同步。

### B4：尺寸、步数和缓存控制

在 B1 的固定 caption/seed 下建立：

| Case | 尺寸 | 步数 | 目的 |
|---|---:|---:|---|
| S512-25 | 512² | 25 | 分离固定开销与空间规模 |
| S1024-25 | 1024² | 25 | 用户目标图像 |
| S1024-8 | 1024² | 8 | 估算固定步开销 |
| S1024-15 | 1024² | 15 | 检查步数线性度 |

每个 case 区分进程 cold 和同进程 warm；不能把加载或首次编译混入 warm 单步性能。

## 4. 可观测性契约

### 硬件校准

独立空卡 probe 只运行一次并版本化保存：

- HBM 读/写/copy 的有效带宽，工作集大于 48 MiB SRAM；
- BF16/FP8/INT8 大矩阵、接近模型形状的矩阵和小 kernel 启动下限；
- hl-smi 的 AIP、HBM、功耗、温度、时钟 CSV；
- 驱动、固件、内核、Habana torch 和 PT_HPU_LAZY_MODE。

Gaudi2 公开参考值为 BF16 432 TFLOP/s、FP8 865 TFLOP/s、HBM 2.45 TB/s；没有可直接套用的 INT8 MME 峰值。roofline 只在 FLOPs、逻辑字节和测量定义明确后计算。

### 静态模型属性

保存 checkpoint 文件大小、dtype、量化布局、层数、词表大小；DiT 按 512²/1024² 计算 latent token、逻辑 FLOPs 和权重字节；文本模型记录每 token 权重字节、lm_head [4096,248320]、MTP 计算/字节；VAE 记录输入输出 shape 和 dtype。静态估算不能冒充硬件计数器。

### 运行时交叉验证

| 层次 | 记录 | 交叉验证 |
|---|---|---|
| 请求 | queue、load、text、DiT、VAE、save、端到端 | history、服务日志、客户端 monotonic |
| TextGenerate | token、token/s、MTP 计数、采样参数 | 原生计数与固定 token probe |
| HPU | AIP/HBM、显存、功耗、温度、时钟 | hl-smi、/system_stats、阶段边界 |
| 算子 | fallback、CPU 等待、kernel/设备时间 | Habana debug、CPU profiler、HPU trace |
| 主机 | CPU、RAM、I/O、线程 | /proc、sysstat、服务日志 |

### Gaudi2 与 MOSS 的归因假设

- MME 主要承载线性层、卷积和大矩阵；TPC/DMA 可能承载 attention、norm、激活、索引、量化/反量化和胶水操作。
- Qwen3.5 的重点是每 token 的 lm_head 解量化、copy、标量回读、MTP 控制和小矩阵启动。
- DiT 的重点是 25 步大矩阵、HBM 读写、TPC 融合和步间空隙。
- VAE 的重点是 layout metadata、残差合并、shape 特化和固定 launch 开销。
- 这些是待 trace 验证的假设；kernel 名称不能单独证明 PCIe 搬运、瓶颈或硬件单元归属。

### 工具状态

hl-smi、HPU torch.profiler activity、Habana fallback 计数、synprof_parser 和 Perfetto SQL 已有实机验证；habana_perf_tool/TensorBoard 高层摘要在真实 MOSS trace 上受 MMEEvents 兼容性限制。原始 .hltv/HPU trace 保留在服务器，不把大型 trace 放进 Git；高层摘要失败本身是工具能力限制，不通过切 lazy 或替换厂商包掩盖。

## 5. 归因与优化顺序

1. 先完成 B2/B4，确认扩写、DiT、VAE 的固定开销和重复实验噪音。
2. 用无 profiler 的阶段计时确认瓶颈，再用短 trace 关联 host API、device event、fallback 和 HPU 状态。
3. 只做一个变量：先验证输出头/解量化/embedding 驻留，再考虑批量 speculative verify、缓存、融合 kernel 或 Graph。
4. 任何优化必须同时复验 token ID/文本、图像 finite 与语义、fallback、显存、功耗、端到端和回滚路径。

## 6. 验收门槛

- 同一启动至少 5 次 warm，收益超过噪音且方向一致；一次最好成绩不算通过。
- 图像无 NaN/Inf、黑图或异常纹理；固定输入保留像素统计和目视检查。
- 文本可读、自然结束；固定 token probe 保存 ID 和 fallback。
- 声称 HPU 加速的目标 case 必须 fallback 为零，或明确量化仍存在的 CPU 路径。
- 报告同时给端到端秒数、阶段秒数、steps/s 或 token/s、显存、功耗、HPU 状态、质量和回滚信息。
- 阶段出口：P0 资料/环境冻结；P1 重复基线；P2 设备/主机归因；P3 单变量优化；P4 将通过/失败路径沉淀为短结论。

## 7. 当前禁止事项

- 不修改原始 /home/sonny/ComfyUI。
- 不在用户服务持卡时启动独立 HPU probe、微基准或 profiler。
- 不先切 lazy、全模型 BF16 常驻、盲目缓存权重或替换量化数学。
- 不以单点 hl-smi 利用率、进度条、profiler 总耗时或一次最好成绩推断硬件上限。
- 不记录 HF token、代理凭据；已有模型组件尤其是 VAE 不重复下载。
