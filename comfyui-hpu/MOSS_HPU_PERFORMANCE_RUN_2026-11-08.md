# MOSS HPU 性能计划恢复与可观测性验证

更新时间：2026-11-09（Asia/Shanghai）

本文只保留本轮性能调查的可复现实验、结果、失败路径和限制。原始大 trace 留在服务器 performance-2026-11-08 目录，不进入 Git。

## 恢复与工具验证

- 服务器恢复后，5.15.0-202 和 5.15.0-186 的 Habana DKMS 均为 installed，设备为 Operational，独立服务版本为 0690d18f，/system_stats 返回 hpu:0 GAUDI2。
- 队列空闲后停止独立服务再运行 probe；BF16 matmul 与 CPU 参考一致。调查期间没有重新下载模型，也没有修改基础 torch、模型或产品推理代码。
- sysstat/linux-tools、HPU torch profiler activity、synprof_parser、Perfetto Python/native trace_processor 和 fallback 计数均完成 smoke。真实 .hltv 转换和 Perfetto SQL 通过。
- TensorBoard Habana 插件和 habana_perf_tool 的高层摘要在短 eager smoke 缺 ProfilerStep/bridge track；真实 MOSS trace 虽有 ProfilerStep，仍因 MMEEvents 不可用失败。该限制保留为工具结论，当前验收路径是原始 HPU trace + Perfetto SQL。

## eager 校准与低精度边界

- 4096³ eager FP8 E4M3/E5M2 torch.mm 原生 HPU、无 fallback，约 525.88/519.63 TFLOP/s；同 shape BF16 约 275.76 TFLOP/s。相对公开 BF16/FP8 理论值 432/865 TFLOP/s，约为 63.8%/60.8%/60.1%，仅是形状和路径特定的微基准。
- INT8 torch._int_mm 约 3.17 TFLOP/s，fallback 103 次；当前 Habana 1.24.2 PyTorch 栈没有可用于该 ConvRot 权重的原生 INT8 linear/mm HPU 路径。Gaudi2 有 INT8 TPC 指令能力，但硬件能力不等于 PyTorch 量化 GEMM 接口。
- E4M3FNUZ/E5M2FNUZ 探针不支持；FP8 custom op schema 存在，但缺 scale/layout contract 时没有硬测。公开 827 TFLOP/s 数据使用不同版本、形状、lazy、scaled_mm、BF16 输出、cache/output reset 和 device event；本机相同外部形状的 eager torch.mm 为 564.11 TFLOP/s，scaled_mm 复核报 synStatus 26，不能直接比较或外推到 MOSS。

## 严格 B0/B1 基线

冻结工作流为 workflows/moss-user-textgen.json，SHA256 为 c283f15bcae32cceebd51c9a8863400f421f46354be32b3c52c4a16743be7bd6；扩散 seed 649029532689846，文本 seed 0。它与用户历史 143.34 s 的 seed 不同，因此本轮不计算相对用户基线的严格提速。

执行控制：

- B0：1 次 process cold + 5 次真实 warm；
- B1：1 次 transition + 5 次真实 warm，首轮 caption 固定；
- 13 个请求均成功，目标计算节点执行断言通过，warm 保留 loader，不重复加载；
- 正常计时无 profiler/逐步同步；另有一次同步诊断请求。

| Case | 样本 | median | min–max |
|---|---:|---:|---:|
| B0 process cold | 1 | 145.521 s | — |
| B0 warm | 5 | 137.282 s | 135.893–137.526 s |
| B1 transition | 1 | 40.305 s | — |
| B1 warm | 5 | 40.183 s | 39.690–40.314 s |
| B1 synchronized diagnostic | 1 | 40.313 s | — |

- B0 warm 节点计时中位数：TextGenerate 96.966 s，文本编码合计 2.013 s，KSampler 37.960 s，VAE 0.132 s，PNG 0.117 s。6 次 B0 都是 327 tokens，token ID 一致，fallback 始终为空，图片语义和 MOSS 灯牌/白大衣/剧院/湿地反光目视通过。
- B1 同步诊断：文本编码 2.034 s、KSampler 38.006 s、VAE 0.136 s、PNG 0.122 s；分段覆盖 99.963%。逐步同步间隔 median 1.5212 s；诊断请求比 B1 warm median 高 0.32%，仅作为扰动参考。
- latent [1,128,64,64] 与 VAE 输出 [1,1024,1024,3] finite；PNG 像素 MAE 为 0–0.1226/255，最大单像素差最高 11/255。AIP/功耗旁采样只作状态证据，不能当 MFU/MBU。
- 首次切换文本 probe 时旧服务尚未完全释放，出现 Device acquire failed；旧 PID 退出、hl-smi Compute Processes 为空后重试成功。该错误是切换次序问题，不是恢复后的驱动故障。

P1 尚欠 B2 模式/长度矩阵、B4 尺寸/步数和跨进程 cold；本轮不宣称 P1 全部完成或已有 P3 收益。

## P2/P3 候选与拒绝

### lm_head 解量化缓存

把 INT8 ConvRot lm_head 一次性解量化为约 2.03 GiB BF16 并复用：

- MTP off、32 tokens 单次样本：16.963 s → 15.920 s，token ID 一致，fallback 为空；
- 目标 MTP auto：10.068 s → 10.622 s，方向相反；
- 额外显存和收益不稳定，候选拒绝，未改产品代码。

### 强制 embedding 驻留 HPU

调查进程中固定 vbar_above_watermark=false，MTP auto 8-token 两次约 4.052 s/3.929 s，差异约 3%，低于短 probe 噪音。未作为产品配置或代码改变。

### 当前 host/device 线索

MTP auto 8-token trace 的主要线索是 aten::_copy_from，约 2.14 s/profile；ConvRot dequant 约 0.21 s，fallback 为空。没有新的 device/host 调用归因前，不缓存权重、不强行改变 embedding 驻留、不切 lazy。

## dtype、INT8 与低 HPU 利用率复核

日志中的 dtype: torch.bfloat16 是计算 dtype；权重仍是 QuantizedTensor。当前 full_precision_mm=true 会逐层解量化到 BF16，再在 HPU 执行 F.linear。

独立探针强制 full_precision_mm=false 后，2-token 从约 1.47 s 变为 12.54 s，调用栈进入 fast_int8_mm → torch._int_mm，并发生 CPU fallback。因此当前栈不能把 native INT8 开关作为优化。

同一 eager、MTP auto、短 prompt 的 32-token 旁采样为 10.10 s，26 个 AIP 样本平均 12.65%、最高 47%。低负载与小批量自回归、.tolist() 控制同步、重复解量化和临时张量搬运一致；不能解释为 HPU 没有加载。

## 外部方案审查

审查 JChen-318/gaudi2-single-card-vllm 后确认：

- native/ 主要是 GDN、拒绝采样、TPC shim 和 NVFP4 MoE，没有通用 INT8 GEMM、INT8 Linear 或 ConvRot 解码 kernel；
- hpu_lm_head_fp8.py 面向 BF16 lm_head → FP8，不是已有 INT8 ConvRot 的直接替换；模型 shape、校准、token/top-1 和 fallback 都需独立验证；
- 该仓库 SDK/torch/ABI/权重布局与本机不同，不能直接替换生产 GC_KERNEL_PATH。

它只能提供 FP8 vocabulary projection、speculative decode、lookahead 和 host/device overlap 的设计参考。

## FP8 文本编码器 A/B

下载 BF16 文件时，已有 HF_ENDPOINT/代理下 Xet 单块长时间重试；未更改端点，停止下载并清理临时文件。随后从现有 INT8 权重在 HPU 解量化生成独立 FP8 E4M3FN 文件，保持 MOSS key layout 和量化元数据。

- 全模型 FP8：561 个量化模块可加载 HPU，但 MTP auto 2-token warmup 16.23 s，慢于原 INT8→BF16 的约 1.47–1.87 s。
- 仅 lm_head FP8：2-token warmup 4.71 s，仍慢于原路径；32-token 在第 17–27 token 间约 1.1–1.3 s/token，未继续跑完。
- 全局 full_precision_mm=false：主体 INT8 层进入 CPU fallback，2-token 约 17.42 s。
- 初版误量化 167 个 1-D norm weight，触发 HPU add_stub 错误；改为保留 norm BF16 后重新加载通过。

结论：当前 Habana 1.24.2 / torch 2.13.0a0 / comfy-kitchen 组合下，自制 FP8 文本编码器没有性能收益，反而变慢；文件未替换生产模型。

## 真实 HPU trace

- 真实用户 prompt、文本 seed 0、MTP auto、sampling on：2-token warmup 后，8-token 无 profiler 4.1018 s，profile 内 5.2150 s（含 prefill），token ID 相同，fallback {}。
- Perfetto SQL 读到 414,226 个 cpu_op、1,030,554 个 kernel 类事件和 ProfilerStep#0 5.216 s；事件存在多 engine/lane、嵌套和重叠，不等于一百万次 kernel launch，duration 不能直接相加。
- kernel 标签包含 memcpy_f32、mult_fwd_f32、cast_i8_to_f32、cast_f32_to_bf16 等，只能作为 ConvRot 解码、prefill/GDN 和 host gap 的线索，不能凭名称认定 PCIe copy 或瓶颈。
- 后续 trace 应拆分 prefill/稳态 decode、缩短范围并默认关闭 shape 记录，先完成事件去重、区间并集和 host/device 关联。

## 当前状态

调查服务和旁采样已停止，日常 ComfyUI-Gaudi 服务恢复，队列为空，/system_stats 为 hpu:0 GAUDI2，部署版本为 0690d18f。下一步是完成 B2/B4 和 _copy_from 调用归因；在此之前不提交猜测性的性能改动。

## 参考

- Gaudi PyTorch profiling：https://docs.habana.ai/en/latest/Profiling/Profiling_with_PyTorch.html
- Gaudi profiling workflow：https://docs.habana.ai/en/latest/Profiling/Profiling_Workflow.html
- Perfetto Python API：https://perfetto.dev/docs/analysis/trace-processor-python
