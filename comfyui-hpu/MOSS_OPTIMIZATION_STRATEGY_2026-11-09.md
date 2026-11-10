# MOSS 后续性能路线评估

日期：2026-11-09（Asia/Shanghai）。基线产品版本 `60da4e33`。本轮仅审查官方资料、产品代码和服务器 checkpoint header；没有停止服务、执行新 HPU 测试或改动产品代码。下文分别标明实测、估算和工程建议。

## 决策

主线继续做 BF16 的 DiT 区域编译/attention 和文本 decode 提交优化，同时设小范围的原生 FP8 Linear 验证分支。优先复用 `fp8_gemm_v2`，必要时只自写它周围的 activation/quant/dequant 融合。暂不立项通用 INT8 GEMM 或从零重写 FP8 GEMM。

BF16 的 shape 管理、状态管理、调度和正确性基线可直接用于 FP8；这些是共同基础。工作流峰值约 44.3 GiB，容量尚未迫使我们换低比特权重；降精度必须体现为更少流量或更快计算。

## 已有事实与未知项

- BF16 完整工作流各 1 cold + 5 warm，优化后 median 60.063 s，TextGenerate 20.156 s，KSampler 37.890 s。按阶段中位数近似分配，DiT 约 63%、扩写约 34%；阶段中位数不要求严格相加。
- BF16 扩写 AIP 平均 15.82%、DiT 48.57%、全请求 36.22%。INT8 的单次诊断扩写约 10.36 token/s、AIP 46.47%，比 BF16 更忙但更慢；两者不是严格同长度 A/B。
- GDN 编译带来的收益已实测。相同输入的 conv/GDN 重放从 6.60/19.13 ms 降至 0.45/0.93 ms，未更换权重精度。
- 当前只给 deferred conv/GDN 增加了 HPU 区域编译；DiT 的 Comfy `malloc_graph` 在 HPU 上不生效。Qwen 主干 Linear/MLP、attention、采样和 Python 控制尚未做同等的整体区域优化。
- HPU attention 入口调用 PyTorch SDPA，没有显式选择 Habana FusedSDPA；最终 bridge 的设备实现仍需 trace 核实，不能仅凭 Python 入口断言未融合。
- 服务器 DiT checkpoint **已经合并 QKV 和 FFN up/gate**，不把再次合并它们当成收益点。
- 旧全模型 FP8 文件实验仍走 `full_precision_mm=True` 解量化路径；旧原生 FP8 head 仅有短 probe，且在 GDN 编译之前。它们不足以否定优化后 FP8 的价值。
- 尚缺新版本的 host/device 时间线、真实 shape GEMM 校准、持续 HBM 带宽、固定 MTP 轨迹对照和完整 FP8 品质校准。

## 硬件上限

公开数据：2 个 MME、24 个 TPC；BF16 MME **432 TFLOP/s**，FP8 MME **865 TFLOP/s**；HBM 96 GB、标称 **2.45–2.46 TB/s**；SRAM 48 MB。[S1][S2]

### 文本 decode：小矩阵与权重流量

对 `[M,K] × [K,N]` 线性层，假设主要流量是每轮从 HBM 读一次权重，忽略输入/输出：BF16 算术强度约 `2MKN/(2KN)=M FLOP/byte`，FP8 约 `2M FLOP/byte`。M=1 或 verify M≤8 时，即使理论 HBM 带宽吃满，weight-only roofline 也仅为 BF16 **2.45–19.6 TFLOP/s**、FP8 **4.9–39.2 TFLOP/s**。两种精度的 compute/bandwidth 平衡点均约 M=176。不能要求单 token 达到大矩阵峰值的 80%，也不能把 AIP 80% 和算力峰值 80% 混为一谈。

只读统计现有 BF16 checkpoint：

| 部件 | bytes | 使用方式 |
|---|---:|---|
| 32 层文本主干 | 13,839,131,136 | 线性层、norm、conv 等静态参数 |
| lm_head | 2,034,237,440 | 每次词表投影的大矩阵 |
| embedding | 2,034,237,440 | 按行查找，不应计作每 token 全量扫描 |
| MTP | 486,581,248 | 草稿额外计算/读取 |
| vision | 912,020,960 | 此次纯文本扩写不执行完整视觉塔 |

主干+lm_head 约 **15.87 GB**。假设完整顺序读一次且无其它成本，BF16 的读权重下界 **6.48 ms**，FP8 约 **3.24 ms**。这不是整模型 token/s 预测：还缺 KV/GDN 状态流量、草稿/验证次数、接受率、prefill 和启动成本；也不能把整个 19.31 GB 文件视为每 token 必须读取的内容。

因此文本 FP8 的主要价值是减少流量。若关键路径仍在 host 提交或状态处理，矩阵本身加速一倍不等于生成加速一倍。

### DiT：大矩阵与 attention

checkpoint 确认为 42 层，d=3072、FFN=8192、24 heads、head_dim=128；QKV `[9216,3072]`，up/gate `[16384,3072]`，down `[3072,8192]`，均 BF16。1024² latent 有 4096 个图像 token，加文本后约数千 token，比单 token 更适合 MME。

只作尺度估算：假设 CFG 每步两份完整前向，两份均 N=4096+303；每层线性 FLOPs 为 `2N(4d²+3df)`，attention 两次矩阵乘为 `4N²d`。42 层×25 步×2 约 **2.59 PFLOP**，除以 BF16 理论峰值得 **约 6.0 s 的矩阵计算下界**。

6 秒不是交付预测：正负条件实际长度、padding/batch、softmax/exp、norm/SiTU、HBM 和运行时成本没有完整计入。真实 shape 尚未采全，不能据此声称精确 MFU。它只支持对当前约 38 秒 DiT 继续调查。如果只把 Linear 换 FP8，attention 或 TPC 操作仍占关键路径，不能套用全流程 2×。

### INT8：不能把 TPC 指令能力当成 MME 吞吐

官方确认 TPC 支持 INT8；MME 数据类型列表和本机实测没有给出可套用的同级 INT8 GEMM 路径。`torch._int_mm` 当前 CPU fallback。[S1][S3] 原硬件记录中的“INT8 与 FP8 同量级”推断已在 `gaudi2-hardware.md` 更正。

自写 INT8 实际有三种路线：TPC 直接做小 M INT8/ConvRot GEMV；TPC 解包后交给 BF16/FP8 MME；一次性离线转换并常驻。前两者必须计入归约、逆旋转、scale、中间张量和同步总成本。第三种是权重准备方案，无需先实现通用 INT8 GEMM。不能 reinterpret ConvRot INT8 字节为 FP8。

## 多维比较

| 路线 | 硬件/算子现状 | 挑战 | 质量与维护成本 | 复用价值 | 优先级 |
|---|---|---|---|---|---|
| BF16 区域编译、执行图、现有 fused ops | 原生 MME；GDN 已证明有效；DiT/主干仍有空白 | 中：shape bucket、graph break、状态/参数 cast、hook/LoRA 边界 | 无重新量化，仍需验证融合舍入和状态语义 | 很高：Qwen、MOSS、其它 DiT，且可用于 FP8 | **主线** |
| 原生 FP8 Linear 适配 | 厂商 `cast_to_fp8_v2`/`fp8_gemm_v2` 已通 | 中高：range、scales、布局、activation quant、缓存失效 | 提示词、MTP、不同 diffusion timestep 校准；敏感层保留高精度 | 很高：通用 Linear 路径，校准按模型进行 | **第二优先，先试点** |
| 定点 TPC 融合：norm/RoPE/SiTU、量化准备、GDN | 适合非 GEMM；编译后是否仍是热点待量化 | 高：reduce、布局、state alias、ABI、设备验证 | 高；状态错误可能延后才表现 | 高；GDN 部分主要限同架构 | **trace 证明后投入** |
| 自写通用 INT8/ConvRot GEMM | 没有已确认可用的同级 MME INT8 路线 | 很高：tiling、packing、归约、多 shape，可能只能用 TPC | 高；ConvRot/W4/W6/codebook 增加组合 | 中：checkpoint 格式绑定，收益不确定 | **暂缓** |
| 从零重写 FP8 GEMM | 已有厂商 MME 原语；TPC kernel 不等于 MME microkernel | 很高，容易重复已有工作且更慢 | 最高，SDK/ABI/shape 维护面大 | 理论高，当前可兑现价值低 | **不立项，做周边适配/融合** |

粗工程预算只作投入上限：BF16 首个热点实验用 2–4 个工程日判断；FP8 shape/质量试点用 3–7 个工程日；可复用 TPC 融合按周评估；通用 INT8 GEMM 是数周以上且可能不产生成品的研究。它们不是交付时间承诺。

## 如何使用 80% 指标

保留“稳定执行阶段 AIP 向 80% 靠近”的目标，但不作为唯一门槛。官方 hl-smi 文档以 1000 ms 活动窗口描述该值；100 ms 轮询不是 100 ms 精度的独立硬件计数。[S4] 本机 HLML 头文件另注明 OAM compute utilization 可大于 100%，memory 字段表示内存占用，须核实接口/版本语义。

INT8 比 BF16 更忙却更慢，说明额外工作也会抬高活动率。建议同时验收：

- 固定 caption/latent 的 steps/s、固定轨迹的 prefill/TPOT/token/s、自然结束用户图时间。
- CPU 等待、launch gap、MME/TPC/DMA 活动区间并集和重叠，不相加嵌套事件 duration。
- 计算密集段与本机真实 shape、相同输出 dtype 的持续 GEMM 基准比较。
- 内存密集段建立逻辑字节模型；有硬件事务计数器再给真实 MBU，不用 `utilization.memory` 替代带宽。
- fallback、显存、文本/图像质量与能耗。`hl-smi -Q power.draw` 只含 54V rail，整卡能耗需再计 12V，或明确只是 54V 估算。[S4]

“匹配 shape 的计算/带宽上限达到 70–80%、host gap 明显缩小、端到端时间下降”可作为工程目标，与粗 AIP 百分比独立。批量/并发可提高利用率，但改变单请求目标，必须另列吞吐实验。

## 执行顺序和转向条件

1. **重新归因。** 用优化后的版本固定最终 caption/latent，拆 DiT Linear、attention、norm/RoPE/SiTU；文本 teacher-forcing 固定轨迹，另测 MTP off/固定树/auto。记录真实 shapes、CPU 等待、device trace；不能沿用旧 GDN 瓶颈占比。
2. **先做 BF16 DiT。** 验证 repeated block 区域编译、shape bucket；既有 QKV/FFN 已融合。按 head_dim=128、真实 mask/长度比较当前 SDPA 最终 backend 与 Habana FusedSDPA operation modes。保留正常 softmax 基线，fast softmax 独立做质量验收。[S5]
3. **扩展文本编译边界。** 从 GDN 扩到匹配 shape 的 Linear/MLP/norm 和解码块；减少控制数据回读。HPU Graph 是后续候选，需要静态 shape/address 和正确处理状态视图，不能直接挪用 CUDA graph 开关。[S6]
4. **FP8 小范围试点。** 与新 BF16 图路径对照，覆盖 M=1/2/4/8 与 prefill/DiT 大 M，统一 BF16 输出，分别计 quant、GEMM、scale/layout 开销。先选高占比 Linear，一次量化常驻；从现有 INT8 转换可以测性能，但不能冒充原始 BF16 校准质量。GDN state/norm/敏感层保持适当精度。[S7]
5. **满足条件才写热点 kernel。** 编译和厂商 op 后，某可融合区域仍占目标阶段约 15% 以上，且 trace 证明是设备计算/重复 HBM 流量，才投入 TPC 融合。若主要是 host gap，继续图执行；若 Linear 成为主耗时且 FP8 在真实形状上有净收益，提高 FP8 优先级。INT8 prototype 必须在包含 unpack/ConvRot/scale 的总耗时上胜过 BF16。

建议本轮投入约 **60% DiT BF16 编译/attention、25% 文本执行与同步、15% native FP8 shape/校准试点**，下一轮按证据调整。

用 60 s≈文本20+DiT38+其它2 做 Amdahl 敏感性分析：

| 假设 | 整图时间 | 降幅 |
|---|---:|---:|
| 只让文本快 2× | 50 s | 约 17% |
| 只让 DiT 快 2× | 41 s | 约 32% |
| 两者都快 2× | 31 s | 约 48% |
| 文本成本降为 0 | 40 s | 只优化文本的理想上界 |

这些是固定工作量假设下的情景，不是预测。沿用单变量、至少五次真实 warm、cold 单列、质量/设备语义通过、收益超过噪音的门槛；自写 kernel 还必须用端到端收益证明维护成本值得。

## 依据

- [S1 Gaudi2 白皮书，第 6–10 页](https://habana.ai/wp-content/uploads/pdf/2021/gaudi2-whitepaper.pdf)
- [S2 Intel Gaudi 硬件对照，第 3 页](https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf)
- [S3 TPC 架构与指令](https://docs.habana.ai/en/latest/TPC/TPC_User_Guide/index.html)
- [S4 hl-smi 指标与功耗 rail](https://docs.habana.ai/en/latest/Management_and_Monitoring/Embedded_System_Tools_Guide/System_Management_Interface_Tool.html)
- [S5 FusedSDPA 与编译约束](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_PyTorch_Models.html)
- [S6 HPU Graph 限制](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html)
- [S7 FP8 原语、校准与 scale](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html)
- [本项目实测](MOSS_GDN_COMPILE_2026-11-09.md)；代码：`ComfyUI-Gaudi/comfy/ldm/moss_image/model.py`、`comfy/ops.py`、`comfy/text_encoders/hpu_gated_delta.py`。
- JChen 快照 `e08710f7` 的 `extension/ops.py` 使用厂商 `fp8_gemm_v2`；`ops/hpu_lm_head_fp8.py` 是权重量化/调用封装。其 vLLM 调度、独立 draft 和性能不能直接外推到本工作流。
