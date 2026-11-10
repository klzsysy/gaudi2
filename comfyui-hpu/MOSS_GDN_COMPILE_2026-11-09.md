# MOSS GDN 区域编译验证

日期：2026-11-09（Asia/Shanghai）。本轮只针对现有 Kitchen deferred conv/GDN 做 HPU 区域编译；不替换依赖、不改变采样或 MTP 状态管理。原始证据保留在服务器 `/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-09-gdn`。

## 已完成的证据

- 原路径在 HPU 上调用 Kitchen 的 torch 参考实现。纯 tensor 区域可由 `torch.compile(backend="hpu_backend", fullgraph=True, dynamic=False)` 编译，包含原地状态/缓冲写入；正常采样控制和模型加载留在区域外。
- 固定真实用户 prompt、文本 seed 0、普通 BF16 文本模型、MTP auto、32-token 上限：各自 1 次同长度 warmup + 5 次 warm；计时没有 profiler/逐算子同步，旁路以 100 ms 采样 `hl-smi`。原路径 median **10.9695 s**（10.4358–11.0499），编译路径 **3.7273 s**（3.2731–3.8272）。每组内部输出 ID 一致，fallback 均 `{}`。跨路径从第 28 个 token 不同，不能当同一 MTP 接受轨迹的严格加速倍数。
- 32-token warm 窗口旁采样（去除边界不完整秒）：原路径 525 个样本，AIP 均值 **12.98%**、最高 39%，零值占 16.95%，平均功耗 **96.11 W**；编译路径 179 个样本，AIP 均值 **16.52%**、最高 41%，零值占 13.97%，平均功耗 **115.54 W**。这些是粗粒度状态采样，不是 MME 利用率/MFU 或 HBM 带宽计数。
- 8-token 同步诊断：72 次 deferred conv 共 **0.4283 s**，72 次 GDN 共 **1.4018 s**，总耗时 **4.3283 s**（包含 prefill、采集首组输入和同步扰动）。不能把诊断占比直接外推到正常请求。
- 捕获真实层输入/状态后分别重放，10 次同步调用中位数：conv **6.6012 ms → 0.4487 ms**，GDN **19.1253 ms → 0.9331 ms**。这是相同输入和初始状态的算子对照，fallback 为零。

## 正确性和边界

- 捕获输入下，卷积 committed state 和 projection buffer 逐值相同；QKV buffer 最大绝对差 **0.03125**。GDN 输出最大绝对差 **0.001953125**、平均绝对差约 **8.58e-6**，该次 committed recurrent state 逐值相同，gate buffer 最大差约 **5.96e-8**。
- 固定原路径的 32 个 token 作为 teacher-forcing 输入，禁用 MTP、逐步比较同一前缀的 logits：全部 finite，top-1 **31/32 一致**，平均 KL(reference || compiled) **0.0003614**；最大 logit 差 0.34375、平均差 0.02182。因此本改动不提供 bitwise/token-ID 一致承诺。误差量级与图编译舍入/融合差异一致，但短样本不能代替系统质量评估。
- 分别只编译 GDN 或只编译 conv 的 teacher-forcing 检查：top-1 分别 31/32、32/32，平均 KL 分别 0.0004233、0.0002226；两者都可能改变采样。无证据表明有状态索引错误。
- 产品回归 **59 passed, 1 skipped**：包括 HPU batch=1/2、1/3/8 行，交替缓冲、零/部分提交、sibling 路径和状态多次复用；CPU 路径不触发 HPU 编译；已有采样和 Qwen3.5 decode 测试通过。跳过项要求 CUDA DynamicVRAM。
- 首次碰到 shape 有编译成本，必须单列 cold。现阶段不覆盖所有 Qwen3.5 型号、所有 dtype、训练或长上下文质量。

## 产品改动

已提交到本地旁边的 `ComfyUI-Gaudi`，提交 **`60da4e335e1ec0be2789b0ed1896df067b5c7219`**（`codex/gaudi2-hpu`，未 push）。`comfy/text_encoders/hpu_gated_delta.py` 延迟创建并共享两个编译函数；`qwen35.py` 仅在 HPU decode 且未设置 `--disable-comfy-compiler` 时选择它们。继续使用 Kitchen 的公开操作及 Comfy 的参数 cast/offload 上下文，不保存模型参数或 KV cache。CPU/CUDA 等继续原路径；现有 `--disable-comfy-compiler` 可关闭新增编译。服务器同名文件与本地 SHA256 一致。

## 完整工作流对照

两组都固定 BF16 文件、prompt、文本 seed=0、扩散 seed=649029532689846、1024²/25 步；每组进程 cold 一次和 5 次 warm，强制计算节点执行但保留 loader。所有请求成功，fallback 始终为零。

| 指标 | 原始 eager | GDN/conv 区域编译 |
|---|---:|---:|
| process cold | 140.726 s | 71.979 s |
| 5 次 warm median | **131.031 s** | **60.063 s** |
| warm min–max | 128.849–132.405 s | 58.370–60.315 s |
| TextGenerate median（含 prefill） | **96.740 s** | **20.156 s** |
| warm 输出 token 数 | 375（5 次） | 303（4 次）、333（1 次） |
| median token/s（含 prefill） | **3.876** | **15.033** |
| KSampler median | 32.262 s | 37.890 s |
| TextGenerate 平均 AIP（各 warm 均值再平均） | 12.44% | 15.82% |
| TextGenerate 平均功耗 | 92.00 W | 110.67 W |
| HPU allocator 全流程峰值 | 44.67 GiB | 44.33 GiB |

此场景端到端延迟下降 **54.16%**（约 2.18×）；自然结束 caption 长度/内容和 DiT 文本输入不同，所以不把所有差异归结为等价工作量下的 kernel 提速。token/s 与相同输入算子重放提供独立支持。

编译组 cold 与前四次 warm 为同一 303-token caption，第五次 warm 为 333 tokens、从第 148 个 token 不同。原始组 cold 357 tokens，五次 warm 为同一 375-token caption。不能宣称跨路径或六次全长 ID 一致。auto MTP 依据运行时间调整树形，编译舍入也会影响 logits；两种因素尚未进一步拆分。编译输出的两种 caption 及对应图片均目视保留 MOSS 灯牌、白大衣、夜景、湿地反光，没有异常纹理。

## INT8 兼容性和最终状态

- 用户原来的 `qwen3.5_9b_int8_convrot.safetensors` 也完成了两次完整 1024²/25-step 工作流：cold **81.512 s**；随后带逐节点/步同步与浮点检查的诊断请求 **70.361 s**。均为 363 tokens、fallback `{}`。这是兼容性/质量 smoke，不是五次 warm 性能基准，诊断请求不计入正常耗时统计。
- 诊断 latent `[1,128,64,64]` 全部 finite，min/max **−6.7640/7.2419**；VAE 浮点输出 `[1,1024,1024,3]` 全部 finite，范围 **0–1**。caption 可读且自然结束，图片目视正确，无黑图或异常纹理。
- 状态测试在补充合法 sibling 执行顺序、同 shape 更换权重后，专项复验 **5 passed**；连同先前 59 passed/1 skipped，覆盖了实际产品代码和更新后的测试。未用同一组重复测试膨胀通过数量。
- 调查服务已退出，日常服务恢复至 `http://192.168.10.21:8189`，部署标记更新为 `60da4e33`；`/queue` 为空、`/system_stats` 为 `hpu:0 GAUDI2`。未修改原始 `/home/sonny/ComfyUI`，未更改用户默认模型选择，现有 INT8 工作流直接获得区域编译路径。

## BF16/INT8 速度与 AIP 补充汇总

由保存的请求/节点时间窗关联 100 ms `hl-smi` CSV，去除边界不完整秒。BF16 为五次正常 warm，INT8 仅一次带同步检查的诊断请求，不能作为严格同条件 A/B：

| 指标 | BF16（5 warm） | INT8（1 diagnostic） |
|---|---:|---:|
| 整图耗时 | median 60.063 s | 70.361 s |
| 扩写 token/s（含 prefill） | median 15.033 | 10.361 |
| 扩写平均 AIP | 各请求均值平均 15.82% | 46.47% |
| KSampler 平均 AIP（样本合并） | 48.57% | 51.35% |
| 全请求平均 AIP（样本合并） | 36.22%（2905 样本） | 48.18%（684 样本） |
| 全请求最高 AIP | 59% | 59% |

INT8 cold 的扩写平均 AIP 42.09%、最高 58%，说明其较高活动率也出现在未开启逐节点同步的 cold 请求。INT8 仍有运行时解量化，因此活动率较高不等于更高的有效生成吞吐；各开销占比尚未量化。本次问答时日常服务队列为空、瞬时 AIP 为 0%、HBM 使用 773 MiB，不能与上述执行期均值混用。

## 复现文件

- `probes/moss_gdn_experiment.py`：短文本重复测试、teacher forcing、GDN 输入捕获、利用率旁采样。
- `probes/gdn_replay.py`：捕获的相同输入逐 tensor 对照及算子计时。
- `probes/run_gdn_workflows.py`：串行启动原始/编译调查服务，执行 cold + 5 warm。
- `probes/moss_observed_server.py`、`probes/moss_baseline.py`：禁用计算节点缓存、保存事件、请求、caption 和图片。
