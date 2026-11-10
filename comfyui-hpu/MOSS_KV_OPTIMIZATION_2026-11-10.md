# MOSS KV attention 与后续 decode 优化

日期：2026-11-10（Asia/Shanghai）；起始产品 `05d2a0a5`，调查仓库起点 `1a12bf6`。只使用已有 BF16/FP32 torch/Kitchen 操作，不自写 kernel。实验目录 `/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-10-kv`。开始时 8189 服务已停止、HPU 空闲，原始 `/home/sonny/ComfyUI` 不修改。

## 已核实的输入与候选

真实 Qwen3.5 9B 主干 verify Q 为 `[1,16,8,256]`、KV 为 `[1,4,6073,256]`，首次 prefix index=1970；MTP draft Q 为 `[1,16,1,256]`，cache 同样分配 6073、首次 index=0。后者的有效缓存更短。短 token probe 显式保留 6073 容量以匹配完整工作流，未通过调低 max_length 缩小容量来制造收益。

原生 GQA/tree merge availability 在 HPU 上为 false，运行 torch FP32 参考路径。候选依次为仅裁剪前缀、仅编译原操作、编译并按 256-token 桶裁剪前缀、现有 SDPA 替换；全部在调查进程中比较，没有修改 Kitchen 安装包。

第一组真实 verify 输入重放（5 组，每组 20 次，HPU 组尾同步）：

| 候选 | median ms/call | 对原结果相对 RMS |
|---|---:|---:|
| 原路径 | 2.7891 | 0 |
| 仅前缀裁剪 | 3.2914 | 0 |
| 原操作区域编译、完整容量 | 0.1541 | 0 |
| 区域编译、前缀分桶 | 0.0869 | 0 |
| SDPA、前缀分桶 | 1.8273 | 0.2648% |
| 编译 SDPA、前缀分桶 | 0.2046 | 0.2648% |

全部 finite、fallback 为空。CPU FP64 attention 参考误差：原路径 0.2235%、SDPA 0.1651%；SDPA 虽更接近这组数学参考，但编译原操作更快且保留现有行为，因此未选择 SDPA 替换。第一份 draft capture 的有效长度只有 1，输出必然等于对应 V，不能把它的零误差外推到所有 draft 状态。

## 文本初步对照

64-token、MTP auto、6073 cache、同一提示词，各 1 cold+5 warm：原路径 4.6990 s；仅编译 3.4096 s；编译+分桶 3.7272 s。随后同一模型进程轮流运行三种模式，分别预热一次再各五次：原路径 4.5411 s；仅编译 3.9309 s；编译+分桶 3.8764 s。

各模式组内 token ID 一致，编译两组之间一致；原路径与编译路径在第 61 个 token 开始分歧。此前 60 个 ID 相同，但这仍不是完全相同解码轨迹。分桶比完整容量仅约 1.4% 的差异，小于已观测波动，且第一组更慢；暂不将其加入产品。产品候选只编译既有完整容量的 decode/tree merge，不改变 cache 状态/长度/生成上限。

## 归因与失败记录

- 已保存 CPU+HPU 原始 trace，8-token、6073 cache、warm 后采集。CPU 汇总的 `run`/`launch` 占 self CPU 时间前两位，原始 trace 约 847 MiB；采集/导出显著放大总时间（约 110.6 s），不作为性能成绩，也不把嵌套事件耗时相加。
- trace 导出阶段仍持有设备，一次过早启动的文本对照进程报 `Device acquire failed`，未产生有效性能数据。保留在 `failed-overlap-*`；确认 profiler 进程完全退出后重跑成功。后续 HPU 任务等待进程 exit 后再启动。

## 回归与原有状态问题

重新测量未改产品的完整工作流基线：1 cold+5 warm，median **36.1551 s**，范围 35.2456–39.3313 s。与之前的 36.648 s 接近。

首次文本回归为 85 passed、1 skipped、1 failed。失败在新增的 HPU tree commit 验证：同一 KV storage 上 `cache[:, :, dst] = cache[:, :, src]` 在多次变更 offset 后可能保持目标行不变。关闭新增 KV 编译也复现，独立 probe 中原路径最大绝对差 4.09375，源输入/write 阶段均无误。这是原有 HPU 状态搬移问题。

两个候选修复——先 clone 源行、以及 index_select/index_copy——均在 eager/compiled 下得到六次零误差。选择改动更小的 HPU 源行 clone；CPU/CUDA 保留原复制方式，没有跨请求 tensor cache。回归扩大到实际 4 KV heads、head_dim=256，以及 64 维、小容量、两种编译状态；KV 专项 **16 passed**。接下来需用同样已修复的 cache 状态做编译开关对照，旧的自然生成时间不作为严格同轨迹证明。

## 扩大 decode 区域的比较

在已修复 tree commit、已编译 KV 操作的基础上，比较完整 GatedAttention、完整 GatedDeltaNet、二者同时编译。64-token、1 cold+3 warm：KV-only 3.9230 s、attention 2.9602 s、delta 3.5788 s、both 3.2312 s。attention/delta 与 KV-only 的 token ID 分别从第 46/61 个分歧，both 在这 64 个 ID 上相同；因此另做固定 teacher 轨迹。

固定 32-token、MTP off、6073 cache、1 cold+5 warm，两组/各候选都包含每步 logits 的 CPU 拷贝：

| 路径 | median | top-1 对照 | 平均 KL 对原路径 |
|---|---:|---:|---:|
| 修复 cache 后的原 attention | 4.1691 s | 参考 | 0 |
| 仅 KV 编译 | 3.6954 s | 32/32 | 0（logits 逐值一致） |
| KV + 完整 attention decode | 3.2089 s | 32/32 | 0.0002996 |
| KV + 完整 delta decode | 3.6752 s | 31/32 | 0.0003345 |
| KV + 两者 | 2.9427 s | 32/32 | 0.0002089 |

delta 在自然生成中没有稳定叠加收益，暂不合入。产品增加 GatedAttention 的普通 operations、BF16 HPU、短步 FixedKVCache、默认 attention 路径编译；prefill/CPU/FP32/训练/自定义 operations/自定义 attention/关闭编译走原路径。检查 root/child/global forward hooks，保留输入、cache 与权重更换语义。最终专项 **33 passed**，覆盖两种 RoPE 表达、batch 1/2、变长、更新权重、hooks、分支回滚和 CPU 等兼容路径。

## 完整产品工作流

均为 1 cold+5 warm，普通 BF16、原始用户提示词、MTP auto、1024²/25 步：

| 路径 | 整图 median | TextGenerate median | KSampler median | warm 输出 tokens |
|---|---:|---:|---:|---|
| 本轮原产品 | 36.1551 s | 14.5493 s | 19.3918 s | 362,277,277,277,362 |
| 仅修复 cache、关闭新增编译的控制组 | 38.6987 s | 17.2748 s | 19.3432 s | 362,362,362,362,277 |
| 最终 attention 产品候选 | **29.9251 s** | **8.3982 s** | 19.6156 s | 305,305,305,305,305 |

最终五次为 `[29.9251,30.2526,29.0930,29.6668,30.3195]`，cold 47.913 s，fallback 全为空。相对此前 36.648 s 约减少 18.34%；自然结束长度与轨迹不同，不作为同轨迹 A/B。严格固定输入结果见 teacher 表。

最终阶段 AIP 均值：整图 **45.22%**、文本 **22.20%**、DiT **57.04%**，峰值 67%；54V rail 平均功耗约 284.03/146.45/355.01 W。仍未达到 80%。实际 median 请求的阶段耗时可相加：文本 8.165403 s、正负编码 1.823840 s、DiT/采样 19.664646 s、VAE 0.132974 s、图片输出 0.119223 s、其它 0.018992 s。

512 auto/768 MTP off 两个变长提示词工作流通过诊断。验收脚本调用 `/free` 时误把空响应按 JSON 解析，中断于重载前；修复调查客户端后只继续未完成的重载/INT8 项，不将中断算作通过。最终独立检查确认全部五项 latent/VAE finite、fallback 为空。BF16 重载诊断 39.093 s/305 tokens，INT8 两次诊断 62.152/52.779 s、均 327 tokens；这些包含同步/finite 检查，不作为性能基准。复核了 BF16/INT8 MOSS 街景、红茶壶和蓝椅上的猫，图像正常；不宣称相同 seed 的画面细节一致或完成了大规模质量评估。

正式产品代码重新做固定 teacher 对照（无 prototype monkeypatch，控制组仅关闭两处新增编译，cache 修复保留）：各 1 cold+5 warm，**4.312471 → 3.002085 s，减少 30.39%**。输出 logits finite，top-1 32/32、平均 KL 0.000299556、最大绝对差 0.3828125、平均绝对差 0.0218026；仍包含每步 logits CPU 拷贝，属于固定工作量诊断。文本模型单独卸载重载，8-token ID 一致，卸载后 allocator InUse **98,549,888 bytes（93.9845 MiB）**。最终 Qwen decode 回归 **42 passed, 1 skipped**；与 33 项专项分开报告，先前失败/重复运行不累加。

用户提供的长上下文 attention 建议另见 [逐项评估](ATTENTION_IDEAS_REVIEW_2026-11-10.md)。已补测厂商 BF16/FP8 FusedSDPA；FP8 未证明优于 BF16，未进入本次产品。

## 交付

产品提交 **`3fec1143298004cc067f0dc37eb6e626e269c4ca`**，本地分支 `codex/gaudi2-hpu`，未 push。三个产品代码文件和测试文件与服务器 SHA256 一致；8189 日常服务已恢复，API 队列为空。调查脚本、失败记录和服务器信息只在本调查仓库。原始 `/home/sonny/ComfyUI` 未修改，没有下载模型。

当前单请求中 DiT/采样约占 65.7%，文本约 27.3%；下一轮优先重新分析 DiT 的剩余执行开销，文本侧仍有 prefill/GDN/采样调度空间。共享 mask、BF16 FusedSDPA 和前缀复用只保留为待完整验证的候选；当前结果不代表 BF16 已逼近硬件上限。
