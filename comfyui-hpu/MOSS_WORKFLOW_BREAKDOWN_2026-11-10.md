# MOSS 36.648 秒工作流拆解与下一步

日期：2026-11-10（Asia/Shanghai）。产品版本 `05d2a0a5`，调查记录此前已提交 `b6210ca`。本次只复算 11 月 9 日的原始事件并审查当前代码，没有重跑 HPU 性能实验、停止服务或修改产品。

## 计时口径

普通 BF16 Qwen3.5 9B、MTP auto、1024²、25 步 er_sde/simple、CFG 2.5；强制计算节点重算、保留 loader。五次 warm 整图为 39.1478、35.8548、38.9186、36.6480、35.2302 秒，median **36.647989080 秒**。首次 cold 为 55.2135 秒，单独列出，不包括在 warm 指标中。

选用实际处于中位数的 **b0-warm-4**，prompt ID `a71ae6dc-82a0-4e6f-a991-5983133cc4e7`，输出 277 tokens。下表各行属于同一次请求，可以严格相加；之前汇报的 DiT 19.3579 秒是该阶段独立的五次中位数，不是这次 36.648 秒请求的 DiT 耗时。

| 阶段 | 节点 | 本次耗时 | 占整图 |
|---|---|---:|---:|
| 文本扩写 | TextGenerate / 18:11 | 14.949898 s | 40.79% |
| 负提示词编码 | CLIPTextEncode / 18:5 | 0.796744 s | 2.17% |
| 正提示词编码 | CLIPTextEncode / 18:4 | 0.957262 s | 2.61% |
| DiT 去噪与采样 | KSampler / 18:7 | 19.666598 s | 53.66% |
| VAE 解码 | VAEDecode / 18:8 | 0.132846 s | 0.36% |
| 图片编码/写入预览文件 | PreviewImage / 26 | 0.123620 s | 0.34% |
| Switch、文本预览及节点间调度 | 其余 | 0.021022 s | 0.06% |
| **合计** | | **36.647989 s** | **100%** |

TextGenerate 内部 `CLIP.generate` 区间为 14.942868 秒，节点其余约 7 ms。其它行使用节点起止事件。全部节点区间无重叠；剩余节点间时间为 0.020557 秒，加上 Switch/PreviewAny 为 0.021022 秒。

这属于服务器执行请求的墙钟时间。非诊断模式没有逐节点 HPU synchronize，因此不能把表格当作纯设备 kernel 时间；异步工作的尾部等待可能跨节点归属。KSampler 包含 DiT、采样器控制和回调；PreviewImage 不包含用户浏览器显示图片的耗时。客户端轮询观察到完成为 37.0104 秒，其额外延迟不算入 36.648 秒。

| warm 请求 | 输出 tokens | TextGenerate 节点 | 正负编码合计 | KSampler | 整图 |
|---|---:|---:|---:|---:|---:|
| 1 | 362 | 17.7419 | 1.8305 | 19.3081 | 39.1478 |
| 2 | 267 | 14.1037 | 1.7945 | 19.6838 | 35.8548 |
| 3 | 362 | 17.7543 | 1.6442 | 19.2526 | 38.9186 |
| 4 | 277 | 14.9499 | 1.7540 | 19.6666 | 36.6480 |
| 5 | 277 | 13.9764 | 1.6208 | 19.3579 | 35.2302 |

全部请求 fallback 为空。既有五次 warm、按阶段筛选的 AIP 采样均值：整图 37.93%、文本 16.79%、DiT 57.23%。这些不是上表单次请求的独立利用率，也不等于 FLOPs 利用率或 HBM 带宽。低活动率提示需检查执行间隙，但不能单凭它判定算子差或 host bottleneck。

## 新核实的文本执行线索

1. **KV attention/tree merge 是尚未覆盖的路径。** 当前默认配置下，`Qwen35.GatedAttention` 的短步 decode 调用 `FixedKVCache.attend`，后者使用 Kitchen 的 `flash_attention_decode_gqa` 和 `flash_attention_decode_tree_merge`。本机 Kitchen `flash_attention.py` 的 native availability 只接受 CUDA/HIP，HPU 进入 `_decode_gqa_torch` / `_tree_merge_torch`。这些 torch 操作仍在 HPU 上，不是 CPU fallback。
2. **参考实现显式用 FP32，并先算完整 KV 容量再 mask。** `_decode_gqa_torch` 对 Q/K/V 调 `.float()`，执行 matmul、logsumexp、exp 和第二次 matmul；tree merge 再做 FP32 计算。当前新增编译仅覆盖 GDN/conv 和 MLP，没有包住这一段。是否存在底层自动融合、这些操作的真实占比仍需新版 trace，不能只按源码行数推算速度。
3. **容量显著大于有效长度。** 记录输入 1970 tokens、max_length=4096；`_generate_mtp` 按 `embeds.shape[1] + max_length + 7` 分配 cache，按记录推算容量约 6073，而本次生成结束约为 1970+277=2247。实际 tensor shape 应由下一次 probe 再确认。可以在保留分配容量/生成上限的前提下，尝试只计算有效前缀或向上分桶的前缀，并保持完整 tree mask/rollback 语义；不能直接截短 max_length 冒充执行优化。
4. **MTP 控制与计时尚有验证空间。** 本次 TextGenerate 有 111 次进度更新，对应当前代码中 111 轮 draft/verify，提交 276 个后续 tokens，约 2.49 token/轮；verify 固定 8 行。每轮已有一次合并 packet 的 `.tolist()`，并非每层读回。HPU 的 auto 策略用 CPU `perf_counter` 测 draft/verify，计时点位于 packet 回读之前，需核实它能否代表异步设备成本。不能据此直接认定 MTP 应关闭或树越大越好。

首个文本进度事件距 generate_start 约 2.0303 秒，但它包含初始化、prefill、首 token 和第一轮 draft/verify，不能标成纯 prefill。现有日志无法准确拆出 prefill、GDN、MLP、KV attention、lm_head、sampling 和 host wait 各占多少，下一次应补设备时间线。

## 更新后的优化顺序

| 顺序 | 工作 | 测量与验收重点 |
|---|---|---|
| 1 | 文本 KV attention/tree merge：先记录实际 shape 与 CPU/HPU 时间线，再分别试有效前缀/shape bucket、现有 torch 区域编译、兼容 tree mask 的现有 HPU attention 路径 | 单变量、固定 token/固定树 A/B；长短前缀、0-length 边界、回滚/分支、数值和卸载；统计整段净收益 |
| 2 | 文本 decode 执行与 MTP：投影/norm/RoPE 的编译边界、采样提交/同步、auto 成本估计；比较 off/固定树/auto | 每轮 draft/verify/接受数、固定工作量耗时；避免仅因输出变短而显得更快；不直接开启 CUDA graph 开关 |
| 3 | DiT 余下 19.67 秒：重做算子与 host gap 归因，关注区域之间的调度、重复位置/条件准备及剩余调制/残差 | QKV/attention/FFN/静态 recipe 已完成；调制/残差保留舍入边界有原因；固定 caption/seed/steps、多提示词数值与质量回归 |
| 4 | 原生 FP8 Linear 适配 | 等实际时间线证明矩阵/权重流量主导，再在真实 shapes 上计入 quant/layout 的总成本和品质校准；继续复用厂商 GEMM |

上图的 BF16 主线仍然成立，但“DiT 和文本主干未优化”的旧状态已经过时：GDN、MLP、DiT attention/投影/FFN 和 shape recipe 都已有收益。下一优先项应明确为**文本 KV attention 与 decode 剩余执行开销**，同时继续分析占总时间最多的 DiT。按照用户最新约束，当前不开展自写 TPC/INT8/FP8 kernel；原生 FP8 也排在剩余 BF16 路径验证之后。

VAE+图片输出仅约 0.26 秒，继续压它们收益有限。正负提示词编码合计约 1.75 秒，可在后续 prefill 工作中顺带评估。固定负提示词的正常节点缓存可能省一次计算，但当前基准刻意重算，不能把缓存跳过当作算子提速。

以这一次请求作固定工作量的情景计算（不是预测）：文本减少 30% 可省 4.485 秒，整图约 32.163 秒；DiT 减少 20% 可省 3.933 秒，整图约 32.715 秒；两者同时达到则约 28.230 秒。两个大阶段必须持续推进，不能只用提高 AIP 百分比代替端到端收益。

## 证据与复算

服务器记录目录 `/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-09-blocks/full-complete-product`；本地原始记录备份在被 git 忽略的 `comfyui-hpu/results/breakdown-2026-11-10/`。

```sh
python3 comfyui-hpu/probes/summarize_moss_workflow_breakdown.py \
  comfyui-hpu/results/breakdown-2026-11-10 \
  --label b0-warm-4
```

原始记录 SHA256：

- `events.jsonl`: `6379b2f646d14d1c1c35afd20125ffa4ca8d8510627ef1ca5f7d11b25c6e89ce`
- `runs.jsonl`: `30c5892ae633cf02e6188302ff81c598f0bc566a922a9c05a28b6b496d4a6428`
- `hpu.csv`: `db2783f67b1c22b789182f1fe86159382cfe67593b37c5840c911f0f1a8dea30`

核对产品部署仍为 `05d2a0a50de8385765f22dbdc12e9669392d3004`；本机 Kitchen `flash_attention.py` SHA256 为 `cd3c66927ffc6ea96d86f524d6e3979350ea036c243bedadfdae9964742db5d8`。相关产品源码为 `comfy/text_encoders/qwen35.py` 的 GatedAttention/_generate_mtp、`llama.py` 的 FixedKVCache、`comfy/ldm/moss_image/model.py` 的 Block；本地和服务器 qwen35.py/llama.py 哈希一致。
