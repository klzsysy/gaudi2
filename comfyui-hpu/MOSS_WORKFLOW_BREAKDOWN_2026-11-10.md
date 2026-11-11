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

## 归因与后续验证

本轮发现HPU KV attention/tree merge走FP32 Torch参考实现，完整容量约6073，已有GDN/MLP编译尚未覆盖此区。111轮draft/verify提交276个后续tokens，首进度包含prefill和第一轮verify，需单独测量。

这些线索随后形成KV编译与状态修复，见 [KV结果](MOSS_KV_OPTIMIZATION_2026-11-10.md)。VAE与图片输出约0.26s，文本和DiT为主要费用。当前路线判断见 [BF16复盘与FP8](MOSS_OPTIMIZATION_STRATEGY_2026-11-09.md)。

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
