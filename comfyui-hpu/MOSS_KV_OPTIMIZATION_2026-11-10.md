# MOSS KV attention与decode编译

2026-11-10，`05d2a0a5`→`3fec1143`。证据根目录为`/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-10-kv`。

## 真实形状与候选

verify Q为`[1,16,8,256]`，KV为`[1,4,6073,256]`，首次index1970。draft Q为`[1,16,1,256]`，同容量，首次index0。HPU走Torch FP32 GQA/tree merge参考实现。短probe保留6073容量。

真实verify输入，5组×20次，组尾同步。

| 候选 | ms/call | 对基线RRMSE |
|---|---:|---:|
| 原路径 | 2.7891 | 0 |
| 仅前缀裁剪 | 3.2914 | 0 |
| 编译完整容量 | 0.1541 | 0 |
| 编译＋256-token分桶 | 0.0869 | 0 |
| SDPA＋分桶 | 1.8273 | 0.2648% |
| 编译SDPA＋分桶 | 0.2046 | 0.2648% |

SDPA对FP64误差0.1651%，原路径0.2235%，但编译原操作更快。自然64-token复验中，完整容量/分桶先为3.4096/3.7272s，同进程交错后为3.9309/3.8764s，分桶收益不稳定。产品采用完整容量编译。

## 状态修复与区域选择

HPU同storage的`cache[:, :, dst] = cache[:, :, src]`在多次offset更新后可能未写目标行，关闭编译仍复现，最大差4.09375。源行先clone和index_select/index_copy均修复，产品采用HPU源行clone。对照统一保留该修复。

固定32-token teacher、MTP off、6073容量、每步拷回logits，各1 cold＋5 warm。

| 原型 | 秒 | top-1 | 平均KL |
|---|---:|---:|---:|
| 修复cache后的基线 | 4.1691 | 32/32 | 0 |
| KV编译 | 3.6954 | 32/32 | 0 |
| KV＋完整attention | 3.2089 | 32/32 | 0.0002996 |
| KV＋完整delta | 3.6752 | 31/32 | 0.0003345 |
| KV＋两者 | 2.9427 | 32/32 | 0.0002089 |

当时delta与attention在自然生成中叠加无稳定收益，选择KV＋完整attention。该结论限于当时递推实现，后续算法改变后已重新评估。

## 正式验收

正式产品的同口径teacher复验4.312471→3.002085s，top-1为32/32，平均KL0.000299556。自然完整图median29.9251s、cold47.913s，输出305 tokens。同期基线36.1551s、277/362 tokens，只作自然工作流观察。

- 33项KV/attention专项及42项decode回归通过，1 CUDA skip，分组报告。覆盖batch1/2、实际KV heads、两种RoPE、分支回滚、权重更新、hooks及关闭编译。
- 512 auto、768 MTP off、1024重载与原INT8图通过，latent/VAE finite、fallback为空。文本单独卸载98,549,888 bytes，重载8-token ID一致。
- 完整attention扩区限定普通BF16 operations、短步FixedKVCache、默认attention推理。hook/自定义路径保留对应调用语义。
- FP8/BF16 FusedSDPA的后续真实decode对照见 [attention实验](ATTENTION_IDEAS_REVIEW_2026-11-10.md)。

局部有效前缀收益须经完整解码验证。状态错误可能由异步设备执行延后暴露，应分别验证读写与编译开关。下一轮费用归因见 [pipeline记录](MOSS_PIPELINE_TIMING_2026-11-10.md)。
