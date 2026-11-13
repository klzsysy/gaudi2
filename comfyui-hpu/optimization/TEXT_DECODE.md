# 文本decode、KV状态与短步提交

覆盖采样索引、MLP/norm、KV attention、RoPE及MTP控制的2026-11-06至11-13实验。权重量化见 [文本精度](TEXT_PRECISION.md)，DeltaNet递推见 [GDN](TEXT_GDN.md)。

## 已采用的执行优化

| 变化 | 固定工作量结果 | 数值与实现要点 |
|---|---|---|
| HPU采样索引修复，`0139b1da` | 原用户Auto扩写图完成，26项回归 | int64 gather触发`GLUE_INCOMPATIBLE_DATA_TYPE`，仅gather前临时转int32，之后恢复long，覆盖高词表编号及top_k0/1/64 |
| 短步MLP编译，`05d2a0a5` | 32-token teacher4.876→4.223s | top-1为32/32、KL0.0003535，普通operations、BF16、KV cache、≤8行 |
| KV及完整attention编译，`3fec1143` | 同口径teacher4.312→3.002s | top-1为32/32、KL0.000299556，普通BF16、短步FixedKVCache、默认推理 |
| 原Qwen35 norm公式编译，`64155fe4` | 独立固定MTP7.481→5.670s，API完整22.461→21.060s | 保留FP32归约和普通编码dtype，1033行top-1为1022/1033、KL0.000719705 |
| 短步RoPE编译，`34e2649e` | FP8 TextGenerate5.227→4.985s，BF16为5.314→5.156s | 两种文本固定logits逐元素一致，采用原生addcmul舍入 |

早期teacher包含逐token logits CPU拷贝，最新固定MTP的正式计时将logits采集单列。不同测试的秒数只能与其同轮基线配对。

norm编译后的两步trace跨度137.309→108.419ms，设备活动61.118→59.396ms，launch1620→964，收益主要来自提交减少。prefill/普通编码另扩大MLP与attention区域，仅约节省0.082秒完整成本。34组真实norm输入含FP32编码及非连续Q/K，BF16误差比为1，FP32主干对FP64误差比0.84450–1.00681。

## KV正确性与候选

真实verify Q为`[1,16,8,256]`，K/V为`[1,4,6073,256]`，有效前缀1970。draft为1行。已有GQA将verify行联合计算并共享历史KV，mask包含树祖先可见性。

HPU同storage的`cache[:, :, dst] = cache[:, :, src]`在多次offset更新后可能未写入目标行，关闭编译仍复现，最大差4.09375。源行先clone及index_select/index_copy均修复，产品使用HPU源行clone。所有性能对照统一保留该修复。

真实verify输入、5组×20次、组尾同步的结果如下。

| 实现 | ms/call | 对原路径RRMSE |
|---|---:|---:|
| 原路径 | 2.7891 | 0 |
| 仅前缀裁剪 | 3.2914 | 0 |
| 编译完整容量 | 0.1541 | 0 |
| 编译＋256-token分桶 | 0.0869 | 0 |
| SDPA＋分桶 | 1.8273 | 0.2648% |
| 编译SDPA＋分桶 | 0.2046 | 0.2648% |

分桶在完整自然64-token首次为3.4096/3.7272秒，同进程交错为3.9309/3.8764秒，优势不稳定，采用完整容量编译。KV＋完整delta当时与attention叠加无稳定收益，后续递推改变后重测并采用八行GDN扩区。

厂商decode SDPA另用同输入、FP64参考比较。当前编译GQA/tree merge为0.1544ms、误差0.2235%，BF16默认0.0848ms/0.5970%，BF16 FP32-softmax0.1085ms/0.1916%，FP8常驻KV0.1064ms/7.5229%。FP8只做初始逐tensor校准，K/V初次量化未计时，缺增量更新及完整模型质量证据。BF16 FP32-softmax尚需验证真实集成收益，每次理论差约0.046ms。

## 短步RoPE的保值实现

2026-11-13在既有`comfy_kitchen::apply_rope_split_half`入口一次性注册HPU实现。范围为BF16 Q/K、FP32频率、64维旋转部分、1–8行，保留训练/梯度、prefill及关闭编译的既有入口。非原地公式使用`torch.addcmul`，编译选项`decomposition_mode=none`保留其原生舍入。

| 候选或测试方法 | 结果与处理 |
|---|---|
| 直接decomposition eager公式 | 临时view上的原地addcmul未通过AOT功能化检查 |
| 功能化公式＋默认算术分解 | 固定logits RRMSE4.6805%，top-1为1007/1033，淘汰 |
| foreach乘加 | 仍有额外数值变化 |
| 原地临时view、关闭分解 | 两个完整测量进程段错误，未采用，底层根因未确认 |
| 运行中反复替换注册kernel | FP8/BF16固定logits差43.7%/43.8%，clone和上下文切换无效，属于对照方法失真 |
| 每进程固定实现、非原地公式、关闭分解 | 两种文本各1033×248320 logits逐元素一致，完整运行通过 |

最终各五轮warm，FP8完整流程16.092→15.902秒，BF16完整流程19.521→19.396秒。六组固定`tree3s1`真实自然采样的token与文字一致，配对RGB MAE0.07597–0.08599，重载、MTP off和原INT8通过。六项RoPE测试覆盖动态1/2/4/8行、非连续部分旋转、输入所有权和回退条件，最终attention/RoPE专项共37项通过。

默认Auto曾在优化版加入文字卡片，回退原文本后也加入标签。工作流参数相同，最初的质量归因据此修正。最终部署又得到与历史基线相同的扩写。Auto变化及基线图像缺陷见 [质量方法](MEASUREMENT.md)。

## 无收益候选与后续方向

- MTP head、采样概率区域、关闭嵌套编译分别约5.636/5.682/5.699秒，对照norm产品5.670秒，均无明确净收益。整Transformer block为6.428秒。
- 厂商`hpu.rms_norm`约快0.32%，对FP64误差最高增至1.711倍，采用原公式区域编译。
- 连续KV已由FixedKVCache提供。当前约2K有效上下文、6K容量，FP8 KV预计只省约106.75MiB，新增镜像或8K分片需新的长上下文证据。
- 跨片attention按log-sum-exp权重合并局部输出。历史无mask条件、末片padding及MTP树可见性分别处理。
- 跨请求前缀复用需要同时恢复KV、GDN recurrent state、conv history、位置及末尾hidden/logits。当前prefill以`initial_state=None`启动，缓存生命周期及权重/LoRA失效还需单独设计。
- 16轮GC控制trace的间隙约39.7%，最长0.966ms，空档原因和可消除部分仍需量化。候选优先沿现有状态边界减少提交，固定决策和自然采样共同验收。

## 证据与复现

原始目录为`performance-2026-11-09-blocks`、`performance-2026-11-10-kv`、`performance-2026-11-11-completion`及`performance-2026-11-12-three-stage`。入口为`moss_fused_decode_probe.py`、`moss_rope_lowering_check.py`、`moss_rope_registration_check.py`和`moss_three_stage_pipeline.py`。[norm证据](../evidence/moss-bf16-completion-2026-11-11.json) · [最终RoPE证据](../evidence/moss-three-stage-2026-11-13.json) · [原地候选补丁](../patches/moss-rope-inplace-rejected-2026-11-13.patch)
