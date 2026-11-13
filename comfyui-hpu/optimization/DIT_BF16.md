# BF16 DiT分区、预计算与norm/RoPE

当前BF16 DiT沿用模型所属attention/投影和FFN分区、局部静态recipe、条件预计算及经系数检查的RoPE实现。2026-11-13再次尝试共享FP8的整block扩区，保留原舍入后无稳定净收益。attention算子与head调度见 [Attention](ATTENTION.md)。

## 分区与shape管理

2026-11-09固定正312/负23 tokens、1024²、25步、CFG2.5、er_sde/simple，各1 cold＋5 warm。首轮attention区域编译`f739cb93`使DiT37.916→28.217秒，含编码整图40.093→30.258秒，allocator峰值44.35→40.11GiB。实际首层N4408的math SDPA11.81→3.39ms，同输入逐值相同。

整block原型DiT19.211秒，但图片MAE7.973，完整block RRMSE0.3260%，首个偏离点在modulation。FP64定位显示modulation误差0.4233%→0.4287%，residual0.2167%→0.1503%，SiTU均0.5058%。产品保留调制/残差边界，将QKV→attention→输出投影和up/gate→SiTU→down分成两区。

| 同轮路径 | 含编码整图 | DiT | allocator峰值 |
|---|---:|---:|---:|
| attention编译基线 | 30.246s | 28.225s | 约40.11GiB |
| 两段编译 | 25.566s | 23.545s | 40.60GiB |
| 再用局部静态recipe，`05d2a0a5` | 21.519s | 19.456s | 39.39GiB |

13种长度切换后默认动态recipe的N4408 block从约9.16升至13.17ms，`force_static_compile`保持约9.21ms，新shape首次0.13–0.21秒。九组层/步真实输入对FP64的候选/基线误差比0.99845–1.00045，完整图MAE3.548，部分细节超过A/A波动。

编译后新增forward hooks曾被忽略，产品补充模块/子模块/全局hook门控。权重更新、别名、mask、参考KV、override、CPU、训练和关闭编译分别回归。两段编译79项、静态recipe20项和MLP/hooks35项测试有重叠，512/768/1024及重载、原INT8完整图通过。

## 条件预计算

同步诊断发现每个请求text_embedder及RoPE各调用50次，分别累计0.797/0.541秒。`b1163be0`在`extra_conds`按本次conditioning预计算，降为正负条件各一次。

固定caption/seed，各五轮，DiT19.581→18.384秒，含编码整图21.642→20.524秒，峰值39.385→39.498GiB。图片RGB MAE0.11504，A/A约0.11488/0.11558。自然完整298-token同ID流程30.786→29.673秒，单列其计时口径。

实现保留原context和FP32 RoPE，预计算tensor归conditioning持有，销毁后weakref释放。适用HPU BF16静态推理，timestep调制每步计算。hooks、区域、ControlNet、参考图、packed latent和不支持options按原入口计算，尺寸、位置、设备变化时重新准备。19项preparation与21项原有attention/block测试通过，含batch2算子、加权mask、合批及跨timestep检查，完整卸载约2.50GiB。

## Native norm与RoPE

`b25001dc`的native modulation norm＋四系数PAIRWISE RoPE使固定caption DiT18.397→16.887秒，18组真实输入对FP64误差比0.99940–1.00108。`1412e1f7`再启用系数验证后的BLOCKWISE，DiT16.893→14.364秒，缓存整图17.146→14.617秒，误差比0.90321–1.03067、中位0.99664。

BLOCKWISE要求两个对角系数相同、非对角互为相反数，一般矩阵用四系数PAIRWISE。早期直接丢弃第二组系数有语义错误，prepared/直接forward统一检查后原逐值断言通过。配对图MAE3.40–13.93，主体通过目视，同seed构图和细节有变化。60项MOSS和5项prefill专项通过，包含hooks、batch2算子、重载及原INT8。

## 无收益或未接受的候选

| 候选 | 同条件实测与误差 | 取舍依据 |
|---|---|---|
| CFG左padding＋mask合批 | DiT19.402s，对照19.581s，三轮，峰值41.241GiB，MAE4.115 | 收益不到1%，适配与显存成本较高 |
| CFG合批＋预计算 | 19.749s，对照预计算18.384s，MAE3.470 | 较慢 |
| modulation/residual短链编译 | 局部快3.5%–4%，FP64误差比最高1.237 | 保留敏感舍入边界 |
| native AdaLN扩区 | 快约2.6%，最差误差增12.95% | 未采用 |
| scale合入gamma | 快0.3%–1.3%，误差比0.859–1.025 | 收益较小 |
| 加权RMSNorm/shift | 快2.4%–3.1%，最差误差增18.5% | 未采用 |
| 双block编译 | 42层正/负293.365/272.404ms，原293.252/272.374ms，输出相同 | 无净收益 |
| 2026-11-13整block扩区 | 正/负6.269/6.697→6.052/6.364ms，RRMSE0.371%/0.380% | 未通过本轮保值要求 |
| 同轮恢复modulation/residual边界 | 输出逐元素一致，正条件6.679ms，对照6.664ms | 无稳定收益，采用原分区 |

## 复用经验与证据

分区大小、Python方法边界、动态图与厂商recipe特化共同决定执行。局部原型的静态shape优势需要经过连续变长和正常产品入口检查。条件预计算由执行所有者持有，避免跨请求模型缓存的失效与卸载问题。递推或底层lowering变化后可重测扩区，现有BF16大Linear进一步提速的预算见 [瓶颈](BOTTLENECKS.md)。

原始目录为`performance-2026-11-09-dit`、`performance-2026-11-09-blocks`、`performance-2026-11-10-dit-next`、`performance-2026-11-11-bf16`、`performance-2026-11-11-completion`及`performance-2026-11-12-three-stage`。入口包括`moss_block_precision.py`、`moss_block_validate.py`、`moss_block_fp64.py`、`moss_block_shapes.py`和`moss_bf16_shared_screen.py`。[norm/RoPE证据](../evidence/moss-bf16-2026-11-11.json) · [权重轮换与双层验证](../evidence/moss-bf16-completion-2026-11-11.json) · [最新保值对照](../evidence/moss-three-stage-2026-11-13.json)
