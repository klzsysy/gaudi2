# MOSS BF16 DiT attention编译

2026-11-09，`60da4e33`→`f739cb93`。证据根目录为`/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-09-dit`。

## 测试条件与正式结果

固定caption正312/负23 tokens、latent seed649029532689846、1024²、25步er_sde/simple、CFG2.5。关闭扩写，每次重算编码/采样/VAE，保留loader。基线与产品各1 cold＋5 warm。

| 指标 | 基线 | 产品 |
|---|---:|---:|
| DiT | 37.915529s | 28.216700s |
| 含编码的整图 | 40.092826s | 30.257719s |
| cold整图 | 51.171390s | 42.644502s |
| allocator峰值 | 44.3546GiB | 40.1147GiB |
| DiT AIP均值 | 47.71% | 52.78% |

图片MAE0.12055/255，两侧自身重复均约0.12。重新开启自然扩写后完整图52.5911s，输出287–333 tokens，作为使用观察单列。

## 候选取舍

真实首层N=4408的原始SDPA为11.81ms，编译后3.39ms且该输入逐值相同。trace确认原始入口为math SDPA。产品只在MOSS所属attention区域编译，继续使用既有dispatcher。

| 候选 | 证据 | 处理 |
|---|---|---|
| 静态attention原型 | DiT25.299s，图片MAE0.122 | 产品采用可适配长度的区域实现，正式值见上表 |
| 整block编译 | DiT19.211s，图片MAE7.973，局部RRMSE0.326% | 转入后续逐区间FP64定位 |
| 默认FusedSDPA | 3.60ms，attention RRMSE0.606% | 数值/速度组合未优于编译math路径 |
| FP32 softmax | attention约11.4ms | 成本较高 |
| Kitchen eager decomposition注入 | AOT对`add_.Scalar`断言失败 | 停止该实现 |
| 共享动态SDPA | 单测通过后，真实图第6步SIGSEGV | 完整模型入口作为必要验收 |
| 共享SDPA加连续化 | DiT28.995s，图片MAE5.656 | 影响文本等额外调用，撤回共享实现 |

共享SDPA失败patch位于`patches/rejected-shared-dynamic-sdpa.patch`。HPU eager FP32全遮挡mask已有NaN现象，产品新增编译范围限定BF16推理，BF16全遮挡mask返回零已验。

## 验收与可复用经验

- 最终产品专项7 passed，覆盖10种长度、同shape换输入/norm参数、mask、参考KV、CPU、训练、override和禁用编译。
- 512/768/1024、加权提示词、卸载重载、原INT8两次完整图通过，fallback为空，latent/VAE finite。图片主体/大字正常，海报含模型原有无意义小字。
- HPU kernel事件分布在多个硬件lane，应统计时间区间并集，launch单独计数。局部静态原型与动态产品的差值需要真实shape切换复验。
- 后续已用保留调制/残差边界的分区编译处理整块差异，见 [block记录](MOSS_BLOCK_OPTIMIZATION_2026-11-09.md)。

复现入口为`run_moss_dit_trials.py`、`moss_dit_server.py`、`moss_dit_replay.py`。正式原始证据在`final/product-final`、`user`，失败实现及日志保留于同一证据根目录。
