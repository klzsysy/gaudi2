# MOSS BF16 block分区与文本MLP

2026-11-09，`f739cb93`→`05d2a0a5`。证据根目录为`/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-09-blocks`。

## 数值定位与实现

整块编译首个偏离点为modulation，局部RRMSE0.3135%，完整block0.3260%。CPU FP64下modulation误差0.4233%→0.4287%，residual误差0.2167%→0.1503%，SiTU两侧均0.5058%。因此用高精度参考逐区间检查，并保留调制/残差的既有舍入边界。

产品分为QKV→attention→输出投影，以及up/gate→SiTU→down两个编译区。首层block约9.22ms，RRMSE降至0.02660%。真实层0/20/41×步0/12/24共9组，完整block对FP64的候选/基线误差比0.99845–1.00045。深层本身更敏感，单首层结果不足以外推。

13种长度切换后，默认动态recipe的N=4408 block由约9.16升至13.17ms，局部`force_static_compile`维持约9.21ms。新shape首次成本0.13–0.21s，算子最大输出差0.0625。该选项用于两个MOSS区域。

## 正式性能

固定caption、1024²/25步，含文本编码的整图，各1 cold＋5 warm。

| 实现 | 整图 | DiT | allocator峰值 |
|---|---:|---:|---:|
| 基线 | 30.2457s | 28.2255s | 前轮同配置40.11GiB |
| 两段编译 | 25.5656s | 23.5449s | 40.60GiB |
| 加静态recipe | 21.5192s | 19.4562s | 39.39GiB |

最终图片对基线MAE3.548/255。多提示词主体与MOSS大字通过目视检查，同seed细节变化超过重复噪声。

文本新增普通operations、HPU BF16、KV cache、≤8-token decode的MLP编译。固定32-token teacher、MTP off、包含逐token logits CPU拷贝，5 warm为4.8757→4.2231s。top-1为32/32，平均KL0.0003535。自然MTP输出会分歧，完整图观察为52.5911→36.6480s，最终267–362 tokens。

## 回归与经验

- 两段编译79 passed/1 CUDA skip，静态recipe复验20 passed，最后MLP/hooks/MOSS复验35 passed。各组有重叠。
- 编译后新增forward hooks曾被忽略，产品增加模块/子模块/全局hook检查并分流eager。覆盖输入别名、权重更新、mask、参考KV、CPU、训练、override及禁用编译。
- 512/768/1024、加权提示词、卸载重载和原INT8图通过，latent/VAE finite、fallback为空。文本单独卸载约93.98MiB。
- 最终全图/文本/DiT AIP为37.93%/16.79%/57.23%，解释为粗活动采样。
- 静态recipe在shape切换后体现收益。跨算子融合须同时验证局部FP64与完整质量，后加hooks和换权重需走正常模型入口。

提交依次为`8ebc3f40`分区、`15c5b17c`静态recipe、`b830b3b8`hooks、`05d2a0a5`文本MLP。探针为`moss_block_precision.py`、`moss_block_capture.py`、`moss_block_validate.py`、`moss_block_fp64.py`、`moss_block_shapes.py`。正式数据在`static/product-final`、`full-complete-product`、`teacher-control`、`teacher-compiled`及`quality/comparison.png`。
