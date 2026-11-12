# MOSS FP8区域融合与数据布局

## 基线与测量

从`f67e7f3f`继续，使用Moody FP8主模型与Qwen3.5 FP8文本，固定1970输入、298输出token的MTP程序，1024²、25步、CFG2.5、`er_sde/simple`。每个独立进程执行1 cold与5 warm，诊断请求另计，HPU任务串行。

本轮重新测得完整流程17.201787s、DiT11.361362s，与前轮17.190751/11.364063s接近。原始数据位于`/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-12-fp8-fusion`。测试checkout通过源码哈希核对基线，原始`/home/sonny/ComfyUI`未作修改。

## 产品实现

- Q/K归一化改为原生`hpu.rms_norm`的FP32输入和权重，再进入原BF16/RoPE转换。三组代表层的整层耗时约5.59→5.27ms，相对本轮基线的整层RRMSE约0.017%–0.075%。直接使用BF16 native norm会放大敏感层差异，因此选用FP32归一化。
- 将FP8 block的调制、投影、attention、FFN和残差交给更大的编译区域。继续使用既有cast/offload、attention dispatcher和参数生命周期。梯度、训练、hooks、attention override、reference、mask等条件沿用相应分区路径。

扩区编译改变部分舍入边界。调制分别放入attention和FFN的私有方法，保留HPU graph break间的区域边界。将两个调制直接内联到总forward时，完整流程16.459s、DiT10.682s，三个代表层约5.21ms。恢复区域边界后，代表层约5.00ms，与调查实现输出相同。它包含编译器生成的多个recipe。

## 层级精度与候选对照

复用Moody真实输入的正/负条件307/23、采样步0/12/24、层0/20/41，共18组。参考block使用相同已解码权重、FP32激活和运算。它衡量当前量化权重上的计算误差，未恢复原始训练权重。

| 候选 | 局部耗时比中位数 | 对f67e7f3f的最大整层RRMSE | 候选/基线的FP32参考误差比 |
|---|---:|---:|---:|
| native FP32 norm与V合并量化 | 0.9097 | 1.088% | 0.99983–1.00041 |
| 再合并FFN调制 | 0.8931 | 2.167% | 0.90015–1.07530 |
| 再合并FFN残差 | 0.8899 | 2.179% | 0.84503–1.17892 |
| 整个block扩区 | 0.8595 | 4.066% | 0.72808–1.29272 |
| 最终产品，逐head量化与明确的区域边界 | 0.8940 | 4.066% | 0.72808–1.29272 |

最终产品重新检查了全部18组，RRMSE范围0.306%–4.066%，对FP32参考的误差比中位数1.00704。最差相对增幅发生在正条件第12步第0层，绝对RRMSE由1.380%变为1.784%。其余17组的误差比低于1.1。所有输出有限，fallback为空。逐层参考通过重建`f67e7f3f`的两分区forward、norm和单head AV路径计算，完整流程基线仍采用实际部署源码的独立测量。

## 完整流程与质量

最终产品为`0995638a`，正式结果取`verified`独立进程的5 warm中位数。实现收益的归因采用DiT节点时间。

| 固定298-token、1024²、25步 | 本轮基线 | 最终产品 |
|---|---:|---:|
| 完整流程 | 17.201787s | 15.925341s |
| DiT | 11.361362s | 10.211354s |
| TextGenerate | 5.194520s | 5.039580s |
| CLIPTextEncode | 0.410186s | 0.386178s |
| VAE | 0.132477s | 0.131646s |
| cold完整流程 | 73.860517s | 77.272611s |
| allocator高水位，含cold | 27.401830GiB | 31.284505GiB |
| 请求结束常驻 | 17.987188GiB | 17.987192GiB |

完整流程减少7.4%，DiT减少10.1%。最终warm范围15.734–16.094s，卸载回到2,685,928,832 bytes。allocator高水位增加3.883GiB，扩大编译区域的显存成本已计入结果。

最终287项回归通过。产品重新完成三组自然1024²图片、卸载重载后的512²/MTP off、原INT8文本/MTP auto，共5次请求和10项latent/VAE有限性检查，fallback为空。自然扩写332/310/453 tokens，两个短流程均32 tokens。红茶壶、MOSS文字与三朵黄花、红围巾白杯人物均通过目视检查。

调查候选的三组配对图使用相同caption、seed、25步和采样设置。候选请求已断言复用文本，六次请求的latent/VAE均有限、fallback为空。茶壶、MOSS海报与三朵花、红围巾白杯人物保留提示词要求。茶壶的光影、背景位置有明显变化，人物和花瓣细节也有变化。三组像素MAE为32.792/2.938/8.662（0–255）。质量判断限于这组目视检查。

## 设备预算

第12步正负条件两段trace按25步外推，并按FP8 MME、其它MME、softmax、其它kernel、间隙的优先级计算互斥时间。参考采用前轮`release`相同步骤与条件的trace，正式延迟采用本轮独立5 warm。

| 互斥设备预算 | 改动前 | 最终产品 |
|---|---:|---:|
| FP8 MME | 3.0464s | 3.0453s |
| 其它MME | 2.3919s | 2.4087s |
| softmax | 1.3562s | 1.3404s |
| 其它kernel | 3.6677s | 2.6509s |
| kernel间隙 | 1.0786s | 0.8194s |
| 合计 | 11.5408s | 10.2647s |
| FP8 MME时间占比 | 26.4% | 29.7% |

两段trace的runtime `run`计数3632→776，`launch`1984→640，`launch_recipe`336→504。更大区域改善了整体调度与设备预算，recipe计数需连同全部提交一起解释。节点首尾各排除1秒的warm遥测中，DiT AIP均值56.14%→58.51%，功率342.99→356.95W。秒级时间戳与100ms采样限制了这些遥测值的精度。

主干矩阵FLOP的FP8覆盖保持约90.6%。本轮主要减少矩阵计算周围的成本，绝对FP8矩阵时间变化较小。

## V量化对照

V批量量化把24个head的动态scale与cast合并。初始三个Moody输入与逐head量化输出相同，完整流程15.674501s、DiT9.847280s、allocator高水位31.293GiB。随机测试随后发现四维BF16广播量化增加约1.5%的独立AV参考差异。

显式使用FP32量化输入后，原有0.5%独立AV参考检查与逐head一致性检查通过，阈值保持原值。该版本完整16.106592s、DiT10.307281s、allocator高水位33.916GiB，288项回归、自然图片与重载通过。18组整层输出与初始候选最大差异0.0131%，局部时间中位数增加约5%。

保留原逐head量化的对照为完整16.078654s、DiT10.196725s、allocator高水位31.299GiB。完整吞吐接近，allocator高水位低2.617GiB，最终产品沿用逐head量化。共享amax、逐head执行cast的后续候选在18组输入中局部时间增加4.54%，最大整层差异1.58%。

## 其它筛选结果

- BF16 native Q/K norm及BF16 RoPE带来更大的敏感层差异，三层最高分别约3.66%和3.34%，RoPE局部收益很小。
- 去掉native FP32 norm后的显式BF16转换，在当前编译结果中与保留转换输出相同，未见额外收益。
- 将QK缩放交给`softmax_fp8`的`inv_attn_heads`，单独局部5.589→5.581ms，敏感层差异3.90%，未采用。本机该参数需要Tensor，首次Scalar调用的schema错误已修正。
- SiTU重用`tanh(gate/4)`表达sigmoid，减少一个非线性函数，但实际block约4.84→8.01ms，未采用。
- 为V量化改用二维逐行scale或显式乘scale，局部约5.10/5.02ms，相对当时产品版本差异约0.21%。最终产品采用原逐head量化路径，独立AV参考检查通过。

## 诊断边界

内存统计采用各独立进程在warm阶段记录的`MaxInUse`高水位，包含此前cold分配，未在请求间重置峰值。请求结束的`InUse`和卸载后的值单独记录。

带profiler的调查服务在完成trace导出和卸载后退出时出现`malloc(): unaligned tcache chunk detected`。前轮`release/server.log`存在相同退出异常。计时排除profiler请求，诊断导出完整，卸载已回到2,685,928,832 bytes。该退出问题的根因尚未确认，普通服务与无profiler重载另行验收。

原生算子的接口和布局依据同时核对本机1.24.2 schema与厂商源码。[RoPE接口](https://docs.habana.ai/en/latest/PyTorch/Reference/Python_Packages.html)、[softmax实现](https://github.com/HabanaAI/gaudi-pytorch-bridge/blob/27eccbcffeab10842c9f29f64c6bf50d8f65d78f/hpu_ops/backend/softmax_fp8_gen.cpp)

## 接续方向

1. TPC逐元素和动态量化仍占秒级预算。本轮广播量化的精度修正抵消部分收益，后续优先验证显式融合kernel的FP32缩放、BF16舍入边界与FP8写出，结合实际中间张量读写计时。服务器已有TPC编译器，尚未为本次改动构建自定义kernel。
2. QK和softmax继续关注中间矩阵的数据流与准确概率计算。全FP8 QK的敏感层问题沿用前轮记录，新实现需通过多层、多timestep和完整图验证。
3. 文本扩写约5.2s，已占完整流程约三分之一。进一步投入需落实到短步提交和GDN状态融合，继续使用固定MTP轨迹、状态回滚与卸载重载检查。

## 部署与复现

`0995638a`已部署8189，九份产品/测试文件哈希一致。普通入口cold79.030s、文本缓存复用10.696s、768²/MTP off 14.295s，三次history均success，最后队列为空，运行时为Habana PyTorch2.13.0a0、Gaudi2。缓存图通过目视检查。

`moss_fp8_fusion_screen.py`在`f67e7f3f`上筛选候选，`baseline-f67,product`重建原计算路径作最终18组对照。`moss_fp8_fusion_release.py`通过`MOSS_FUSION_RELEASE_CASE=verified`复现最终回归、固定工作量、自然图片和重载。`moss_fp8_fusion_deploy.py`验证正常服务入口。[结构化证据](evidence/moss-fp8-fusion-2026-11-12.json)
