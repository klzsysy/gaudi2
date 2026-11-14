# Qwen Image PE文本进一步加速

2026-11-15，针对用户关于文本带宽与进一步优化手段的追问，在`dfca7cd6`基础上验证跨checkpoint草稿头。普通8189队列为空后暂停，HPU实验串行，结束后恢复原服务并再次生成原2MP图。

## 带宽判断与投入方向

当前42.8 tokens/s乘以每token约15.873GB主要权重，得到约0.679TB/s逻辑权重流量。真实HBM访问还包括重复读取、KV、激活、布局和写回，硬件计数尚未校准。这个平均值支持继续挖掘执行效率，单个GEMV活动期间的带宽仍需独立测量。

已确认每token约146次recipe提交，24个GDN、8个完整attention和32个MLP。可投入方向包括更长生命周期的decode重放、更大的纯tensor编译区、GDN复合kernel和多token验证。既有单GDN扩区的收益受到投影舍入变化制约，关闭eager流水线明显变慢。单token持续执行约23.4ms，纯权重读取的理想6.5–7.4ms之外还包含状态操作、非线性、提交与同步等成本。

本机安装的`habana_frameworks/torch/hpu/graphs.py`明确将HPUGraph限定于lazy mode，当前为eager。持久decode图需要处理执行模式、静态buffer、KV/GDN状态和模型卸载生命周期。[Intel HPU Graph文档](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html)

投机解码的一般思路是由较便宜的模型生成提案，目标模型并行验证多个位置。分布保持依赖具体验证/采样算法及目标概率计算的正确性。[原始论文](https://arxiv.org/abs/2211.17192)

## 现有MTP头的隔离试验

目标模型使用当前`qwen3.5_9b_qwen_image_2.1_pe_t2i.bf16_from_int8.safetensors`。从已有`qwen3.5_9b_bf16_from_int8.safetensors`读取15个`mtp.*`张量，共243,290,624参数，约0.487GB BF16。严格加载到同架构MTPHead，仅在调查进程内挂接。目标PE继续承担验证和最终token选择。

使用用户最新提示词、thinking开启、temperature1、top_k20、top_p0.95、presence_penalty1.5和原文本seed。两种运行模式都保持BF16。普通模式关闭MTP，候选复用现有固定MTP树、KV rollback及commit逻辑。草稿参数从已有文件按需读取，并通过内存对象挂接。

### 固定512-token窗口

每个模式一次首次运行、三次warm。所有模式均输出512 tokens，尚未到自然EOS。各自重复运行token序列相同，模式间token序列不同。

| 模式 | warm中位秒 | tokens/s | 目标decode/verify调用 | 草稿调用 | 每次验证推进token |
|---|---:|---:|---:|---:|---:|
| 单步off | 12.182 | 42.029 | 511×1行 | 0 | 1.000 |
| tree3s1 | 9.156 | 55.917 | 243×8行 | 729 | 2.103 |
| 2 | 8.742 | 58.566 | 281×4行 | 562 | 1.819 |

四行候选吞吐提高39.35%，相同512-token工作量的耗时减少28.24%。八行树的推进token更多，同时草稿成本更高，最终速度低于四行候选。每次验证推进token数包含该轮目标选择的最后一个token，边界处还受512-token截断影响。

### 完整自然扩写

恢复4096输出上限，每个模式一次首次运行加两次warm。均自然结束于stop token248046，每个模式的三次token序列一致。off序列与上一轮已验产品1784-token序列的哈希一致。

| 模式 | 自然输出tokens | warm中位秒 | tokens/s | 每次验证推进token |
|---|---:|---:|---:|---:|
| 单步off | 1784 | 41.507 | 42.981 | 1.000 |
| 四行MTP | 1911 | 31.279 | 61.097 | 1.830 |

自然运行吞吐提高42.15%。两种模式生成内容和长度不同，不能将自然耗时差全部视作固定计算量的收益。该表覆盖独立文本阶段，完整出图时延另需验收。

## 数值边界

额外在第一次四行验证前克隆完整KV/GDN状态，为根token构造相同输入和位置的单token对照。单token分支清除tree元数据并使用自己的状态副本。两侧经过同一目标模型及输出头比较。

首个根token logits的RRMSE为2.0221%，最大绝对差0.3125，argmax均为1118。该结果说明多行验证相对现行单token执行已有数值变化。现有MTP采样路径另将验证logits转为FP32处理，而普通PE采样保留其原dtype。论文中的分布保持结论不能直接替代这个实现的目标概率与状态验证。

当前结论是草稿复用的性能潜力已得到实测，质量验收仍待完成。还需定位多行目标计算的数值差异、对齐采样语义、覆盖更多提示词/seed/长上下文并验证最终图片，以及完成正常loader/模型管理生命周期接入。当前线上版本为已验`dfca7cd6`。

## 复现与恢复

入口为`probes/qwen21_draft_screen.py`，参数支持`--length`、`--runs`、`--modes`和`--audit-root`。执行时设置`PT_HPU_LAZY_MODE=0`、`GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so`及产品checkout的`PYTHONPATH`。

原始资料保存在服务器`/home/sonny/gaudi2/comfyui-hpu/performance-qwen21-text-next-2026-11-15`，包括各次tokens、解码文本、timings、MTP来源清单和根token logits。紧凑证据为`evidence/qwen21-text-draft-2026-11-15.json`。

实验后普通8189恢复，原2MP工作流94.763秒成功，最终RGBA像素与上一轮已验基线一致，队列为空。
