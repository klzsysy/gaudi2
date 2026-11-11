# MOSS GDN区域编译

2026-11-09，产品`60da4e33`。Gaudi2/Habana 1.24.2，eager，普通BF16文本模型。证据根目录为`/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-09-gdn`。

## 测试与结果

Kitchen deferred conv/GDN在HPU使用Torch参考实现。将纯tensor区域交给`hpu_backend`静态编译，采样、MTP状态管理及参数cast/offload由既有调用层管理。各1 cold＋5 warm，计时与同步诊断分开。

| 测试 | 基线 | 候选 | 口径 |
|---|---:|---:|---|
| 真实conv输入重放 | 6.6012ms | 0.4487ms | 相同输入/初始状态，10次同步中位数 |
| 真实GDN输入重放 | 19.1253ms | 0.9331ms | 同上 |
| 32-token MTP auto | 10.9695s | 3.7273s | 第28个token起分歧 |
| 完整1024²/25步 | 131.031s | 60.063s | 自然输出375→303–333 tokens |
| TextGenerate | 96.740s | 20.156s | 包含prefill |
| 完整cold | 140.726s | 71.979s | 加载/首次编译计入 |
| allocator峰值 | 44.67GiB | 44.33GiB | 完整工作流 |

自然caption变化会改变解码和DiT工作量。相同输入重放提供局部加速证据，完整图记录实际使用延迟。所有成功请求fallback为空。

## 正确性与实现边界

- conv committed state/projection buffer逐值相同，QKV buffer最大差0.03125。GDN输出最大差0.001953125、平均差8.58e-6，该次committed state相同。
- 固定32-token teacher、MTP off，top-1为31/32，平均KL0.0003614。编译舍入会改变自然采样轨迹。
- 59 passed、1 CUDA skip，后续状态/权重复用专项5 passed，测试有重叠。覆盖batch1/2、1/3/8行、交替缓冲、零/部分提交及sibling路径。
- 原INT8完整图两次、BF16/INT8 latent与VAE finite检查通过。街景主体和MOSS灯牌保持，质量范围限于这些样本。
- 编译callable按需缓存，状态由调用方持有。CPU/CUDA、关闭编译走既有路径，新shape具有编译成本。

## 对后续优化的意义

HPU通用参考实现中的小算子提交曾是主要成本。编译后仍需按真实工作量重新归因。INT8诊断扩写AIP约46.47%，BF16正常warm约15.82%，但INT8吞吐较低且带同步检查，活动率只能配合延迟解释。

可复用探针为`moss_gdn_experiment.py`、`gdn_replay.py`、`run_gdn_workflows.py`和`moss_observed_server.py`。后续递推与外围投影改进见 [最新GDN记录](MOSS_BF16_ATTENTION_AND_SUBMISSION_2026-11-11.md)。
