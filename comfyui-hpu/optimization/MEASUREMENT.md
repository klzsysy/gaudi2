# 性能与质量验证方法

整理范围为2026-11-06至2026-11-13的单卡Gaudi2实验。环境、模型位置与运行命令见 [环境](../ENVIRONMENT.md)。本篇记录已经用实验证明会影响结论的测量方法。

## 固定工作量与真实入口

| 对照 | 固定内容 | 用途 |
|---|---|---|
| 完整自然工作流 | 原始API图、两类seed、采样设置、模型、尺寸、步数 | 用户实际等待时间、自然结束及图片质量，同时记录caption和token数 |
| 固定caption整图 | 同一正负caption、latent seed、尺寸、25步、CFG2.5、er_sde/simple | 隔离DiT与文本扩写变化，注明是否重算编码 |
| 固定MTP程序 | 1970输入、298输出、258次draft/top-k、130次sample/control及4次width决策 | 同轨迹计算与提交成本，执行原运算后代入记录决策 |
| 同输入局部重放 | 真实shape、stride、权重、初始状态和编译上下文 | 定位收益、舍入和状态差异，随后进入完整模型验证 |

常用1024² DiT正负条件为307/23 tokens，总序列4403/4119。早期记录为312/23，长图为8583/8629。跨轮数字需核对这些条件。独立文本脚本与API节点的计时边界不同，各自使用同轮基线。

固定MTP程序会强制决策，只能说明相同工作量下的执行行为。自然质量测试使用真实采样，最新可重复对照固定`tree3s1`。Auto按测得耗时和接受情况调整树，同seed跨进程也可能改变扩写文字。

## 计时与资源

- 完整候选通常各做1次进程cold和5次真实warm，报告中位数、范围及实际样本数。小于约3%的候选采用交错A/B/A或独立复验，五个样本的分位数仅作描述。
- 修改计算节点fingerprint并断言实际执行，保留loader。缓存文本的整图、完整重算、模型加载和首次shape编译分别记录。
- 正式wall time使用无profiler、无逐算子同步、无logits导出的请求。同步诊断与trace单独运行。
- 使用同流程比较allocator峰值、请求结束InUse和卸载稳定值。完整流程常见卸载约2.50GiB，文本单独约94MiB，混合shape可达5,393,876,352 bytes。
- 同进程A/B高水位包含此前编译和两套recipe。2026-11-13的42.63→47.27GiB属于累计高水位，单请求增量还需隔离进程测量。
- 核对实际运行文件哈希和模型metadata。曾出现版本标记为`b1163be0`、源码已含native norm/RoPE，恢复真实基线后才继续比较。

## 数值、图像和状态

1. 局部参考覆盖层0/20/41、步0/12/24、正负条件、长短shape。用同一份实际输入构建CPU FP64或FP32参考，同时比较候选对原实现的变化和两者对高精度参考的误差。
2. 质量保值候选优先检查逐元素一致。存在差异时定位首个舍入边界，不能用放宽原阈值代替判断。历史已接受量化算法的误差单列，新实现相对当前基线的额外误差另行检查。
3. 完整文本比较固定轨迹logits、top-1、KL与自然输出。大型logits的相对RMS用分块平方和计算。状态检查覆盖commit/rollback、pending、sibling、行重排、同shape换输入和换权重。
4. 图像用同caption/seed配对及新鲜A/A对照，记录RGB像素MAE的0–255量纲，保存原PNG并目视主体、文字、数目、手部和细节。finite、history success与低局部误差只能覆盖各自检查内容。
5. 按改动范围验证512/768/1024及长图、MTP off/3/tree3s1、原INT8、hooks后加、mask、参考KV、CPU、训练/梯度、关闭编译和卸载重载。batch2仅做过部分算子回归。

2026-11-13的保值对照为18组真实block和两种文本固定logits逐元素一致，六组固定树自然token相同。默认Auto的额外文字在原始文本回退版和优化版都出现，修正了最初将其归因于RoPE优化的判断。BF16茶壶把手不可见、FP8海报低对比度在两侧样本均存在，质量结论限于实际检查集合。[最新证据](../evidence/moss-three-stage-2026-11-13.json)

## 已发生的测量陷阱

| 现象 | 修正与经验 |
|---|---|
| warm保存logits增加约1.3–2秒 | 保存放到独立diagnostic，受影响样本从性能统计排除 |
| 两步文本profile很慢 | GC控制后的16轮窗口额外开销约1.24%，两轮约31.0%，未控制GC的两轮曾约45.8%，需分别标定扰动 |
| 负条件trace多出约2秒Cache Lookup | GC移出profiler窗口后停顿消失，重新采样设备预算 |
| 百万级kernel事件 | 多engine/lane嵌套与重叠，使用区间并集，launch单独计数 |
| `_copy_from`耗时很大 | 关联host等待和设备事件后归因，算子名不足以计算PCIe搬运 |
| 运行中替换RoPE custom-op | 固定logits偏差约43.7%/43.8%，改为每进程固定实现后逐元素一致，热替换数据退出候选评价 |
| 只设Synapse配置环境变量 | API读回未生效，该轮排除。有效试验保存初始化、Set/Get状态和实际值 |
| CPU测试改变全局attention backend | HPU fixture显式核对dispatcher，避免把不同算法混入A/B |
| 临时PNG随服务重启消失 | 32MiB候选缺完整图片证据，未完成质量验收。测试时持久化history、caption和PNG |
| 自然输出长度、加载和缓存状态不同 | 只记录实际使用观察，固定工作量另测，收益百分比不能跨口径计算 |

## 证据与复现入口

紧凑结果保存在`evidence/`，复现实验在`probes/`，失败实现位于`patches/`。服务器大型资料均在`/home/sonny/gaudi2/comfyui-hpu/performance-*`，固定MTP程序为`performance-2026-11-11-completion/mtp-record-298/program.pt`。本地`results/`是忽略的大型资料副本。

常用入口为`moss_observed_server.py`、`moss_mtp_program.py`、`moss_fp8_logits_compare.py`、`moss_fp8_text_quality.py`与`moss_three_stage_summary.py`。工具已验证HPU torch.profiler、synprof_parser、Perfetto SQL和fallback计数。TensorBoard Habana插件及habana_perf_tool在本栈存在MMEEvents兼容问题，分析采用原始trace。

后续报告保留问题、实验条件、候选结果、质量/资源代价、取舍和复测触发条件即可。服务启停、工具安装重试和中间排队过程只有在改变实验有效性或解释故障时才进入结论。
