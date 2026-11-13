# GDN递推、prefill与区域编译

当前采用Torch参考公式的区域编译、静态prefill分桶、FP32短序列并行递推及受控八行扩区。2026-11-12的自写TPC四列核心因完整质量变化撤回。范围为Qwen3.5 MOSS文本，单卡Gaudi2 eager。

## 已采用路径及测试要点

| 改动 | 测试与结果 | 质量和边界 |
|---|---|---|
| deferred conv/GDN区域编译，`60da4e33` | 同输入/初始状态重放conv6.601→0.449ms，GDN19.125→0.933ms | conv committed state/projection buffer相同，QKV最大差0.03125，GDN输出最大差0.001953125 |
| prefill静态分桶，`d49bdd14` | 1970 BF16单层74.477→5.473ms，FP32正负编码1.804→0.364s | 输出/状态RRMSE2.93e-7/1.39e-7，307-token FP32误差5.22e-8 |
| FP32有限nilpotent级数并行递推，`41f01e68` | 独立固定MTP5.720→5.261s | 80组CPU代数验证最大误差1.19209e-7，15组HPU状态/输出RRMSE最大8.03901e-8/2.66693e-5 |
| 八行投影/conv/递推扩区，`a8f42df2` | 同轮原区域5.240s，原型4.883s，产品4.867s | 1033行logits相对并行核心top-1为1023/1033，平均KL0.000655902 |

首轮自然完整图131.031→60.063秒、文本96.740→20.156秒，但输出375→303–333 tokens。局部重放提供相同计算的加速证据，自然图保留为实际延迟观察。最初32-token teacher的top-1为31/32，平均KL0.0003614，历史编译改动存在舍入变化。

prefill按至少64和2的幂补齐，1970→2048、307→512。beta/value/log-decay补零使新增chunk为恒等状态更新，最后裁回输出。普通编码FP32与扩写BF16分别检查。自动动态编译曾因symbolic slice的SymEngine `evalf`失败，采用静态分桶。

短递推最多8行，FP32状态保持deferred commit、rollback、任意合法行重排和零decay/beta语义。八行扩区限定普通BF16 operations、标准LinearKV、decode、无hooks/梯度。状态由调用方持有，换权重、交替buffer及禁用编译有回归。

## 整体收益与真实模型故障

`64155fe4`→`a8f42df2`同时包含GDN改进及DiT单head attention。固定MTP完整图21.044→19.978秒，TextGenerate6.109→5.251秒，独立文本5.720→4.867秒，各五轮。两步trace中CPU Linear154→10、launch964→820，活动并集59.396→47.885ms，间隙49.023→37.755ms。短trace用于归因。

真实MTP `3`触发`synStatus26`，异步栈最初落在attention/MLP。逐段同步定位四行GDN，补到八行虽通过离线回放，完整模型仍失败。逐组重建输入后仅激活x经HPU clone或CPU重建有效，产品采用四行x在编译区外clone。状态/gate/sumsq相同，输出RRMSE2.24574e-10。厂商内部原因仍未知。复现入口为`moss_four_input_trials.py`，输入为`performance-2026-11-11-bf16-next/mtp-four-fences/delta-four.pt`。

最终194项广泛回归通过、1项CUDA跳过，含16项GDN扩区测试。512/768/1024、MTP off/3/tree3s1、重载及原INT8完整图通过。全部文本变化相对早期基线logits top-1为1026/1033、KL0.000609445，历史自然图细节有变化。

## 无收益与被替代的方案

| 候选 | 实测 | 判断 |
|---|---|---|
| host pending/parity专门化 | 7.481→7.473s | 无收益 |
| conv＋DeltaNet专门化 | 7.381s | 约1.33%，波动内 |
| 早期整段DeltaNet＋norm | 5.861s，norm单独5.670s | 叠加更慢，递推改写后才重测扩区 |
| pending分桶/精确裁剪 | 5.472/5.464s，对照5.720s | 后续并行递推更优 |
| 共享CPU conv索引 | 5.675s，与并行递推组合5.216s | 无稳定叠加收益 |
| 改动递推编译选项 | 0.335–0.338ms | 无明显收益 |
| sparse-parent gather | 32头8行0.183→0.220ms | 小矩阵改成gather后变慢 |

## TPC四列核心的收益与撤回

2026-11-12在`0995638a`上，将已接受行提交和当前树输出合入一个TPC算子，保留FP32状态、投影/conv/gate/norm及buffer协议，设备读取pending并跳过填充行。32头8行的核心迭代为逐列约0.88ms、vector广播0.34ms、共享Q/K/gate0.254ms、仅提交3行0.207ms、四列并行0.084ms，原编译基线约0.182–0.188ms。pending0时约0.075ms。

固定298-token程序中pending0/1/2/3分别出现2/16/57/55次。完整API各五次干净warm为16.036→15.751秒，文本5.178→4.905秒，独立BF16文本4.804→4.492秒。带logits的warm已剔除，独立FP8文本的4.716→4.387秒只包含四次干净warm。

15组真实GDN输入输出最大RRMSE1.033e-5，状态7.794e-8，gate/sumsq相同，但完整1033行logits的FP8/BF16 RRMSE达到5.68%/1.07%，top-1分别988/1033、1028/1033。自然茶壶出现顶部和侧面双把手。用户要求质量保值后撤回该候选，回退的固定logits与基线逐元素一致。局部小误差和287项回归通过均未替代完整质量判断。[撤回补丁](../patches/moss-gdn-tpc-rejected-2026-11-12.patch)

工具链经验保留如下。

- C++扩展应链接当前eager的`libhabana_pytorch2_plugin.so`。误链lazy插件会重复注册。
- HPU dispatch、`hpu_custom_op_pt2.h`、`custom_op`命名空间、fake metadata和shape inference均影响完整图接入。
- launcher覆盖外部`GC_KERNEL_PATH`曾导致GUID找不到和`synStatus26`，需核实最终库链。
- 本轮编译模式可用VLM为16KiB。20/32KiB缓存及占满后spill的版本构建失败，不能把该报错泛化为设备所有模式的容量。
- 向量化复用和四列并行带来局部收益，后续若重启该方向，应先解决完整logits保值，再评价速度。

## 证据与复测依据

算法、递推结构或运行时改变后可以重测原先失败的融合边界。八行扩区就是递推改写后重新有效的案例。自写核心后续需要定位累积舍入并通过固定树自然质量集合。

原始资料为`performance-2026-11-09-gdn`、`performance-2026-11-11-bf16`、`performance-2026-11-11-completion`、`performance-2026-11-11-bf16-next`、`performance-2026-11-12-optimization`和`performance-2026-11-12-quality-preserving`。TPC源码/构建归档在optimization目录的`rejected-product-source`与`rejected-product-build`。可复用入口为`gdn_replay.py`、`moss_gdn_experiment.py`及`moss_four_input_trials.py`。[BF16证据](../evidence/moss-bf16-next-2026-11-11.json) · [TPC实验](../evidence/moss-optimization-execution-2026-11-12.json) · [回退对照](../evidence/moss-quality-preserving-2026-11-12.json)
