# Qwen无MTP执行优化逐项验证

2026-11-15。按用户要求逐项验证GDN状态更新、完整attention及整步执行，再验证组合的真实工作流收益、自然出图和生命周期。产品基线为用户已接受的`dfca7cd6`。候选通过调查进程安装，普通服务恢复相同产品源文件。

## 完整工作流结果

沿用原1088×1920、8步官方Turbo调度、CFG1.2、thinking开启的工作流及同一BF16 PE文件，MTP调用为0。固定teacher轨迹为2803输入、1784输出tokens，两侧正常执行模型与采样，再代入相同目标token。每侧1次cold、2次warm，表中warm取两次中位数。

| 指标 | 当前产品 | GDN直接更新＋SDPA＋整步编译 |
|---|---:|---:|
| 完整warm秒 | 60.6615 | 44.1780 |
| TextGenerate秒 | 41.6929 | 25.1831 |
| PE tokens/s，含prefill与节点费用 | 42.789 | 70.841 |
| 采样秒 | 16.5826 | 16.5832 |
| cold整图秒 | 96.804 | 111.894 |
| 两次完整warm | 60.720 / 60.603 | 44.236 / 44.120 |

PE耗时降低39.60%，吞吐提高65.56%，完整图耗时降低27.17%。固定teacher的扩写和RGBA输出与新鲜基线逐元素一致，新鲜基线也匹配此前已验收的第1张产品图。实际自然采样的质量另见后文。[结构化证据](../evidence/qwen21-execution-2026-11-15.json)

cold增加15.090秒，包含整步编译的首次成本。当前整步图采用静态KV容量，首次遇到新的容量可能再次编译。本轮热运行与自然质量样本使用同一个原始提示词，输入长度相同。

## 逐项与组合筛选

首先使用已有`qwen21-user-bf16-api.json`的427-token固定轨迹，每项1次warm-up、3次无profiler正式计时、1次独立完整logits导出。该短用例与上面的原始2MP长扩写分别报告。

| 路径 | 完整固定文本秒 | 对10.2283秒基线降低 | logits RRMSE | top-1相同/427 |
|---|---:|---:|---:|---:|
| 当前产品 | 10.2283 | — | 0 | 427 |
| 已有单token GDN TPC核心 | 9.6415 | 5.74% | 0.9861% | 421 |
| GDN直接状态更新 | 9.3472 | 8.62% | 0.9851% | 422 |
| 厂商SDPA，FP32 softmax | 10.1686 | 0.58% | 0.9750% | 423 |
| 原attention＋1024-token KV分桶 | 10.1166 | 1.09% | 0 | 427 |
| 当前数学路径＋整步编译 | 9.9089 | 3.12% | 0.9846% | 421 |
| GDN直接更新＋SDPA | 9.4188 | 7.91% | 1.0034% | 422 |
| GDN直接更新＋整步编译 | 6.6444 | 35.04% | 0.9829% | 420 |
| GDN直接更新＋SDPA＋整步编译 | 6.5333 | 36.13% | 0.9883% | 419 |

主收益来自GDN简化与整步执行共同作用。SDPA在最终组合上仅带来约1.67%的短用例增益，这个小差值仍需要更多交错样本才能确认稳定幅度。分桶增加首次遇到桶容量时的编译工作，本轮保留为单项实验。已有TPC候选速度低于新的直接更新，完整图阶段采用表中最快组合。

所有logits有限。最终组合平均`KL(reference || candidate)`为0.000601，首个prefill输出行逐元素一致。完整浮点输出存在变化，后续自然扩写和图像检查继续执行。

## 实现与适用范围

### 单token GDN直接状态更新

新核心保留BF16投影和卷积舍入、FP32 recurrent state以及现行gated norm公式。每个token立即更新conv history及recurrent state，执行单步rank-1更新与归约。单token路径省去了通用八槽deferred/tree结构中的历史重放、索引、padding及小矩阵幂运算。

该实现使用现有torch操作及`hpu_backend`编译，复用产品加载的权重与cast上下文。它改变了调查cache内部状态的提交时机，闭合在无MTP的普通自回归生成中。当前调查范围为匹配PE、batch1、单token解码，正常产品接入需要明确cache类型及其它调用路径的选择。

### 完整attention

比较了原FP32参考attention的1024-token容量分桶，以及已安装厂商`FusedSDPA`的BF16 Q/K/V、FP32 softmax。最终组合使用后者，按设备seqlen构造有效前缀mask，保持完整容量及现有Q/K norm、RoPE和输出gate。

研究实现的SDPA mask针对单token解码。多行验证、树形MTP和其它attention配置需要各自入口及回退协议。

### 整步执行

完整32层和最终norm放入一个`torch.compile(fullgraph=True)`入口。真实cache的prepare、位置和advance仍由生成调用维护。传给图的浅层cache代理共享当前请求的张量buffer，固定Python索引以避免逐token值触发recompile。代理仅存在于这次generate调用，finally恢复原forward。

prefill沿用已有区域编译。外层整步图追踪时，内部区域入口返回原tensor函数。该实现已通过真实ComfyUI请求、多个seed、512²及卸载重载，但仍属于调查进程中的候选安装方式。

## HBM与执行事件复测

复用此前已经用copy/fill/GEMV校准的16端口AXI返回beat口径，capture_window=16383、bandwidth_window=255、128 bytes/beat。使用原2MP提示词，4096输出上限，在第512至528次采样之间采集16轮，最终强制560-token结束以缩短诊断时间。正式窗口采用无profiler时间，计数字节来自独立profile轮。

| 指标 | 已验产品此前校准 | 本轮候选 |
|---|---:|---:|
| 读取GB/token | 17.1680 | 16.3224 |
| 写入GB/token | 1.2609 | 0.2400 |
| 总读写GB/token | 18.4289 | 16.5625 |
| 无profiler晚位置ms/token | 23.0122 | 14.1656 |
| 有效TB/s | 0.8008 | 1.1692 |
| 相对2.45TB/s | 32.69% | 47.72% |
| Recipe Start/token | 306 | 84 |
| TPC执行事件覆盖ms/token | 7.627 | 2.116 |
| MME/TPC/EDMA事件未覆盖ms/token | 8.370 | 5.272 |

读写量降低10.13%，写入量降低约81%。新counter取整敏感性上沿对应48.31%峰值带宽。候选profile窗口0.23466秒，前后两个无profiler窗口为0.23077/0.22665秒，本表采用后者，与旧汇总的单窗口口径对应。

新trace中约3.929ms/token位于recipe区间内且未被上述执行事件覆盖。事件空档仍需关联具体依赖、调度与其它活动，当前数据保留事件覆盖口径。纯HBM传输预算约6.76ms/token，实际晚位置约14.17ms，剩余执行费用仍值得继续定位。[原HBM校准方法](QWEN21_BF16_SINGLE_TOKEN.md)

## 自然出图与生命周期

自然质量阶段关闭teacher代入。三组1088×1920分别使用原提示词及连续三个配对文本/图像seed，另补一组512×512。前三组基线复用用户刚验收的原PNG和history，新鲜原seed基线确认与旧图一致。512²基线本轮新生成。所有history核验TextGenerate与SamplerCustomAdvanced实际重算。

| 自然用例 | 候选输出tokens | 整图秒 |
|---|---:|---:|
| 原seed，1088×1920 | 1867 | 45.285 |
| seed＋1，1088×1920 | 2115 | 49.222 |
| seed＋2，1088×1920 | 1717 | 43.261 |
| 原seed，512×512 | 1867 | 29.329 |
| 卸载后重载原seed，1088×1920 | 1867 | 71.608 |

四组候选均成功自然结束。扩写文字、姿态与构图发生变化，RGBA与旧版对应图不同。人工预览已检查三组2MP概览与512²并排图，主体、花束和蜂鸟均可见，最终视觉质量接受仍由用户检查。

原seed候选在卸载重载后扩写及RGBA逐元素一致。卸载后10/20/30/40秒的占用逐步下降，40秒为9.08GiB，随后进入重载。该轮未采集基线的同条件稳定卸载值。完整固定工作流记录的进程峰值约72.89→72.92GiB，后续跨caption样本最高约73.05GiB，峰值含进程累计高水位。

检查材料位于本地`results/qwen21-execution-2026-11-15/review`，包含8张原PNG、扩写文本、4张原尺寸并排图、概览及离线HTML。每个并排面板已用ImageMagick AE=0核验原图像素。ZIP完整性及8张原图计数通过。

## 调查启动器问题与恢复

首次完整API请求在Dynamo解析诊断包装函数的`__main__.mode`时失败。嵌套runpy替换了main模块，改为稳定模块导入后继续验证。

第二次启动把服务循环直接置于模块导入期间，解码worker随后停在Python锁等待。C栈已保存，系统gdb缺少`py-bt`命令。结合启动结构推断为导入锁等待。将服务启动移入`main()`，在模块导入完成后显式调用，第三次相同请求及全部后续请求成功。两个失败目录和jobs记录均保留。

独立整步文本probe曾在解释器退出期报告`ModelPatcher.__del__`访问已清除的callbacks。补充显式同步、Dynamo reset、模型卸载及GC后，64-token cold/warm复测正常退出且该告警消失。普通API的卸载重载已独立通过。

每个串行批次均先检查空队列、停止普通服务，在finally恢复。最终普通8189服务可访问、队列为空，部署标记及5个产品源文件SHA256与实验前一致。产品源码保持用户接受的`dfca7cd6`。

## 证据与复现

- 紧凑结果、全部样本、logits指标、质量request ID、源文件哈希、恢复信息和ZIP摘要见[证据](../evidence/qwen21-execution-2026-11-15.json)。
- 服务器根目录为`/home/sonny/gaudi2/comfyui-hpu/performance-qwen-execution-2026-11-15`。`screen/`、`combined/`、`workflow/`、`hbm/`、`cleanup/`分别保存原始输出，`workflow-failed-main-module/`和`workflow-failed-import-lock/`保留失败资料。
- 候选安装入口为`probes/qwen_execution_candidates.py`。`qwen_execution_jobs.py`管理串行执行、空队列检查及服务恢复，`--stage text --logits --teacher <json>`复现固定文本，`--stage workflow --teacher <json> --case-graphs <json> --reload`复现完整流程与自然质量，`--native`复现晚位置计数。根目录中的两个case JSON和`latest-teacher.json`保留了具体工作量。
- `qwen_execution_summary.py`比较CPU logits与计时，`qwen_execution_review.py --collect`从history收集自然图，省略`--collect`则在有ImageMagick的本地生成检查页与ZIP。原生计数及时间线继续复用已有`qwen_hbm_summary.py`和`qwen_native_timeline.py`。
