# MOSS BF16 attention与文本解码优化

从产品`64155fe4`继续。用户在利用率复核后明确要求继续深挖attention设备工作、文本解码提交/状态/同步。证据目录为服务器`/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-11-bf16-next`。

最终产品为`a8f42df21a5698c32708d1a2f4d0305019bb9d32`，包含三个独立提交。

- `41f01e68`采用并行FP32 GDN递推，并处理四行HPU激活进入编译区的兼容性。
- `700bdb6d`在合适的长序列形状下按独立head调用现有attention。
- `a8f42df2`将普通BF16八行GDN的外围投影纳入区域编译，覆盖hooks、权重更新与状态回退。

沿用固定caption307/23、1024²、25步、CFG2.5、er_sde/simple，以及1970输入/298输出的完整MTP程序。此前默认/fast/fp32 FusedSDPA、recompute、query tiling、K/V padding、整block和整GDN扩区均已有测量，优先验证新的费用来源。

第一组候选按独立heads拆分现有SDPA，测试24/12/8/6/4/3/2/1 heads每组。每组沿用相同的完整序列、原算子和softmax，目标是观察中间张量规模和调度变化。先测真实正负输入的attention及完整block，再决定是否扩到多层多步和完整工作流。

第二组候选针对文本中尚在eager路径的Linear。前轮两步trace有154次Linear调用，24层GDN×2步×3次外围投影占144次，另外为4次MTP输入FC及6次词表输出头。当前CombinedLinear仅合并符合VBAR条件的量化权重，普通BF16仍使用独立投影。先将这些外围调用单独编译，保持已有conv/GDN/KV区域及控制逻辑，按GDN、head、全部三个范围与新基线对照。

开始前确认8189队列为空，停止独立服务以串行占卡。验收后日常服务已恢复并重放原工作流。

## 局部筛选

独立heads分组均与原输出逐位相等。307-token条件的单head SDPA从3.681降到3.201ms，完整block只从7.090降到6.965ms。23-token条件的完整block6.579→6.605ms。重新排列拼接、折叠batch/head未取得更好组合。复用现有`attention_basic`的post-QK缩放路线更慢且有额外舍入差异。单head候选继续做完整25步测试，避免把局部算子收益当作整图收益。

固定完整MTP程序的新文本基线5.719843s，仅编译外围GDN Linear为5.667515s，仅head为5.674143s，全部为5.656065s。幅度约1%，处于这组运行波动内。

此前每层都回放8个pending slots。完整固定程序实际pending分布为2×57、3×55、1×16、0×2。仅跳过已用decay=1、beta=0中和的尾部槽位，静态分桶为5.472340s，精确长度为5.463734s。此实验通过进程内源码生成进行验证，尚未形成产品实现。

卷积结构索引在CPU每步共享准备一次的候选为5.675497s，单独收益同样较小。与并行GDN叠加后5.216281s，未优于单独并行GDN初测5.039439s。

## 并行短序列GDN候选

当前Kitchen的deferred接口在HPU使用通用Torch回退，已核对现有公开操作，无匹配的原生HPU tree decode实现。新候选使用Torch矩阵操作改写短递推，仍采用FP32递归状态，保留BF16 gate、输出norm和silu的舍入边界。

将每行沿父节点的decay传播写成有限幂级数，再用小型nilpotent矩阵求出DeltaNet修正量。最多8行，用重复平方展开得到有限阶精确代数形式，无除法，允许decay为0。验证树的父状态从实际commit程序恢复，可覆盖行重排。历史pending按链重放，仅历史已接受部分写回state，当前验证行继续保留deferred语义。

80组CPU FP64/FP32代数检查已通过，覆盖1/2/3/4/8行、batch=2、任意有效程序行重排、零decay和零beta。FP32最大绝对误差1.19209e-7。固定完整MTP原型初测5.039439s，相对本轮5.719843s约缩短11.9%。正式实现与后续编译区叠加结果见下表。

真实HPU参考采样为5个decode阶段×3个GDN层，共15组。candidate相对原实现的state RRMSE最大8.03901e-8，输出RRMSE最大2.66693e-5。gate与sumsq全部逐位一致。CPU FP64参考下，candidate输出误差为原误差的0.99997564–1.00000068倍，state误差最大5.87472e-8，原实现最大6.85577e-8。完整固定MTP程序的1033×248320 logits，top-1为1024/1033一致，平均KL0.000597717，RRMSE0.0107172。逐层小差异在多步BF16舍入后会改变自然采样轨迹。

## Attention完整工作量复核

按基线A、候选、基线B顺序，各1 cold＋5 warm，固定caption缓存整图中位数为14.638745、14.401615、14.637128s，DiT为14.386597、14.150001、14.388378s。原型净收益约0.237s、1.6%。进入产品时仅对已验证的HPU BF16、batch=1、24×128 heads、N≥4096、无mask/参考缓存、默认attention的编译路径按独立head调用原优化dispatcher。每个head继续带`AttentionTensorContainer`与`preferred_attention`，产品没有全局替换SDPA。

第一次完整head原型被Dynamo的模块guard拒绝，错误为`module '__main__' has no attribute 'sdpa_original'`。原因是多层runpy使用`__main__`切换，已将实验wrapper迁入独立可导入模块后重跑，失败日志保留在`head-candidate-main-guard-failed*`。

## 回归中的执行环境核对

第一次合并运行文本和MOSS测试出现4项失败。收集期已有CPU测试把全局attention选成sub-quadratic，HPU专项因此偏离日常启动的PyTorch backend。原提交`64155fe4`与当前候选在三种小shape场景的编译输出逐位相同，CPU FP64核验也相同，证据为`attention-test-audit*.json`。更新HPU attention测试fixture明确选择日常HPU路径的PyTorch attention，原数值阈值保留，专项10项全部通过。首次失败和审计日志一并保留。

合并重跑为163 passed、1 CUDA专属skip。覆盖新增并行公式、FP32/BF16、batch=1/2、1/3/8行、pending达到全部槽位、非顺序接受分支、权重更新、动态长度、CPU分流、hooks、参考KV和mask。产品的15组真实GDN参考误差与原型相同。

## 正式产品工作量验收

各1 cold＋5 warm。固定MTP程序通过强制既有token和控制决策维持同工作量，沿用并验证全部258次draft top-k、130次sample、4次宽度决策和130次控制行检查，最终298个token ID全部相同。自然生成的文本变化另行检查。

|工作负载|本轮基线|正式候选|耗时下降|
|---|---:|---:|---:|
|独立固定MTP文本|5.719843s|4.866763s|14.91%|
|固定MTP完整工作流|21.043838s|19.978488s|5.06%|
|固定MTP的TextGenerate节点|6.109491s|5.250935s|14.05%|
|固定MTP的KSampler节点|14.368867s|14.133343s|1.64%|
|固定caption缓存整图，产品调度路径复验|14.627998s|14.386117s|1.65%|
|同一固定caption的DiT|14.375917s|14.134462s|1.68%|

完整cold为58.163453→64.665240s，新增编译成本需要计入首次使用。最终完整warm范围19.905970–20.121573s，缓存14s目标还差0.386117s。固定caption产品复验与前一组A/B/A共同确认小幅稳定收益。较早仅采用并行GDN和head分组的中间版本为独立文本5.261498s、完整20.339359s。

同一固定caption的attention产品图片MAE为0.09832/255，基线自身重复为0.10406/255，候选自身重复为0.10419/255，三者最大像素差均为5。这组对照隔离了扩写文本变化，差异与重复运行波动相当。

最终自然扩写的三组图片已逐张检查。红色陶瓷茶壶、蓝色花瓶/MOSS大字、红围巾老人/白茶杯等主体保持。海报候选画出三朵黄花，基线画出四朵。人物与茶壶的光线、衣物、背景和构图细节变化明显。对应caption长度为339→340、260→303、289→312，图片MAE30.918–69.111/255，逐像素差异同时包含扩写文本的变化。图片检查范围为这三组样本。

## 四行MTP验证输入的修复

自然质量流程切换到MTP `3` 时触发`synStatus 26`。异步错误最初分别出现在attention与MLP调用栈，逐段同步定位到新并行GDN的四行decode。真实输入保存在`mtp-four-fences/delta-four.pt`。将attention/MLP改为静态编译的隔离实验仍失败。

补齐到8行的候选在离线回放及31项单元测试中通过，完整MTP `3`仍报同样错误。逐组把原HPU输入经CPU重建，state、qkv、权重、控制和侧缓冲等组仍失败，单独重建激活`x`成功。HPU内部直接clone `x`同样成功。使用原始四行公式加HPU clone，完整自然生成551 tokens通过，fallback为空，因此撤回了补齐方案。

正式实现仅在四行输入进入编译函数前执行`x.clone()`，保留调用参数及梯度连接，临时张量由本次调用持有。实际内核内部触发机制尚未由厂商确认，已用明确的模型入口复现与输入隔离实验约束处理范围。新增测试验证clone发生在编译边界外，并覆盖1–8行、真实32 heads、FP32/BF16、batch=2和GQA分组。每个硬件参数组重置Dynamo，确保所有形状实际进入编译路径。真实MTP `3`和`tree3s1`完整出图已通过。

最终四行算子回放相对原Kitchen的输出RRMSE为2.24574e-10，最大绝对误差1.86265e-9。state、gates、sumsq逐位相同。`moss_four_input_trials.py`使用`MOSS_FOUR_RAW_INPUT=1`绕过正式输入隔离，保留失败复现入口。

## 递推重构后的编译区复测

同一组真实输入的原型、正式函数、额外`force_static_compile`均约0.335–0.338ms，编译选项差异处于波动内。改写递推后，更大GDN区域在固定程序中达到4.883419s，同轮原区域复测5.240083s。将该区域落实为普通BF16、8行、标准LinearKV、无hooks且关闭梯度的专用分流后，正式独立文本为4.866763s。

增量区域的1033行logits top-1为1023/1033，平均KL0.000655902。最终全部文本改动相对本轮原始基线为1026/1033，平均KL0.000609445。冷/热candidate与baseline各自的重复logits逐位相同。

两步固定MTP trace记录了CPU侧Linear调用154→10、`launch`事件964→820。设备kernel区间并集59.396→47.885ms，间隙49.023→37.755ms，总跨度108.419→85.639ms。kernel事件包含硬件lane，计数与launch分开保存。短trace用于观察调度变化，完整耗时采用无profiler的五次warm结果。

## 最终验收与部署

- 最终整组回归194 passed、1 CUDA专属skip。新增区域专项16项通过，已计入整组总数。
- 三组512/768/1024图片、卸载重载、MTP off/3/tree3s1和原INT8完整图均成功，latent/VAE检查finite，fallback为空。
- 同工作量完整pipeline的allocator峰值为基线39.4984GiB、最终39.3754GiB。各自卸载稳定2,685,928,832 bytes。多形状及INT8质量流程最后卸载为2,686,190,976 bytes。
- 六份产品文件SHA256与服务器一致。部署标记为`a8f42df2`，8189原图cold重放66.725s，后续缓存请求14.751s，TextGenerate和正负编码命中正常节点缓存。最终只读检查确认设备为`hpu:0 GAUDI2`，运行与待执行队列均为空。
- 结果与文件哈希保存在 [紧凑证据](evidence/moss-bf16-next-2026-11-11.json)，性能、数值、失败路径和真实输入可按证据目录复查。

这轮同时减少了递推的设备工作和外围CPU提交。当前缓存目标的差距为0.386s，完整工作流在20s附近波动。attention设备费用及文本运行时间隙仍有余量，硬件峰值判断继续结合真实shape测量与分段trace。
