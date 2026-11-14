# 通用 INT8 模型的 Gaudi2 算子开发计划

2026-11-14（Asia/Shanghai）。计划状态为已停止、已放弃。用户后续明确要求仅在具有原生INT8 MME能力时继续。安装库HAL的MME类型掩码`0x7906`排除INT8，直接Synapse的9组INT8 GEMM均在节点创建阶段失败，3组BF16对照编译成功，已据此关闭计划。[能力验证](INT8_NATIVE_MME.md) · [证据](../evidence/int8-native-mme-2026-11-14.json)

以下保留停止前收集的经验和设计，供历史查阅。原基准环境为Gaudi2 HL-225、Habana1.24.2、Habana torch2.13.0a0、eager＋区域编译，产品基线`ed9552e5`。

目标是让具有明确量化语义的INT8模型通过通用算子使用Gaudi2设备计算，提高实际推理效率，并记录精度、显存和加载成本。算子选择依据格式、shape、dtype及设备能力，模型名称和文件名仅用于验收样本索引。首阶段按推理场景规划，首先交付Linear/GEMM，再推进Embedding、batched matmul、卷积及专家层。

## 1. 硬件与软件前提

此前将INT8慢路径概括为“缺少软件适配”遗漏了硬件能力核验。仓库`gaudi2-hardware.md`已记录这一边界，本轮再次核对官方资料。

| 层次 | 当前证据 | 开发含义 |
|---|---|---|
| Gaudi2 MME | 官方列FP32、TF32、BF16、FP16、FP8，累加为FP32 | 原生INT8输入的MME路径作为待证能力，首个里程碑核查SDK、lowering和设备trace |
| Gaudi2 TPC | 官方列INT8/INT16/INT32和多种浮点类型 | 可开发整数运算、反量化、旋转、量化和epilogue融合，吞吐需独立测量 |
| 官方PyTorch算子表 | Integer Types中`mm`、`matmul`的INT8栏为No | 现有高层矩阵接口需要专门验证和适配 |
| 当前本机 | 历史`torch._int_mm`出现CPU fallback。BF16 GEMM及`fp8_gemm_v2`已有真实模型验收 | BF16/FP8可作为硬件矩阵执行基础，整数路线保留独立能力与精度契约 |
| 当前SDK头文件 | `/usr/include/habanalabs/synapse_common_types.h`含`syn_type_int8`等枚举 | 继续查对应设备、节点GUID、输入输出类型和编译支持 |

官方来源于本轮读取，文档显示1.24.2。白皮书第6页说明MME类型，第10页说明TPC整数类型。架构和算子表共同支持将原生INT8 MME列为待核验项。[Gaudi2架构](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html) · [Gaudi2白皮书](https://habana.ai/wp-content/uploads/pdf/2021/gaudi2-whitepaper.pdf) · [PyTorch算子支持表](https://docs.habana.ai/en/latest/PyTorch/Reference/Pytorch_Operators/Pytorch_Operators.html) · [仓库硬件记录](../../gaudi2-hardware.md)

通用INT8后端需要分别描述checkpoint存储格式、激活量化规则、实际乘法类型、累加类型及输出dtype。INT8文件可以使用设备端解码和BF16 MME执行，也可以经过校准转换使用FP8 MME。整数输入/INT32输出的算子另有精确累加要求。各路线共享格式解析和生命周期管理，能力结果决定其适用范围。

## 2. 仓库经验收集与复用

仓库主要完成的是BF16和FP8适配。FP16线索包括官方MME类型、外部attention中间score，以及本栈FP16→FP8 cast失败记录。FP16需要补齐本机primitive和模型验证。

| 已有经验与证据 | 可直接复用 | INT8开发中的约束 |
|---|---|---|
| FP8最初每层解码，DiT约205.8s，接入原生Linear后完成同源BF16/FP8对照 | `MixedPrecisionOps.Linear`、`CastBiasWeightContext`、`hpu_fp8_linear`，设备和bias dtype传递 | 从普通loader/forward验证实际命中，记录解码次数、设备计算类型和fallback。见[FP8接入](DIT_FP8.md) |
| FP8文件范围448与Gaudi2范围240，曾产生inf/nan。安全权重范围224 | 全编码解码测试、metadata保留、来源及重载检查 | INT8检查signed/unsigned、零点、scale轴、group尾部和舍入。FP8转换沿用已有范围处理。见[正确性适配](../ADAPTATION.md) |
| FP8文本曾因force-cast走慢路径，完整流程740.685s。修复后主要收益为约7.51GiB峰值下降 | 量化权重与计算dtype分离、显存/时延分别报告 | 低位存储、设备矩阵调用和完整模型速度分别验收。见[文本精度](TEXT_PRECISION.md) |
| BF16两段编译、静态recipe有效，连续变长会改变性能 | 模型所属编译区、shape复用、混合shape测试、hooks门控 | INT8 backend显式提供可编译能力，按格式及dtype验证，避免通过模型全局单一格式判断所有层。见[BF16分区](DIT_BF16.md) |
| BF16/FP8融合会改变norm、调制、残差和scale舍入。全FP8 QK曾有18.5%–48.1%误差 | CPU FP32/FP64参考、首个偏离点定位、敏感边界 | 先保持量化模型现有数学语义，再单独评估新增激活量化或FP8重编码。见[Attention](ATTENTION.md)、[全BF16验收](../QWEN21_BF16.md) |
| 输出头绕过层入口曾产生CPU fallback。权重合并依赖成员scale和生命周期 | `BaseGenerate.logits`层调用、`CombinedLinear`及其已有测试 | Linear、lm_head、tied embedding和合并QKV均从正常入口命中，保留每个输出通道scale。见[文本精度](TEXT_PRECISION.md) |
| 小M FP8额外编译产生舍入及重载`synStatus26`。KV同storage复制需clone | 交替buffer、重复请求、换权重、卸载重载测试 | 性能probe与真实生命周期共同验收，异步故障追踪最早DFA。见[运行时](RUNTIME.md) |
| GDN TPC局部约0.185→0.084ms，完整logits仍变化，候选撤回 | eager扩展构建、host glue、fake/meta、GUID库链、TPC lane与VLM经验 | 手写kernel需要完整logits/自然输出验证，构建按本机ABI。见[GDN](TEXT_GDN.md) |
| FP8大Linear局部活动约806TF/s。低扰动文本trace显示大量调度与空档 | 引擎区间并集、host/设备关联、读取写入预算 | decode、prefill、DiT采用各自shape测量，TPC与MME重叠重新确认。见[预算](BOTTLENECKS.md) |
| 自然生成长度、缓存、加载会改变总时间。profiler及logits导出会扰动 | 固定token/固定caption、cold＋5 warm、A/B/A、独立诊断 | 正式速度数据使用相同工作量与实际执行断言。见[测量方法](MEASUREMENT.md) |

### 可复用代码与资料入口

以下产品路径相对于兄弟`ComfyUI-Gaudi`仓库，实际开发继续遵循该仓库的AGENTS。

- `comfy/quant_ops.py`已有QuantizedTensor、layout注册、QUANT_ALGOS及HPU FP8入口。INT8元数据和公共数学放在这一层附近。
- `comfy/ops.py`已有量化加载、Linear、Embedding、CombinedLinear、cast上下文和保存流程。设备能力与执行方式在这些入口接入。
- `comfy/model_management.py`已有设备能力查询，当前`supports_int8_compute`的含义需保持清晰。权重格式支持、浮点解码执行、精确整数计算各自记录能力。
- `comfy/sd1_clip.py`、`comfy/text_encoders/qwen35.py`展示目前文本加载和编译门控，改造通用入口后以这些调用者验证。
- `tests-unit/comfy_test/hpu_fp8_linear_test.py`、`hpu_quant_test.py`、`combined_linear_test.py`、`qwen35_decode_test.py`覆盖字节编码、metadata、patch、输出头和合并权重，可扩展对应回归。
- 本仓库`probes/convert_qwen35_int8_to_bf16.py`提供原INT8解码参考。产品`scripts/gaudi/convert_qwen35_fp8.py`提供原子发布和来源记录经验，其模型键名筛选需要改为格式适配后再用于通用转换。
- `probes/hpu_fp8_shape_benchmark.py`提供同步、重复测量及fallback增量框架。新增整数测试应增加随机/极值输入、结果消费和CPU INT64参考。
- `probes/moss_fp8_logits_compare.py`、`moss_fp8_text_quality.py`、`qwen21_text_profile.py`和固定MTP资料可用于模型层诊断。

## 3. 通用格式和算子契约

### 格式覆盖顺序

| 范围 | 首轮语义 | 扩展方式 |
|---|---|---|
| 普通signed INT8权重 | per-tensor、per-output-channel scale，浮点激活W8A16 | 先复用当前`int8_tensorwise`的真实metadata，名称与scale粒度逐项核实 |
| ConvRot INT8 | 读取rotation类型、方向、group及归一化，覆盖16/64/256和合法尾部 | 作为可选权重变换插件，使用标准Linear测试和多种模型调用 |
| W8A8 | 静态或逐行动态激活scale、INT32累加及浮点/重定标输出 | 单列激活量化契约、舍入、饱和和零点处理 |
| affine INT8/UINT8 | per-tensor/per-channel零点、scale，group沿K轴 | 使用显式axis、group_size及尾部有效长度，验证零点修正 |
| 混合量化模型 | 每层INT8/BF16/FP16/FP8组合，bias/norm保持其契约 | 按层能力分派，记录每种执行路径覆盖率 |
| 外部checkpoint格式 | 已有Comfy metadata先交付，后续逐个接compressed-tensors、torch.ao等实际格式 | adapter负责格式版本、axes、packing与来源，先建立可复算的小fixture |
| outlier分解、特殊packing、W4A8/W6A8 | 解析格式自身的残差支路或低位编码 | 接入相应adapter与测试，复用兼容的INT8核心 |

适配器输出使用已有量化表示，并补齐实际需要的字段，包括logical shape、storage dtype、scale与zero point的轴、group、变换、compute/output dtype及版本。未知格式在加载边界给出具体信息，已支持格式保持现有回退能力。通用性按格式和算子矩阵逐项公布。

### 数学与数值契约

令`D(q, scale, zero_point, transform)`为该格式规范的权重解码。W8A16参考计算为`Y = X @ D(Wq).T + bias`，舍入位置与现有格式实现一起固定。参考对象使用同一份量化权重，来源浮点模型的误差另列。

W8A8对逐行激活量化和逐输出通道权重量化满足以下关系。`Qa`形状为`[M,K]`，`Qw`为`[N,K]`，激活零点`za[m]`、权重零点`zw[n]`在当前K归约段内固定，对应scale为`sa[m]`、`sw[n]`。

```text
C[m,n] = sum_k(Qa[m,k] * Qw[n,k])
         - za[m] * sum_k(Qw[n,k]) - zw[n] * sum_k(Qa[m,k])
         + K * za[m] * zw[n]
Y[m,n] = sa[m] * sw[n] * C[m,n] + bias[n]
```

groupwise scale在各组完成修正和缩放后归并，沿K逐元素变化的scale/zero point按其实际轴解码或细分归约。量化padding填入该量化域中的零。整数累加与修正项采用宽参考检查溢出，明确生产算子的有效范围和溢出行为。舍入模式、clamp范围、零输入、极小scale、非有限激活和输出类型均进入fixture。

ConvRot中若`W_rot = W R`且R正交，实数公式可写为`Y = (X R) @ W_rot.T`。按checkpoint的scale轴与group边界判定该变换是否合法。在线旋转会改变浮点舍入与激活量化误差，和“按原定义还原权重再计算”分别建立数值对照。Hadamard、归一化系数和尾部语义均由metadata决定。

### 计划交付的计算单元

1. 权重准备与设备反量化，覆盖scale、零点、分组、ConvRot、layout/padding，提供可验证的浮点输出。
2. 激活量化及可选旋转，提供整数值、scale/zero point，合并有实际收益的归约和写回。
3. 矩阵核心，分别注册或选择已验证的MME路径与TPC整数候选，清楚声明乘法、累加和输出类型。
4. epilogue，处理零点补偿、group归并、bias、输出转换/重定标及裁剪。
5. 通用Linear与Embedding入口，使用既有权重所有权和patch流程，再将矩阵核心复用于bmm、卷积和专家层。

函数和模块的数量随实际共享逻辑确定。初期可在现有ops/quant_ops旁增加一个小型HPU INT8实现模块，底层接口接收tensor与量化参数。

## 4. 执行路线与选择条件

| 路线 | 计算和容量特征 | 开发位置与验收重点 |
|---|---|---|
| 原生INT8 MME，能力待证 | INT8乘法及声明的累加类型 | P0取得可调用API、设备执行和正确性证据后接入公共矩阵接口 |
| W8A16设备解码＋BF16/FP16 MME | INT8存储，浮点矩阵计算。解码表示增加临时或常驻容量 | 复用厂商GEMM，测准备一次、每层按需和分块流水三种成本 |
| 浮点MME实现整数点积，研究候选 | INT8数值可精确表示为BF16/FP16，累加和输出仍需证明 | 核查FP32输出与累加路径，进行极值、消去和大K整数精确性测试 |
| TPC整数GEMM/GEMV，研究候选 | 设备整数运算，适合性随M/N/K与访存改变 | 从M=1/2/4/8筛选，计入tile、packing、epilogue和整体带宽 |
| INT8来源转换FP8 MME | 权重重编码与激活FP8量化，通常保留低位设备存储 | 复用已有FP8入口，记录额外量化误差及校准，作为独立精度策略交付 |

浮点MME整数点积路线需要特别验证输出。INT8元素的可精确表示性只覆盖输入。FP32对整数精确表示的范围、累加归约顺序、途中舍入和输出dtype共同决定结果。按有符号INT8最坏乘积16384估算，K分块上限1024使绝对乘积和不超过2^24。该界仅作为实验起点。分块结果需要FP32精确输出，再按范围转换并累加到整数域，所有额外buffer/转换计入时间。使用BF16输出后再转FP32会丢失这一保证。UINT8及零点中心化后的更大取值重新推导分块界。

以上路线按语义兼容性和实测成本选择。引入新激活量化或FP8重编码的候选通过独立配置和质量报告启用。默认格式行为以已验参考为基准，当前可执行路径继续提供覆盖。

TPC公共CustomOp适合实现归约、旋转、反量化和epilogue。官方`UserCustomOpDescriptor`描述单个TPC kernel，MME与TPC多阶段融合需要评估bridge建图、编译器及调度接口。当前eager扩展链接`libhabana_pytorch2_plugin.so`，按文档提供`custom_op`注册、Meta及连续布局处理。[CustomOp API](https://docs.habana.ai/en/latest/PyTorch/Reference/PyTorch_CustomOp_API/page_index.html) · [固定bridge注册](https://github.com/HabanaAI/gaudi-pytorch-bridge/blob/27eccbcffeab10842c9f29f64c6bf50d8f65d78f/scripts/hpu_op.yaml) · [已有工具链经验](TEXT_GDN.md)

## 5. 停止前的分阶段设计

### P0　能力核验与成本基线

- 固定驱动、torch、bridge、TPC编译器、comfy-kitchen版本和源码hash。沿已有`27eccbcf`bridge审查`mm/matmul`、quant/dequant、FP8 GEMM，核对本机SDK类型与节点支持。
- 设计独立小测试覆盖`torch.mm`、`matmul`、`_int_mm`、可用Synapse GEMM和TPC整数候选，保存结果dtype、输入/输出值、fallback增量、host拷贝及MME/TPC时间线。
- 单独验证BF16/FP16 GEMM、FP16相关cast、FP32累加输出能力。编译接受、结果正确和设备引擎使用各自给出证据。
- 复现最新INT8 Linear的反量化/旋转/矩阵乘/调度分项，使用现有同源BF16常驻作为对照，并从同源权重构建FP8候选。
- 能力核验已交付`evidence/int8-native-mme-2026-11-14.json`及Synapse probe。按最新用户条件，阴性结果已触发整体关闭，其余阶段转为历史设计归档。

### P1　通用量化表示与参考算子

- 从现有`QuantizedTensor`和layout读取元数据，为普通INT8、ConvRot、per-channel/group、非零zero point建立小fixture及CPU FP64/INT64参考。
- 固定Linear输入`[..., K]`、权重`[N,K]`、输出`[...,N]`和bias契约，补空输入、非连续输入、tail、混合dtype及张量共享测试。
- 定义W8A16、W8A8、FP8转换三种执行语义的能力记录，复用既有loader和层对象传递。
- 验收量化decode、scale广播、零点与group参考逐项通过，同shape换值/换权重能够改变结果。

### P2　通用Linear首版与重复准备消除

- 将已有BF16解码经验接到通用Linear入口，首先消除同次生成中的重复权重还原，保留当前格式的舍入定义。
- 优先采用现有加载期权重准备机制，或由顶层执行上下文拥有的有预算临时表示。跨请求准备结果若需要保留，则作为现有模型管理体系中的正式权重表示，参与加载、patch、显存估算、保存与卸载。
- 明确每层INT8本体、scale、解码权重、workspace和recipe占用，权重/scale/zero point/LoRA变更均使准备结果失效。设备、shape、stride、dtype和变换参数也纳入有效性判定。
- 基于相同契约接入P0证明可用的矩阵路线。通过能力门控允许已验Linear进入区域编译，合并QKV、MLP和lm_head分别检查。
- 普通CLIPLoader与UNETLoader入口验证混合INT8/BF16/FP8模型。输出头和tied embedding沿同一权重所有权运行。
- 交付通用Linear产品改动和针对性回归。报告每token反量化次数、固定工作量时延、峰值及清理后稳定占用。

### P3　低位常驻和小M优化

- 对保留8-bit设备权重的路线测量反量化tile与MME流水、TPC GEMV及FP8转换，权重访问按真实层轮换，记录转换后的额外字节。
- W8A8开发激活量化、零点修正和输出重定标。ConvRot分别测试准备期还原和在线激活旋转，保留两侧完整数值与质量结果。
- 先组合已有Torch/厂商原语进行区域编译，trace确认独占热点后再写TPC融合。每个候选报告整体调用成本、workspace及新增kernel数量。
- 小M与prefill分别选择tile及编译shape策略，通过连续长度切换、tail和非连续布局验收。编译失败采用已验路径并记录具体原因。
- 交付按格式/shape/dtype的实测选择表，低位驻留路线与BF16容量路线分别提供时延和显存数据。

### P4　跨算子与跨模型验收

- Embedding按索引读取并还原需要的行，检查重复索引、tied lm_head和权重更新。
- batched matmul覆盖batch广播、transpose、stride和多头布局。量化attention的QK/softmax/KV单独建立精度契约。
- Conv1d/2d/3d验证stride、padding、dilation、groups/depthwise，比较设备解码＋原生卷积与降低到GEMM的临时内存和性能。
- 专家层验证expert维scale、token路由、空专家与grouped GEMM，复用公共矩阵核心及各层格式adapter。
- 验收模型采用不同结构，覆盖Qwen3.5/GDN、Qwen3-VL/标准attention和DiT。优先复用服务器已有组件，对缺少的普通INT8格式使用可复算fixture及已有浮点层构建测试权重，并明确其生成方式。
- 交付格式×算子×dtype×模型验收矩阵，完成独立checkout正常API、自然生成、清理后重载及回归测试。

## 6. 测量与接受依据

### 矩阵形状和输入集合

以真实模型捕获的M/N/K/stride为主，并补以下边界。矩阵组合按代表场景选取，避免对全部尺寸做无意义笛卡尔积。

| 维度 | 覆盖 |
|---|---|
| M | 1/2/4/8 decode，16/32/64/128中等batch，512/2048/4096及真实长序列prefill/DiT |
| K/N | 模型hidden/FFN大小，256/1024/4096/8192/16384，真实大词表输出头，以及对齐值±1 |
| layout | 连续/转置/非连续stride、rank2/3/4输入、batch1与多batch、同shape交替buffer |
| dtype | BF16/FP16/FP32激活与输出，INT8/UINT8存储，INT32累加输出及CPU INT64参考 |
| metadata | tensor/channel/group scale、zero point、16/64/256旋转组、尾部、旧metadata及混合格式 |
| 数值 | 全零、全部极值、正负消去、稀疏、异常值重尾、随机scale、极小scale及真实激活 |

### 精度与性能报告

- 整数核心使用CPU INT64参考，在声明的INT32范围内逐元素验证，单列overflow与zero-point修正边界。
- 相同数学契约的替换优先逐元素一致，出现差异时定位归约或舍入变化并记录对FP32/FP64参考的误差。沿用已验模型的精度条件，已有回归阈值持续有效。
- 新激活量化、在线旋转和FP8重编码报告额外RRMSE、max error、logits KL/top-1、自然文本和图像质量，先建立独立策略的验收基线再选择默认值。
- 性能同时比较当前INT8路径、同源常驻BF16、最佳已验FP8和候选。每组记录cold、准备/编译时间、5次warm中位及范围、首token与稳态tokens/s、执行节点和固定token数。
- 小于约3%的差异复用A/B/A规则。自然输出长度单列，正式计时与profile/导出logits分开。fallback计数读取失败标成缺失，trace提供矩阵执行位置及数据搬运证据。
- 显存统计区分源权重、准备表示、workspace、recipe、allocator峰值、请求结束和卸载稳定值。报告达到准备成本盈亏平衡所需的重复次数。
- 首版目标为在代表工作量上明显改善当前INT8慢路径，并说明相对同源BF16和FP8的差距。低位常驻目标还需证明设备驻留容量收益。选择表按测得的时延/容量/精度权衡落地。

### 生命周期与正常入口

覆盖加载失败清理、请求中断、shape切换、重试、换权重、scale/zero point变更、LoRA应用及移除、保存重载、CPU/HPU切换、hooks后加、禁用编译和卸载。共享权重、并发准备和一次准备失败后重试应保持一致。临时tensor由执行所有者管理，长期权重表示纳入已有模型管理。

产品测试围绕新增契约和已发生回归扩展现有测试。实际shape性能测试、固定token和自然输出对照作为本计划交付的一部分，执行记录保存在本仓库。

## 7. 实施顺序、风险与当前完成状态

原拟依赖顺序为P0/P1、P2、P3及P4。后续用户将原生INT8 MME设为继续条件，本轮P0阴性结果已结束整项开发，后续阶段归档。

| 具体风险 | 处理与下一步 |
|---|---|
| 原生INT8 MME能力核验阴性 | 已保存HAL与直接SDK结果，按用户条件关闭整项开发 |
| 解码表示容量增加导致offload | 在层级和请求级预算内准备，比较分块、按需与低位常驻路线 |
| INT8→FP8或在线旋转引入数值变化 | 保留源格式参考，先完整质量对照再确定可启用范围 |
| 小M量化/调度成本高于矩阵收益 | 专门测GEMV、准备复用及编译分区，按shape选择已验路径 |
| TPC或ABI集成异常 | 复用现有eager host glue，保存编译器/库hash及最早错误，最小复现后重试 |
| 模型文件名或全局dtype造成错误分派 | 以每层量化语义和算子能力决定入口，混合格式作为首版回归 |

调查、计划、probe和失败记录保存在本仓库，产品代码、便携脚本与回归放在独立ComfyUI-Gaudi。服务器原始`/home/sonny/ComfyUI`遵循只读约束。后续硬件测试使用独立测试checkout并按已有单卡规则调度，下载复用已有组件和既有HF环境。

本轮已完成仓库经验、通用入口、官方资料核查，以及安装库HAL和直接Synapse能力测试。原生INT8 MME核验阴性，开发已按用户条件停止并放弃。后续TPC实现、转换路线和通用产品实现归档。

已有外部源码快照及哈希见[内核参考](KERNEL_REFERENCES.md)、[固定来源清单](../kernel-audit-2026-11-10/sources.json)和[JChen源码清单](../kernel-audit-2026-11-10/jchen-sources.json)。当前用户任务的慢路径现场见[2026-11-14证据](../evidence/qwen21-int8-slow-2026-11-14.json)。
