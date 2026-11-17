# Qwen扩写第一个decode步骤长等待

## 2026-11-17 修复验收

修复版本`07fedda3+staged.c8ad74f79c7e`通过普通服务验收后，用户确认并要求补齐缓存回归后提交。当前部署已关联正式产品提交`a5308be1c7dd56d8ebb694a8a9a630a072089a77`，状态为`accepted_committed`。本轮优化基线为`07fedda3278b5c4be8c036fe245dc6991b91dd4a`，初次修复暂存tree为`cfdce5bf049c851dd042a84ab7b85248f535b3cd`，patch SHA256为`c8ad74f79c7e76d7e88f679b24276a051dbd609df441810636e3161db1cc7d96`。

实现仅在符合现有整步解码准入条件的路径中，把KV的容量维标记为动态，并使用`dynamic=None`复用Dynamo图。HPU区域继续采用已有静态recipe编译策略。RoPE切片使用`clone(memory_format=torch.contiguous_format)`规范包括大小为1的维度在内的步长。KV按实际请求容量分配，状态由本次生成持有。

实机guard日志补齐了第二个触发条件。仅标记KV容量动态时，仍因`rotary`第0维步长`883072 → 883328`重新编译。普通`clone()`默认保留步长，大小为1的维度仍携带总表容量。完成上述两处调整后，13次短生成跨多个提示词、最大生成长度4095–16384及KV容量最高19187，全程仅编译一次解码图。新容量的首个decode迭代约0.580–0.663秒，重复容量约0.017秒。

### 同进程完整工作流对照

| 请求变体 | 修复前整图 | 修复后整图 | 修复前首个decode | 修复后首个decode |
|---|---:|---:|---:|---:|
| 原提示词，thinking关闭，容量6901 | 78.872秒 | 26.470秒 | 52.397秒 | 0.560秒 |
| 第一处颜色修改，容量6900 | 76.789秒 | 24.531秒 | 52.479秒 | 0.624秒 |
| 两处颜色修改，thinking关闭，已有容量6899 | 26.372秒 | 26.081秒 | 0.017秒 | 0.018秒 |
| 两处颜色修改，thinking开启，容量6897 | 89.024秒 | 35.288秒 | 53.626秒 | 0.629秒 |

首个decode采用前两次采样完成后的设备同步差值，覆盖解码、输出头和采样。各对照使用相同自然采样参数和seed，强制实际计算节点重算并保留loader。四组变体及原提示词的全部token、caption和RGBA逐元素一致。

五轮交错warm每侧五次，整图中位38.503→38.629秒，变化+0.327%，范围分别38.397–38.645及38.293–38.826秒。warm测量关闭首步同步和logits导出。此前热运行收益保留。

193项回归通过、1项CUDA专属测试跳过。新增BF16/FP8回归覆盖12次容量切换、11个不同容量32–8193、状态推进及仅一次解码图编译。固定1909个有效token的248320维logits与上一部署逐元素一致，RRMSE、最大绝对误差、KL均为0，top-1为1909/1909。

22次普通节点请求全部成功。新路径在这些请求中`hpu::batch_as_strided`警告为0次。原路径四次解码图编译合计出现8次该警告，产品路径在模型重载前只编译一次。重载后原提示词121.725秒，随后修改提示词34.662秒、首个decode0.182秒，输出仍与对应基线一致。

### 初始编译边界

新进程或重新加载模型后，仍需一次约55秒的解码图初始编译。同进程首次产品对照93.069秒已经复用了prefill及DiT等先前缓存，进程cold基线140.020秒包含这些初始化，两者采用各自的计时条件。当前修复覆盖提示词和生成长度变化导致的重复整图编译。

### 普通服务部署验收

普通`main.py`入口六次完整请求均通过，TextGenerate、文本编码、DiT、VAE实际重算。冷图142.963秒，四组编辑变体依次27.236、24.754、26.232、36.046秒，最后warm38.330秒。全部caption和RGBA匹配独立验证版本。该次服务日志的`hpu::batch_as_strided`警告为0次。部署六个文件的SHA256与暂存快照一致，结束时队列为空，服务已预热。两个仓库保持暂存，等待人工验收。[结构化修复证据](../evidence/qwen-decode-startup-fix-2026-11-17.json)

### 提示词改动与缓存隔离补验

按用户要求核对三层生命周期。`CacheKeySetInputSignature`把节点全部字面输入值以及递归上游输入纳入输出缓存签名，包括完整prompt、seed、thinking和system_prompt上游节点值。每次实际执行`BaseGenerate.generate`都会调用`init_kv_cache`重新分配并清零本次KV、卷积及递推状态，创建本次RNG和回读缓冲。`_generation_forward`的代理和RoPE表位于生成调用闭包中。已编译函数接收当前hidden、rotary及cache张量，seqlen参与运行时attention mask。

动态标注作用于KV容量维，规范RoPE步长用于稳定编译输入布局。当前提示词首先经完整tokenize/prefill更新本次状态，后续decode读取这些状态。完整节点输入签名负责结果复用，张量形状和编译guard负责算子图复用。

补充普通8189七次完整请求，使用固定文本/latent seed并沿用正常节点缓存。深蓝色与深红色两组提示词均为2803个输入tokens，token序列内容不同。二者扩写分别描述深蓝色与酒红色衣服，出图目视对应颜色，caption和RGBA哈希不同。执行蓝→红→蓝后，回切蓝色的caption/RGBA与第一次蓝色逐元素一致。随后分别以MTP off触发红色/蓝色重算，该模型无MTP头，实际生成路径与auto一致，两组重算结果均逐元素匹配首次结果。修改上游system_prompt也触发TextGenerate实际执行，最后恢复原始提示词的caption/RGBA匹配已部署基准。

七次请求的TextGenerate均未列入`execution_cached`，耗时37.902/31.794/37.805/31.540/37.758/34.153/38.494秒。别名警告0次，源码哈希匹配部署版本，队列为空。该补验覆盖同长度换内容、跨请求回切、强制重算及上游系统提示变化，结合上一轮变长、thinking切换和重载验证确认当前工作流的缓存失效与状态隔离。[补验原始记录](../evidence/qwen-prompt-cache-2026-11-17.json)

补验原始资料位于`/home/sonny/gaudi2/comfyui-hpu/performance-qwen-prompt-cache-2026-11-17`，入口为`probes/qwen_prompt_cache_audit.py`。该轮沿用已部署产品源码，调查脚本与证据保留暂存。

### 固化自动回归与人工验收

用户于2026-11-17确认验收，并要求把缓存case加入回归测试后提交。产品新增14项自动回归，专项14项通过，完整相关测试207项通过、1项CUDA专属跳过。节点缓存12项另在CPU模式独立运行通过。

- `tests-unit/execution_test/test_text_generation_cache.py`通过实际`CacheKeySetInputSignature`和`BasicCache`检查同长度换词、尾部空格、长度变化、thinking、max_length、seed、temperature、模板开关、MTP以及上游system_prompt变化。另验证A→B→A取回各自结果、NaN fingerprint强制重算。
- `tests-unit/comfy_test/hpu_qwen_generation_test.py`增加BF16/FP8非零权重小模型的真实`generate`入口测试。相同长度的两组token产生不同首步logits，A→B→A及清除编译缓存后的重新生成均匹配各自首次token和完整logits。每次请求检查KV/GDN状态清零、索引归零和与既有请求的存储隔离。
- 既有12次KV容量切换、11个不同容量仅一次图编译、RoPE连续步长及四token回读回归继续通过。

专项耗时32.77秒，完整集151.86秒，CPU节点缓存4.66秒。完整集32条告警包含厂商弃用、torchvision扩展加载、Dynamo包装器及BF16合成模型中的`hpu::batch_as_strided`别名告警。该BF16合成路径的重复/重编译结果逐元素一致。已部署FP8普通工作流此前两轮验收中的该别名告警均为0次。

本轮追加内容为测试与调查记录，产品暂存tree为`f68d0361edcc01559eacb4f68345c30e9349ccfe`，暂存身份`07fedda3+staged.44b0b24b6fca`。原始日志、验证哈希、普通服务恢复结果和最终提交关联位于`/home/sonny/gaudi2/comfyui-hpu/performance-qwen-cache-regression-2026-11-17`。

最终产品提交`a5308be1c7dd56d8ebb694a8a9a630a072089a77`的tree与该暂存tree一致。普通服务恢复验证142.826秒，caption/RGBA匹配已验版本，七个部署文件哈希匹配提交，队列为空。部署标记已切换正式commit并保留暂存快照，人工验收及提交状态已归档。[最终验收证据](../evidence/qwen-cache-acceptance-2026-11-17.json)

### 实验归档

服务器根目录为`/home/sonny/gaudi2/comfyui-hpu/performance-qwen-startup-fix-2026-11-17`。`dynamic`保存只标记容量后因RoPE步长重新编译的guard证据。首次`dynamic_compact`调查复制的函数命名空间引用了旧编译getter，该次数据排除。修正调查脚本后，`screen2/dynamic_compact`记录有效候选。`validation`保存产品源码快照、193项测试、22次API history、编译次数、首步计时与阶段事件；`logits-validation`保存独立数值对照。`deployment`保存暂存快照、回退源文件及普通服务验收。

## 初次调查的生产现场

用户报告`Generating tokens ... 1/4096 [00:00<48:05, 1.42it/s]`间歇性停顿，并伴随警告。已在普通8189服务日志中定位完全相同的行，对应请求`c1f4339f-ecd9-497b-9d0f-63c38d0462c3`。下一次进度为`2/4096 [00:56...]`，随后恢复约100 token/s，整图90.924秒。

当时运行版本为`07fedda3+staged.87ea451fce53`。初次调查采用生产history、原始日志、部署源码和CPU上的实际产品tokenizer核对，状态为根因已定位、修复待验证。该次调查结束时队列为空。

## 触发条件

`BaseGenerate.generate`按输入token数加`max_length`分配KV。本轮`max_length=4096`。`Qwen35._generation_forward`从第二次模型调用开始使用`_compiled_hpu_decode`，其配置为`fullgraph=True, dynamic=False, force_static_compile=True`。KV容量参与整个解码图的静态形状专门化。

| 按生产执行顺序 | 输入tokens | KV容量 | 第一个decode等待 | 整图 |
|---|---:|---:|---:|---:|
| 原提示词，thinking关闭，首次 | 2805 | 6901 | 约54秒 | 80.401秒 |
| 第一处灰褐色改白色，thinking关闭 | 2804 | 6900 | 约54秒 | 78.394秒 |
| 两处改白色，thinking关闭 | 2803 | 6899 | 无长停顿 | 26.433秒 |
| 原提示词，thinking关闭，再次 | 2805 | 6901 | 无长停顿 | 25.926秒 |
| 原提示词，thinking开启，再次 | 2803 | 6899 | 无长停顿 | 38.689秒 |
| 两处改白色，thinking开启 | 2801 | 6897 | 约56秒 | 90.924秒 |

6899容量已由部署验收请求预热。thinking模板关闭时多两个token，因此切换该参数也会改变容量。两处颜色文字修改也改变分词长度。新容量首次等待、重复容量恢复正常，与静态图重新编译的触发条件一致。原请求未启用Dynamo guard trace，具体guard失败行及56秒内部的编译/执行分项仍待插桩确认。[Gaudi动态形状说明](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Dynamic_Shapes.html)

进度到1时已完成prefill迭代；后续第一次单token decode触发整图编译。四token回读使首批可见token等待到buffer刷新。`48:05`按最初一个迭代外推，随后会随吞吐变化更新。生成进度的1表示循环迭代数。

## 警告来源

警告为`UserWarning: hpu::batch_as_strided`，内容指出自定义算子返回值与输入或其它返回值共享存储，违反其当前声明的别名约束，并提示未来PyTorch版本将升级为错误。

已核对服务器Habana1.24.2包和固定源码版本。`pass_batch_as_strided_groups`在编译后端合并多个`aten.as_strided`为`hpu.batch_as_strided`。该算子由厂商编译器插入，属于后端别名语义与PyTorch契约的兼容性问题。警告与新图路径同时出现，警告文本本身无法量化长等待，也无法证明图像输出损坏。当前请求均成功完成。[厂商固定源码](https://github.com/HabanaAI/gaudi-pytorch-bridge/blob/27eccbcffeab10842c9f29f64c6bf50d8f65d78f/python_packages/habana_frameworks/torch/dynamo/compile_backend/_passes/batch_as_strided.py)

## 初次调查的归因与验证安排

整步静态解码编译已存在于产品基线`07fedda3`，随后暂存版本增加四token回读和专用RoPE等优化。此次定位的是整步静态编译的冷启动代价，以及精确KV容量使普通提示词编辑频繁付出该代价的性能缺陷。仅凭本次现场数据，对最新RoPE改动单独造成的冷编译增量尚无法定量归因。

此前验收覆盖固定提示词、seed变化、分辨率、thinking切换和卸载重载，重点指标是固定工作量warm耗时。连续编辑提示词造成多个相邻KV容量的使用体验被遗漏，应增加同进程多提示词/多容量和首个decode延迟验收。

初次调查提出验证KV容量分桶或缩小受动态容量影响的编译区域，同时检查有效长度mask、输出质量、HBM增量、热运行收益、桶边界、模型重载和编译缓存上限。随后执行中选用了容量维动态标注及规范RoPE步长，结果见本文开头的修复验收。

## 证据

- [结构化生产证据](../evidence/qwen-decode-startup-2026-11-17.json)包含11次history的输入长度、容量、首尾进度、警告次数、部署源码哈希和最终队列状态。
- 服务器原始目录为`/home/sonny/gaudi2/comfyui-hpu/performance-qwen-startup-2026-11-16`。目录于调查开始时创建，报告完成日期为2026-11-17。保存`history-before.json`、`log-before.txt`、`token-capacities.json`和`diagnosis.json`。
