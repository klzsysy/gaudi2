# Qwen DiT attention D1执行记录

2026-11-17，Asia/Shanghai。产品基线为`a5308be1c7dd56d8ebb694a8a9a630a072089a77`。产品提交`22e3972af91d44c7af7a6fa3fb371e4b3dd2e3f4`已部署，用户已确认人工验收通过。283项回归、41次普通节点对照及七次普通服务验证及两次连接恢复后复验完成。

## 工作量与捕获

沿用1088×1920、八步Euler、官方Turbo sigmas、CFG1.2、FP8 DiT及FP32来源FP8 PE。当前已验PE自然生成2094 tokens。正负条件prefix分别为1285和447 tokens，图像query为8160，正负KV长度分别9445和8607。各方案使用同一caption、latent seed和KV auto设置。

初筛捕获第1/4/8步、0/15/31层、正负条件18组图像attention。随后捕获第4步全部32层，合计76组图像attention及6组masked prefix。对每组的全部32个head各取17行，以CPU FP64矩阵乘和softmax为参考，同时保留完整输出对基线的误差。普通请求通过计算节点fingerprint强制Sampler与VAE执行，loader和文本结果的缓存状态写入事件记录。

首轮捕获中prefix与image用了相同文件名，6个prefix文件被同层image覆盖。该轮18个image输入完整保留。第二轮增加masked后缀，重新收齐prefix与全部层，最终评价采用明确标识的捕获文件。

基线1069个Python及shell文件SHA256与本地提交一致。服务器原始资料根目录为`/home/sonny/gaudi2/comfyui-hpu/performance-qwen-d1-2026-11-17`。

## 外部项目核查

用户提供的`jchen-38/gaudi2-single-card-vllm`地址返回404，既有审计记录指向`JChen-318/gaudi2-single-card-vllm`。本轮固定该项目`72fafcf21430eda364fe45290908860e37b11a31`，以及1Cat的`e2266e0efa381e6649e0f25b789cbb61b21b0e1f`。源码文件及SHA256见[外部审计清单](../evidence/qwen-d1-external-2026-11-17.json)。

1Cat的diffusion入口采用连续BHND布局、明确的softmax模式和FusedSDPA recompute参数。其MLA selected-row入口使用共享KV及FP32模式，适用的稀疏decode形状与本轮密集DiT分别评估。相关实现用于确定本轮布局与厂商入口的对照项。[diffusion attention](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/e2266e0efa381e6649e0f25b789cbb61b21b0e1f/vllm_gaudi/omni/attention.py) · [MLA](https://github.com/1CatAI/1Cat-vLLM-Gaudi/blob/e2266e0efa381e6649e0f25b789cbb61b21b0e1f/flashinfer_gaudi/mla.py)

JChen的新版把非128对齐KV拆为对齐主体和短尾段，分别获得SDPA输出、行最大值及逆行和，再按归一化权重合并。它还指出切片必须连续化。本轮已把这一算法移植为独立候选，并实测当前Qwen的两种KV长度。[固定实现](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/72fafcf21430eda364fe45290908860e37b11a31/vllm-gaudi/vllm_gaudi/omni/attention.py)

JChen直接MME/TPC实现使用私有SCAL资源及预生成地址计划，现有bucket从37760 keys起步。本机Qwen约8K/9K KV，需要重新构造形状、SDK互操作和同步方案，其SRAM流水留作D2参考。本轮D1使用已安装厂商算子。[原生运行时](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/72fafcf21430eda364fe45290908860e37b11a31/native/mme-direct-attention/attention/h3_direct_attention.py)

## 筛选结果

两种真实KV长度分别比较32/8/1 head分组、普通/fast/FP32 softmax、recompute开关、FP16输入、KV拆分、mask padding、K中心化、BF16 QK＋FP8 AV及概率scale/query tile。局部时间包含输入布局、中心化、量化和输出合并。

18组完整输入的各组中位数如下。FP64误差为各输入的整体相对L2范围。

| 路径 | attention中位 | FP64误差范围 |
|---|---:|---:|
| 当前数学attention | 18.176ms | 0.638%–1.810% |
| BF16 QK＋FP8 AV | 11.195ms | 2.320%–4.571% |
| FusedSDPA fast | 10.718ms | 1.146%–2.987% |
| 对齐拆分＋普通softmax | 16.658ms | 0.437%–1.598% |
| 对齐拆分＋fast | 9.570ms | 1.156%–2.965% |
| K中心化＋对齐拆分＋fast | 9.608ms | 0.790%–2.951% |

首层第8步部分head相对误差最高19.31%。选中组合保留首层原实现，其余层使用K中心化、对齐拆分和fast softmax。补充全部32层后，加速范围内70组输入的整体FP64误差最高2.781%，单head最高6.503%。首层的6组输入继续由原实现承担。

FP32 softmax约64ms，FP16输入约77ms，均明显更慢。mask补齐未获收益。FP8概率scale2048产生非有限值，该候选退出。1024/2048 query tile在中间层约14.08/13.08ms，对照未分块混合路径11.62ms，后续投入转向实际片上数据流。

第一轮筛选的`functools.partial`外层共享Dynamo wrapper，达到8次编译上限后部分候选进入eager，出现sections validation错误和异常输出。该轮全量归档于`excluded-wrapper-limit/`。修正为各候选独立code object及函数默认参数后，完整筛选重新运行，表格采用重测结果。

## 完整候选流程

同进程交错运行，每侧五次warm，计时阶段无逐算子同步和输出张量导出。

| 路径 | 固定caption整图 | DiT |
|---|---:|---:|
| 基线 | 17.158s | 15.239607s |
| 全层中心化拆分fast | 12.779s | 10.859259s |
| 首层保留原实现 | 12.907s | 10.984515s |

选中组合DiT减少27.92%，达到约11–12秒的阶段时间预算。

候选覆盖原seed、第二latent seed、512²、1024²及KV off。512²保持逐像素一致。原图目视主体、花束、蜂鸟及整体构图接近，脸部、衣料和花朵细节改变。完整轨迹另存CPU张量，导出请求从性能样本中排除。

## 产品准入与回归

产品候选由Qwen Image 2.1原生FP8模型请求加速，首层保留原实现。共享dispatcher继续处理preferred attention、override、mask及精度要求。实际快路径限定编译推理、HPU BF16激活、batch1、32×128 heads，以及Q/KV均至少4096行。显式`hpu_fast_attention=False`进入原路径。

K中心化和KV拆分使用本次调用的Q/K/V，生成的临时张量随调用释放。首层判断在编译调用外完成，进入编译图的是布尔选择。既有PE编译缓存、节点输入签名和状态隔离回归保留。

初版合成测试统一采用3%误差界，实测分散随机attention的厂商fast近似误差约4.0%–6.5%。验证拆为厂商fast及完整实现对FP64的8%压力界、CPU高精度归一化合并检查、完整正常入口对未拆分fast的1%比较，以及A→B→A重算一致性。短KV下拆分前后的fast近似本身可相差约3%–4%，小shape合并正确性采用独立精确softmax计算验证。真实模型76组输入与配对图的误差独立保留。

最终专项33项、相关集238项和CPU cache admission 12项分别通过，合计283项通过，1项CUDA专属测试跳过。专项补充连续11次调用、10种KV长度及128对齐边界，编译图数保持≤4，长度回切及新V值生效。相关集继续覆盖PE同长度提示词、system prompt、seed、thinking、编译缓存重算及请求状态隔离。BF16合成用例仍有既有厂商别名告警，普通FP8请求中该告警及Dynamo重编译上限告警均为0。

产品验证目录为`product3/`。`product/`和`product2/`保留合成数值界限诊断的两轮失败日志，三轮产品运行时实现相同，测试逐步补齐独立参考。

最终33项专项和前缀实验位于`followup/`，产品运行时代码与`product3/`相同，新增KV边界回归已固化到测试文件。

## 前缀物化与生命周期

两组真实中间层分别实测普通cat、稳定buffer、将cat纳入编译区和分段KV归一化合并，每种方案五组交错计时。

| 路径 | 正条件 | 负条件 | 对普通路径输出 |
|---|---:|---:|---|
| 普通cat＋选中attention | 10.031ms | 9.482ms | 基准 |
| 稳定buffer更新target | 10.006ms | 9.471ms | 逐元素一致 |
| cat纳入编译区 | 9.997ms | 9.475ms | 逐元素一致 |
| prefix/target分段attention | 11.669ms | 10.754ms | RRMSE约1.03% |

单独cat约0.168/0.192ms。稳定buffer和编译cat的完整attention收益均≤0.34%，按512次调用估算仅约0.01秒整图量级。若正负条件32层均保留全长稳定buffer，总量约9.46GB，较现有prefix增加约8.56GB。分段attention则变慢约13%–16%。产品继续使用现有prefix物化方式。

## 产品完整请求与质量

41次普通节点请求覆盖五轮固定caption、五轮完整重算、跨尺寸、latent seed、KV off、轨迹、提示词蓝→红→蓝和卸载重载。以下各侧均为五次交错warm中位数。

| 指标 | 基线 | 产品 |
|---|---:|---:|
| 完整自然重算 | 38.508s | 34.138s |
| 完整重算范围 | 38.249–38.590s | 34.090–34.271s |
| 固定caption整图 | 17.170s | 12.924s |
| 固定caption DiT | 15.243372s | 10.994672s |
| 完整请求内TextGenerate | 20.832164s | 20.714699s |
| 输出tokens | 2094 | 2094 |

完整时间减少11.35%，固定caption DiT减少27.87%。PE阶段各组输出均为2094 tokens，文本阶段差异按本轮波动记录。产品输出逐像素匹配已验证的首层保留候选。基线和产品各自的cold、五次warm、五次完整重算均保持各自图像一致，产品重载121.010秒后再次匹配。蓝/红的caption与图片不同，蓝色回切的caption/RGBA匹配其首轮，基线与产品的对应caption一致。

| 配对场景 | RGB MAE，0–255 | PSNR |
|---|---:|---:|
| 原提示词/seed | 1.770 | 34.04dB |
| 第二latent seed | 1.825 | 31.70dB |
| 512² | 0 | 逐像素一致 |
| 1024² | 1.520 | 36.06dB |
| KV off | 2.364 | 31.66dB |
| 深蓝裙提示词 | 4.645 | 26.14dB |
| 深红裙提示词 | 1.951 | 31.31dB |

真实采样轨迹的16次DiT输出相对基线RRMSE为1.97%–7.73%，最大值位于第3步负条件。局部attention参考、实际轨迹差异及像素误差各自保留。fast softmax改变数值与图片，配对图已获用户人工验收通过。

## HBM流量与硬件目标

两个独立进程使用同一冻结caption和第2步负条件forward，每侧三次plain及一次原生采集。采用既有AXI返回beat校准配置，按plain中位时间归一。

| 指标 | 基线 | 产品 |
|---|---:|---:|
| HBM读取 | 0.790308TB | 0.553992TB |
| HBM写入 | 0.596155TB | 0.319141TB |
| 总读写量 | 1.386463TB | 0.873133TB |
| plain forward | 0.918757s | 0.662545s |
| 采集扰动 | −0.318% | −0.229% |
| 有效HBM带宽 | 1.509064TB/s | 1.317848TB/s |
| 2.45TB/s峰值占比 | 61.59% | 53.79% |

总流量减少37.02%，forward减少27.89%。按当前0.873133TB流量，80%带宽预算为`0.873133 / 1.96 = 0.445476s`，相对当前还需减少约32.76%时间。本轮已经达到D1的约11–12秒DiT预算，80%资源目标继续保留。

具名MME/TPC kernel事件的区间并集覆盖率约96.48%→95.43%。MME事件并集543.25→341.00ms，TPC事件并集581.51→391.35ms，两类存在重叠。汇总排除起止名称不同的`START_EVENT/MMEH_WB1_MON_TS_*`及`TPC_SPU_START*`硬件标记，BMON字节计算独立于这些标记。覆盖率衡量事件时间，后续阶段效率仍需结合有效工作量、TPC依赖和必要流量审计。

混合对照进程在warm、完整重算和蓝/红提示词后，allocator InUse稳定为39,273,374,208 bytes，累计峰值54,141,253,504 bytes包含两套路径及多个shape。该进程卸载后9,866,979,200 bytes，独立普通基线服务卸载后数值相同。

后续主要机会仍在attention内部MME/TPC数据衔接及邻接算子融合。按新流量重新核算80%预算，沿D2验证SRAM驻留与执行流水。

## 普通服务交付

入口为`http://192.168.10.21:8189`，普通`main.py`运行暂存版本`a5308be1+staged.319d9026b5e9`，tree为`9d01642e8deff0c4cb461a34b808115968ba8729`。产品已提交为`22e3972af91d44c7af7a6fa3fb371e4b3dd2e3f4`，调查记录随本轮归档，状态为`accepted_committed`。

| 普通请求 | 时间 |
|---|---:|
| cold | 139.686s |
| 完整warm | 34.175s |
| 深蓝裙 | 34.985s |
| 深红裙 | 28.008s |
| 深蓝回切并重算 | 33.362s |
| 恢复原提示词 | 34.568s |
| 卸载重载 | 118.747s |

七次请求均实际执行TextGenerate、TextEncode、Sampler及VAE，caption/RGBA匹配产品验证结果。普通日志的`batch_as_strided`和重编译上限告警均为0。卸载后9,866,979,200 bytes，与普通基线相同。三份变更文件及1070份Python/shell源码哈希核对通过，队列为空，原工作流已预热。初次模型加载仍包含PE约55秒编译成本。

收尾时出现短暂SSH和HTTP连接关闭，随后两种连接均恢复。主机uptime持续，部署版本及空队列再次确认。恢复后两次完整warm为34.264/34.399秒，caption/RGBA再次匹配验证版，最后使用原MTP auto。该连接现象的根因未定位。

原始部署身份、暂存diff、回退文件、实际SHA256及普通history在`deployment/`，紧凑记录见[交付证据](../evidence/qwen-d1-2026-11-17.json)。

## 复现入口

捕获与候选分别由`qwen_d1_jobs.py`、`qwen_d1_screen.py`及`qwen_d1_validation.py`执行。产品入口为`qwen_d1_product_jobs.py`，负责串行回归、普通节点对照、硬件计数及服务恢复。`qwen_d1_server.py`仅用于调查进程，普通服务使用产品`main.py`。`qwen_d1_summary.py`汇总计时、图像、轨迹及经过AXI返回beat校准的HBM数据。

## 人工验收与提交关联

用户确认“验证通过，提交代码”。产品提交`22e3972af91d44c7af7a6fa3fb371e4b3dd2e3f4`的tree为`9d01642e8deff0c4cb461a34b808115968ba8729`，与已验暂存tree一致。服务器manifest保留原暂存身份、差异和文件SHA256，补充人工验收记录及正式commit关联。
