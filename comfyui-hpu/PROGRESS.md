# 当前状态与未决问题

截至2026-11-17，产品分支`codex/gaudi2-hpu`，当前产品提交与部署为`22e3972af91d44c7af7a6fa3fb371e4b3dd2e3f4`，D1已人工验收并提交。本文件提供交接快照，测试过程和候选取舍集中于 [主题索引](README.md)。

## 2026-11-17 D1 attention验收提交

选中Qwen原生FP8 DiT的K中心化、128对齐KV拆分及厂商fast softmax，首层使用原路径。76组真实attention与6组prefix捕获、全部32层补查、283项自动回归及41次普通节点验证完成。连续10种KV长度、128对齐边界和长度回切保持≤4份编译图。普通服务七次完整生成及重载均匹配验证版caption/RGBA，提示词蓝→红→蓝回切通过。连接恢复后又完成两次完整warm，结果一致。

五轮交错完整重算38.508→34.138秒，减少11.35%，固定caption DiT15.243→10.995秒。负条件流量1.386→0.873TB，时间0.919→0.663秒，有效带宽1.318TB/s、峰值53.79%。D1时间预算已达成，按新流量达到80%需≤0.445秒。fast softmax带来图像细节变化，实际轨迹输出RRMSE最高7.73%，用户已确认人工验收通过。

产品暂存tree为`9d01642e8deff0c4cb461a34b808115968ba8729`。普通cold139.686秒、warm34.175秒，蓝/红/回切34.985/28.008/33.362秒，恢复原图34.568秒、重载118.747秒。三份变更文件哈希匹配，1070份Python/shell源码核对通过，队列为空。产品已提交，调查记录随本轮归档。普通日志的别名及重编译上限告警均为0。[D1记录](optimization/QWEN_D1_ATTENTION_2026-11-17.md) · [交付证据](evidence/qwen-d1-2026-11-17.json)


用户明确确认“验证通过，提交代码”。产品提交`22e3972af91d44c7af7a6fa3fb371e4b3dd2e3f4`的tree为`9d01642e8deff0c4cb461a34b808115968ba8729`，与已验证部署快照一致。部署保留`a5308be1+staged.319d9026b5e9`历史身份并关联正式提交，状态为`accepted_committed`。

## 2026-11-17 回归补齐与验收提交

用户确认验收并授权补齐回归后提交。新增14项自动回归覆盖同长度内容变化、A→B→A回切、编译缓存清除后的重算、独立KV/GDN状态，以及prompt/system_prompt/seed/thinking等节点输入变化。专项14项、完整相关集207项通过，1项CUDA专属测试跳过；节点缓存12项CPU独立复验通过，计数存在重叠。

产品提交`a5308be1c7dd56d8ebb694a8a9a630a072089a77`的tree为`f68d0361edcc01559eacb4f68345c30e9349ccfe`，与最终已验证暂存tree一致。服务普通完整图142.826秒恢复预热成功，caption/RGBA匹配已验版本，七个代码/测试文件哈希与提交一致，队列为空。`.run/deployed-version`已关联正式commit，manifest保留原暂存身份和人工验收记录，状态为`accepted_committed`。[提交验收证据](evidence/qwen-cache-acceptance-2026-11-17.json)

BF16合成回归仍出现厂商`batch_as_strided`别名告警，状态隔离及重编译输出逐元素比较通过。已验FP8普通流程该告警为0次。模型首次加载约55秒图编译成本及80%硬件利用率目标继续保留为已知边界。

## 2026-11-17 扩写启动延迟修复部署

将KV容量维标记为动态，并规范RoPE切片的singleton步长。新容量首个decode从52.4–53.6秒降至0.56–0.63秒，三组编辑请求整图78.872/76.789/89.024→26.470/24.531/35.288秒。每侧五次交错warm中位38.503→38.629秒，变化+0.33%。193项回归通过、1项跳过，固定1909-token logits逐元素一致，22次普通节点验证通过，全部对照token及RGBA一致。模型重载仍需一次约55秒初始图编译，后续编辑复用成功。

暂存版本`07fedda3+staged.c8ad74f79c7e`已部署8189。普通服务六次完整计算成功，冷图142.963秒，四个编辑变体27.236/24.754/26.232/36.046秒，warm38.330秒。六张图及caption匹配验证版本，别名警告0次，六个文件哈希匹配、队列为空。两个仓库保持暂存，等待人工验收。[修复与边界](optimization/QWEN_DECODE_STARTUP_2026-11-17.md) · [证据](evidence/qwen-decode-startup-fix-2026-11-17.json)

随后补验七次普通请求，深蓝/深红提示词同为2803 tokens、固定seed，均实际执行TextGenerate并生成对应颜色。蓝→红→蓝回切、分别强制重算的caption/RGBA逐元素匹配各自首轮，上游system_prompt修改正确失效，恢复原提示词匹配部署基准。KV/GDN状态与RoPE表按生成调用持有，节点输出签名包含完整输入及上游值。别名警告0次、哈希匹配、队列为空。[缓存隔离补验](evidence/qwen-prompt-cache-2026-11-17.json)

## 2026-11-17 扩写启动延迟反馈

已定位用户`1/4096 ... 48:05`停顿现场。普通提示词编辑和thinking切换使输入2801–2805 tokens，对应KV容量6897/6899/6900/6901。新容量首次decode等待54–56秒，重复容量恢复约100 token/s。整步静态编译按精确容量专门化，形成频繁冷编译的性能缺陷；伴随的`hpu::batch_as_strided`警告来自厂商编译器插入算子的别名约束问题。调查状态为根因已定位、修复待验证，当前暂存部署仍待人工验收。[调查及边界](optimization/QWEN_DECODE_STARTUP_2026-11-17.md) · [生产证据](evidence/qwen-decode-startup-2026-11-17.json)

## 2026-11-16 P1/P2/P3继续执行

按用户要求保留暂存并继续推进。完整token图及1/2/4/8展开、设备RNG、lazy Graph、共享量化/组合权重、GDN TPC与conv、M1–32布局/K填充、有效长度/GQA/RoPE均已实测。选中FP8生成调用级紧凑频率表、立即状态控制清理和独立Qwen原生RoPE入口。191项回归通过、1项跳过，19次普通节点请求及卸载重载通过。

固定1909-token交错A/B各五次warm，PE26.843→19.070秒，整图44.370→36.593秒，分别减少28.96%/17.53%。两组自然整图减少13.21%/10.27%，扩写与图片改变。有效logits RRMSE5.1793%，人工质量待验收。64-token原生计数扰动−0.70%，HBM0.893TB/s、峰值36.44%，80%目标仍未达到。产品`07fedda3+staged.87ea451fce53`已部署，待人工验收。普通首图141.564秒，完整warm38.635/38.316秒，重载121.408秒，扩写及RGBA均匹配已验自然候选。六个文件哈希匹配、队列为空。[报告](optimization/QWEN80_PE_EXECUTION_2026-11-16.md) · [证据](evidence/qwen80-pe-execution-2026-11-16.json)

## 2026-11-16 硬件80%计划首批执行

首批低扰动早段64-token trace扰动3.96%，后段15.79%。四token回读产品候选完成同进程交错A/B各五次warm，PE27.554→26.254秒，整图45.139→43.819秒，分别减少4.72%和2.92%，十次扩写及RGBA均与原图一致。132项回归通过、1项CUDA跳过，另一组seed的512²成功。RoPE原公式内联和原生候选分别变慢38.8%/35.3%，logits有约4%差异，作为失败路径归档。产品暂存版本`07fedda3+staged.d39241e049d0`已部署、待人工验收。普通冷启动145.016秒，两次完整warm43.856/43.957秒，卸载后重载125.588秒，四次扩写与RGBA均匹配原图，文件哈希匹配、队列为空。80%阶段硬件效率目标仍待后续实现。[首批报告](optimization/QWEN80_FIRST_DELIVERY_2026-11-16.md) · [证据](evidence/qwen80-first-delivery-2026-11-16.json)

## 2026-11-16 硬件80%优化计划

按用户要求完成[深度优化计划](optimization/QWEN21_HARDWARE_80_PLAN.md)，以当前单卡batch1工作流为主场景，目标为主导资源及阶段硬件效率≥80%。固定当前流量时，PE带宽目标1.96TB/s、约4.45ms/token、完整扩写约8.93秒，DiT负条件forward目标≤0.707秒。先建立低扰动FP8执行账本，再推进图外输出头/采样/回读闭合、共享FP8量化和GDN状态融合、有效长度attention及MME/TPC片上流水。第一阶段整图预算22–24秒，深度数据流优化探索预算17–20秒，最终80%达标须逐阶段实测证明。计划及索引已暂存供审阅。

## 2026-11-16 最后一次FP8生成的性能调查

冻结用户13:54:50的`Qwen_image_2.1_00062.png`请求，原执行45.465秒。实际为1088×1920、八步、CFG1.2、thinking开启、FP32来源FP8 PE和DiT KV auto。完整1 cold＋3 warm，cold146.058秒，warm中位44.773秒，PE27.267秒占60.9%、DiT15.144秒占33.8%。2803输入、1909输出tokens，thinking段812 tokens约11.9秒。四张复现图扩写与RGBA均匹配原图。

PE后段16-token计数为8.730GB/token，按plain时间归一的HBM带宽估算0.630TB/s、峰值25.72%，采集扰动71.70%。DiT负条件为1.386TB/forward、1.520TB/s、峰值62.04%，采集扰动0.18%。DiT kernel活动覆盖96.81%，attention BatchGemm及softmax独占合计约59.3%设备跨度。完整warm频率1650MHz、47–55°C、allocator峰值49.955GiB。

文本详细trace触发重编译，已排除定量结论。native DiT进程在成功采集和出图后的退出阶段发生段错误，保留待定位。普通8189已恢复`07fedda`，原图147.813秒冷启动成功，扩写和RGBA一致，队列为空，1067份产品源码哈希匹配。调查文档、工作流及脚本已暂存供审阅。[报告](QWEN21_LATEST_BOTTLENECKS_2026-11-16.md) · [证据](evidence/qwen21-latest-bottlenecks-2026-11-16.json)

## 2026-11-16 提交归档

用户明确要求“提交代码”。产品仓库已提交为`07fedda3278b5c4be8c036fe245dc6991b91dd4a`，覆盖Qwen原生FP8接入、PE整步/DiT FFN编译及两类转换器和回归测试。提交树为`d4264962aec7a96f7318ceca52263edf9b6ee5b7`，与已部署的最终暂存树一致，服务器源文件SHA256核对通过。调查、工作流、失败候选和测量证据在本仓库归档。下方阶段记录保留各次实验的快照、数据和当时验收状态，数值与图像差异继续按原报告记载。

## 2026-11-16 PE从INT8经FP32直接转FP8

按用户要求新增`qwen3.5_9b_qwen_image_2.1_pe_t2i.fp8_from_int8_fp32.safetensors`，10,890,431,404 bytes。201个Linear直接由FP32反量化结果量化FP8，559个保护张量与上一版相同。全量权重对FP32参考的RRMSE从2.65482%降至2.64975%，201层均改善，相对降幅0.1911%。7项转换测试通过。自然扩写420→806 tokens，新模型五次完整warm中位30.248秒，上一版23.943秒；固定432-token为24.478秒，上一版24.129秒，固定图RGBA逐元素一致。512²和卸载重载通过。

新增独立工作流`Qwen-Image-2.1-Turbo-FP8-PE-From-FP32-HPU.json`。普通8189冷图131.448秒成功，RGBA匹配独立新模型流程，源文件哈希匹配、队列为空。暂存部署`b89280c0+staged.d5bda4e78d95`，模型文件和两个仓库相关改动已交付待人工验收。自然图片改变，权重重建精度与使用体验分别记录。[转换与验证](QWEN21_PE_FP32_CONVERSION_2026-11-16.md) · [证据](evidence/qwen21-pe-fp32-2026-11-16.json)

## 2026-11-16 FP8达到BF16无缓存warm水平

按用户要求继续逐项优化并保留暂存。接入原生FP8生成调用级整步编译及DiT FFN区域编译，产品完整5次warm中位23.943秒，原BF16基线25.425秒，同轮BF16确认25.526秒。固定同一432-token轨迹，产品3次warm为24.129秒，BF16确认25.526秒。自然PE稳定420 tokens，PE6.368秒、DiT15.185秒。cold128.631秒，对照BF16 cold104.261秒，首次编译成本仍较高。203项回归通过，1项CUDA专属测试跳过，512²和卸载重载通过。普通8189冷图129.466秒、两次完整重算24.277/24.105秒成功，候选/产品/普通服务扩写及RGBA逐元素一致，源文件哈希匹配、队列为空。

PE窗口HBM 0.214→0.652TB/s，FP8有效算力0.0367%→0.1368%峰值。DiT负条件读写量1.892→1.431TB，时间1.081→0.950秒。FFN三组真实输入的FP64抽样误差更低。PE对首轮FP8的固定logits RRMSE5.465%、top-1为419/432，自然扩写与图片改变，人工质量验收待确认。产品暂存树`b294c6a6d796176956e3f4b0f49d0a9f107ac75a`，状态为已部署、待人工验收。[完整报告](QWEN21_FP8_OPTIMIZATION_2026-11-16.md) · [证据](evidence/qwen21-fp8-optimization-2026-11-16.json)

## 2026-11-15 Qwen Image 2.1 FP8转换

已本地转换192个DiT Linear及201个匹配PE Linear，复用现有Qwen3-VL 8B FP8并接入其HPU原生矩阵入口。53项相关回归通过，2MP完整流程、512²和卸载重载通过。最新请求thinking关闭，2805输入tokens。无节点结果缓存且DiT KV关闭，各5次warm，BF16为25.425秒，FP8自然为71.507秒，扩写432→1061 tokens。固定432-token轨迹后FP8仍为40.149秒，PE20.645对6.529秒，DiT17.205对16.655秒。PE采样HBM约0.214TB/s、8.72%峰值，FP8有效算力0.0367%峰值，逐层提交和量化开销突出。DiT约1.75TB/s、71.42%带宽，FP8 Linear有效算力12.85%峰值，attention/softmax及中间量读写是主要后续方向。容量降低约20GiB，当前FP8完整流程速度退化。产品基线`b89280c0`，本轮改动已暂存并部署，待人工验收。普通8189完整图122.256秒成功，RGBA与独立自然FP8图逐元素一致，源文件哈希匹配暂存快照，队列为空。[完整报告](QWEN21_FP8_2026-11-15.md) · [证据](evidence/qwen21-fp8-2026-11-15.json)

## 2026-11-15 Qwen执行优化上线

用户要求先部署体验，已将候选接入普通产品入口，主实现`28794c4b`、最终版本`92d15ff7`。适用于常驻普通BF16、HPU、batch1、无MTP头的Qwen，生成调用持有独立cache代理，入口兼顾hooks、在线权重变换、低显存、attention配置及中断。155项回归及末次19项专项通过，有17项重叠。固定427-token产品logits与实验候选逐元素一致，三次warm中位6.4555秒。普通8189接口两组自然图103.705/49.905秒、卸载后重载83.915秒、最终提交重启预热107.747秒，扩写及RGBA均匹配对应候选图。MOSS BF16固定tree3s1三组扩写匹配此前基线。最终源文件哈希与提交一致，队列为空，原工作流已预热。新KV容量仍有首次编译成本。[上线记录](optimization/QWEN21_EXECUTION_DEPLOYMENT.md) · [证据](evidence/qwen21-execution-deployment-2026-11-15.json)

## 2026-11-15 Qwen Image 2.1 最新2MP工作流

用户要求逐项推进后，在独立调查进程验证9条GDN/attention/整步执行路径。最终候选为单token GDN立即状态更新、BF16 Q/K/V＋FP32 softmax SDPA和整步编译。固定427-token三次warm为10.228→6.533秒。原2MP同1784-token工作量，每侧1 cold＋2 warm，PE41.693→25.183秒，42.79→70.84 tokens/s，整图60.662→44.178秒，耗时降低27.17%。cold96.804→111.894秒。新HBM计数为16.562 GB/token、1.169TB/s、峰值47.72%，recipe306→84/token。完整logits约0.988% RRMSE，4组自然扩写和图片发生变化，检查页已提供，人工质量接受待用户确认。512²及卸载重载通过，重载前后候选扩写和RGBA一致。调查启动器的模块解析与导入等待、独立probe退出析构告警已修复并复测。普通服务最终恢复已接受的`dfca7cd6`，5份产品源文件哈希一致、队列为空。[逐项验证](optimization/QWEN21_EXECUTION_VALIDATION.md) · [证据](evidence/qwen21-execution-2026-11-15.json)

用户人工确认8组版本对照质量正常，接受`dfca7cd6`并要求保持提交。新增根因分析以真实第33步状态重放核心，未插桩20.902ms、主线程CPU17.276ms。已编译MLP及输出头权重有效带宽约2.005/2.200TB/s。修正后的同轮消融基线20.853ms，跳过GDN核心15.796ms，组合跳过GDN核心与完整attention9.566ms。原生晚位置16-token trace为306 recipes/token，执行事件未覆盖的8.370ms/token中约7.751ms位于recipe区间内，具体等待类型仍待归因。消融改变计算工作和执行重叠，不能视为上线收益。后续优先处理GDN/attention执行链，并将有数值差异的候选推进完整质量验证。两次隔离诊断后普通服务恢复、队列为空、源文件哈希相同。[根因报告](optimization/QWEN21_DECODE_ROOT_CAUSE.md) · [证据](evidence/qwen21-decode-root-cause-2026-11-15.json)

按用户确认，产品`dfca7cd6`与上一提交`ed9552e5`各独立生成8张原提示词2MP图，两类seed逐组配对。16份history确认完整重算扩写和采样，8组扩写文字及RGBA像素均逐元素一致。原图、扩写、8张原尺寸并排图、概览和ZIP已保存，供用户人工检查。当前服务恢复`dfca7cd6`，源文件哈希一致、队列为空。[图组记录](optimization/QWEN21_PRODUCT_AB.md) · [证据](evidence/qwen21-product-ab-2026-11-15.json)

用户要求先推进无MTP的BF16路径。本轮以AXI返回beat完成HBM有效载荷校准，128/512/2048 MiB copy及GEMV读取回算误差约−0.43%至−0.26%。真实解码约17.168 GB读取和1.261 GB写入/token，16轮无profiler为0.36819s，约0.801 TB/s，即峰值32.69%，取整敏感性上沿33.38%。整步eager编译与持久HPUGraph将decoder核心20.871→约17.8ms，存在0.79%–0.93% logits差异。GDN单tokenTPC使固定427-token耗时10.220→9.567s，但完整logits仍有0.986%误差，两个舍入修订也未对齐。普通服务最终运行`dfca7cd6`，原2MP图100.598s成功且像素一致、队列为空。DMA校准首版的地址使用错误曾触发设备自动复位，已恢复并以独立Synapse控制完成补测。[完整校准与验证](optimization/QWEN21_BF16_SINGLE_TOKEN.md) · [证据](evidence/qwen21-bf16-single-token-2026-11-15.json)

后续隔离试验复用已有Qwen3.5的243M参数MTP头。固定512-token窗口三次warm，单步42.03→四行验证58.57 tokens/s，吞吐提高39.35%。完整自然扩写两次warm为1784 tokens/41.507秒→1911 tokens/31.279秒，约61.10 tokens/s。相同前缀首个根token的单行/四行验证logits仍有2.022% RRMSE，完整图片和正常loader接入尚待验收。候选只在调查进程运行，线上使用`dfca7cd6`。实验后原2MP图94.763秒成功且像素一致，队列为空。[候选与边界](optimization/QWEN21_TEXT_NEXT.md) · [证据](evidence/qwen21-text-draft-2026-11-15.json)

已冻结最后一次用户请求，实际1088×1920、8步官方Turbo调度、CFG1.2、thinking开启、匹配PE的BF16文件，2803输入/1784输出tokens。原请求66.274秒。三次完整warm初始基线中位62.250秒，回退原产品三次为61.563秒，最终三次为60.689秒。按回退基线，完整耗时减少1.42%，PE42.575→41.713秒，约41.9→42.8 tokens/s。DiT约16.59秒，文本缓存图约18.47秒。

接入单token普通BF16的conv/delta联合编译及MLP残差融合，GDN两端投影维持原执行边界。16-token cProfile中recipe提交2720→2336、eager add1024→512、Python调用389151→313781。固定427-token全logits逐元素一致，五组不同seed/尺寸/thinking/官方调度的token和RGBA图片逐元素一致。初始组合另有五次完整warm中位60.713秒。157项回归通过、1项跳过，最终含新增单token hooks的39项专项通过，两组测试有重叠。

普通8189冷启动95.048秒，新提示词512²为10.614秒，卸载后原图重载91.124秒。卸载40–60秒稳定约8.984GiB，该2MP尺寸的原版卸载对照未采集。MOSS BF16固定MTP三组新鲜原版/新版扩写文字一致，历史样本漂移在原版中也存在，尚未归因。最终恢复Qwen原2MP图95.721秒成功，cold/reload/final像素与基线一致，队列为空，四份部署文件SHA256与产品提交对应文件一致。[证据](evidence/qwen21-performance-2026-11-15.json)

扩大整GDN编译区仍有约0.98% logits误差，审计确认输入和输出投影均有舍入变化。2MP DiT分组最快局部约5%，尚有数值变化，本轮继续采用原DiT数学路径。单行commit无稳定净收益，RoPE预计算变慢，关闭eager流水线使固定文本10.259→32.620秒。后续HBM校准结果及新的单token候选见本节开头。[完整调查](QWEN21_PERFORMANCE_2026-11-15.md)

## 2026-11-14 原生INT8 MME验证与开发终止

用户明确要求有原生INT8 MME才继续，能力核验阴性后已停止并放弃通用INT8开发。服务器安装库`Gaudi2HalReader::getSupportedMmeTypes()`返回`0x7906`，INT8位为零。直接Synapse的默认、inference、inference＋quantization模式共9组INT8 GEMM均在节点创建阶段失败，3组BF16对照编译成功。

本轮验证范围为已安装HAL与直接SDK建图/编译，INT8在执行前被拒绝。验证前后普通服务PID相同、队列为空。[验证结果](optimization/INT8_NATIVE_MME.md) · [证据](evidence/int8-native-mme-2026-11-14.json) · [历史计划](optimization/INT8_DEVELOPMENT_PLAN.md)

## 2026-11-14 INT8扩写慢调查

已核对普通8189最后一次120.523秒请求，Qwen Image 2.1 PE选择原INT8 ConvRot文件，文本循环约102秒、进度737、平均7.23 tokens/s。此前同提示词/文本seed的同源BF16扩写进度456、平均39.12 tokens/s，中段约42 tokens/s。thinking关闭，两个文件均无MTP权重。

部署源码确认INT8使用`full_precision_mm=True`，每次量化Linear调用先反量化/ConvRot还原为BF16，同时跳过短步MLP及完整attention区域编译。生成长度增长进一步增加耗时。已存在的同源BF16文件提供预先解码及编译路径。本轮完成只读历史、源码和模型header核查，各算子独立耗时尚未profile。[调查](optimization/TEXT_PRECISION.md) · [证据](evidence/qwen21-int8-slow-2026-11-14.json)

## Qwen Image 2.1 全BF16优化

当前组件为Turbo BF16、Qwen3-VL 8B BF16、匹配文生图PE的BF16解码文件和Qwen Image 2.1 VAE BF16。`17b4b231`接入原生RoPE与无mask attention编译，`ed9552e5`编译temperature=1.0的HPU采样概率。投影/RMSNorm保留原舍入边界，有mask前缀采用原数学执行。

用户832×1504、8步Euler/simple、CFG1、PE开启且thinking关闭，五次完整warm中位21.028→16.079s，PE11.416→10.593s，DiT8.089→4.006s。复用文本结果后的整图9.2185→5.130s，进程峰值76.645→60.335GiB。回退原数学路径复验为20.790s，收益可复现。

12组真实block逐元素一致，六组完整扩写token与RGBA PNG逐元素一致，含不同seed、512²/1024²、thinking开启与官方Turbo调度。159项回归通过、1项跳过。普通8189首次49.833s、换seed18.267s、重载42.926s成功，清理稳定占用约3.15GiB。当前验证范围为batch1文生图，2K及编辑另验。[深入记录](QWEN21_BF16.md) · [证据](evidence/qwen21-bf16-2026-11-13.json)

## Qwen Image 2.1 Turbo 首版

`0c764d59`修复官方增强器 presence penalty 的 HPU 布尔 mask 更新，`9f2ff96e`接入普通 BF16 FFN 区域编译。使用官方基础文生图结构与 Turbo 8 步调度，复用已有 BF16 主模型、Qwen3-VL 8B、Qwen Image 2.1 VAE。新文生图增强器已下载并校验，默认 HPU 图使用它的本地 BF16 解码文件。

1024²直接图五次 warm 中位 7.726s，基线8.262s。开启 thinking 增强器的完整图为42.340/42.229s，2548输入/1316输出tokens。原 INT8 同提示词自然生成1124 tokens，完整热运行161.042s。512²及卸载重载通过，清理稳定后约0.849GiB。80项回归通过、1项跳过。九组真实 block 相对 eager RRMSE 为0.00180%–0.01142%，九组 FFN 抽样对CPU FP64的误差比0.99988–1.00228。相同seed图像细节存在变化，本轮覆盖512²和1024²。普通8189最终直接图17.389/7.712/7.733s、首次增强图69.834s成功，工作流已安装，队列为空。[首版记录](QWEN21_FIRST.md) · [证据](evidence/qwen21-first-2026-11-13.json)

## 当前接受的实现

| 方向 | 现行实现与主要提交 | 详细记录 |
|---|---|---|
| 设备与模型 | HPU后端、E4M3解码表、VAE非原地残差 | [适配](ADAPTATION.md) |
| BF16 DiT | 两段编译/静态recipe、条件预计算、native norm、验证后BLOCKWISE | [BF16](optimization/DIT_BF16.md) |
| FP8 DiT | `2227f7d3`原生Linear、`0995638a`FP32 Q/K norm及block扩区 | [FP8](optimization/DIT_FP8.md) |
| Attention | `700bdb6d`单head、`f67e7f3f`混合AV、`69fe4a54`移除8192上限、`ef16c195`长序列分批 | [Attention](optimization/ATTENTION.md) |
| 文本精度 | INT8解码BF16、`148b252a`原生FP8文本 | [量化](optimization/TEXT_PRECISION.md) |
| 文本执行 | KV行clone、MLP/norm/attention编译、`34e2649e`短步RoPE | [decode](optimization/TEXT_DECODE.md) |
| Qwen单token上线试用 | `28794c4b`直接GDN状态、SDPA与整步编译，`92d15ff7`可选norm兼容 | [上线](optimization/QWEN21_EXECUTION_DEPLOYMENT.md) |
| GDN | `60da4e33`区域编译、prefill分桶、`41f01e68`并行递推、`a8f42df2`八行扩区 | [GDN](optimization/TEXT_GDN.md) |
| 运行时 | `441d459d`初始化后恢复SIGPIPE忽略策略，独立eager环境 | [运行时](optimization/RUNTIME.md) |

## 最近一轮 MOSS 验收

各项为自己的同轮五次warm中位数。

| 场景 | 基线 | 最终候选 |
|---|---:|---:|
| 1088×1936固定caption整图 | 29.645s | 28.892s |
| 1024² Moody FP8＋FP8文本固定MTP完整流程 | 16.092s | 15.902s |
| 上述TextGenerate | 5.227s | 4.985s |
| 1024² SFT BF16＋BF16文本固定MTP完整流程 | 19.521s | 19.396s |
| 上述TextGenerate | 5.314s | 5.156s |

18组真实block、两种文本固定logits逐元素一致，六组固定tree3s1自然扩写token相同。BF16整block共享候选保留舍入后无稳定收益。341项广泛回归通过、1项CUDA跳过，动态padding整理后最终37项专项通过，源码树为`4d9cc16d8fc372dc580511bed073bb0ce606672e`。临时原文本回退另过335项，测试范围存在重叠。

最后一次普通8189验收原始1088×1936工作流三次成功，cold100.467秒，文本缓存29.113/28.887秒，四份产品文件哈希核对一致、队列为空。该状态记录来自本轮优化验收。[结构化证据](evidence/moss-three-stage-2026-11-13.json)

## 质量与调查边界

- 默认Auto额外文字在优化版和原始文本回退版均出现，已修正最初的单项归因。固定树保值对照通过，两侧样本仍有BF16茶壶把手不可见、FP8海报低对比度等表现。
- TPC GDN候选因完整logits变化和双把手样本撤回。32MiB SRAM设置、额外小M FP8编译及BF16扩区未达到各自接入条件，原因保留在主题候选表。
- QK scores仍落DRAM，概率tile已在SRAM。文本HBM访问已用AXI返回beat校准，DiT尚待按新口径重测。约0.015秒kernel归属和文本空档因果归因仍需补齐。
- 调查进程的若干段错误、profiler退出tcache异常、部分vendor异步输入问题仍缺底层根因。已验产品路径和各故障触发条件分别记录。
- 质量范围限于已有模型/样本/参数组合。完整批量、多卡、编辑/控制LoRA、SLG、其它VAE/custom nodes及超大尺寸需要独立验证。

后续投入依据见 [优化方向](optimization/STRATEGY.md)，测量边界见 [验证方法](optimization/MEASUREMENT.md)。
