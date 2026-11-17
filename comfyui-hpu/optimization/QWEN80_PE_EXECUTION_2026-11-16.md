# Qwen FP8 P1、P2、P3执行结果

2026-11-16，Asia/Shanghai。按用户要求继续推进P1/P2/P3，改动保留暂存。本轮完成完整token图、设备状态与展开、FP8共享量化及组合投影、GDN TPC、矩阵布局、有效长度attention和RoPE各项候选实现与实测。有效产品版本为`07fedda3+staged.87ea451fce53`。

**已部署，待人工验收。硬件80%目标尚未达成。** 当前PE约9.75ms/token、HBM约0.893TB/s，为2.45TB/s峰值的36.4%。按当前8.706GB/token流量，80%带宽预算为4.442ms/token，仍需约2.20倍吞吐提升。P2提出的6–7ms/token阶段预算也尚未达到。

## 完整工作流

固定用户原始1088×1920、八步Euler/官方Turbo调度、CFG1.2、thinking开启、FP32来源FP8 PE、FP8 DiT/CLIP、BF16 VAE及DiT KV auto。输入2803 tokens，固定扩写1909 tokens。每次真实执行模型和采样，再以同一teacher轨迹控制工作量。两侧计算节点强制重算，loader与已编译recipe常驻。

基线为上一轮四token回读暂存部署`07fedda3+staged.d39241e049d0`。同进程交错顺序为基线/候选/候选/基线/基线/候选/候选/基线/基线/候选，各五次warm。

| 项目 | 基线中位 | 本轮中位 | 降幅 |
|---|---:|---:|---:|
| 完整出图 | 44.370秒 | 36.593秒 | 17.53% |
| TextGenerate | 26.843秒 | 19.070秒 | 28.96% |
| CLIP编码 | 0.509秒 | 0.500秒 | — |
| DiT采样 | 15.159秒 | 15.161秒 | — |
| VAE | 1.098秒 | 1.097秒 | — |
| 保存 | 0.755秒 | 0.756秒 | — |

完整耗时范围分别为43.188–44.506秒与36.468–36.744秒。十次固定轨迹扩写和RGBA均与原始图片一致。本轮PE约占整图52.1%，DiT约占41.4%。各节点为墙钟计时，原始events保存了每次请求的边界。

## 自然输出与质量

| 自然请求 | 基线tokens/整图 | 本轮tokens/整图 | 整图降幅 |
|---|---:|---:|---:|
| 原提示词及原seed | 1909 / 44.449秒 | 2094 / 38.576秒 | 13.21% |
| 两类seed均加1 | 1481 / 38.284秒 | 1692 / 34.354秒 | 10.27% |

原生RoPE改变浮点执行与舍入，完整有效1909行logits相对基线RRMSE为5.1793%，top-1相同1835/1909，平均KL为0.006207。自然扩写与图片发生变化。两组图片均保留人物、花束、蓝色蜂鸟与绿色植物背景，构图、服装和花冠细节改变，人工质量验收待用户确认。

质量统计仅覆盖有效输出。基线的自然轨迹和同teacher复测在前1909行logits逐元素一致。四token回读实际计算1912行，1910/1911两行位于EOS之后，已排除出质量统计。

512²、更换seed、1024²且thinking关闭、卸载重载均成功。重载图122.551秒，2094 tokens，扩写和RGBA与对应卸载前候选一致。测试进程峰值活跃分配约50.092GiB，卸载后40–60秒稳定分配9,866,979,200 bytes，约9.19GiB。

191项回归通过、1项CUDA专属跳过。覆盖生成回读、EOS、RNG隔离、异常清理、FP8/tied与untied配置、缓存及回退、任意矩阵RoPE、标准旋转的FP64参照、MLP、norm、attention和相关Qwen block。19次普通节点请求的fallback计数均为空。

## P1完整token执行链

| 候选 | 固定短窗口结果 | 取舍 |
|---|---|---|
| 完整单token图 | 256 tokens约4.809秒 | 包含embedding、32层、head、penalty、采样和设备状态，速度低于最终候选 |
| 两步静态展开 | 256 tokens约3.763秒，cold280.983秒 | 图外控制减少，剩余边界及kernel费用仍突出 |
| 四步静态展开 | 256 tokens约3.280秒，cold1381.120秒 | cold包含四步和尾部三步两个shape |
| 四步＋原生RoPE＋改进分区 | 对齐257 tokens约2.927秒，cold85.164秒 | 与最终模块化候选的收益接近 |
| 八步＋原生RoPE＋改进分区 | 对齐257 tokens约2.974秒，cold154.836秒 | 展开规模增加未形成额外净收益 |
| 完整单步lazy HPUGraph | 256 tokens约3.95秒 | 已验证请求独立Generator及持久重放 |
| 四步函数式状态HPUGraph | 256 tokens约4.17秒，prefill/准备约1.02秒 | 块内函数式更新、块尾提交可运行，速度低于最终eager候选 |
| 频率表及旧控制提交清理 | 与原生RoPE组合约2.876秒/256 tokens | 保留FP8请求内频率表、紧凑输入和立即状态控制清理 |

完整图的显式设备seed采样曾被后端错误放入CPU fallback。独立进程placement修正后的最小采样测试可复现且draw随step变化。实际sample作为输出保留，确保teacher forcing不会消除head或采样。早期未暴露真实sample的`screen3b/token1`已排除正式比较。

FX分区器的可达集合与循环检测形成严重冷编译开销。八步旧实现RSS约63GB，随后主动终止以验证更紧凑的表示。bitset表示通过1000组随机DAG分区集合等价检查。后续四/八步试验同时调整了分区方式、RoPE边界和窗口对齐，cold差异包含这些条件。bitset代码保留在调查脚本中。

lazy Graph先遇到QuantizedTensor weight-sharing兼容问题及高维/index dtype的CPU fallback，已分别通过独立环境选项和二维/int32索引修订。四步原地状态图在capture_end出现`target node not found`，块内函数式状态及块尾提交后通过。这些原型具备各自实验条件，当前部署使用已验证的eager执行路径。

## P2共享量化、GDN与矩阵供数

| 方向 | 实测结果 |
|---|---|
| QKV/QKVZ共享量化 | 256 tokens约3.780秒，对照3.912秒，收益较小 |
| 按原scale合并FP8投影 | 约3.588秒，逐输出通道scale可运行 |
| 扩展到MLP gate/up | 在原生RoPE组合上约2.954秒；频率表等全组合约2.924秒，最终无组合副本候选约2.876秒 |
| 立即更新GDN TPC | 实现4/8/16个value列，16列约3.886秒，4列约3.903秒 |
| 持久value-major状态布局 | 4列约3.809秒，未形成充分的组合收益 |
| 原生causal conv | BF16且关闭融合激活时，局部输出/state相同；完整短文本约3.858秒 |
| M=1/2/4/8/16/32布局 | 三份真实MLP权重共150,994,944 bytes，超过48MB SRAM；M1有用权重带宽约1.515TB/s，M16最高约1.756TB/s |
| 转置与K填充 | 转置在M8有局部改善，M1接近；K填充16/32/64/128/256/512的有用带宽均低于原始约1.529TB/s |

FP8组合保留每组原scale，准备权重副本后测量实际收益。最终组合缺少稳定叠加收益，产品草稿中的额外权重副本实现已撤回。小M数字采用有用权重字节/时间，完整HBM计数使用下文的原生测量。

新的TPC基于当前立即更新公式，保留FP32状态及周围BF16边界，沿用现有构建工具链。原生conv的正确ABI为BLC输入、BKC状态、KC权重。合并SiLU省略了中间BF16舍入，局部差异约0.292%，测试使用独立激活边界。各失败实现、首次接口错误及修订源码保存在对应试验快照。

## P3有效长度、GQA与RoPE

256/512/1024有效容量分桶均已实测。512分桶的完整组合warm约22.977秒，首轮cold414.129秒，对照组合warm约23.116秒。新shape的编译代价明显。

实际Q16头、KV4头、D256、容量6899的GQA微基准中，`valid_seq_len`与causal/right-padding接法在有效长度2804/4097/4712均出现约100%相对误差。显式FP32分组matmul约97–99微秒，现行FusedSDPA约54–65微秒。`recompute=True`输出相同，局部时间接近。

原生RoPE保留独立编译边界后形成主要收益。前轮的整图内联曾变慢约35%–39%。本轮选中`comfy_hpu::qwen35_rope`入口，由FP8请求缓存显式选择，使用内建Qwen频率函数及64维标准旋转。任意2×2矩阵通用入口的精确回归通过。[原生RoPE参数与建图源码](https://github.com/HabanaAI/gaudi-pytorch-bridge/blob/27eccbcffeab10842c9f29f64c6bf50d8f65d78f/hpu_ops/backend/rotary_pos_embedding_gen.cpp)

频率表与逐token计算在多个早晚位置逐元素相同。未经紧凑复制的切片引入数值差异与额外编译，clone后的256行logits与逐token原生RoPE候选逐元素一致。底层视图/缓存机制尚待更低层定位。产品使用紧凑频率输入，并检查自定义频率函数的回退。

## 新硬件测量与剩余差距

原生HBM窗口为第1537–1600个输出附近的64个decode间隔。无profiler中位0.624044秒，采集0.619661秒，扰动−0.70%。窗口读540.778GB、写16.387GB，约8.706GB/token。

| 指标 | 本轮 |
|---|---:|
| 单token无profiler时间 | 9.751ms |
| HBM带宽 | 0.893TB/s，峰值36.44% |
| 有效FP8 Linear算力 | 1.626TFLOP/s，峰值0.188% |
| 64-token trace扰动 | 5.04% |
| kernel区间并集/设备跨度 | 399.890 / 643.110ms，62.18% |
| 未被kernel事件覆盖的区间 | 243.220ms，约3.80ms/token |
| host launch_recipe | 18/token，原调查为26/token |
| host launch | 39.25/token，原调查为90/token |
| 主线程CPU时间 | 6.76ms/token |

trace的FP8 GEMM独占区间约202.741ms，其它GEMM45.438ms、softmax5.934ms、其它kernel145.777ms。kernel活动区间与主线程CPU时间存在重叠，未覆盖区间还需要进一步区分供数、依赖、提交及其它设备活动。

现有kernel并集约6.25ms/token，已高于80%带宽预算4.44ms/token。后续仍需降低kernel执行及调度费用，并提升小M矩阵供数。文本预算与DiT的D1/D2工作继续保留，当前数据不足以认定阶段硬件效率或理论资源利用率达到80%。

HBM汇总脚本现从窗口元数据读取间隔数，支持本轮64-token与历史16-token资料。一次补充温度/功率遥测出现不完整记录，已从该汇总舍弃。最终部署的单一写入器采集1663条完整记录，其中1350条位于请求执行区间且频率有效。SOC频率1650MHz，温度43–55°C，功率80–404W，功率/温度违规标志均为零。

## 普通服务部署验收

8189已部署`07fedda3+staged.87ea451fce53`。普通首图141.564秒，两次完整warm为38.635/38.316秒，卸载后重载121.408秒，四次扩写与RGBA均匹配已验证的自然候选图。最后恢复原MTP auto参数，六个文件哈希匹配暂存快照，验收时队列为空。改动保留暂存，等待人工质量验收。

## 验证修订与交付身份

首轮原生RoPE直接替换通用矩阵入口时，回归发现其标准旋转假设，已通过独立Qwen入口和缓存准入修正。新增测试夹具补齐独立FP8输出头，覆盖tied/untied配置。当前191项回归、19次普通节点请求、两种尺寸及重载均通过。

- 产品baseline为`07fedda3278b5c4be8c036fe245dc6991b91dd4a`。
- 暂存tree为`7f6b735fa3cd5350e9d6cbfdebb9fe8b79d9eb2d`。
- 暂存binary patch SHA256为`87ea451fce5358377e83f3836aeda430420fef2206e600251b33d0a1387268be`。
- 六个产品/测试文件的SHA256已与服务器验证快照逐项匹配。
- 当前状态、部署文件哈希及普通API验收写入[结构化证据](../evidence/qwen80-pe-execution-2026-11-16.json)。

服务器原始资料根目录为`/home/sonny/gaudi2/comfyui-hpu/performance-qwen80-pe-2026-11-16`，正式验证在`product-validation4`，部署及回退快照在`deployment`。`screen*`、`full-quality`、`reference-check2`、`rope-quality`、`refinements2`、`token-refinement`、`token-completion`、`graph-*`、`padding`保留各项条件、源码快照与失败日志。

日期化候选脚本针对四token回读基线生成源码变体。复现使用相应`*-source`及`baseline-source`快照，当前已部署实现的后续试验需要适配新的入口。脚本检查这一源码依赖，并记录产品文件哈希。

自然配对图位于独立产品checkout的`output/gaudi/qwen80_pe`，文件为`natural-baseline_00001.png`、`natural-product_00001.png`、`seed-baseline_00001.png`、`seed-product_00001.png`。本地检查副本在`comfyui-hpu/results/qwen80-pe-2026-11-16/images`。固定工作量、自然长度变化和人工质量验收共同决定本轮交付评价。
