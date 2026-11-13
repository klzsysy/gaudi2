# Qwen Image 2.1 全 BF16 工作流深入优化

2026-11-13 开始，产品基线 `9f2ff96e`。用户要求同时深入分析文本扩写与 DiT，并验证完整 BF16 工作流。

## 固定基线

保存用户最后一次实际运行的 UI/API 图到 `workflows/qwen21-user-bf16.json` 和 `qwen21-user-bf16-api.json`。组件为 Turbo BF16、Qwen3-VL 8B BF16、Qwen Image 2.1 VAE BF16、匹配 T2I PE 的 BF16 解码版本。参数保持用户当前 9:16、1.2MP、8步 Euler/simple、CFG1、thinking关闭、PE开启、seed1。扩写上限4096、presence penalty1.5等采样设置照原图。

测量采用每次重算 PE/编码/采样/解码、保留 loader 的 cold＋五次 warm，以及单独缓存文本的五次出图。同步分阶段诊断和16步文本/一步DiT profiler使用独立请求。固定输入与实际 token 数一并保留，最终普通服务再次验收。

## 调查顺序

1. 拆解 PE prefill/decode、输出头与采样，以及编码、DiT、VAE 和加载费用。设备 trace 按区间并集计算活动与空档。
2. DiT 首版扩大编译受 prefix-cache closure 重编译影响，本轮把 QKV/RoPE 和 SDPA 放在稳定边界内，分别比较，再检查组合收益与质量。
3. PE 无 MTP 权重，检查 MOSS 的八行 GDN 编译在单 token 生成上的适用范围，评估逐 token 固定开销、概率处理和递推。
4. 同输入数值、自然输出、不同形状和卸载重载通过后接入产品，保留失败路径及具体原因。

原始资料在服务器 `/home/sonny/gaudi2/comfyui-hpu/performance-qwen21-bf16-2026-11-13`。执行入口为 `probes/qwen21_deep_jobs.py`，诊断和临时候选分别为 `qwen21_deep_server.py` 和 `qwen21_deep_candidates.py`。

## 基线测量

当前实际输出为832×1504。五次完整warm为21.089/21.057/20.833/21.028/20.819秒，中位21.028秒。节点中位PE11.416s、基础编码0.356s、KSampler8.089s、VAE0.656s、PNG保存0.448s。PE每次427输出tokens。

切换到缓存文本测试的第一条请求因fingerprint变化重新运行PE，已从缓存统计排除。后四次为9.223/9.216/9.221/9.215秒，中位9.2185秒。

独立同步诊断记录prefill0.447s、426次decode合计9.738s、427次输出头0.646s、427次采样1.744s。DiT累计attention及投影7.820s，其中SDPA3.605s，FFN1.130s，modulation0.752s，residual0.494s。同步边界有额外扰动，这组分项用于定位。

首轮在HTTP执行线程启动torch profiler后出现 `External init callback must run in same thread as registerClient`，收集期间history请求30秒超时导致调查进程终止。已有完整warm和阶段数据保留，trace请求退出统计。后续改为主线程独立重放并允许诊断请求的暂时HTTP超时，避免让采集过程取消自身。

## 组件候选与数值边界

| 候选 | 相同工作量或实际运行结果 | 核查结论 |
|---|---|---|
| 稳定 QKV/SDPA 两个编译入口 | 完整图19.050/19.324s，缓存文本7.395/7.397s | 避开首版 cache closure 重编译上限，DiT约6.27s，扩写427 tokens相同 |
| 整体编译 QKV/norm＋原生 RoPE | DiT约3.480s，缓存文本4.600s | 实际block对FP64误差比1.0903，显式FP32 norm仍为1.0910，继续拆分舍入边界 |
| 保留投影和RMSNorm边界，原生RoPE＋独立SDPA编译 | 真实block30.932→15.521ms，输出逐元素一致，FP64误差比1 | 完整图仍有局部像素差异，差异已定位到前缀attention的编译舍入 |
| 上述DiT与概率编译组合 | 五次完整warm中位16.168s，缓存文本5.229s | 427-token自然扩写完全相同，最终质量验收见后文 |
| 单head attention | 真实block13.077→13.875ms | 在本组32-head、4888 query＋前缀形状下较慢 |
| 单token整GDN编译 | 固定427-token三次中位11.960→9.918s | 固定logits RRMSE0.9846%、top-1相同421/427、平均KL0.000580，撤回接入候选 |
| GDN关闭算术分解 | 首次运行报8与32维度不匹配 | 记录Habana lowering失败，原路径可正常执行 |
| 采样概率编译 | 单独完整图19.832/19.860s，PE约10.311s | 自然427-token序列相同，整图像素与基线相同 |

单GDN候选的自然输出缩短至331 tokens，DiT输入也随之改变，因此其自然整图19.2s仅作使用观察。固定teacher独立测试才提供相同427-token工作量的计时，logits导出另测，避免CPU复制污染性能样本。

## 主线程 trace

HTTP线程采集失败后，主线程16-token窗口成功导出。无profiler两轮窗口0.4204/0.4241s，采集窗口0.5314s，存在约26%扰动，导出本身另有约一分钟费用。trace观察到6032次launch、2704次recipe launch、2064次Dynamo cache lookup。单步执行包含24个GDN、8个完整attention和32个MLP。

该trace可见设备事件跨度529.55ms，事件并集251.82ms，空档277.73ms。MME相关区间95.57ms，其它事件包含TPC、布局操作与设备同步。计数包含多engine事件的区间并集，CPU多线程Self/Total时间不能直接相加。空档受profiler扰动影响，尚未将每段空档逐一归因到主机依赖。

真实DiT block的trace跨度30.87ms，未覆盖区间0.317ms，主要成本为float32广播乘法、布局拷贝、softmax和GEMM。原生RoPE候选将广播乘法热点显著压缩。此处文本decode偏提交与细碎操作，DiT更偏实际设备算子和内存布局，两者需要各自验证分区。

## 完整图差异定位

相同seed的基线重复图逐像素一致。较快候选虽在选定target block上逐元素一致，整图仍有约1.70/255的RGBA MAE，差异主要可见于人脸。将RoPE替换成保留原addcmul的编译版本后，整图差异完全相同，排除了该旋转公式作为这一整图差异的来源。

进一步将有mask的文本前缀attention交回原执行路径，target图像的无mask attention继续区域编译，完整图恢复逐像素一致，warm16.022s。最终接入仅编译无mask attention，保留投影和RMSNorm的舍入边界，原生PAIRWISE RoPE保留四个系数。这样同时避免闭包重编译和前缀softmax舍入变化。

四个输入文件的header均为BF16，分别为DiT265、Qwen3-VL750、PE760、VAE238个张量。用户当前PE输入为2804 tokens，输出427 tokens。PE的310个量化Linear已由先前INT8文件解码，本轮继续使用该BF16文件。


## 最终接入与验收

产品提交 `17b4b231` 接入保留舍入边界的原生RoPE与无mask attention编译，`ed9552e5` 接入temperature=1.0的HPU概率编译。非单位temperature的新增测试有三项BF16概率舍入差异，编译门控据此收窄，原测试条件继续逐元素验证。

| 指标 | 原产品 | 最终产品 | 时延减少 |
|---|---:|---:|---:|
| 每次重算全部节点的完整图，五次warm中位 | 21.028s | 16.079s | 23.54% |
| TextGenerate | 11.416s | 10.593s | 7.21% |
| TextEncodeQwenImage21 | 0.356s | 0.353s | 约0.003s |
| KSampler | 8.089s | 4.006s | 50.47% |
| VAEDecode | 0.656s | 0.655s | 约0.001s |
| SaveImageAdvanced | 0.448s | 0.447s | 约0.001s |
| 复用扩写和编码结果后的整图 | 9.2185s（四次） | 5.130s（五次） | 44.35% |

完整图范围原产品20.819–21.089s，最终15.905–16.147s。随后回退运行原数学路径，三次warm中位20.790s、缓存文本9.219s，确认收益可复现。独立进程前六次同条件请求的allocator峰值为76.645→60.335GiB，减少16.31GiB，fallback计数为空。

12组实际block对照包括层0/16/31的三个前缀，以及步0/4/7的九个target，全部逐元素一致。最终真实block重放为30.932→15.100ms，完整输出和CPU FP64抽样误差均与基线一致。

六组完整对照覆盖seed1、seed2、512²、1024²、thinking开启以及官方Turbo ManualSigmas。每组的完整扩写token序列和最终RGBA PNG均逐元素一致。输出长度分别427、498、427、427、1877、427，基线和最终相同。159项回归通过、1项跳过，覆盖新采样/attention边界以及已有Qwen decode、MOSS block、GDN区域和缓存路径。

普通8189服务最终验收成功。原用户图首次49.833秒，换seed2完整生成18.267秒，清理后重载原seed1为42.926秒。调用`/free`后10/20/30秒读数仍在下降，40/50/60秒稳定为3,383,652,608 bytes（约3.15GiB）。全BF16模型的卸载过程需要等待实际完成，25秒观测不足以代表稳定占用。

本轮覆盖用户832×1504及512²/1024²、batch1文生图。2K、完整批量和编辑工作流需要独立验收。文本PE当前约占完整耗时的66%，主要剩余机会在逐token执行中的GDN、KV和主机提交。已撤回的单tokenGDN扩区需要解决固定logits变化后再评价接入。单步prefill约0.45秒，跨请求前缀缓存的收益预算较小。

原始trace和张量保存在服务器调查目录，紧凑结果见 `evidence/qwen21-bf16-2026-11-13.json`，普通服务history与完整文件哈希保存在忽略的本地 `results/qwen21-bf16-2026-11-13`。

消融实验中，单独原生RoPE的DiT为5.596s、进程峰值73.48GiB，单独attention编译为6.491s、峰值60.43GiB，各一条warm样本。结合trace，这组结果将时延收益分别关联到RoPE广播/布局开销和attention执行，将主要峰值下降关联到attention编译。最终组合另有五次warm和回退复验。

最后普通服务的cold与卸载后reload扩写文字、PNG像素均相同。当前服务日志没有新增Traceback、执行异常或recompile_limit，队列为空，四个产品/测试文件的本地与部署SHA256一致。
