# Qwen Image 2.1 Turbo FP8最后一次生成的性能调查

2026-11-16，Asia/Shanghai。产品版本`07fedda3278b5c4be8c036fe245dc6991b91dd4a`，单卡Gaudi2 HL-225、Habana1.24.2、eager＋现有区域/整步编译。本轮完成工作流复现、节点计时、硬件采样与瓶颈分析。调查记录及脚本暂存供审阅，普通8189服务已恢复并验证。

## 原始请求与复现范围

原图为`Qwen_image_2.1_00062.png`，prompt ID为`3b17fc44-fda6-4009-b813-2199c4e13653`。服务history记录开始于13:54:50.177，结束于13:55:35.642，实际执行45.465秒。原始history包含节点缓存清单和整图时间，节点分项与硬件指标来自本轮同参数复现。

| 项目 | 实际设置 |
|---|---|
| 分辨率 | 1088×1920，batch1 |
| 采样 | Euler、官方Turbo八步sigmas、CFG1.2 |
| DiT | `qwen_image_2.1_turbo_fp8.safetensors` |
| PE | `qwen3.5_9b_qwen_image_2.1_pe_t2i.fp8_from_int8_fp32.safetensors` |
| CLIP | `qwen3vl_8b_fp8_scaled.safetensors` |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` |
| PE参数 | thinking开启，max_length4096，MTP auto，temperature1、top_k20、top_p0.95、presence_penalty1.5 |
| seed | PE447606998181262，latent284487731643277 |
| DiT前缀KV | auto、default精度 |

完整冻结图见[API工作流](workflows/qwen21-latest-2026-11-16-api.json)。原请求的计算节点全部执行，缓存清单主要包含loader、尺寸、采样设置和cache配置节点。复现强制重算PE、CLIP、采样和VAE，保留loader及已编译recipe，沿用用户的DiT KV设置。

完整复现1 cold＋3 warm。四张复现图的扩写文本、RGBA像素均与原图逐元素一致。cold为146.058秒，三次warm为45.262、44.553、44.773秒，中位44.773秒。所有完整计时请求的HPU fallback计数为空。

## 耗时分布

下表取总耗时居中的完整warm请求，分项可相加。其它包含节点调度、切换及预览等开销。

| 阶段 | 秒 | 占整图 |
|---|---:|---:|
| PE扩写 | 27.267 | 60.90% |
| CLIP正负编码 | 0.488 | 1.09% |
| DiT采样 | 15.144 | 33.82% |
| VAE解码 | 1.100 | 2.46% |
| PNG保存 | 0.758 | 1.69% |
| 其它 | 0.017 | 0.04% |
| 合计 | **44.773** | **100%** |

PE输入2803 tokens，实际输出1909 tokens，日志进度最后显示1908。按token ID定位`</think>`，thinking包含结束标记共812 tokens，后续正文、换行及结束符共1097 tokens。三次warm中thinking段中位11.900秒，后续段中位15.440秒，两列为独立中位数。首token中位0.433秒，文本总吞吐约70 tokens/s，首token后的吞吐约71 tokens/s。长自回归循环占据PE主要时间。

PE模型header的201个FP8矩阵合计7,929,331,712个元素，未包含MTP头张量。产品`Qwen35.generate`在该文件下选择普通单token生成，界面`auto`设置按这一模型结构执行。

复用文本结果的一次有效测量为17.023秒。首次从强制重算切换回节点缓存的请求发生fingerprint变化，PE再次执行，耗时45.254秒，该轮归入缓存过渡记录。有效缓存样本数为1。

本次thinking开启，区别于前轮报告中thinking关闭、420或806 tokens的自然扩写。历史23.943秒和30.248秒对应各自工作量。本轮13:53相邻用户请求thinking关闭、耗时29.693秒，latent seed也发生变化，保留为历史上下文。

## 硬件利用率

普通完整warm期间每250ms读取hl-smi和进程/线程CPU时间。三个样本中PE平均板卡功率153–155W、DiT361–369W。PE进程CPU约3.76–3.89个逻辑核等价，DiT约1.72–1.88个逻辑核等价，均为全进程各线程累计CPU时间除以墙钟时间。主机为32核64线程。

warm阶段频率均为1650MHz，观察温度47–55°C，功率/热限制计数均为0。HPU allocator峰值最高49.955GiB，结束时36.412GiB，设备总容量96GiB。该完整工作流的容量、温控和频率采样均保留在证据中。

本机驱动的`utilization.aip`来自功率归一公式。PE均值约19–21%、DiT约60–61%。算术和带宽利用率采用下面的工作量及硬件计数口径。[本机AIP公式核验](optimization/BOTTLENECKS.md#aip口径修正)

### HBM及有效矩阵算力

沿用已经用copy/GEMV校准的AXI返回beat计数，峰值分母为HBM2.45TB/s和FP8稠密865TFLOP/s。PE抽样第1537–1552 token，16次forward，前1567 tokens与完整自然生成逐元素一致，截断只在第1568个token代入结束符。DiT抽样第4次forward，负条件、目标行8160，前缀KV开启。

| 指标 | PE后段16 tokens | DiT一次负条件forward |
|---|---:|---:|
| HBM读取 | 135.593GB | 790.296GB |
| HBM写入 | 4.086GB | 596.148GB |
| 总读写量 | 8.730GB/token | 1.386TB/forward |
| 无profiler窗口 | 0.221664秒，两次plain中位 | 0.912191秒 |
| 归一化有效带宽估算 | **0.630TB/s** | **1.520TB/s** |
| HBM峰值比例 | **25.72%** | **62.04%** |
| FP8 Linear有效算力 | 1.145TFLOP/s | 124.867TFLOP/s |
| FP8 Linear峰值比例 | 0.132% | 14.44% |
| 计数采集窗口 | 0.380603秒 | 0.913831秒 |
| 采集相对plain扰动 | **+71.70%** | **+0.18%** |

PE带宽采用采集字节数除以未插桩窗口时间，采集窗口自身的速率为0.367TB/s。其较大扰动限制了对原执行重叠及等待分布的解释精度。每token8.730GB的访问量与此前同产品早位置8.732GB相近。计数取整的单向敏感性约为PE2.20%、DiT1.22%，这些范围不构成统计置信区间。

FP8有效算力仅计算FP8 Linear的有用矩阵工作，FMA计2 FLOPs。DiT共有192个FP8矩阵、6,979,321,856个元素，缓存后的计算量为`2 × 8160 × 6979321856 = 113.903 TFLOP`。编译FFN内部的64个Linear未出现在host-level FP8调用清单中，因此使用完整权重元素数计算，host可见的128次调用只覆盖35.047TFLOP。attention、非线性、量化、状态更新及调度另有成本。

### DiT设备时间线

独立torch profiler窗口为0.937942秒，比native任务中的plain窗口高2.82%。设备kernel跨度916.471ms，活动区间并集887.230ms，覆盖率96.81%，kernel间隙29.241ms、占3.19%。多engine/lane事件使用区间并集，以下按MME优先归类互斥时间。

| 设备活动类别 | 毫秒 | 占设备跨度 |
|---|---:|---:|
| BatchGemm，主要为attention QK/AV | 386.536 | 42.18% |
| 其它GEMM，包含FP8 Linear | 157.380 | 17.17% |
| softmax独占 | 156.572 | 17.08% |
| 其它kernel独占 | 186.741 | 20.38% |
| kernel间隙 | 29.241 | 3.19% |

softmax自身活动并集379.637ms，与矩阵运算存在重叠。表中156.572ms为扣除MME重叠后的独占时间。`GEMM`、`gemm`与`BatchGemm`原始名称及EventName均随trace保留，部分eager矩阵事件只有`n_ab`名称，因此对全部矩阵活动采用上述语义边界。

Qwen21的FP8路径覆盖Linear，当前attention使用BF16执行。CFG1.2使八步采样包含正负条件共16次DiT forward。长序列attention的矩阵运算、softmax和中间量访问是DiT主要成本。

## 瓶颈判断

1. **首要瓶颈是PE的长单token解码链。** 1909个输出tokens累计27.3秒，包含约11.9秒thinking。单token FP8矩阵工作量约15.86GFLOP，相对8.73GB读写量的算术强度约1.82 FLOP/byte。单token执行受权重流量、小矩阵效率、状态更新与提交延迟共同制约。按峰值HBM折算的读写下界约3.56ms/token，实际窗口约13.85ms/token。主机等待、小kernel与状态计算各自的份额需要低扰动时间线继续定位。
2. **第二瓶颈是DiT的长序列attention及访存。** 采样约占33.8%，设备活动覆盖率96.8%，HBM有效带宽约峰值62%。BatchGemm与softmax独占时间合计约59.3%的设备跨度，表明attention融合、中间量读写和相关kernel效率具备主要优化空间。FP8 Linear全forward有效峰值比例需要连同这些成本解释。
3. **CLIP、VAE、保存及节点外开销合计约5.3%。** 即使把这些阶段全部省去，当前44.8秒流程的理论节省也只有约2.36秒。cold146.1秒的加载/首次编译成本作为单独场景保留。

性能投入顺序据此为PE单token执行/访存效率，其后为DiT attention与中间量处理。thinking输出长度是本次实际工作量的重要组成，改变该参数会改变扩写及图片，应通过用户体验与质量对照评价。

## 测量问题与恢复验证

- 文本详细torch trace在切换record_function分支后触发重编译，16-token窗口从0.219758秒变为53.535390秒。该trace已排除出设备利用率和主机空档定量结论。普通完整warm在插桩之前完成。
- native DiT采集的所有Synapse API状态均为0，history与图像验证通过。该独立进程在请求完成后的退出阶段出现`prompt_worker`栈的段错误，记录于`native-dit/server.log`。退出异常根因尚未确定。
- 普通服务已恢复到`07fedda`，原工作流经普通API冷启动147.813秒成功，恢复后的扩写文本及RGBA与原图逐元素一致，队列为空。调查前后1067个产品Python/shell文件哈希一致，并与本地`07fedda`源码核对一致。
- 四个调查脚本通过`py_compile`。trace分析器补充EventName识别、完整权重FLOP选项及精确GEMM名称区间并集，并通过构造trace检查。硬件汇总脚本支持显式PE矩阵元素数和DiT trace摘要路径。

## 证据与复现

紧凑结果见[结构化证据](evidence/qwen21-latest-bottlenecks-2026-11-16.json)。服务器完整目录为`/home/sonny/gaudi2/comfyui-hpu/performance-qwen21-latest-2026-11-16`，包含原始history/PNG、节点事件、原生计数、profile配置、模型header、源码哈希、telemetry及恢复验证。

入口为`probes/qwen21_latest_jobs.py`，在独立产品checkout内通过其venv运行。它读取调查目录的`source-manifest.json`，该清单由本地产品仓库`git ls-files`筛选Python/shell文件生成。运行前确认8189队列为空，串行停止普通服务、执行调查，finally恢复服务并提交冻结工作流。`qwen21_latest_summary.py`生成节点与遥测汇总。大型原始trace和日志留在服务器，本地`results/qwen21-latest-2026-11-16`保存摘要副本。
