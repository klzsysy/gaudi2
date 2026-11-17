# 硬件80%计划首批执行

2026-11-16，Asia/Shanghai。基线`07fedda`，冻结1088×1920、八步、CFG1.2、thinking开启、1909-token FP8工作流。本轮完成首批低扰动采集、token回读候选、短RoPE边界筛选及产品自动验证。产品暂存版本为`07fedda3+staged.d39241e049d0`，当前已部署，待人工验收。

80%硬件目标继续保留在[实施计划](QWEN21_HARDWARE_80_PLAN.md)。本轮得到一个小幅、保值的执行优化，PE和DiT的80%阶段上限证明尚待后续实验。

## 低扰动采集

旧文本trace在编译区内切换record_function包装，引发额外编译。调查脚本增加`--no-region-labels`和可配置窗口长度，直接采集已编译模型的执行事件，logits采集只在显式质量诊断中安装。

| 64-token窗口 | 两次plain中位 | profiler | 扰动 |
|---|---:|---:|---:|
| 第65–128 token | 0.893779s | 0.929179s | 3.96% |
| 第1537–1600 token | 0.900299s | 1.042468s | 15.79% |

两个位置的plain前缀均与原1909-token轨迹一致。原生早段trace的设备跨度927.580ms、kernel活动并集424.130ms、未覆盖区间503.450ms，分别为45.72%和54.28%。这些是事件覆盖口径，具体硬件等待还需关联recipe、依赖及其它活动。

早段64步记录1664次host `launch_recipe`、5760次`launch`，即26/90次每token。主线程CPU约8.1ms/token。另有每token八次短RoPE编译区调用，以及`.item()`引发的逐token回读。后段较高扰动限制了精细时间归因。采集中的profile结束/导出成本单独保留，正式性能采用plain请求。

为控制开销，本轮文本trace未记录Input Dims，形状不完整的FP8 FLOP字段标为空。必要矩阵工作量通过此前核对的权重元素数单独计算。旧71.7%扰动HBM样本继续作为访问量参考，本轮尚未完成新的低扰动HBM计数和完整阶段下界。

## 回读候选与产品实现

独立256-token筛选，各1 warm-up＋3 plain，单token回读4.205秒、四token回读3.946秒、八token回读3.896秒，分别减少6.16%和7.35%，返回token全部一致。四/八方案差值接近本轮波动，产品选择四token回读。

产品复用原Qwen整步生成准入条件。CPU、其它dtype、batch、MTP头、hooks、在线权重变换、低显存/VBAR、特定attention配置和自定义model forward继续由原准入判断处理。符合条件的HPU单请求生成每步真实执行模型、采样、penalty及cache更新，把已采样token暂存到当前generate调用的局部列表，每四步回读一次。

返回结果截到第一个EOS，max_length最后不足四个token的尾块也会回读。RNG由每个请求自己的generator管理，取消和异常通过既有finally清理。该路径可能在EOS之后额外执行最多三个forward，当前1909-token请求最多执行1912次采样，结果与进度只包含有效tokens。所有可变KV/GDN状态均归本次generate调用所有，准入条件排除了模型forward hooks等可观察副作用路径。

实现位于产品`comfy/text_encoders/llama.py`和`qwen35.py`，新增private readback策略及共享准入判断。原有模型默认一次回读一个token。

## 完整工作流交错A/B

同一产品代码、同一进程和模型，调查入口分别选择一次及四次回读，强制重算PE、CLIP、采样和VAE。顺序为1/4/4/1/1/4/4/1/1/4，各五次warm，cold单列。以下为各列独立中位数。

| 阶段 | 一token回读 | 四token回读 |
|---|---:|---:|
| PE | 27.554s | 26.254s |
| CLIP | 0.544s | 0.514s |
| DiT | 15.175s | 15.164s |
| VAE | 1.099s | 1.100s |
| 保存 | 0.754s | 0.754s |
| 完整 | **45.139s** | **43.819s** |
| 完整范围 | 44.642–46.095s | 43.645–44.300s |

PE减少1.300秒、4.72%，整图减少1.320秒、2.92%。十次warm返回1909 tokens，扩写和RGBA均与原用户图逐元素一致，fallback计数为空。候选cold146.164秒。另一组seed的512²请求成功。

## RoPE候选筛选

基于早段trace中的八个短RoPE边界，使用同一256-token轨迹、同源FP8模型，真实执行采样后代入固定tokens。每项1 warm-up＋3 plain＋一次独立logits采集。

| 候选 | plain中位 | 64步窗口ms/token | logits RRMSE | top-1相同 |
|---|---:|---:|---:|---:|
| 当前路径 | 4.137s | 14.593 | 0 | 256/256 |
| 原公式内联进入默认整步编译 | 5.742s | 21.028 | 4.506% | 250/256 |
| 原生FP32 RoPE后转BF16 | 5.597s | 19.778 | 3.988% | 252/256 |

整步采用`decomposition_mode=none`的候选在首轮遇到`aten::split_with_sizes`未注册错误。其余两个候选分别变慢38.8%和35.3%，并改变logits，均记录为失败路径。扩大编译区改变了后端执行效果，下一轮需结合实际生成图和数据流继续定位。

## 自动验证、修正与部署

132项回归通过，1项CUDA专属测试跳过。覆盖EOS位于首token及块中间、非整块max_length、空生成、采样/RNG隔离、异常清理，以及Qwen原有准入、状态、数值及DiT回归。

调查过程修正了两个客户端问题。动态生成候选的源码替换缩进断言最初失败，修正后四/八token候选完成测试。独立API卸载接口返回空响应，调查客户端最初按JSON解析并中断重载验证，现已改为支持空响应。完整交错数据与512²请求已完成，卸载重载已通过普通部署验证闭合。

普通8189冷启动145.016秒，切换MTP off→auto使完整计算节点重算，两次warm为43.856/43.957秒。该PE文件采用单token路径，两次请求的原始采样参数及最终auto设置均有history记录。卸载20–60秒的API占用稳定为9,691,059,584 bytes，约9.03GiB。重载125.588秒成功。四次普通请求均确认PE、CLIP、采样和VAE执行，扩写及RGBA与原图逐元素一致，最终队列为空，部署文件SHA256匹配暂存快照。

产品基线`07fedda3278b5c4be8c036fe245dc6991b91dd4a`，暂存树`0f4df163a296fd3829d2344ab7385158510797dd`，patch SHA256为`d39241e049d0f58c80f7c6581ebc884d96eebf3f27c090da5a2dbd60a08c54bd`。四份部署文件及SHA256见结构化证据。人工验收按仓库既有交付流程执行。

## 后续优先级

首批工作确认了回读同步的一部分成本。当前剩余成本需要继续从GDN、FP8小矩阵供数、量化准备及recipe内依赖查找。RoPE边界的单纯移除已取得阴性结果，后续复测绑定具体lowering或内核变化。DiT主线继续按计划推进真实attention精度筛选与MME/TPC片上数据流。

## 证据入口

[结构化证据](../evidence/qwen80-first-delivery-2026-11-16.json)包含五轮交错、原始阶段时间、token/像素检查、profile扰动和RoPE误差。服务器原始目录为`/home/sonny/gaudi2/comfyui-hpu/performance-qwen80-2026-11-16`，包含`trace-early`、`trace-late`、`readback`、`product-validation`、`rope`和`deployment`。

入口为`qwen80_initial_jobs.py`、`qwen80_candidate_jobs.py`、`qwen80_validation_jobs.py`、`qwen80_rope_jobs.py`和`qwen80_deploy.py`。正式产品文件的验证副本位于`product-validation/candidate-source`，失败候选限于独立进程。大型trace与logits留在服务器。
