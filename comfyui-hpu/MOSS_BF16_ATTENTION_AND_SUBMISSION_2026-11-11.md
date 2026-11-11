# MOSS BF16 attention与并行GDN

2026-11-11，`64155fe4`→`a8f42df2`。证据根目录为`/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-11-bf16-next`。指标、源码哈希及部署检查见 [紧凑证据](evidence/moss-bf16-next-2026-11-11.json)。

## 正式改动与测试条件

- `41f01e68`将短序列DeltaNet递推转为FP32有限nilpotent矩阵级数，保留BF16 gate/norm/silu舍入及deferred状态语义，最多8行，允许零decay和分支重排。
- `700bdb6d`在HPU BF16、batch1、24×128 heads、N≥4096、默认无mask/参考KV的编译推理路径按独立head调用原attention dispatcher，每个head携带原container/preferred_attention。
- `a8f42df2`编译普通BF16八行GDN的外围投影、conv及递推。标准LinearKV、decode、无hooks/梯度等条件门控，四行路径在编译函数外clone激活。

固定caption正307/负23、1024²、25步、CFG2.5、er_sde/simple。固定MTP使用1970输入/298输出，先执行原运算再强制记录token/控制决策，校验258次draft、130次sample、4次width、130组控制表。各1 cold＋5 warm，自然质量另测。

| 工作量 | 基线 | 最终产品 | 耗时下降 |
|---|---:|---:|---:|
| 独立固定MTP文本 | 5.719843s | 4.866763s | 14.91% |
| 固定MTP完整图 | 21.043838s | 19.978488s | 5.06% |
| TextGenerate节点 | 6.109491s | 5.250935s | 14.05% |
| 同工作流KSampler | 14.368867s | 14.133343s | 1.64% |
| 固定caption缓存整图 | 14.627998s | 14.386117s | 1.65% |
| 同caption DiT | 14.375917s | 14.134462s | 1.68% |

完整cold58.163453→64.665240s，最终warm19.905970–20.121573s。缓存14s目标差0.386117s。head原型A/B/A各5 warm为14.638745/14.401615/14.637128s，产品复验确认小幅收益。

## 筛选结果与思考

| 候选 | 结果 | 取舍 |
|---|---|---|
| 每组24/12/8/6/4/3/2/1 heads | 单head正分支SDPA3.681→3.201ms、block7.090→6.965ms，负block6.579→6.605ms | 用完整25步确认约0.24s收益 |
| 重排布局、折叠batch/head、attention_basic | 无更好block组合，basic更慢且有舍入差异 | 保留原dispatcher |
| 只编译外围GDN/head/全部Linear | 5.667515/5.674143/5.656065s，基线5.719843s | 约1%，处于波动内 |
| pending分桶/精确裁剪 | 5.472340/5.463734s | 被并行递推方案替代 |
| 共享CPU conv索引 | 5.675497s，与并行递推组合5.216281s | 无稳定叠加收益 |
| 并行递推正式中间版本 | 文本5.261498s、完整20.339359s | 继续查外围提交 |
| 递推函数prototype/product/force_static | 0.335–0.338ms | 单独编译选项无明显收益 |
| 改写后扩大GDN区域 | 原区域复测5.240083s，扩区原型4.883419s，正式4.866763s | 采用八行受控分流 |

最新两步trace相比基线，CPU Linear154→10、launch964→820，设备活动并集59.396→47.885ms、间隙49.023→37.755ms、跨度108.419→85.639ms。短profile用于归因，正式速度采用无profiler完整测量。递推结构变化后应重新测试区域边界。

## 数值与完整模型故障

80组CPU FP64/FP32代数验证覆盖1/2/3/4/8行、batch2、任意合法行重排、零decay/beta，FP32最大误差1.19209e-7。15组实际HPU输入覆盖5个阶段×3层，state RRMSE最大8.03901e-8、输出2.66693e-5，gate/sumsq相同。输出FP64误差比0.99997564–1.00000068。

最终全部文本改动的1033行logits相对原基线top-1为1026/1033，平均KL0.000609445。扩大GDN区相对并行核心为1023/1033、KL0.000655902。各实现自己的重复logits相同，自然采样会受这些差异影响。

MTP `3`真实模型触发`synStatus26`，异步栈最初指向attention/MLP，逐段同步定位四行GDN。补齐到八行在离线回放/单测通过，完整模型仍失败。逐组重建输入，仅激活x经CPU重建或HPU clone有效，最终采用四行x在编译函数外clone。厂商内部原因尚未确认。最终回放输出RRMSE2.24574e-10、最大差1.86265e-9，state/gate/sumsq相同。

复现入口为`moss_four_input_trials.py`及`MOSS_FOUR_RAW_INPUT=1`，真实输入在`mtp-four-fences/delta-four.pt`。另外HPU专项曾被CPU测试全局选成sub-quadratic backend，原版与候选的编译输出相同，显式选回日常PyTorch attention后按原阈值通过。测试fixture需核实实际dispatcher。

## 最终验收

- 194 passed、1 CUDA skip，包含16项GDN扩区测试。覆盖1–8行、batch2、权重/状态更新、hooks、CPU/梯度/禁用编译、mask和参考KV。
- 512/768/1024、卸载重载、MTP off/3/tree3s1与原INT8完整图通过，latent/VAE finite、fallback为空。
- 固定caption attention图片MAE0.09832/255，基线/候选自身重复0.10406/0.10419，最大像素差均5。
- 自然茶壶、海报、人物三组caption长度339→340、260→303、289→312，图像MAE30.918–69.111/255，含扩写变化。主体与MOSS大字保持，光线/构图细节明显改变，海报黄花基线4朵、候选3朵。质量范围限于这三组。
- 完整allocator峰值39.4984→39.3754GiB，同流程卸载均2,685,928,832 bytes，多形状/INT8最后为2,686,190,976 bytes。
- 六份部署文件哈希一致，8189原图恢复cold66.725s、缓存14.751s，最终检查队列为空。此为该轮部署记录。

后续投入分析见 [BF16复盘与FP8路线](MOSS_OPTIMIZATION_STRATEGY_2026-11-09.md)。
