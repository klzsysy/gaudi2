# MOSS BF16 prefill与norm/RoPE优化

2026-11-11，最终产品`1412e1f7`。新增证据根目录为`/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-11-bf16`，恢复的前轮数据在`performance-2026-11-10-bf16-deep`。checkpoint、代码哈希及指标见 [证据摘要](evidence/moss-bf16-2026-11-11.json)。

## 测试条件与基线校准

Gaudi2/Habana1.24.2，eager，1024²、25步、CFG2.5、er_sde/simple，固定caption正307/负23，N=4403/4119。自然扩写输入1970 tokens，上限4096。各1 cold＋5 warm，index6诊断排除。

服务器版本标记为`b1163be0`时实际已有native norm/RoPE源码，先按哈希恢复并提交`b25001dc`。由此确定本轮实际基线，避免把已有收益重复计算。普通正负编码为FP32激活，扩写为BF16，按真实dtype验证。

## 已采用的改动

| 改动 | 同条件结果 | 数值证据 |
|---|---|---|
| native modulation norm＋保留四系数PAIRWISE RoPE | 历史固定caption DiT18.397218→16.887383s | 18组真实输入，FP64误差比0.99940–1.00108 |
| GDN prefill静态分桶 | 正负FP32编码1.804091→0.364218s | 307-token误差5.22e-8，23-token输出相同 |
| 同一GDN prefill算法编译 | 1970 BF16单层74.477→5.473ms | 输出/状态相对误差2.93e-7/1.39e-7 |
| GDN固定298-token teacher，MTP off | 13.988245→12.048124s | top-1为295/298，平均KL0.00063676，两组均含logits采集 |
| 系数验证后的BLOCKWISE RoPE | 固定caption DiT16.893310→14.364312s，缓存17.146270→14.616611s | 18组FP64误差比0.90321–1.03067，中位0.99664 |

prefill长度补到至少64及2的幂，beta/value/log-decay补零，新增chunk为恒等状态更新，再裁回输出。1970→2048、307→512的额外工作已计入时间。BLOCKWISE只在对角系数相同、非对角互为相反数时启用，一般矩阵保留四系数PAIRWISE。prepared与直接forward执行相同检查。

## 失败候选与复测条件

| 候选 | 结果 | 后续启示 |
|---|---|---|
| prefill自动动态编译 | 307-token symbolic slice lowering在symengine `evalf`失败 | 本栈采用静态分桶，新编译器可复测 |
| modulation/residual短链编译 | 局部约3.5–4%收益，FP64误差比最高1.237 | 保留敏感舍入边界 |
| native AdaLN扩区 | 局部约2.6%，最差FP64误差增12.95% | 收益与误差联合判断 |
| 合并scale到gamma | 局部0.3–1.3%，误差比0.859–1.025 | 收益较小 |
| 加权RMSNorm/shift编译 | 局部2.4–3.1%，最差误差增18.5% | 暂缓 |
| attention query分块 | block无稳定净收益，native-256负分支误差0.322%→0.400% | 计入适配及完整block成本 |
| 显式contiguous | block约0.5%变化 | 暂缓 |
| K/V补齐64/128并mask | 负分支block6.636→6.803/6.929ms | 复制、mask和新增计算抵消收益 |

早期直接BLOCKWISE丢弃第二组系数存在语义错误，正式版本的系数检查与一般矩阵回归是必要条件。prepared/直接路径最初4项逐值一致测试失败，统一分流后原断言通过。

## 完整验收与解释

自然完整图从27.807305s降至GDN阶段24.592002s，再降至最终22.231077s，输出298→296 tokens。最终warm范围21.705189–22.590189s，cold56.876466s。严格固定工作量的结论采用上表teacher与caption数据。

- GDN CPU FP64输出/状态误差基本相同。六组GDN配对图MAE0.078–0.126/255，与重复波动相当。
- RoPE配对图MAE3.40–13.93/255，主体、大字、人物/手部通过该组目视检查，同seed细节/部分构图变化明显。
- 最终60项MOSS回归及5项prefill专项通过，覆盖512/768/1024、batch2算子、权重/状态复用、hooks、CPU、梯度、关闭编译、重载及原INT8。fallback为空。
- 完整allocator峰值39.512→39.498GiB，卸载2,686,190,976 bytes，较旧稳定值多256KiB。SOC1650MHz，记录期与limit一致。

产品提交为`b25001dc`、`d49bdd14`、`1412e1f7`。后续42层权重轮换、固定MTP与执行模式验证见 [预算与norm记录](MOSS_BF16_PLAN_COMPLETION_2026-11-11.md)。
