# 执行记录

本文保留每个问题的验证逻辑、核心证据、成功/失败结论和边界。服务 PID、重复启停、工具安装流水账和已完成计划不再逐次记录。

## MOSS长序列FP8 attention修复（2026-11-12）

- 用户原两张预览为1024×1824／1088×1936，24.965／35.748s。正负文本355／401 token，两尺寸总序列7651／7697和8583／8629。旧代码在8192上限退出融合FP8 attention，原始调查数据保留在[定位证据](evidence/moss-resolution-cliff-2026-11-12.json)。
- 产品仅移除该上限，长序列继续使用原BF16 QK、融合FP8 softmax概率和FP8 AV。新增8192／8193／8629／16385分支回归及BF16长度8193逐元素一致性检查。
- 同进程、同caption、相同三组seed对照，1088×1936旧路径35.753s／修复29.572s，减少17.29%，KSampler35.266→29.084s。修复后1024×1824中位23.721s。
- 宽1024、高1936／1952／1968的warm结果由27.784／31.102／34.303s变为27.757／27.286／26.746s。BF16 SFT的对应实测34.862／36.166／36.896s，代码无8192上限，跨界持续使用原BF16路径。
- 26个完整请求及512²卸载重载成功，fallback均为空。混合BF16/FP8、多shape进程卸载稳定5,393,876,352 bytes。306 passed、1 skipped的广泛回归通过，最终attention专项30 passed（含重复覆盖）。
- 18组1088×1936真实block覆盖正负条件、0/12/24步及0/20/41层。相对旧路径变化0.32%–3.42%。对FP32参考误差由0.42%–3.62%变为0.44%–3.97%。三组同seed图已保存，目视第一组人物、蜂鸟和花束正常，脸部、花枝等细节变化。
- 分散随机概率压力测试8193／8629／16385的FP8输出RRMSE为7.57%／7.74%／8.83%。长序列新增专项按其量化误差检查，原短序列阈值保留。逐行概率量化候选误差约3.7%–3.8%，8629长度0.627ms高于普通SDPA0.562ms，未采用。
- 初次扩展旧AV的CPU量化参考检查时，8629误差1.515%超过0.5%。分解观察到编译上下文的V量化差异约1.42%，该参考包含量化差异。原AV测试保留原覆盖，新增分支测试比较完整输出与BF16参考。详细失败日志、误差与耗时见[修复证据](evidence/moss-resolution-fix-2026-11-12.json)。独立调查文档已按用户要求删除。
- 产品提交`69fe4a54`已部署普通8189。原用户完整扩写图cold100.072s，文本缓存运行29.910／29.618s，均success，队列为空。两份产品变更文件与服务器SHA256一致。

## MOSS分层边界复核（2026-11-12）

- 对现有5 warm事件区间只读复算，节点之外耗时15.50–16.79ms，中位15.90ms，约为服务端执行窗口0.1%。
- Attention同一FX图含24个QK、24个softmax_fp8及AV。正负条件编译后图中的24个完整分数张量均`persistent=false`、`in_persist_storage=false`，最终位于DRAM。32MiB对照保留相同驻留关系。
- 对GC控制的16轮文本trace重新合并，设备跨度548.497ms、活动330.621ms、间隙217.876ms，约39.7%，最长单次间隙0.966ms。该窗口profiler扰动约1.24%，空档原因及可消除比例仍待分离。
- 本机bridge `1.24.2.481`的`compilers.py`及`fuse_attention.py`与官方`v1.24.2`提交`27eccbcf`的Git blob一致。公开bridge可进行源码级对照，标准CustomOp接口只描述单个TPC核，跨MME／TPC规划涉及更深接口。
- 决策为主攻第四层及第三／第四层交界，文本提交作为第二主线。第三层全部耗尽的证明仍不足，代码级厂商栈净收益尚待验证。详情见 [分层边界复核](MOSS_LAYER_BOUNDARY_AUDIT_2026-11-12.md) 与 [证据](evidence/moss-layer-boundary-audit-2026-11-12.json)。

## MOSS质量保值与GDN撤回（2026-11-12）

- 用户拒绝上一轮logits漂移和茶壶双把手的质量代价。GDN产品改动已撤回，本地恢复干净的`0995638a`，服务器代码与启动脚本同步恢复，扩展源码及二进制已归档。
- 回退后固定298-token程序完成，1033×248320 logits与上一轮基线逐元素一致。9种布局保持两组整block逐元素一致，8组有效编译器配置完成筛选。32MiB候选的90组输入及固定logits均逐元素一致，24MiB候选出现数值差异。
- 基线／32MiB完整流程5 warm中位数16.101／16.064s，均值16.009／16.099s。90组局部耗时比中位数1.00668，收益未确认。编译后图仍将QK分数放在DRAM，概率tile与AV共享SRAM。
- 茶壶已恢复单侧把手。三组自然图与历史样本存在像素差异，海报扩写也不同，来源尚未定位。普通8189入口cold79.338s、缓存10.693s、768² 14.213s通过，队列为空。10份运行源码哈希一致，产品工作区干净。范围、失败路径和完整数据见 [质量保值优化](MOSS_QUALITY_PRESERVING_OPTIMIZATION_2026-11-12.md)。

## MOSS第9节首轮优化，GDN候选已撤回（2026-11-12）

- SiTU代数改写、TPC SiTU／量化、query分块及GDN gather路径均完成筛选，相关实现耗时高于编译基线。
- GDN四列TPC核心在32头8行、pending3下约0.182→0.084ms。完整API流程5 warm中位数16.036→15.751s，减少1.8%，其中生成5.178→4.905s。BF16独立文本5 warm为4.804→4.492s，减少6.5%。
- 产品287项回归通过，15组真实GDN输入的输出最大相对误差1.033×10⁻⁵、状态7.794×10⁻⁸。完整FP8／BF16 logits相对差异5.68%／1.07%，top-1分别988／1033、1028／1033。
- 三组自然图片、512²重载／MTP off、原INT8文本运行成功。茶壶图出现双把手，记录保留该细节缺陷。峰值增加约9.6MiB，卸载回到2,685,928,832 bytes。
- 候选当时的普通8189入口cold76.413s、缓存10.721s、768²／MTP off13.768s，均success、队列为空。7份运行源码哈希一致，候选版本为`0995638a`加GDN改动。构建、精度和失败路径见 [历史执行记录](MOSS_OPTIMIZATION_EXECUTION_2026-11-12.md) 与 [证据](evidence/moss-optimization-execution-2026-11-12.json)。

## MOSS性能证据补齐（2026-11-12）

- 按用户指定顺序先补全栈报告第10节证据。两组独立5 warm完整流程16.009／15.995s、DiT10.190／10.183s，六段干净trace外推10.261s。GC移出profiler窗口后，负条件约2秒的Cache Lookup诊断停顿消失。
- 本机AIP由驱动功率公式计算，源码、已加载模块反汇编及531组功率配对确认。此前通用活动窗口解释已修正。
- 编译后图确认QK scores落DRAM、FP8概率tile与AV共享SRAM。原5.50TB双矩阵物化情景修正为3.664TB分数写读逻辑量。约99.4%的其它kernel已归因，SiTU约1.01s且已融合。
- 有效copy带宽1.98–2.15TB/s，真实block的HBM、MME和TPC计数已取得。BMON字节校准仍未通过，实际HBM总字节和饱和率保留为主要未决项。
- 新增72组真实输入及4组图像，FP32计算参考RRMSE0.37%–4.20%，finite及fallback检查通过。16轮文本profile经GC控制后的窗口扰动约1.24%。
- 最后恢复后正常入口cold79.834s、缓存10.693s、768²／MTP off14.071s，均success、队列为空。9份产品／测试哈希对应`0995638a`。首次旧wrapper的warm段错误、采集配置失败和字节口径限制见 [证据补齐记录](MOSS_EVIDENCE_BOUNDARIES_2026-11-12.md)。

## 阶段一：设备后端（`8326476f`）

- 在独立 HPU checkout 和独立 venv 中增加 HPU 设备探测、设备/内存查询、同步、bf16 策略及 CUDA 专用路径的安全关闭；没有使用全局 CUDA monkeypatch。
- 实机 smoke：bf16 matmul 与 CPU 参考一致，ComfyUI attention 通过，ModelPatcher 可加载到 HPU 并卸载到 CPU；HTTP `/`、`/object_info`、`/system_stats` 返回成功，设备为 `hpu:0 GAUDI2`；CPU quick-test 不分配 HPU。
- 失败路径：非交互 SSH 不加载 Habana 环境，缺少 `GC_KERNEL_PATH` 时 attention 编译报 `synStatus 26`；显式设置 `/usr/lib/habanalabs/libtpc_kernels.so` 后通过。服务持卡时独立 probe 会 `Device acquire failed`，改为串行执行并等待旧进程退出。pip resolver 会尝试替换 Habana torch，改为继承基础 site-packages、只安装缺失依赖；`pip check` 仅剩已知 torchvision 声明冲突。
- 阶段边界：未验证模型工作流、其它实体加速器或所有自定义节点；阶段一不代表完整生图通过。

## 阶段二：Krea2（`fc1c891f`）

- 初次流程能返回 history，但文本改到 HPU bf16 后保存阶段出现 `invalid value encountered in cast`，图片异常，不能通过。
- 根因验证：HPU 将 E4M3FN 存储字节 `[240,256,288,320,384,448]` 转 float 得到 `inf/nan`，第一处非有限值在 Qwen3VL 第 0 层 q_proj；不是 prompt、采样或 VAE 问题。
- 修复 `comfy/quant_ops.py::_TensorCoreFP8LayoutBase.dequantize`：HPU 使用完整 256 值解码表，保留权重/scale/量化状态；其它设备走原路径。10 项回归通过，256 个 FP8 字节在 bf16/fp32 下与 CPU 逐值一致（合法 NaN 编码除外）。
- 修复后 512²、1024² 和卸载重载均正常出图，文本/DiT/VAE 均使用 HPU bf16。量化权重仍是解量化计算，未宣称原生量化 GEMM。

| 验证 | 配置 | 端到端耗时 | 结果 |
|---|---|---:|---|
| 修复后首次 | 512²，seed 42，8 steps | 15.02 s | 红茶壶正常，无 NaN warning |
| 大尺寸 | 1024²，seed 43 | 10.93 s | 图片正常，标准差 75.36 |
| 卸载重载 | 512²，seed 44 | 13.75 s | 文本/DiT/VAE 重新加载并成功 |

## 阶段三：Qwen-Image-2.1 Turbo（`6db162dd`）

- 初次 512²/1024² history 均 success，但图片异常纹理；禁用 prefix cache、改 CPU 文本编码器均无效。固定随机输入的 CPU/HPU DiT 中间层有限，不能解释异常图。
- 同一 latent 经 CPU fp32 VAE 得到正确红茶壶，证明文本和 DiT 已产生有效潜变量；HPU VAE 逐层对照显示，带卷积 layout metadata 的 `strip_apply` 原地残差累加在分支合并处产生大误差。取消分块、`contiguous()` 和 `copy_` 均不能修复；最小复现的 4/64 通道最大绝对误差分别为 48/993。
- 修复 Wan 2.2 `strip_apply` 的 HPU 分支：保留分块，但将 `out + residual` 改为返回新张量，避免写入卷积视图；CPU/CUDA 行为不变。VAE 分块/不分块和真实工作流回归通过。
- 修复后 512²、1024²、长提示词和卸载重载均通过；14 项回归通过。CPU fp32 与 HPU bf16 VAE PNG 对照 MAE 0.2102/255、PSNR 54.49 dB；这是 VAE 对照，不是全生成器逐像素等价声明。

## 阶段四：MOSS-Image SFT（`21e98390`）

- 组件为 MOSS SFT bf16 DiT、Qwen3.5 9B INT8 ConvRot 文本模型和复用的 Flux2 VAE；通过 header/shape/dtype 对照确认旧 VAE 与官方文件的 251 个张量一致，不重复下载 VAE。
- Qwen3.5 `torch_chunk_gated_delta_rule` 长度 17/65/128 与 CPU 参考输出有限，最大绝对差 0.0004883/0.0009766/0.0009766；VAE roundtrip 通过。
- 512²、1024²、长提示词和卸载重载均通过，配置为 25 步 `er_sde/simple`、cfg2.5；图片像素范围 0–255，无非有限值或异常纹理。验证只覆盖 SFT 文生图组合，不包括 Base、编辑/控制 LoRA、SLG、prompt enhancer 或其它量化变体。

## MOSS TextGenerate 采样修复（`0139b1da`）

- 官方模板启用 `mtp=auto`、sampling 后，HPU 在 top-k 采样的 `top_indices.gather(1, next_token)` 报 `gather_elements_fwd_i32` / `GLUE_INCOMPATIBLE_DATA_TYPE`。随机 logits 最小复现确认根因是索引 int64，不是 `.item()` 或 `GC_KERNEL_PATH`。
- 修复只在 HPU 将词表索引临时转 int32，gather 后恢复 `torch.long`；模型计算、随机采样和返回值仍在 HPU，没有关闭 MTP 或把文本模型搬到 CPU。
- 原样用户图重放成功，扩写约 251 tokens 后正常结束并完成 1024² 出图；诊断模式端到端 350.39 s，不作为性能基准。回归 **26 passed**，覆盖 CPU/HPU、bf16/fp32、top_k=0/1/64、batch、高位词表编号、固定 seed 和返回设备/类型。

## MOSS 输出头 CPU fallback 修复（`0690d18f`）

- 8-token 诊断发现 `BaseGenerate.logits` 和 MTP `verify_logits` 直接调用 `F.linear`，绕过量化 `Linear` 的 `_full_precision_mm` 策略；INT8 词表矩阵每步进入 `aten::_int_mm` CPU fallback。
- 改为在存在独立 lm_head 时调用该层本身，复用既有解量化/精度策略；普通生成从 10.19 s 降至 4.47 s，token ID 完全一致，fallback 为空。真实 MTP auto 32-token 为 10.412 s，fallback `{}`；CPU 解码回归 **42 passed, 1 skipped**。
- 同一完整用户请求从 337.013 s 重放到 134.586 s；新扩写约 307 tokens、约 89 s，图片语义正确。旧请求与新服务缓存/生命周期不同，因此不能作为严格多轮性能基准。
- HPU 利用率仍有 0% 间隙，84 个扩写阶段样本平均约 22%、最高 57%；本修复只消除了已定位的输出头 CPU fallback。

## 驱动恢复与多内核检查

- 故障原因：重启进入 `5.15.0-202` 时，Habana DKMS 只安装在 `5.15.0-186`；内核自带模块没有 Gaudi2 PCI alias，`hl-smi` 无设备。不是 ComfyUI、Python、模型或 PCI 枚举故障。
- 修复：为 `202` 安装现有 Habana DKMS，按既有模块顺序加载并刷新 initramfs；随后修复驱动安装脚本，使安装/verify 覆盖当前内核和所有有效 `/boot/vmlinuz-*`，校验模块、vermagic、路径、alias 和补丁字符串。
- 验证：`186`/`202` 双内核 DKMS installed，重启后 `hl-smi` Operational，BF16 HPU matmul 与 CPU 参考一致，ComfyUI-Gaudi `/system_stats` 返回 `hpu:0 GAUDI2`；19 项模拟回归、实际双内核 verify 和完整重启通过。审查补丁为 `patches/install-gaudi2-all-boot-kernels.patch`。
- 边界：未配置开机自启；驱动恢复后没有修改原始 `/home/sonny/ComfyUI`。

## 模型集中存储

- 两个 checkout 的模型已合并到 `/home/sonny/ComfyUI-Models/models`，去重 36 项、冲突 0；两个 checkout 的分类 symlink 和相同 `extra_model_paths.yaml` 已验证。
- Gaudi checkout `/system_stats` 返回 HPU，模型列表包含 Qwen 2.1、MOSS、Qwen3、T5 和 VAE；Qwen 2.1 Turbo 512² 实际出图成功。原始 ComfyUI 没有启动。
- 旧配置备份路径和最终目录结构见 `MODEL_STORAGE_PLAN_2026-11-06.md`；共享目录不包含代码、venv、output 或 custom nodes。

## MOSS性能优化里程碑

2026-11-11按用户要求精简各轮记录。当前路线与完整结果索引见 [BF16复盘与FP8判断](MOSS_OPTIMIZATION_STRATEGY_2026-11-09.md)，各轮保留测量条件、数值/质量边界、失败原因和原始证据位置。

| 产品 | 已验收变化 | 记录 |
|---|---|---|
| `60da4e33` | GDN/conv区域编译，同输入重放显著减少小算子执行成本 | [GDN](MOSS_GDN_COMPILE_2026-11-09.md) |
| `f739cb93` | MOSS attention编译，固定caption DiT37.916→28.217s | [DiT](MOSS_DIT_OPTIMIZATION_2026-11-09.md) |
| `05d2a0a5` | 投影/FFN分区、静态recipe、MLP与hooks兼容，DiT28.225→19.456s | [block](MOSS_BLOCK_OPTIMIZATION_2026-11-09.md) |
| `3fec1143` | KV与完整短步attention编译、分支行clone，固定teacher4.312→3.002s | [KV](MOSS_KV_OPTIMIZATION_2026-11-10.md) |
| `441d459d` | 恢复Python SIGPIPE策略，浏览器取消下载后设备可继续执行 | [恢复记录](HPU_SIGPIPE_RECOVERY_2026-11-10.md) |
| `b1163be0` | 条件/RoPE预计算50→2次，DiT19.581→18.384s | [预计算](MOSS_DIT_PRECOMPUTE_2026-11-10.md) |
| `1412e1f7` | native norm/RoPE、prefill分桶、验证后BLOCKWISE，DiT至14.364s | [深挖](MOSS_BF16_DEEP_EXECUTION_2026-11-11.md) |
| `64155fe4` | 原norm区域编译，固定MTP完整22.461→21.060s，补全P0–P6证据 | [预算](MOSS_BF16_PLAN_COMPLETION_2026-11-11.md) |
| `a8f42df2` | 并行递推、八行GDN扩区、单head attention，固定完整19.978s/缓存14.386s | [最新验收](MOSS_BF16_ATTENTION_AND_SUBMISSION_2026-11-11.md) |

早期BF16常驻、FP8/INT8 probe及失败路径见 [初期调查](MOSS_HPU_PERFORMANCE_RUN_2026-11-08.md)。拆解36.648s工作流和30.786s节点耗时的证据分别见 [工作流拆解](MOSS_WORKFLOW_BREAKDOWN_2026-11-10.md) 和 [节点计时](MOSS_PIPELINE_TIMING_2026-11-10.md)。

## BF16阶段验收与后续试点

- 最新产品194 passed、1 CUDA skip，512/768/1024、MTP off/3/tree3s1、原INT8及卸载重载通过。自然文本与同seed图像细节会改变，cold64.665s，完整warm19.906–20.122s。
- 同工作量allocator峰值39.3754GiB，卸载2,685,928,832 bytes，多形状/INT8验收为2,686,190,976 bytes。六份部署文件哈希一致，上一轮8189恢复cold66.725s、缓存14.751s、检查队列为空。[部署证据](evidence/moss-bf16-next-2026-11-11.json)
- 下一阶段建议验证DiT原生FP8 Linear的真实shape、量化总成本和多timestep质量。BF16已有大Linear接近同shape可用速度，attention数据流和文本提交保留按证据启动的研究机会。详细决策见复盘。
- 官方/外部实现依据分别见 [INT8/FP8调查](GAUDI2_INT8_FP8_KERNEL_SURVEY_2026-11-10.md)、[JChen核查](JCHEN_GAUDI2_KERNEL_REVIEW_2026-11-10.md) 和 [硬件下界/5090](MOSS_HARDWARE_LIMIT_AND_5090_2026-11-10.md)。

## Moody MOSS原生FP8主模型

- 处理用户导入的`Moody-Moss-Mix-V4.1_00001__clean-fp8.safetensors`。修复去模型前缀后重复注入量化元数据的警告，通过既有MixedPrecision Linear接入Gaudi E4M3量化/GEMM。加载时适配448→224范围，权重生命周期沿用现有加载、patch和卸载。
- 同文件常驻BF16参考/原生FP8固定caption缓存端到端13.993→12.810s，DiT13.749→12.565s，各5 warm中位数。完整自然扩写warm中位数20.666s，首次诊断70.292s。原用户请求222.10s，稳定DiT约7.89s/step。
- 三组图像、512/768/1024及卸载重载通过，fallback为空，显存峰值39.510→34.842GiB，卸载均回到2,685,928,832 bytes。文本实际使用现有BF16文件，遵循优先优化主模型的范围。精度与测试边界见 [FP8报告](MOSS_FP8_ADAPTATION_2026-11-11.md)。

产品`2227f7d3`已部署8189，正常入口cold70.970s/缓存13.600s，文件哈希一致、队列为空。核心回归240 passed，最后量化补充29 passed（有重叠），加载重复metadata warning消失。

## Qwen3.5 FP8文本与全流程预算（2026-11-12）

- 产品`148b252a`接入HPU原生E4M3文本Linear与已有区域编译，修复CLIP强制compute dtype导致的逐次解量化。转换201个语言Linear，文件11.377GB，MTP、门控、norm等保持源精度。源权重来自INT8解码BF16，属于二次量化。
- 同298-token程序，完整warm18.954→18.933s，扩写5.307→5.154s，普通编码0.317→0.411s，DiT约13.07s。allocator峰值34.777→27.267GiB。吞吐提升未超出波动范围。
- DiT六段trace互斥预算为FP8 MME2.515s、其它MME3.508s、softmax2.700s、其它kernel3.604s、间隙1.041s。大Linear活动阶段估算806TF/s，后续优先attention数据流、量化/逐元素融合和文本提交粒度。
- FP8 fused attention、FP8 QK/AV分解、全tensor激活scale均无整体收益。小M额外动态编译存在数值与重载问题，已撤回。251项最终回归、三组自然图片、默认异步重载/MTP off/INT8通过，fallback为空。
- 8189已恢复，版本和8份文件哈希一致，正常入口cold74.786s/缓存13.466s、队列为空。详见 [全流程记录](MOSS_FP8_SYSTEM_OPTIMIZATION_2026-11-12.md)。

## FP8 attention覆盖与数据流（2026-11-12）

- 产品`f67e7f3f`采用BF16 QK、融合softmax量化与FP8 AV，经现有attention dispatcher启用。固定完整流程18.936→17.191s，DiT13.068→11.364s，分别减少9.2%和13.0%。
- 两段trace外推的FP8 MME时间2.515→3.046s，占比19.0%→26.4%；softmax2.701→1.356s。42层主干矩阵运算量的FP8覆盖81.2%→90.6%。AIP均值仍约56%。
- 全FP8 QK/AV的末层误差最高48.1%，逐head/逐行scale、中心化与exp方案未获得更好的速度/误差组合。所选方案18组真实输入误差1.76%–3.07%，分散概率随机压力测试约7.3%，厂商softmax额外误差已记录。
- 静态Linear scale存在局部收益，尚需跨输入校准；文本共享量化完整17.166s、文本5.157s，和产品持平，保留为调查候选。
- 266项回归、三组相同caption/seed图片、重载MTP off/512²、原INT8通过。8189正常cold76.278s、缓存11.881s、768² 13.676s，七份文件哈希一致、队列为空。恢复被CUDA torch遮蔽的既有Habana环境，冲突包保留备份。详见 [进一步优化](MOSS_FP8_DEEP_OPTIMIZATION_2026-11-12.md)。

## FP8 block区域融合（2026-11-12）

- 产品`0995638a`采用native FP32 Q/K RMSNorm与FP8整block编译，并保留attention/FFN调制的区域边界。固定完整流程17.202→15.925s、DiT11.361→10.211s，分别减少7.4%和10.1%。
- FP8 MME采样时间占比26.4%→29.7%，绝对时间约3.045s。其它kernel预算3.668→2.651s，launch1984→640。allocator高水位27.402→31.285GiB，卸载2,685,928,832 bytes。
- 最终18组真实输入RRMSE0.306%–4.066%，对FP32参考的误差比中位数1.007、最高1.293。配对图主体/文字通过，光影、背景和细节改变。V批量量化修正精度后完整16.107s、高水位33.916GiB，最终采用原逐head路径。
- 287项回归、三组自然图片、512²重载/MTP off及原INT8文本通过，fallback为空。8189已恢复，正常cold79.030s、缓存10.696s、768² 14.295s，九份文件哈希一致、队列为空。
- 完整试验、编译分区退化与恢复、未采用的SiTU/量化候选及profiler退出异常见 [区域融合记录](MOSS_FP8_FUSION_OPTIMIZATION_2026-11-12.md)。
