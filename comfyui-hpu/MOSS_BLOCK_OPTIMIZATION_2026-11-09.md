# MOSS BF16 block 分区优化

起始产品版本 `f739cb93`。本轮继续使用现有 BF16/Kitchen/torch 算子，不自行编写 kernel。服务器原始 `/home/sonny/ComfyUI` 不修改；实验目录 `/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-09-blocks`。

## 数值差异定位

`moss_block_precision.py` 对前轮保存的真实首层输入逐个扩展计算前缀，而非返回所有中间结果来改变融合边界。整块编译的首个偏离点是 modulation：相对 RMS 误差 0.3135%，约 10.24% 元素改变。完整 block 为 0.3260%。独立残差 `addcmul_` 编译误差为 0.2652%，SiTU 单独编译在此输入上逐值一致。

点运算使用 CPU FP64 算术参考（首轮汇总误差在 FP32 中计算）：

| 区域 | eager 对 FP64 的相对 RMS 误差 | compiled 对 FP64 的相对 RMS 误差 |
|---|---:|---:|
| modulation | 0.4233% | 0.4287% |
| residual | 0.2167% | 0.1503% |
| SiTU | 0.5058% | 0.5058% |

残差编译结果反而更接近高精度参考，因此不能把所有像素差异解释为精度下降。为保留当前产品的数值行为，本轮先把 modulation 与 residual 留在原执行路径，编译 QKV→attention→输出投影，以及 FFN up/gate→SiTU→down 两个区域。

## 首层同输入重放

| 实验 | block median | 相对 RMS 误差（对当前产品） |
|---|---:|---:|
| 整块编译 | 9.055 ms | 0.3260% |
| 保留残差原执行 | 9.169 ms | 0.2058% |
| 保留调制与残差 | 9.233 ms | 0.02660% |
| 显式划分两段，静态 | 9.217 ms | 0.02660% |
| 显式划分两段，自动动态 | 9.208 ms | 0.02660% |

这不是完整图收益。首层剩余差异在 QKV GEMM 已可见：约 0.0196% 元素改变，相对 RMS 误差约 0.00371%；FFN up/gate 类似。不是 SiTU 自身首先产生的误差。

## 不同层和采样阶段

`moss_block_capture.py` 在同一真实固定 caption 工作流中，采集第 0/12/24 个正条件前向的第 0/20/41 层，共 9 组输入，权重按层复用。`moss_block_validate.py` 对同一输入分别执行当前产品和候选分区。

- 第 0 层：相对 RMS 误差 0.0254%–0.0338%。
- 第 20 层：0.0328%–0.0362%。
- 第 41 层：0.1162%–0.1614%，是这组样本较敏感的位置。
- 全部输出 finite，fallback 为空。
- 对三层第一阶段 QKV 取发生差异的行及均匀取样行，做 CPU FP64 GEMM 参考；eager/compiled 的相对 RMS 误差均约 0.163%–0.166%，两者差别小于 1e-8。结果支持剩余差异主要是 BF16 输出舍入边界附近的累加变化，但不代替完整模型质量验收。

## 完整高精度参考与图片对照

`moss_block_fp64.py` 仅在 CPU 上，用 FP64 计算完整 block 的 norm、调制、QKV、RoPE、attention、FFN 和残差。输入/权重仍是采集的 BF16 数值，位置编码沿用采集值；这是给定真实输入的局部数学参考，不是整模型 FP64 生成结果。九组参考均 finite，每组约 4.6–4.7 s。

| 层 | 采样阶段 | 原产品对 FP64 相对 RMS | 分区候选对 FP64 相对 RMS |
|---|---|---:|---:|
| 0 | 0 / 12 / 24 | 0.31410% / 0.30409% / 0.29169% | 0.31393% / 0.30418% / 0.29182% |
| 20 | 0 / 12 / 24 | 0.29776% / 0.40035% / 0.27908% | 0.29730% / 0.40049% / 0.27899% |
| 41 | 0 / 12 / 24 | 2.48146% / 2.52695% / 1.78725% | 2.48127% / 2.52658% / 1.78724% |

候选/原产品误差比值为 0.99845–1.00045，在这九组输入上未见系统性精度恶化。单独编译 SiTU 的九组输入全部逐值一致，约 0.90–1.00 ms → 0.53–0.54 ms；它不能解释分区方案剩余的图像差异。

固定 caption 的候选完成 1 cold + 5 warm，DiT median **28.2255 → 23.4820 s**、整图 **30.2457 → 25.6161 s**。MOSS 街景图片 MAE 7.202/255；另外三组成对固定 seed 的图片 MAE：512 红茶壶 2.472/255、768 加权海报 1.682/255、1024 重载后的猫 6.026/255。三例和主测试均无异常纹理，主体、构图与指定的 MOSS 大字正常；海报的额外无意义小字两组都有。没有据此声称逐像素一致或大规模质量评测通过。

与上一轮仅凭图片差异暂缓整块编译不同，本轮增加了中间结果定位、完整 FP64 block 对照，以及多提示词成对图片。最终仍选择保留 modulation/residual 原执行方式，只扩大两个投影区域，接受 BF16 GEMM 累加顺序造成的细节变化。

## 产品候选与回归

产品候选将 `Block` 分为 attention 投影和 FFN 两个私有方法，仅 HPU BF16 推理编译；CPU、训练、关闭编译和 attention override 保留原路径。调制、参考 token 处理与残差原地更新在编译区外。`_fp16_linear_wanted` 先判断 CUDA/FP16，再读取 CUDA 配置，避免 HPU tracing 访问不可追踪的 CUDA getter，原设备选择语义不变。

**79 passed, 1 skipped**：新增 block 13 项、已有 attention 7 项以及 GDN、文本采样/解码回归；覆盖 fused/unfused QKV、mask、参考 KV、参考 token、输入别名、更换权重、10 个不同长度、CPU/训练/override/禁用编译。跳过为 CUDA 专属项。分区实现已提交 `8ebc3f40`。

## 动态形状线索

本机 Habana 1.24.2 的 `compile_backend/config.py` 和 `recipe_compiler.py` 支持 `force_static_compile`；`backends.py` 可按单次 `torch.compile(options=...)` 设置。它保持 Dynamo 符号图，按实际符号值选择静态 recipe。没有改变全局缓存上限、padding、有效 token、位置编码或 lazy 模式。

`moss_block_shapes.py` 比较 13 个不同长度，再返回 4408/4119。默认动态 recipe 在切换长度后，N=4408 由约 9.16 增至 13.17 ms/block；强制静态 recipe 保持约 9.21 ms。新 shape 的首轮约 0.13–0.21 s，稳定 warm 获益；两组 HPU allocator 峰值均约 4.016 GiB，全部 finite、fallback 为空。算子级输出并非全部逐值一致，最大差异 0.0625。该选项仅设置于两个 MOSS 编译区域，提交 `15c5b17c`，MOSS 20 项复验通过。

正式产品代码固定 caption、1024²/25 步，各 1 cold + 5 warm：

| 实现 | 固定 caption 整图 median | DiT median | DiT AIP 平均 | allocator 峰值 |
|---|---:|---:|---:|---:|
| 本轮基线 `f739cb93` | 30.2457 s | 28.2255 s | 前轮同配置约 52.8% | 前轮约 40.11 GiB |
| 两段编译 | 25.5656 s | 23.5449 s | 50.19% | 40.60 GiB |
| 两段编译 + 静态 recipe | **21.5192 s** | **19.4562 s** | **56.50%** | **39.39 GiB** |

最终固定工作量 DiT 比本轮基线减少约 31.07%。最终图片对基线 MAE 3.548/255、PSNR 29.63 dB；与不加静态 recipe 的分区图 MAE 7.743/255，表明微小累加差异可经扩散轨迹放大。固定 seed 的画面细节会变化，不宣称逐像素等价。512/768/1024、加权提示词、卸载重载和两次原 INT8 图均通过，latent/VAE finite、fallback 为空。

## 文本 MLP 区域编译

当前 Qwen3.5 BF16 checkpoint 的 gate/up 投影仍分开存储。先在调查进程编译现有 `MLP.forward`：同一真实提示词，MTP auto、32-token 上限、各 1 cold + 5 warm，median **3.7102 → 3.3322 s**；这两组输出 ID 相同。该全 MLP 原型同时影响 prefill，因此产品进一步收窄。

最终候选只在 Qwen3.5 的普通 operations（manual_cast/disable_weight_init）、HPU BF16、存在 KV cache、每步不超过 `SPEC_ROWS=8` 时编译 MLP。量化/自定义 operations、CPU、训练、关闭编译、大段 prefill 或无 cache 走原路径。BF16 实测 33 个 block 具备该能力（32 主干 + 1 MTP），原 INT8 文件为 **0**；原模型/权重文件均未改动。

产品 32-token 五次 warm median **3.2076 s**，相对 3.7102 s 约少 13.54%；组内输出 ID 相同，但与原路径第 27 个 token 开始分歧，不能沿用全 MLP 原型的跨组 ID 相同声明。固定 32-token teacher forcing、MTP off：logits 全部 finite，top-1 **32/32 一致**，平均 KL **0.0003535**、最大绝对差 0.265625、平均绝对差 0.02107。此为短轨迹数值检查，不是完整语言质量评测。

最后另做相同当前代码、仅切换 MLP 编译的严格定轨迹对照：固定相同 32 个 token、MTP off，各 1 cold + 5 warm，median **4.8757 → 4.2231 s（-13.38%）**。此诊断包含逐 token logits 的 CPU 拷贝，两组口径相同，用于固定工作量 A/B，不代替自然生成速度。记录见 `teacher-control` 与 `teacher-compiled`。

回归发现 PyTorch 默认可能忽略编译后新增的 forward hooks。新增 `comfy.ops.has_forward_hooks` 检查模块/子模块及全局 hooks，有 hooks 时使用 eager；同时修补 MOSS block。根模块、Linear、norm、全局 hook 的添加/移除均通过。初轮文本回归 69 passed/1 skipped、1 个 hook 测试失败；修复后 MLP 14 项、MOSS block 14 项、attention 7 项合计 **35 passed**。这些与前述回归有重复，不累加成独立测试数。hook 修复提交 `b830b3b8`。

文本模型卸载重载后 8-token 输出 ID 一致，HPU allocator `InUse` 在卸载后约 **93.98 MiB**，没有保留整份模型权重；最终原 INT8 完整图两次通过，诊断 latent/VAE 全部 finite、fallback 为空。

## 完整用户图结果

均为 BF16 文本模型、原用户提示词、MTP auto、1024²/25 步、每组 1 cold + 5 warm，强制计算节点重算、保留 loader：

| 实现 | 整图 warm median | 扩写 median | DiT median |
|---|---:|---:|---:|
| 本轮开始（上一轮产品） | 52.5911 s | 19.6479 s | 31.3022 s |
| 两段 DiT 编译 | 45.4458 s | 19.9031 s | 23.4260 s |
| 加静态 recipe | 41.2054 s | 19.6675 s | 19.5603 s |
| 再加短步 MLP 编译 | **36.6480 s** | **14.9429 s** | **19.3579 s** |

最终 warm 耗时 `[39.1478,35.8548,38.9186,36.6480,35.2302]`，自然结束 tokens `[362,267,362,277,277]`；上一轮为 `[333,303,303,287,290]`，所以整图约 30.32% 的降幅是实际工作流观察，不是同一 token 轨迹的严格 A/B。严格 DiT 对照见固定 caption 表；文本另补固定 teacher 轨迹的五次 warm 对照。

最终 warm AIP：全请求 **37.93%**、扩写 **16.79%**、DiT **57.23%**，峰值 68%；平均 54V rail 功耗分别约 242.54/118.61/355.74 W。仍未达到用户要求的 80%，没有以活动率代替有效吞吐，也不宣称已逼近硬件上限。

主要证据为 `static/product-final`、`full-complete-product`、`text-product*`、`text-reload`、`phase-summary.json`；固定 caption 街景和多提示词配对图在 `quality/comparison.png`。所有调查代码和失败路径只在本仓库，产品代码仅包含执行优化和回归测试。

最终 BF16 完整诊断请求的 latent `[1,128,64,64]` 和 VAE `[1,1024,1024,3]` 全部 finite，fallback 为空；文本、图像和权重都没有做额外量化。最终图片与扩写文本不保证相同 seed 的逐位一致。

## 交付状态

- `8ebc3f40`：保留调制/残差边界，编译 MOSS 投影与 FFN。
- `15c5b17c`：按实际 shape 使用厂商静态 recipe。
- `b830b3b8`：在存在 forward hooks 时保留 eager 调用。
- **`05d2a0a50de8385765f22dbdc12e9669392d3004`**：普通 BF16 Qwen3.5 的 KV-cache 短步 MLP 编译。

服务器产品文件与本地 SHA256 一致，8189 日常服务已恢复、队列空闲；本地 `codex/gaudi2-hpu` 分支未 push。原始 `/home/sonny/ComfyUI` 未修改，没有下载模型或自写 kernel。后续仍需对 KV attention/tree merge、采样同步和余下 norm/RoPE 做更新后的归因；80% 利用率目标尚未实现。
