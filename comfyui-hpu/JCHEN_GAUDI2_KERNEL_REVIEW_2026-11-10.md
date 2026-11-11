# JChen Gaudi2 仓库专项复核

日期：2026-11-10（Asia/Shanghai）。用户指定 [JChen-318/gaudi2-single-card-vllm](https://github.com/JChen-318/gaudi2-single-card-vllm)。本轮固定审查提交 **`6da6fb16892f0e6ec3a7e1900942dec4e47a8706`**（2026-11-09），下载源码到临时目录，核对 native、vllm-gaudi/csrc、Python 调用及相关验证记录；没有编译、运行下载的代码或改动服务器。

## 结论与前轮修正

这是包含真实 Gaudi2 内核、运行时和模型集成的工程快照，不能仅以 `hpu_lm_head_fp8.py` 的厂商算子封装概括整个仓库。

- 相对此前审查的 `e08710f7fe4c953529c728ec0b3a661f922d07c5`，新增 **2 个提交、297 个变化文件**：`03f0dc502088` 增加直接 MME/TPC attention，`6da6fb16892f` 更新说明。295 个变化文件位于 `native/mme-direct-attention/`，其余为 README 和优化日志。[固定比较](https://github.com/JChen-318/gaudi2-single-card-vllm/compare/e08710f7fe4c953529c728ec0b3a661f922d07c5...6da6fb16892f0e6ec3a7e1900942dec4e47a8706)
- **新增直接 MME/TPC 执行路线确实存在**，包含描述符、队列/同步、SRAM 调度、自写 exp/归一化、Python 集成。上一轮只重新读取了 main 的输出头文件，漏查了这两次提交。
- **此前已有的 `vllm-gaudi/csrc` 也应补充到清单**：真实 FP8 TPC QK 点积、FP8 量化、残差/RMSNorm/量化融合、低比特解码及 GDN。源码快照包含继承/移植内容，不把所有文件都归为 JChen 独立原创。
- **通用 INT8 GEMM 的结论不变**：本轮未找到可用于 `_int_mm` 的替代实现；该仓库的 W8A8 INT8 路径仍明确转 BF16。

## 1. FP8：厂商 MME 与自写 TPC 两类都包含

| 路径 | 源码实际行为 | 适用边界 |
|---|---|---|
| `ops/hpu_lm_head_fp8.py` → `extension/ops.py::apply_fp8_linear_hpu` | BF16/FP16 dense 权重按输出通道一次量化；activation 动态量化；调用官方 `hpu.fp8_gemm_v2` | 仍依赖厂商 GEMM；保留原权重，不是 INT8 ConvRot 的直接替代 |
| `csrc/flashinfer_gaudi/kernels/tp2_dynamic_quant_bf16_gaudi2.c` | TPC-C 动态 FP8 量化 | Python 路由另有 token/CGUID 条件；存在不等于默认开启 |
| `.../add_rmsnorm_quant_bf16_gaudi2.c` | 残差相加、RMSNorm、absmax/scale、FP8 输出，在 TPC 内融合 | host glue 限制 BF16、宽度 256–17408、128 对齐；输出含 FP8、scale、normalized 和 residual |
| `.../silu_mul_quant_bf16_gaudi2.c` | 读取已计算的 SiLU 和 up，融合相乘、行量化、FP8 写出 | **不是整个 SiLU 都在一个 TPC kernel 内完成**；bridge 先建 split + 厂商 `silu_fwd_bf16`，再接此 kernel |
| `csrc/deepseek_v4/include/deepseek_v4_paged_sparse_attn_fp8.h` | 自定义分页稀疏 attention，QK helper 直接调用 **`v_f8_mac_acc32_b`**，FP32 累加 | 模型专用布局：head_dim=512、其中 448 维 FP8、分块 scale；不是通用 MME GEMM，也不能直接替代 MOSS SDPA |

可复核源码：[FP8 Linear 调用](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/vllm-gaudi/vllm_gaudi/extension/ops.py#L1301)、[RMSNorm/量化 kernel](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/vllm-gaudi/csrc/flashinfer_gaudi/kernels/add_rmsnorm_quant_bf16_gaudi2.c)、[SiLU 组合图](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/vllm-gaudi/csrc/flashinfer_gaudi/bridge/silu_quant.cpp#L31)、[TPC FP8 MAC](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/vllm-gaudi/csrc/deepseek_v4/include/deepseek_v4_paged_sparse_attn_fp8.h#L101)。

稀疏 attention 的 `.c` include、host glue、PyTorch HPU/Meta 注册及 `ops/deepseek_v4_native.py` 调用都在仓库，故不是只有函数名。其 E4M3 helper 对扩展编码做调整和补偿后执行 FP8 MAC，说明它处理了 Gaudi2 的 FP8 格式差异。本轮只确认实现存在和调用关系，没有验证该模型或算子的速度/质量。

## 2. 直接 MME/TPC attention：新增的重要路线

[总说明](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/native/mme-direct-attention/README.md) 描述并提供两种不同阶段的实现，不能混为一谈：

1. **裸 hl-thunk 固定 GEMM 原型。** `direct-mme/mme_runner.c` 使用 `desc128.h`、WREG 命令和 DMA/SOB/monitor，不加载 PyTorch/Synapse/SCAL 执行。M=N=K=128，BF16；描述符最初由 SynapseAI 1.24.1 导出。作者报告 HBM、SRAM 及两次 GEMM 串联的 12 个用例通过，输入是小整数，不能外推任意浮点分布和 shape。[源码与边界](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/native/mme-direct-attention/direct-mme/README.md)
2. **H3 的框架互操作 attention。** 在已初始化的 Synapse 环境内借助私有 SCAL 流调度 MME 描述符和 TPC，运行 QK→exp/行和→PV→归一化。S/P 中间张量复用 SRAM，减少 HBM 往返；包含 descriptor replay、head 地址缓存、流水、同步与输出所有权管理。[运行时源码](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/native/mme-direct-attention/attention/runtime_quad.hpp)

**这条优化路线不是 FP8 或 INT8 attention。** 交付路径的 Q/K/V 为 BF16，score 为 FP16，probability 为 BF16；TPC 做受上界约束的 exp 和归一化。`sim_fp8pv.py` 是 CPU 数值模拟，不能记作已部署 FP8 kernel；四层证据中一些 FP8 方案超过其 1% RMS 门槛。[模拟与结果](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/native/mme-direct-attention/attention/evidence/sim-fp8pv-4layers.json)

Python 入口 [h3_direct_attention.py](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/native/mme-direct-attention/attention/h3_direct_attention.py#L269) 检查 batch=1、seq_len=37750、head_dim=128、Q/K/V 同 shape、BF16 HPU；外层适用于无 mask、非因果的 H3 attention。其它形状或不满足数值门限的 head 回原实现。当前 MOSS DiT 虽也使用 head_dim=128，但 token 数、模型和工作流不同，不能只替换函数名接入。

作者结果与本机结果严格分开：

| 作者报告的 H3 稳态配置 | 请求时间 |
|---|---:|
| 原正式配置，同环境复测 | 100.45 s |
| 直接 attention + head streaming + compiled gates | 84.41 s |
| 再加文本编码器锁页和 t2va DiT 常驻、VAE tile=4 | 79.7 s |

后两行的差异不能全归因于 kernel。逐层采样最大 RMS 约 0.78%，但视频与正式配置约 18 dB PSNR，作者明确未完成生产质量验收；45/48/49 层保留 FusedSDPA，默认启动器未启用实验路径。[作者实验记录 §23.10](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/docs/OPTIMIZATION.md#2310-直接编程-mmetpc-注意力实验路径)

该路线依赖固定 descriptor fixtures、地址生命周期及私有同步资源；源代码要求同进程调用时没有其它设备工作并发。移植需重做 shape 描述符和当前 SDK 的互操作验证，不能在本项目日常服务进程中直接试运行。

## 3. INT8 与 NVFP4 不能混称

[HPUCompressedTensorsW8A8Int8_BF16Fallback](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/vllm-gaudi/vllm_gaudi/ops/hpu_compressed_tensors.py#L320) 仍声明 INT8 GEMM 不可用，并在加载后执行 INT8×scale→BF16、随后 `F.linear`。对 `native/`、`vllm-gaudi/csrc` 和相关 Python 路径的检索没有找到通用 INT8 MAC GEMM 或 ConvRot INT8 kernel；`uint8_t` 命令字节、NVFP4 packed bytes、INT8 LUT 常量不属于 INT8 GEMM 实现。

`native/qwen4exp-nvfp4-moe` 确实有自定义 GEMV，但采用 **NVFP4 packed 权重→寄存器内 BF16 解码→`v_bf16_mac_acc32_b`**，不是 INT8 MAC，也没有先把整个专家权重物化为 BF16。包含专家 count/prefix/scatter 等配套 kernel；这个独立库的 README 明确只完成 host/模拟器验证，没有设备实测，prefill grouped GEMM 不在该库内。[实际 GEMV kernel](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/native/qwen4exp-nvfp4-moe/kernels/qwen4exp_nvfp4_moe_gemv_bf16_gaudi2.c#L34)、[验证边界](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/native/qwen4exp-nvfp4-moe/README.md)。

## 4. GDN、采样和图片适配

- **GDN 是真实 TPC kernel。** `gdn_mtp_fused_f32_gaudi2.c` 合并 slot 选择、Q/K normalization、decay 和 8-token recurrence；BF16 输入、FP32 state/checkpoints。固定 B≤16、T=8、H=16、HV=48、K=V=128，且 `key_head=head/3` 和 packed 偏移写在源码中。当前 ComfyUI 的区域编译复用 torch 路径，不等于已经移植此 kernel；移植必须匹配 heads、状态布局、MTP checkpoint/接受语义。[源码](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/vllm-gaudi/csrc/flashinfer_gaudi/kernels/gdn_mtp_fused_f32_gaudi2.c)
- `native/mtp-sampling` 是拒绝采样相关 TPC 源码；`native/tpc-kernel-shim` 是 GUID/库转接，服务于多个 TPC 库与 recipe cache 共存，不是数学 kernel。[shim](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/native/tpc-kernel-shim/README.md)
- **图片/视频适配也有参考价值。** `vllm_gaudi/omni` 包含 HPU attention、区域编译、H3/VAE 路径。[regional_compile.py](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/vllm-gaudi/vllm_gaudi/omni/regional_compile.py) 把重复 block 接到 HPU backend，hooks 在编译区外。
- Qwen-Image 2.1 的作者记录是 BF16 HTTP 约 4.4 s；在线 FP8 的服务端时间从 3.59→3.2 s，但构图差异较大，未启用。其 `qwen_worker.py/image_api.py` 明确不在这个仓库，所以不能宣称仅凭该仓库已经拿到完整 Qwen 部署代码。[作者记录 §24](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/docs/OPTIMIZATION.md#24-vllm-omni-媒体模型qwen-image-21)

## 5. 对当前 ComfyUI 的优先级判断

| 内容 | 可复用性与建议 |
|---|---|
| 厂商 FP8 Linear 的量化/scale/weight preparation | 优先候选，仍需真实 shape 和模型质量验证；已有 lm_head probe 未证明净收益 |
| RMSNorm＋FP8 量化、乘法＋量化 TPC 融合 | 后续 FP8 路径建立后再测；先看量化准备是否占关键路径，按当前 1.24.2 重编译 |
| GDN fused/prepared kernel | 有实际开发价值；当前 BF16 编译基线已明显提速，要重新归因并适配 state/heads，而非复制即用 |
| 自定义 FP8 稀疏 attention / NVFP4 MoE | 模型和 layout 专用，与当前 MOSS dense DiT/ConvRot 文件不直接匹配 |
| 直接 MME/TPC attention | 值得作为独立研究路线；固定 shape、SDK/descriptor、同步/allocator 与质量验收工作较多，暂不替换当前产品路径 |
| 区域编译、mask、卸载生命周期 | 可以逐项对照当前产品，避免把已完成的 BF16 编译优化重复计入收益 |

作者环境为 Ubuntu 24.04、Python 3.12、SDK 1.24.1、torch `2.11.0a0+git009b5f6`；本项目为 Python 3.10、Habana 1.24.2、torch `2.13.0a0+git2469d18`。源码可参考，二进制、私有 bridge ABI 与固定 SCAL 配置不能直接混用。[安装说明](https://github.com/JChen-318/gaudi2-single-card-vllm/blob/6da6fb16892f0e6ec3a7e1900942dec4e47a8706/docs/INSTALL.md#L3)

## 证据与范围

- HEAD commit 与递归 tree 已分别读取，tree 未截断，共 10,251 个文件；临时目录提取相关顶层目录 3,449 个文件。完整目录检索不等于逐行审计全部文件。
- [jchen-sources.json](kernel-audit-2026-11-10/jchen-sources.json) 记录固定 commit、tree、前版比较及关键文件哈希/永久链接。外部代码保留在临时目录，调查与结论只进入本仓库。
- 本轮未运行作者的硬件测试或 CPU 模拟，所有性能数字均标为作者报告。证据层级为源码确认、作者记录、本项目既有实测；不混淆。
