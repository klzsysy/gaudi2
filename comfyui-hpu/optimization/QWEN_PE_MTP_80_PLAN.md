# Qwen Image 2.1 FP8 PE 解码 80% 硬件上限优化计划（含 MTP）

制定日期：2026-11-19（Asia/Shanghai）。基线产品提交为已验收的 D2 方案 B `26a170f97b360a61cc597c0908df294eea03dfa5`（单卡 Gaudi2 HL-225、Habana 1.24.2、eager）。

本计划以当前 1088×1920 2MP 完整工作流中的 PE 文本扩展阶段（`qwen3.5_9b_qwen_image_2.1_pe_t2i.fp8_from_int8_fp32.safetensors`）为目标，引入 MTP 投机解码并逐项根治已定位的硬件与执行瓶颈，将 PE 解码阶段的主导硬件资源（HBM 带宽）和阶段关键路径效率推进至 **≥80% 硬件上限**。

---

## 1. 现状诊断与硬件物理边界

### 1.1 当前基线（最后一次生产生成与纯热态参考）

- **最后一次实际生成**（`Qwen_image_2.1_00084.png`，`2026-11-19 13:04:17`）：总耗时 `33.74 s`（PE `~21.66 s` 占 `64.2%`，DiT `~9.62 s` 占 `28.5%`，CLIP/VAE/Save `~2.46 s`）。
- **纯热态基线**（[qwen-d2-plan-b-2026-11-18.json](../evidence/qwen-d2-plan-b-2026-11-18.json)）：同提示词、同 seed、`thinking=True`、2094 输出 tokens，整图中位 `32.077 s`，其中 PE `20.708 s`（`64.6%`）、DiT `8.895 s`（`27.7%`）、其余 `2.474 s`。
- **实测硬件计数**（64-token 低扰动窗口，[qwen80-pe-execution-2026-11-16.json](../evidence/qwen80-pe-execution-2026-11-16.json)）：
  - 单 token 耗时：`9.751 ms`（稳态 `103–105 tok/s`），板卡平均功耗 `~183 W`（动态功耗 `~100 W`）。
  - HBM 访问量：`8.706 GB/token`（读 `8.450 GB` + 写 `0.256 GB`），有效 HBM 带宽 **`0.893 TB/s`**（2.45 TB/s 峰值的 **`36.44%`**）。
  - 设备时间线：`device_span = 643.11 ms` 中，kernel 活跃 `399.89 ms`（`62.18%`），**设备空档 `243.22 ms`（`37.82%` = `3.80 ms/token`）**；活跃时间内 `fp8_gemm = 3.17 ms/token`，其余 `3.08 ms/token` 消耗在非 FP8 GEMM（`0.71 ms`）、softmax（`0.09 ms`）与大量碎片 TPC 算子（`other_kernel = 2.28 ms`）。
  - Host 提交：`18.0 recipes/token`、`39.25 launches/token`、主线程 CPU `6.76 ms/token`。

### 1.2 精确权重与状态字节账本（Qwen3.5-9B FP8 PE + MTP）

基于服务器现有模型头校验结果：

| 组件 | 张量与形状特征 | 精确字节量 | 备注 |
|---|---|---:|---|
| 32 层 MLP FP8 权重 | 96 个矩阵（`gate`/`up`/`down`） | `4.8318 GB` | 占 FP8 权重 60.9% |
| 24 层 GDN FP8 权重 | 72 个矩阵（`qkv`/`z`/`out`） | `1.6106 GB` | 另含 `in_proj_a/b` BF16 `12.6 MB` |
| 8 层 Full Attention FP8 权重 | 32 个矩阵（`q`/`k`/`v`/`o`） | `0.4698 GB` | `q` 含 gate 双倍输出 |
| `lm_head` FP8 权重 | `[248320, 4096]` | `1.0171 GB` | 与 MTP draft 共享 |
| **目标主干 FP8 权重合计** | **201 个 2D FP8 矩阵** | **`7.9293 GB`** | 另有 `embed_tokens` BF16 `2.034 GB`（按行查表） |
| 8 层 Full KV 读取（BF16） | `[1, 4, cap, 256] × 2` | `0.2261 GB`（满容 6899） / `0.1245 GB`（中段 3800） | 当前扫描满容浪费 ~45% |
| 24 层 GDN FP32 状态 R+W | `[1, 32, 128, 128] FP32` | `0.1007 GB`（读 `50.3 MB` + 写 `50.3 MB`） | 未融合时被重复读 2–3 次 |
| **现有 MTP 头权重**（来自已有 `qwen3.5_9b`） | 8 个 Linear + 7 个 norm | `0.4865 GB`（BF16） / **`0.2433 GB`（量化 FP8 后）** | 服务器已有 15 个 `mtp.*` 张量 |
| 1 层 MTP KV 读取（BF16） | `[1, 4, cap, 256] × 2` | `0.0283 GB`（满容 6899） / `0.0156 GB`（中段 3800） | 独立 1 层 Full Attention |

### 1.3 为什么引入 MTP 是达成 ≥80% 且突破物理墙的关键

- **为什么仍以 HBM 带宽为 80% 主导资源**：即使引入 MTP（`M=4` 或 `M=8`），每轮 Verify 算术强度约 `7.1–13.8 FLOP/byte`，远低于 Gaudi2 FP8 MME 岭点（`865 / 2.45 = 353 FLOP/byte`），仍是严格 Memory-Bound。因此 **80% 硬件目标定义明确为**：
  1. **理论资源利用率**：校准有效 HBM 带宽 **≥ 1.96 TB/s**（`2.45 TB/s × 80%`）；
  2. **同 shape 可持续效率**：实际吞吐 / 同精度同布局真实权重轮换微基准 **≥ 80%**；
  3. **阶段关键路径效率**：经审计阶段硬件下界 / 真实 PE 墙钟耗时 **≥ 80%**；
  4. **用户体验指标**：严禁以增加无效流量凑带宽，必须同时显著降低每 token HBM 字节、单图 PE 耗时与整图耗时。
- **MTP 的双重物理红利**：
  1. **解除 `M=1` 小矩阵效率瓶颈**：此前实测 `M=1` 下真实 FP8 MLP 独立微基准仅能跑到 `1.515 TB/s`（`61.8%`），而 `M=4/8/16` 可提升至 `1.756 TB/s+`，且 32 层固定 launch、GDN 状态读写与 KV 扫描成本可跨 4–8 行摊薄，使整轮持续 HBM 带宽真正具备冲击 **≥1.96 TB/s（80%）** 的物理条件。
  2. **打破单 token 最低字节下界**：无 MTP 每 token 至少读 `8.71 GB`（100% 带宽物理极限 `3.55 ms/token` / `281 tok/s`）。引入 FP8 MTP 后：
     - `M=4`（2 drafts + 1 verify）：单轮约 `8.9 + 2 × 1.29 = 11.48 GB`，平均推进 `~1.9–2.2 tokens` → **每 token 仅需 `5.2–6.0 GB`**；
     - `M=8` 树（3 drafts + 1 verify）：单轮约 `9.2 + 3 × 1.29 = 13.07 GB`，平均推进 `~2.1–2.5 tokens` → **每 token 仅需 `5.2–6.2 GB`**；
     - 在 **80% HBM 带宽（`1.96 TB/s`）**下，有效吞吐可达 **`~320–400 tok/s`**（2094 tokens 仅需 **`~5.7–7.0 s`**，反超当前 `8.90 s` 的 DiT！）。

---

## 2. 已定位的六大核心瓶颈（逐项对照当前源码）

| # | 瓶颈 | 当前产品代码现场（`ComfyUI-Gaudi`） | 对性能的影响 |
|---|---|---|---|
| B1 | **MTP 路径完全绕过整步编译** | `qwen35.py:1065` `_can_compile_generation()` 在 `self.mtp is not None` 时直接返回 `False`；`_generate_mtp()` 走 `llama.py:1070` 逐层循环；`qwen35.py:410` GDN 区域编译仅限 `x.shape[1] == 8`，导致 `M=4` 连 GDN 区域编译都未命中 | 大量 Python 分发与上百次 recipe 提交，MTP 直接接入 FP8 会严重损失加速比 |
| B2 | **37.8% 设备空档与图外碎步控制** | `qwen35.py:1140, 1240–1370`：单步图外残留 embedding、`lm_head`、penalty、采样与回读；MTP 的 `draft()` 未编译且保留 CUDA VBAR `prefetch_queue_pop`；`verify_sample()` 含 Python `for r` 掩码循环；HPU 上 `auto` 模式缺 `torch.hpu.Event`，用无同步 `time.perf_counter()` 错误估算 `draft_ms`/`verify_ms` | 单步 `3.80 ms/token` 空档、`18 recipes/token`、`6.76 ms` CPU；MTP `auto` 树形决策失真 |
| B3 | **24 层 GDN 状态多次读写与固定 8 槽冗余** | `qwen35.py:517–521` 单步多次读写 50.3 MB FP32 状态；`hpu_gated_delta.py:103–148` MTP 延迟提交固定按 `SLOT_MAX=8` 执行两次幂级数（`_commit` + `_tree_output`）并二次读写状态，且 `M=4` 触发 Habana 1.24.2 输入别名 `x.clone()` | 占非 FP8 GEMM 活跃时间的最大头，严重拉低整步平均带宽 |
| B4 | **小尺寸 FP8 投影重复量化与碎片 GEMM** | `ops.py` 在 `--disable-dynamic-vram` 下 `CombinedLinear` 返回 `NotImplemented`；`quant_ops.py:214` 每个 Linear 独立做 `amax + cast_to_fp8_v2`；MLP `gate`/`up`、GDN `qkv`/`z`、Attn `q`/`k`/`v` 被拆成小矩阵 | 重复激活量化 TPC 开销 + FP8 矩阵过小（32–64 MB）导致 `fp8_gemm_v2` 突发带宽偏低 |
| B5 | **8+1 层 Full Attention 全容量扫描与慢速树合并** | `llama.py:98–109` 与 `hpu_kv_attention.py:25–27`：MTP 树验证调用 `comfy_kitchen.flash_attention_decode_gqa(..., return_lse=True)` + `tree_merge`（HPU 上为未融合 FP32 路径），且始终扫描满容 6899（平均浪费 ~45% KV 读流量） | Attention 耗时高且多读 ~110 MB/步无效零填充 KV |
| B6 | **Draft 步 `lm_head` 1.02 GB 权重搬运与 MTP BF16 权重** | 现有 `qwen3.5_9b_fp8_from_int8.safetensors` 中 8 个 MTP Linear 仍为 BF16（`486.5 MB`）；每步 `draft()` 均完整读取 `1.017 GB` 全词表 `lm_head`（占单次 draft 流量 80%） | 3 步 draft 仅 `lm_head` 就消耗 `3.05 GB`，拖高单轮总流量 |

---

## 3. 分阶段实施路线图（M0 → M5）

### 阶段 M0：FP8 MTP 资产构建、小 M 硬件上限审计与低扰动基线账本

**目标**：在不触碰线上 8189 服务（串行隔离进程）的前提下，建立可复算的 PE+MTP 硬件下界模型和瓶颈排序。

1. **构建带 MTP 头的 FP8 PE 独立模型文件**：
   - 扩展 `scripts/gaudi/convert_qwen35_fp8.py`，增加 `--attach-mtp <mtp_source>` 和 `--quantize-mtp` 选项。
   - 复用服务器已有 `qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors`（FP32 反量化路径）与已有 `qwen3.5_9b_int8_convrot.safetensors` / `qwen3.5_9b_fp8_from_int8.safetensors` 中的 15 个 `mtp.*` 张量，生成独立新文件（保留原无 MTP 文件不覆盖）。
   - 同时测试 MTP 8 个 Linear 保持 **BF16（486.5 MB）** 与量化为 **FP8 E4M3（243.3 MB，范围 224）** 对接受率、精度和草稿延迟的影响。
2. **真实形状小 M 带宽上限扫描（权重轮换集合 > 48 MB SRAM）**：
   - 测量 `M = 1, 2, 4, 8, 16` 在 Qwen3.5-9B 真实形状下的可持续 HBM 带宽：
     - 独立 `gate`/`up`/`down` vs 逐输出通道 scale 合并 `gate_up`（`[24576, 4096]`）；
     - 独立 `qkv`/`z` vs 合并 `qkvz`（`[12288, 4096]`）；
     - 独立 `q`/`k`/`v` vs 共享激活量化；
     - `lm_head`（`[248320, 4096]`）。
   - 建立精确的“理论 2.45 TB/s 上限”与“同 shape 可持续上限”对照表。
3. **建立 FP8 PE+MTP 基线账本**：
   - 测量 `mtp = off / 2 / 3 / tree3s1 / auto` 在 2MP 原提示词、蓝/红变体、`thinking=True/False` 下的真实接受率（tokens/verify）、BMON 校准 HBM 读写（采集扰动 ≤ 5%）、原生时间线（MME/TPC 占比、gap、recipe/launch 计数）与功耗温度遥测。

**阶段退出标准**：产出经实测验证的 M0 硬件账本与各子阶段下界公式，锁定优先验证树形（4 行链 vs 8 行树）。

---

### 阶段 M1：整步执行图闭合与 MTP 控制链重构（消灭 37.8% 设备空档）

**目标**：解决 B1、B2，把设备空档从 `37.8%` 压至 **`<10%`**，消除 MTP 在 FP8 下的逐层回退。

1. **MTP Verify 全步闭合编译（同时支持 `M=4` 与 `M=8`）**：
   - 解除 `_can_compile_generation` 对 `self.mtp is not None` 的阻断，新增专用 `_compiled_hpu_verify(model, hidden, rotary, caches, ...)`，覆盖 **32 层主干 + `final_norm` + `lm_head` + 向量化树 penalty 掩码与 top-k/top-p 概率准备**。
   - **严守动态容量边界**：对全部 8 层 Full KV 保留 `torch._dynamo.mark_dynamic(cache.key/value, 2)` 和规范 RoPE 连续步长（沿用 `a5308be1` 修复），保证跨提示词编辑、切换 `thinking`、改 `max_length` 时首个 decode 仍 **< 1 s**，0 次 `batch_as_strided` 别名告警、0 次 `recompile_limit`。
   - 将 `verify_sample()` 中的 Python `for r in range(1, tree.rows)` 替换为基于 `VerifyTree` 预计算祖先位图的纯张量掩码操作，并核对采样 dtype 边界（消除无谓 FP32 全词表 cast）。
2. **MTP Draft 全步闭合编译与连续展开**：
   - 清理 HPU 常驻路径下 `draft()` 中的 CUDA `prefetch_queue_pop` 包装，将 `embed → MTPHead (fc + 1 层 GatedAttention + MLP + norm) → lm_head → argmax/topk` 整体编译为单一静态图（`mtp_kv` 容量维动态）。
   - 对比“逐次调用编译 draft（2–3 次）”与“2 步/3 步静态展开连续 draft 图（1 次提交完成整棵草稿树构建）”，以完整热耗时与冷编译代价综合选型。
3. **修复 HPU `auto` 树自适应计时与单步兜底**：
   - 将 `qwen35.py:1219` 的 `timers` 扩展支持 `torch.hpu.Event(enable_timing=True)`（或基于实测校准成本表），修复 HPU 上 `mtp="auto"` 的树形选择。
   - 同步把无 MTP 单步路径的 `lm_head + 采样准备` 纳入整步闭合评估，确保 `mtp="off"` 也获得收益。

**阶段预算**：每 verify 周期 recipe 提交降至 **≤3–5 次**，设备空档 `<10%`，HBM 带宽提升至 **~1.35–1.60 TB/s（55–65%）**，2094-token PE 从 `20.71 s` 降至 **`~8.5–10.0 s`**。

---

### 阶段 M2：FP8 共享激活量化、权重合并与动态有效 KV Attention

**目标**：解决 B4、B5，将 FP8 矩阵突发带宽和 Full Attention 效率推至硬件可持续区。

1. **共享 FP8 激活量化（复用 DiT 方案 B 已验证模式）**：
   - 沿用 `comfy/ldm/qwen_image21/hpu.py` 已通过 100% 逐位一致验证的 `_quantize(x, tensor_divisor)` 方案（设备张量除数 `240.0` 保持 eager 舍入），对共享输入的投影只做一次行 amax + `cast_to_fp8_v2`：
     - 32 层 MLP：`gate_proj` + `up_proj`（省 1/2 量化）；
     - 24 层 GDN：`in_proj_qkv` + `in_proj_z`（省 1/2 量化）；
     - 8+1 层 Full Attention：`q_proj` + `k_proj` + `v_proj`（省 2/3 量化）。
2. **常驻 FP8 投影输出维合并（依据 M0 实测准入）**：
   - 针对 `--disable-dynamic-vram` 下 `CombinedLinear` 失效的问题，在模型加载/权重版本稳定时构造按原权重 scale 保留的逐输出通道 scale 合并矩阵（`gate_up [24576, 4096]` = `100.7 MB`、`qkvz [12288, 4096]` = `67.1 MB`），复用 `fp8_gemm_v2` 逐输出通道 scale 支持（2026-11-16 P2 已验证可运行），大幅改善 FP8 窄 M 带宽；严格核算额外显存（或原地替换原权重指针实现零显存增量）与 LoRA/hooks/卸载重载回退。
3. **Full Attention（8 层主干 + 1 层 MTP）升级**：
   - 将 MTP 树验证的 `FixedKVCache.attend()` 从 FP32 `comfy_kitchen` 参考实现迁移至编译图内的 **`FusedSDPA`（BF16 QKV + FP32 softmax + 树可见性 mask）**，消除 FP32 K/V 转型与低效树合并。
   - 接入 **粗粒度动态有效 KV 切片**（按 `512` 或 `1024` 对齐有效前缀长度，动态维标注复用单份编译图），避免每步扫描满容 6899 槽位，将 8+1 层 KV 读取从 `254 MB` 降至 `~140 MB` 左右。

**阶段预算**：持续 HBM 带宽提升至 **~1.70–1.85 TB/s（70–75%）**，2094-token PE 降至 **`~7.0–8.0 s`**。

---

### 阶段 M3：24 层 GDN 状态流重构与融合核（突破 80% 硬件上限）

**目标**：解决 B3，压缩非 FP8 GEMM 独占时间，达成 **≥1.96 TB/s（≥80% 理论峰值）** 与 **≥80% 阶段关键路径效率**。

1. **重构 MTP GDN 状态提交与树查询数据流**：
   - 评估并对比两种零额外读方案：
     - **A. 紧缩合并 Deferred**：在单次编译区/融合核内按真实 `pending`（1..4）而非固定 `SLOT_MAX=8` 执行状态提交，并与当前 `M=4/8` 树查询共享同一次 `50.3 MB` FP32 `recurrent_state` 加载，消除二次读取与冗余幂级数计算；
     - **B. Immediate 轻量提交**：Verify 阶段只读一次当前已提交状态计算树输出，确认 `path` 后仅对接受的 1..4 行执行一次融合提交，消除双缓冲 `qkv_buf`/`gates_buf`/`sumsq_buf` 写入。
   - 消除 `conv` 路径的冗余 `flip`/cast 与 Habana 1.24.2 4 行输入 alias 触发条件。
2. **专用 TPC-C GDN 融合核（吸取历史撤回教训）**：
   - 复用本机已验证的 16KiB VLM、4/8/16 value 列并行工具链（`libhabana_pytorch2_plugin.so` + `hpu_custom_op_pt2.h`），实现单次片上 VLM 驻留完成：`state` 读取 → `pending` 提交 → 当前树 `M=4/8` 递推输出 → `state` 写回 → `gated RMSNorm`。
   - **吸取 2026-11-12/11-15 教训**：核内显式对齐 FP32 累加顺序与 BF16 舍入点；先验单步与多步 `state`/输出 RRMSE，再验完整 2094-token 固定轨迹 logits 与自然扩写配对图，确保无长序列漂移或图像语义退化。
3. **（条件触发）公共 Synapse C API 持久化执行模板**：
   - 若完成上述步骤后原生 trace 显示设备空档仍 `>8%` 或带宽停在 `75–79%`，复用 D2 方案 A 已在生产验证的公共 Synapse C API（`synLaunch` on `_compute_stream()` + 基地址绑定）封装高频重复块，消除剩余框架调度延迟。

**阶段预算**：**实测持续 HBM 带宽 ≥ 1.96 TB/s（≥ 80% 峰值）**、阶段关键路径效率 **≥ 80%**；PE 吞吐达 **`~320–400 tok/s`**，2094-token PE 耗时 **`~5.7–7.0 s`**（整图进入 **`~17.1–18.4 s`**，PE 耗时低于当前 DiT 的 `8.90 s`）。

---

### 阶段 M4：MTP 高杠杆降本分支（Draft `lm_head` 减负与接受率优化）

**目标**：在硬件跑满 80% 之后，进一步压低每 accepted token 的逻辑 HBM 字节（单列筛选，严格以端到端净收益和分布保真门控）。

1. **Draft 阶段高频词表 Shortlist（严格不改变目标分布）**：
   - Draft 的 `lm_head`（`1.017 GB`）占单步 draft 流量的 80%，但目标模型在 Verify 阶段仍完整运行全部 248,320 词表。
   - 构建 PE 高频词表子集（如 32K–48K 常用中英文及 `<think>`/`</think>`/标点/EOS，仅 `~130–196 MB` FP8）供 Draft 阶段投影，实测接受率变化与每轮省下的 `~2.5 GB` HBM 流量净收益。
2. **投机验证采样策略对比**：
   - 在 `temperature=1.0, top_k=20, top_p=0.95, presence_penalty=1.5` 下，对比现有 **树多分支精确命中（`tree3s1` / 自适应宽度）** 与 **严格无损 Rejection Sampling（$\min(1, p/q)$ + 残差重采样 $(p-q)_+$）** 的平均接受 token 数、重复性及图像质量。
3. **（远期可选，不阻塞本轮）跨请求 System Prompt 前缀缓存**：
   - 本工作流 `system_prompt` 约 2600+ tokens 固定不变，后续如需继续压缩 `~0.43 s` prefill，可在 PE decode 验收完成后单独设计带完整失效契约的 KV+GDN 前缀缓存。

**探索预算**：若 Draft shortlist 保持 ≥95% 原接受率，单轮流量可压至 `~9.6–10.5 GB`，80% 带宽下 PE 吞吐可达 **`~410–520 tok/s`**（2094 tokens **`~4.5–5.5 s`**，整图 **`~16 s`**）。

---

### 阶段 M5：全维度验证、质量验收与默认部署交付

严格遵循 [MEASUREMENT.md](MEASUREMENT.md) 与仓库默认交付流程：

1. **性能与 80% 达标证明**：
   - 1 cold + 5 轮交错 A/B warm（含固定工作量对照与完整自然扩写对照）。
   - 同时给出四项强制指标：校准 BMON 有效 HBM 带宽（TB/s 及占 `2.45 TB/s` 比例）、同 shape 微基准可持续效率、经审计阶段关键路径效率、完整端到端用户体验（PE/整图/cold/连续编辑首步 `<1 s`）。
   - 记录 250ms 遥测功耗、温度、1650 MHz 频率与 `violation=0`、独立进程 `InUse` / `MaxInUse` / 卸载稳定显存。
2. **正确性、缓存隔离与质量验收**：
   - 完整自动化回归：覆盖 `mtp = off / 2 / 3 / tree3s1 / auto`、KV/GDN `rollback` & `commit_path`、EOS 截断、`presence_penalty` / `repetition_penalty`、RNG 请求级隔离、蓝→红→蓝回切、同长度不同内容、跨容量动态切换、`recompile_limit == 0`、`batch_as_strided == 0`、hooks/LoRA/CPU/禁用编译回退及卸载重载。
   - 多组自然扩写与图像配对人工验收：原提示词、seed+1、蓝/红变体、`thinking=True/False`、512²/1024²/1088×1920，保存原图与扩写文本供人工检查。
3. **默认交付流程**：
   - 验证通过 → `git add` 暂存 `gaudi2` 与 `ComfyUI-Gaudi` 改动（暂不提交） → 部署至 `/home/sonny/ComfyUI-Gaudi`（普通 8189 全流程验证、100% 源文件 SHA256 核对、空队列） → 交付人工验收 → 验收确认后再提交。

---

## 4. 风险清单与应对边界

| 风险项 | 应对策略 |
|---|---|
| 跨 checkpoint 嫁接 `qwen3.5_9b` 基础 MTP 头在 T2I 提示词风格下接受率偏低 | 2026-11-15 BF16 实测已证明同提示词接受率达 `1.83–2.10 tok/verify`；M0 第一步即在 FP8 + 多组提示词/thinking 开关下复核；若某场景接受率低于收益阈值，`auto` 模式自动收敛树宽或回退单步 |
| 合并权重（`gate_up`、`qkvz`）导致显存 `MaxInUse` 暴涨 | 优先采用加载期释放原分开权重或共享 storage 的方式，单次增量严格以独立硬件进程 `MaxInUse` 验收（参考 D2 方案 B 守住 ≤40 GiB） |
| 整步编译或动态有效长度切片引入 55 s 冷编译或 `batch_as_strided` 警告 | 所有新编译图均强制经过“12 次连续 KV 容量切换 + 蓝/红/回切 + `thinking` 切换”回归，硬门槛：`recompile_limit == 0`、新容量首个 decode `<1 s` |
| TPC GDN 融合核引入长序列数值漂移或画面缺陷 | 先收割 M1/M2/M3.1 的零误差/低误差收益；TPC 核作为独立候选严格经 2094-token 固定 logits + 多组自然配对图人工验收，不合格即回退 |
| Gaudi2 硬件上 FP8 窄 M 可持续微基准低于 80% 理论峰值（`1.96 TB/s`） | M0 先用独立真实权重轮换微基准标定硬上限；通过 M=4/8 + 合并大矩阵突破小 M 瓶颈，同时如实汇报理论峰值与同 shape 可持续两套占比，不虚报达标 |

---

## 5. 阶段里程碑汇总

| 阶段 | 核心交付 | 预期 HBM 带宽（占比） | 预期 PE 耗时（2094 tok） | 预期整图（配 8.90s DiT） |
|---|---|---:|---:|---:|
| 当前基线（`26a170f9`） | 无 MTP，单步 `M=1` | `0.893 TB/s`（`36.4%`） | `20.71 s` | `32.08 s` |
| **M0** | FP8 MTP 权重构建、小 M 上限表、基线账本 | 测量定标 | 基线定标 | 基线定标 |
| **M1** | Verify/Draft 全步闭合编译、修复 HPU `auto` 计时 | `~1.35–1.60 TB/s`（`55–65%`） | `~8.5–10.0 s` | `~19.9–21.4 s` |
| **M2** | 共享 FP8 量化、权重合并、`FusedSDPA` 树验证 + 动态有效 KV | `~1.70–1.85 TB/s`（`70–75%`） | `~7.0–8.0 s` | `~18.4–19.4 s` |
| **M3** | GDN 状态流重构 + TPC 融合核（+ 按需 Synapse 模板） | **≥ `1.96 TB/s`（≥ `80%`）** | **`~5.7–7.0 s`** | **`~17.1–18.4 s`** |
| **M4（探索）** | Draft 轻量词表 Shortlist + 投机采样策略调优 | **≥ `1.96 TB/s`（≥ `80%`）** | **`~4.5–5.5 s`** | **`~15.9–16.9 s`** |
| **M5** | 回归、质量配对、暂存部署与人工验收 | 完整实测证明 | 验收定版 | 验收定版 |
