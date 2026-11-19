# Qwen Image 2.1 FP8 PE + MTP 优化执行与 M2 部署报告

日期：2026-11-19（Asia/Shanghai）。基线提交：`26a170f97b360a61cc597c0908df294eea03dfa5`（已验收 D2 方案 B，单卡 Gaudi2 HL-225、Habana 1.24.2、eager `PT_HPU_LAZY_MODE=0`）。

本轮按既定 [PE+MTP 80% 优化计划](QWEN_PE_MTP_80_PLAN.md) 与用户确认条件执行 **M0 → M1 → M2**，生成**独立** FP8 PE+MTP 权重文件（不覆盖原无 MTP 文件），完成全套回归、BMON/原生时间线审计、暂存与普通 `http://192.168.10.21:8189` 端到端验证部署。用户已确认“验证通过，提交代码并推进M3”，产品已提交为 **`96a337d58b3674593c252d804755a6cb7c5d8d36`**（tree `0848de81178693597c3ba5542d4b272693489166`），状态为 **`accepted_committed`**。

完整结构化指标、哈希与日志见 [qwen-pe-mtp-m2-2026-11-19.json](../evidence/qwen-pe-mtp-m2-2026-11-19.json)。服务器原始实验资料位于 `/home/sonny/gaudi2/comfyui-hpu/performance-qwen-pe-mtp-2026-11-19`。

---

## 1. 交付标识与独立资产

| 项 | 值 |
|---|---|
| 部署状态 | `accepted_committed`（已通过人工验收并提交；正式产品提交 `96a337d58b3674593c252d804755a6cb7c5d8d36`） |
| 产品暂存身份 | `26a170f9+staged.dfe536e987e3`（基线 `26a170f97b360a61cc597c0908df294eea03dfa5`） |
| 产品暂存树 (`git write-tree`) | `0848de81178693597c3ba5542d4b272693489166` |
| 产品暂存补丁 SHA256 | `dfe536e987e3b92a68d0b91f0d5aa2a233b20ce1523871dc8a43c806dd325b34` |
| 服务地址与队列 | `http://192.168.10.21:8189`（PID `145830`，验收后已热态留存，队列为空） |
| 独立 PE+MTP 模型文件 | `/home/sonny/ComfyUI-Models/models/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i_mtp.fp8_from_int8_fp32.safetensors`（`11,133,746,108` bytes，SHA256 `468cdb77ae8ba0bc0ad906af4282755f90bf12358f1113e935cce36fc4c864bd`，209 个 FP8 Linear = 原 201 + 8 个 MTP Linear） |
| 原无 MTP 模型（未改动） | `/home/sonny/ComfyUI-Models/models/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.fp8_from_int8_fp32.safetensors`（`10,890,431,404` bytes；新文件中全部 961 个非 MTP 张量与其逐字节一致） |
| 独立 UI 工作流 | `/home/sonny/ComfyUI-Gaudi/user/default/workflows/Qwen-Image-2.1-Turbo-FP8-PE-MTP-HPU.json`（仓库副本 [qwen21-fp8-pe-mtp.json](../workflows/qwen21-fp8-pe-mtp.json) · [API 图](../workflows/qwen21-fp8-pe-mtp-api.json)） |
| 全仓源码哈希核对 | `1,074 / 1,074` 份已跟踪 `.py`/`.sh` 文件与本地暂存树一致（`0` 项不一致） |

产品仓库改动（6 份文件，已暂存）：
- `comfy/text_encoders/hpu_kv_attention.py` (`209de7f5...`)：新增 `tree_decode_sdpa`（`FusedSDPA` 树验证注意力）。
- `comfy/text_encoders/hpu_qwen35.py` (`478c0ef9...`)：新增模块化 HPU MTP 编译器与运行时（`gdn3`/`attn`/`tree_mask`/`verify_head`/`mtp_draft_chain`/`kv_commit_slot`）。
- `comfy/text_encoders/qwen35.py` (`edd97f25...`)：放开 MTP 检查点下 `mtp="off"` 单步编译准入；接入 `hpu_qwen35.build_mtp_runner`；紧凑 `mtp_cap = max_length + 7`；修复 Habana 1.24.2 采样掩码重叠 `copy_` 静默丢位 bug；新增 HPU 采样步级 `Event` 计时与 `skip_timing` 防抖。
- `scripts/gaudi/convert_qwen35_fp8.py` (`0b0b5121...`)：新增 `--attach-mtp` 与 `--quantize-mtp`。
- `tests-unit/comfy_test/qwen35_fp8_conversion_test.py` (`4b06116b...`)：补齐 MTP 嫁接与量化转换单测（8 项通过）。
- `tests-unit/comfy_test/hpu_qwen_mtp_test.py` (`2d1179ad...`)：新增 32 项 HPU MTP 专项回归（树 SDPA、向量化 penalty 掩码、跨容量图复用、隔离与 15 项回退门控）。

---

## 2. M0 微基准定标与关键技术取舍

### 2.1 真实形状小 M 带宽上限（`m0/matrix-micro.json` 与 `m0/attention-micro.json`）

权重轮换集合均大于 48 MB SRAM，避免片上缓存失真：

| 算子组 | 变体 | M=1 (TB/s, 占比) | M=4 (TB/s, 占比) | M=8 (TB/s, 占比) | 数值 RRMSE | 结论 |
|---|---|---:|---:|---:|---:|---|
| 4L MLP (`604 MB`) | `mlp_sep` | `1.713` (`69.9%`) | `1.667` (`68.0%`) | `1.835` (`74.9%`) | `0.0` | 基线 |
| 4L MLP (`604 MB`) | **`mlp_shared`** | `1.702` (`69.5%`) | `1.692` (`69.1%`) | **`1.850` (`75.5%`)** | **`0.0`** | **采纳（共享量化零误差且最快）** |
| 4L MLP (`604 MB`) | `mlp_merged` | `1.666` (`68.0%`) | `1.676` (`68.4%`) | `1.804` (`73.6%`) | `0.0221` | **放弃（更慢且引入 2.2% 误差）** |
| 6L GDN in (`302 MB`) | **`gdn_qkvz_sep`** | `1.508` (`61.5%`) | `1.510` (`61.6%`) | **`1.799` (`73.4%`)** | `0.0` | **采纳共享激活量化 + 独立权重** |
| 6L GDN in (`302 MB`) | `gdn_qkvz_merged` | `1.181` (`48.2%`) | `1.031` (`42.1%`) | `1.870` (`76.3%`) | `0.0` | 仅 M=8 略快、M=1/4/16 大幅退化，放弃 |
| 6L Attn QKV (`252 MB`) | **`attn_qkv_sep`** | `1.072` (`43.8%`) | `1.263` (`51.6%`) | **`1.464` (`59.8%`)** | `0.0` | **采纳共享激活量化 + 独立权重** |
| 6L Attn QKV (`252 MB`) | `attn_qkv_merged` | `0.907` (`37.0%`) | `0.740` (`30.2%`) | `0.926` (`37.8%`) | `0.0` | **放弃（全尺寸变慢 37%）** |
| `lm_head` (`1017 MB`) | `fp8` | `2.145` (`87.5%`) | `2.150` (`87.8%`) | **`2.141` (`87.4%`)** | `0.0` | 已超 80% 峰值 |
| 8L Tree Attn (M=8, pre=3800) | `kitchen_tree` → **`fused_sdpa`** | — | `1.130` → **`0.726 ms`** | `1.252` → **`0.855 ms`** | `0.0036` | **采纳（提速 32%）** |

按 `M=8` 同 shape 微基准加权，主干 201 个 FP8 Linear（`7.929 GB`）可持续下界为 **`4.303 ms`（`1.843 TB/s`，占 2.45 TB/s 的 `75.2%`）**；叠加 8L `FusedSDPA`（`0.855 ms`）与 2 步 FP8 draft（`~2.0 ms`），MME+SDPA 纯可持续下界约 **`7.16 ms/verify`**。

### 2.2 关键架构突破与放弃路径

1. **放弃 32 层单片整步编译，改用原始张量元组模块化编译（Verify 提速 `3.09×`、冷启动提速 `20×`）**：
   - Habana 1.24.2 Synapse 在 32 层单片 MTP 图上出现严重调度退化（单 verify `34.40 ms`、冷编译 `74 s`、换 KV 容量重编译 `8.33 s`）。
   - 拆分为 6 个可跨层复用的原始张量元组图（`8× gdn3 + 8× attn + tree_mask + verify_head + mtp_draft_chain + kv_commit_slot`）后，24 层 GDN 完全不依赖 KV 容量，单 verify 降至 `11.15 ms`，冷编译降至 `~5 s`，编辑提示词换容量首次 decode `<0.9 s`（通过 `<1 s` 硬门槛）。
2. **GDN 通道末维布局 `[B, T, C]` + 预归一化 FP32 乒乓状态（24 层 GDN 核心 `15.02 ms → 4.56 ms`，提速 `3.3×`）**：
   - 原 `[B, 8192, 11]` 沿最内维 `index_select(dim=2)` 触发标量 TPC gather（仅 conv 耗时 `6.41 ms`）。改为通道末维 `[B, T, 8192]`、主机预计算树控制表、片上静态 `onchain_ctl` 与预归一化 FP32 `[B, vh, rows, hd]` 乒乓缓冲后，24 层 GDN 核心降至 `4.56 ms`。
3. **消除约 `4.5 ms/verify` 主机/eager 尾开销**：
   - 融合逆 CDF 采样进 `head_compiled`（`1.355 ms → 0.085 ms`，独立 CPU `Generator` + 32 字节 `u_buf` 保请求隔离）；
   - 2D 扁平化 `ftab_2d` 零维 RoPE 索引（`0.61–1.00 ms → 0.127 ms`）并内联 `_native_rope_forward`（消除嵌套 custom_op `recompile_limit`）；
   - 8 层 KV 编译式 sibling 提交 `_compiled_kv_commit_slot`（`2.549 ms → 0.105 ms`，提速 `24×`）；
   - `auto` 模式改为采样步级 `torch.hpu.Event` 计时（前 8 步跳过、16 步收敛、之后每 16 步采样），消除每步 HPU Event 管线空档（`native-auto` 24-verify 窗口 `378.6 → 365.2 ms`，节省 `0.56 ms/verify`）；
   - `mtp_base_buf` 静态 1 元素载体消除 `storage_offset` 变化导致的每步 Synapse 重编译。
4. **明确放弃的候选**：
   - `pe_mtp_bf16.safetensors`（FP8 MTP 在 512 tok `tree3s1` 上比 BF16 MTP 快 `5.7%`：`4.459 s` vs `4.714 s`，且省 `226 MiB` 显存）；
   - `kv_bucket=1024` 非连续切片（切片拷贝叠加额外编译，反而变慢且冷启动增至 `125 s`）；
   - `fuse_stage4=True`（热态与 `gdn3+attn` 持平，但编辑换容量时重编译慢 `350 ms`：`2.218 s` vs `1.870 s`）；
   - 权重合并 `gate_up` / `qkv`（M0 微基准证实变慢且 `mlp_merged` 引入 2.2% 误差）。

---

## 3. 性能实测结果

### 3.1 独立 9B 文本生成：512-token 消融与完整 4096 自然扩写（同提示词 `thinking=True`）

| 阶段 / 模式 | 512-tok 热耗时 (`tok/s`) | 4096 自然热中位（范围） | 自然输出 Tokens | 吞吐 (`tok/s`) | 冷启动 | 接受率 (`tok/verify`) | `MaxInUse` | 告警 (`alias/recomp`) |
|---|---:|---|---:|---:|---:|---:|---:|---|
| 基线 `26a170f9` (`mtp="off"`) | `5.276 s` (`97.05`) | **`20.924 s`** (`20.74–21.11 s`) | `2094` | `100.08` | `89.66 s` | `1.000` | `14.402 GiB` | `0 / 0` |
| M0 未优化 MTP `auto`（eager 回退） | `8.570 s` (`59.74`) | — | — | — | `~35 s` | `2.120` | — | — |
| M1 模块化 FP8（无共享量化/kitchen） | `4.712 s` (`108.66`) | — | — | — | `24.28 s` | — | `14.31 GiB` | `0 / 0` |
| M2 v1（`m2_fp8` / `m2_bf16`） | `4.459 / 4.714 s` | — | — | — | `23.86 s` | — | `14.31 GiB` | `0 / 0` |
| **M2 终版 `auto`（默认）** | **`3.958 s` (`129.36`)** | **`14.426 s`** (`14.21–17.19 s`*) | `1909` | **`132.33`** (`134.34` 稳态) | **`33.60 s`** | `2.166` | `14.422 GiB` | **`0 / 0`** |
| **M2 终版 `tree2s2`** | **`3.962 s` (`129.24`)** | **`13.820 s`** (`13.79–13.85 s`) | `1943` | **`140.60`** | **`33.09 s`** | `2.180` | `14.432 GiB` | **`0 / 0`** |
| **M2 终版 `tree3s1`** | `4.136 s` (`123.79`) | **`13.795 s`** (`13.66–13.84 s`) | `1871` | **`135.63`** | **`33.33 s`** | `2.200` | `14.400 GiB` | **`0 / 0`** |
| M2 终版 `auto` (`thinking=False`) | — | **`5.112 s`** (`5.08–5.15 s`) | `605` | `118.36` | `28.15 s` | `1.993` | `14.413 GiB` | **`0 / 0`** |

> \*注：`auto_full_v2` 的第 2 次 warm 触发一次 `widths=(7,)` 深度 1 一次性冷编译（`17.19 s`），第 3 次稳态恢复 `14.210 s`（`134.34 tok/s`）。编辑提示词换新 KV 容量 64-token 总耗时 `1.870 s`（prefill `0.43 s` + 首个 decode `<0.9 s` + 64 tokens），满足 `<1 s` 硬门槛。

### 3.2 普通 8189 ComfyUI 端到端 1088×1920 完整出图（5 轮交错热态 A/B）

所有热态请求均通过切换 `mtp` 保持 4 个模型加载器 100% 命中缓存，且强制完整重算 `TextGenerate` + `TextEncode` + `Sampler(DiT)` + `VAEDecode`：

| 场景（普通 `8189` API） | 执行耗时 | 相对基线变化 | 备注 |
|---|---:|---:|---|
| `cold-auto`（独立 PE+MTP 首次冷启） | **`81.034 s`** | **−42.5%**（对比历史基线 `~141 s`） | 模块化编译省去 ~55 s 冷启动停顿 |
| **`warm-off`（基线单步，5 轮交错中位）** | **`32.354 s`** (`32.073–32.413 s`) | 基线 | 与原无 MTP 文件像素 100% 一致 (`MAE=0.0`) |
| **`warm-auto`（M2 PE+MTP，5 轮交错中位）** | **`25.660 s`** (`25.569–29.243 s`*) | **−6.694 s / −20.69%** | 稳态 4 轮稳定在 `25.569–25.695 s` |
| **`warm-tree3s1`（M2 固定树）** | **`25.165 s`** | **−7.189 s / −22.22%** | 确定性固定树，像素/文本完全复现 |
| `blue-tree3s1` → `red-tree3s1` → `blue-return-tree3s1` | `25.316 / 24.595 / 25.111 s` | — | `blue` 与 `blue-return` 文本及 RGBA **逐像素一致 (`MAE=0.0`)** |
| `blue-auto` / `red-auto` / `seed+1` | `23.425 / 23.392 / 23.756 s` | — | 提示词编辑无长卡顿 |
| `thinking-off` / `sq512` (528²) / `sq1024` (1056²) | `17.470 / 18.937 / 20.515 s` | — | 跨模式与跨分辨率正常 |
| `baseline-nonmtp-auto` / `baseline-nonmtp-off` | `96.692`（重载冷）/ `31.986 s` | 无回归 | 与 `warm-off-0` 文本及 RGBA **逐像素一致 (`MAE=0.0`)** |
| `/free` 60 秒卸载稳态 → `reload-auto` | **`9.189 GiB`** → **`61.379 s`** | 重载快 `1.93×`（基线 `118.4 s`） | 卸载稳态与 D2 方案 A `9.189 GiB` 完全一致，无泄漏；重载出图与 `warm-auto-0` 逐像素一致 |

---

## 4. 硬件效率深度审计（BMON HBM + 原生时间线 + 遥测）

使用校准后 AXI 返回 beat BMON 配置（24 步 verify 低扰动窗口，起始步 32，覆盖全部 32 个 HBM 端口）：

| 指标 | 基线单步（`2026-11-16` 64-tok） | M2 `auto` (`native-auto-v2`) | M2 `tree3s1` (`native-tree3s1`) | 80% 硬件目标 |
|---|---:|---:|---:|---:|
| Profiler 采集扰动 | `−0.70%` | **`+0.78%`** (`≤5%` 通过) | **`+1.57%`** (`≤5%` 通过) | `≤ 5%` |
| 每轮 Verify HBM 流量 | `8.706 GB`（1 token） | `12.200 GB`（48 drafts / 24 verify） | `13.303 GB`（72 drafts / 24 verify） | — |
| **每接受 Token HBM 流量** | `8.706 GB/tok` | **`5.324 GB/tok`（−38.9%）** | `7.256 GB/tok`（全长按 2.20 折算 `6.05 GB/tok`） | 降低逻辑流量 |
| 有效持续 HBM 带宽（占 2.45 TB/s） | `0.893 TB/s` (`36.44%`) | `0.802 TB/s` (`32.72%`) | `0.887 TB/s` (`36.21%`) | **`≥ 1.96 TB/s` (`≥ 80%`)** |
| 同 shape MME+SDPA 下界效率 | — | `~47.0%`（`7.16 / 15.22 ms`） | `~53.8%`（`8.16 / 15.18 ms`） | `≥ 80%` |
| 设备计算覆盖率 / 空档 | `62.18%` / `37.82%` | `67.12%` / `32.88%` (`5.03 ms/verify`) | `70.61%` / `29.39%` (`4.46 ms/verify`) | 空档 `< 10%` |
| MME / TPC / EDMA 活跃（每 verify） | — | `6.01 / 4.48 / 0.97 ms` | `6.56 / 4.40 / 0.98 ms` | — |
| Recipe 提交数 | `18.0 / token` | `31.0 / verify` (**`13.53 / token`**) | `30.38 / verify`（全长 `13.81 / token`） | `≤ 3–5 / verify` |
| 250ms 遥测（均值功率 / 峰值 / 温度 / 频率） | `182.3 W` / `318 W` / `45°C` / `1650 MHz` | `180.6 W` / `305 W` / `45°C` / `1650 MHz` | — | 无降频 |

### 如实说明：为什么 M2 整图已降 6.7 秒，但距离 80% 硬件带宽上限仍差 M3？

硬件时间线把剩余瓶颈量化得一清二楚：
1. **M2 的核心成果**：
   - 把**每接受 token 的逻辑 HBM 字节从 `8.71 GB` 压到 `5.32 GB`（−38.9%）**，并把 PE 实测耗时从 `20.92 s` 压到 `13.80–14.43 s`（冷启动 `89.7 s → 33.6 s`），整图从 `32.35 s` 压到 `25.17–25.66 s`。
   - **MME 突发带宽已达到硬件高位**：`auto_v2` 窗口内 MME 仅用 `6.01 ms/verify` 就搬运了 `12.20 GB` 绝大部分权重（等效突发 **`~2.0 TB/s`，>80% 峰值**）。
2. **尚未达到 80% 持续带宽与高功耗的精确原因（留待 M3 攻克）**：
   - **24 层 PyTorch 编译态 GDN 的 TPC 开销（`4.48 ms/verify`，占周期 ~30%）**：每层仍需在 PyTorch 算子级执行两次 `_delta_vh` 与若干小 TPC 算子；
   - **模块化 31 recipes/verify 带来的设备空档（`5.03 ms/verify`，占周期 32.9%）**：受限于 Habana 1.24.2 单片 32 层编译器的调度退化，当前保持 8× `gdn3` + 8× `attn` + 尾胶水提交，主机/调度间隔限制了平均 HBM 利用率与板卡动态功耗（故 `hl-smi` 平均仍约 `181 W` / `29%`）。
3. **M2 验收通过后的 M3 精确发力点**：
   - **M3.1 专用 TPC-C GDN 融合核**：单次 VLM 驻留完成 state 读写 + `pending` 提交 + `M=8` 树递推 + gated RMSNorm，目标将 `4.48 ms` TPC 压至 `~0.8 ms`；
   - **M3.2 公共 Synapse C API (`synLaunch`) 持久执行模板**（沿用 D2 方案 A 生产路径）：绕开 PyTorch 32 层单片编译退化，将 `31 recipes/verify` 与 `5.03 ms` 空档压至 `<0.8 ms`；
   - **M4 Draft 高频词表 Shortlist**：把每步 draft 的 `1.02 GB` 全词表 `lm_head` 压降至 `~160 MB`，进一步冲刺 `~5–7 s` PE。

---

## 5. 回归测试、日志与质量核验

1. **自动化回归（全量在暂存一致源码上运行）**：
   - 23 个 HPU/Qwen/生成/缓存测试套件：**`454 passed, 2 skipped, 0 failed`**；
   - 转换器专项 `qwen35_fp8_conversion_test.py`：**`8 passed`**；
   - 新增 MTP 专项 `hpu_qwen_mtp_test.py`：**`32 passed`**（含 6 组连续 KV 容量图复用与 15 项回退门控，0 次 `batch_as_strided` 告警）。
2. **线上服务日志审计（27 次完整请求，324,692 字节日志）**：
   - `UserWarning: hpu::batch_as_strided`: **`0`**
   - `hit config.recompile_limit`: **`0`**
3. **视觉与扩写质量抽查**（人工复核样张位于服务器 `/home/sonny/ComfyUI-Gaudi/output/gaudi/qwen_pe_mtp_m2/`）：
   - `warm-off-0_00001.png`（基线单步，`32.39 s`）与 `baseline-nonmtp-auto_00001.png` 逐像素一致（`MAE=0.0`）；
   - `warm-auto-0_00001.png`（M2 `auto`，`25.57 s`）与 `tree3s1-1_00001.png`（M2 `tree3s1`，`25.17 s`）：主体亚洲女性、无肩带灰褐色长裙、桃色/白色玫瑰花束、花卉头冠、悬停蓝色蜂鸟、鼠尾草绿背景与手部解剖结构均自然完整，英文扩写流畅遵循系统提示词规范；
   - `blue-tree3s1_00001.png`（深蓝长裙）、`red-tree3s1_00001.png`（深红长裙）忠实响应提示词颜色修改，且 `blue-return-tree3s1_00001.png` 与 `blue-tree3s1_00001.png` 逐像素完全一致（`MAE=0.0`），证明无跨请求状态串扰。
   - **已知行为差异说明**：启用 MTP 投机采样后，由于树验证路径使用每步独立随机流与 FP8 MTP 草稿提案，同一 `seed` 下自然扩写文字和出图构图会与 `mtp="off"` 不同（两者各自在固定树/同模式下 100% 可复现）；如需与原无 MTP 版本逐像素一致，将 `mtp` 设为 `off` 或选用原无 MTP 模型文件即可。
