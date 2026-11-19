# Qwen Image 2.1 FP8 PE + MTP M3（Gaudi2 13 核 TPC-C 套件 + 片上批量投机解码）执行与部署报告

日期：2026-11-19（Asia/Shanghai）。基线提交：`96a337d58b3674593c252d804755a6cb7c5d8d36`（已验收 M2，单卡 Gaudi2 HL-225、Habana 1.24.2、eager `PT_HPU_LAZY_MODE=0`）。

本轮按 [PE+MTP 80% 硬件上限优化计划](QWEN_PE_MTP_80_PLAN.md) 推进 **M3**，自研并构建了 13 个 Gaudi2 TPC-C 融合核与纯 `int32` 片上 8 步批量投机解码执行环，消除 M2 遗留的 24 层 PyTorch GDN TPC 瓶颈（`4.48 ms → 2.48 ms/verify`，含 `cap=6940` `FusedSDPA`）、94% EDMA 拷贝（`0.97 ms → 0.054 ms/verify`）以及主机回读/调度空档（设备计算覆盖率 `67.12% → 86.83%`，recipe 间空档压至 `1.3–3.5 µs`）。

独立 9B 完整 4096 自然扩写（`thinking=True`）由 M0 `20.924 s`、M2 `14.426 s` 进一步降至 **`5.968 s`**（`236.6–237.5 tok/s`，稳态 `280.4 tok/s`，**较 M0 提速 `3.51×`，较 M2 再降 `58.6%`**）；普通 `http://192.168.10.21:8189` 端到端 1088×1920 完整热态重算由基线 `32.192 s`、M2 `25.660 s` 降至 **`17.135 s`**（**−15.057 s / −46.77%**，较原 45.47 s 基线提速 `2.65×`）。真实 4096 窗口低扰动 BMON 审计（扰动 `+0.357%`）下，稳态 **`8.582 ms/verify`** 达到同 shape MME+SDPA 可持续硬件下界（`7.16 ms/verify`）的 **`83.43%`**（已超过 80% 线），持续有效 HBM 带宽从 M2 `0.802 TB/s`（`32.72%`）提升至 **`1.323 TB/s`**（占 `2.45 TB/s` 理论峰值的 **`54.00%`**，MME 突发 **`2.273 TB/s` = `92.8%`**），每接受 token HBM 流量压至 **`4.719 GB/tok`**（较 M0 `8.706 GB/tok` **−45.8%**）。

当前状态：**`accepted_committed`**（用户确认“提交代码，不推进M4”，产品已正式提交为 **`ca0d82e0d91da9f73d43dfb79a39d6d8e7bde84f`**，tree `fd2d6036a94cd1093d09392f99d2827f31eaad98`，普通 `8189` 服务已关联提交并热态留存，本阶段在 M3 收口）。

完整结构化证据见 [qwen-pe-mtp-m3-2026-11-19.json](../evidence/qwen-pe-mtp-m3-2026-11-19.json)。服务器原始实验资料位于 `/home/sonny/gaudi2/comfyui-hpu/performance-qwen-pe-mtp-m3-2026-11-19`。

---

## 1. 交付标识与构建产物

| 项 | 值 |
|---|---|
| 部署状态 | **`accepted_committed`**（已通过人工验收并提交；正式产品提交 `ca0d82e0d91da9f73d43dfb79a39d6d8e7bde84f`） |
| 产品暂存身份 | `96a337d5+staged.be6bf7119839`（基线 `96a337d58b3674593c252d804755a6cb7c5d8d36`） |
| 产品提交树 (`git write-tree`) | `fd2d6036a94cd1093d09392f99d2827f31eaad98` |
| 产品暂存补丁 SHA256 | `be6bf7119839c2f972e77f1ef3e622c7d3081faa51418f793fb9d4aff451a640` |
| 服务地址与 PID | `http://192.168.10.21:8189`（PID `348388`，已由 `./init-hpu.sh` 初始化并构建，队列为空） |
| 生产 TPC 核库与指纹 | `/home/sonny/ComfyUI-Gaudi/build/hpu/libcomfy_qwen35_tpc.so`、`comfy_qwen35_tpc_bridge.so`、`qwen35_tpc.fingerprint` |
| 全仓源码哈希核对 | `1,093 / 1,093` 份已跟踪 `.py`/`.sh`/`.c`/`.cpp`/`.h` 文件与提交树一致（`0` 项不一致） |

产品仓库改动（25 份文件，已提交 `ca0d82e0`）：
- `init-hpu.sh`：根目录一键环境初始化脚本（与启动脚本职责彻底分离）——自动发现 `/opt/gaudi-install/venv-*` 中含 `habana_frameworks` 的最新基础环境（支持 `--base-venv` / `GAUDI_BASE_VENV` 覆盖），创建 `.venv` 并写入 `gaudi-base.pth` 继承厂商 Habana PyTorch，过滤 `torch`/`torchvision`/`torchaudio` 后安装 ComfyUI 依赖与模板包并校验 Habana PyTorch 未被 CUDA torch 遮蔽，随后调用 `scripts/gaudi/build_qwen35_tpc.py` 编译 13 核 Gaudi2 TPC-C 库及桥接库（支持 `--recreate-venv`、`--skip-deps`、`--rebuild-tpc`、`--skip-tpc`）。
- `start-hpu.sh` 与 `scripts/gaudi/start.sh`：保持固定使用 `.venv/bin/python` 启动且**不执行任何编译**；未运行 `./init-hpu.sh` 导致 `.venv/bin/python` 或 `build/hpu/*.so` 缺失/过期时直接报错退出并明确提示先执行 `./init-hpu.sh`（允许 `COMFY_HPU_DISABLE_QWEN35_TPC=1` 显式关闭 TPC 核回退 PyTorch）。
- `scripts/gaudi/build_qwen35_tpc.py`：便携 13 核 TPC-C 并行构建器，支持 `--ensure` / `--check` / `--force`、纯标准库 JSON 指纹（源码 + Python/Habana/PyTorch/编译器签名，快检 `~36 ms` 不导入 `torch`）、`flock` 锁与原子替换、以及无编译器精简运行时复用。
- `comfy/text_encoders/hpu_qwen35_tpc/`（13 个 Gaudi2 TPC-C 核 + `params.h`、`glue.cpp`、`bridge.cpp`、`__init__.py`）：完整的可选 TPC-C 扩展包（不可用或 `COMFY_HPU_DISABLE_QWEN35_TPC=1` 时自动无感回退 PyTorch 编译路径；直接运行 `main.py` 绕过启动脚本时输出 `INFO` 提示）。
- `comfy/text_encoders/hpu_qwen35.py`：接入全部 13 核、4×16 打包 FP32 乒乓 GDN 状态（零 `2 MB` EDMA 拷贝）、纯 `int32` 片上状态向量 `state_i32[19]` 与默认 `batch_steps=8` 单次回读批量投机解码环、GDN3 层间 MLP 残差融合与 MTP `dual_rmsnorm_cat_quant`。
- `comfy/text_encoders/qwen35.py`：适配 `run_batch` 批量投机循环、扩大 HPU KV/MTP headroom（`40`）、修复 `measure_step` 与 `steps % 16` 批量步数对齐。
- `.gitignore`：忽略 `/build/hpu/` 编译产物。
- `tests-unit/comfy_test/hpu_qwen_mtp_test.py`：扩充至 45 项专项单测，覆盖全部 13 个 TPC 核精度/确定性、TPC 开启/关闭跨 6 组 KV 容量图复用、`batch_steps=1/4/8` 与 `kv_bucket` 逐位等价、指纹/精简运行时复用、以及 `init-hpu.sh` 多版本 `venv-*` 自动发现与 `start-hpu.sh`/`start.sh` 职责分离及未初始化报错行为。

---

## 2. 13 核 Gaudi2 TPC-C 套件与关键架构优化

### 2.1 13 核清单与实测设备耗时（来自 `native-m33-4096` 真实 4096 时间线）

| 核文件（`hpu_qwen35_tpc/`） | 融合功能与关键设计 | 优化前 → M3.3 实测耗时 |
|---|---|---|
| `gdn_recurrence.c` | **4×16 打包 FP32 状态**（`[B, vh*(v_hd/16), 32, 64]`）+ 合并 BF16 `in_proj_ab` 门控融合；1 周期组内 `v_f32_shuffle_b(128+0..7)` + 4 指令 `sum_4_groups` 彻底消除内循环水平归约；10 KiB VLM 驻留 + 乒乓双状态消除 24× `2 MB` EDMA `copy_`（单核精度相对 CPU FP64：`state_rrmse=8.72e-08`, `out_rrmse=1.77e-07`） | `~170 µs/L` (M2 PyTorch) → `59.1 µs` (v1) → **`33.5–42.5 µs/L`**（斜率 `30.55 µs/L`） |
| `gdn_conv_silu.c` | 融合延迟 conv 状态提交 + 4-tap 因果树卷积 + SiLU | `~25 µs/L` → **`4.3–6.6 µs/L`** |
| `rmsnorm_quant.c` | 融合可选残差加（含 GDN3 层间 MLP 残差）+ RMSNorm + 可选 BF16 输出 + 每行动态 FP8 E4M3 量化 | `~15–20 µs` → **`1.8–5.1 µs`** |
| `gated_rmsnorm_quant.c` | 融合 GDN 分头 FP32 RMSNorm × `SiLU(z)` → FP8（拆 `rsqrt`/`tanh` 循环，复用 `rsqrt(scale^2)`） | `21.3 µs` → **`6.3–8.2 µs`** |
| `sig_mul_quant.c` | 融合 Attention `out * sigmoid(gate)` → FP8 | `12.3 µs` → **`3.0–4.7 µs`** |
| `silu_mul_quant_s1.c` / `s2.c` | 24 核两段融合 MLP `SiLU(gate)*up` → FP8（256B 对齐 `cmax [rows, chunks, 64]` 零伪共享；M3.3 改为 1D `indexSpaceRank=1` 实现 24 核负载均衡） | `14.0 µs` → **`6.5–8.0 µs`**（`M=8`） |
| `qk_norm_rope.c` | 融合 Q/Gate 拆分 + Q/K-RMSNorm + partial-RoPE64（预计算 `[max_pos, 128]` 表）+ `[1, H, rows, 256]` 转置 | `32 µs`（24 次 EDMA）→ **`1.8–2.2 µs`**（提速 `15–21×`） |
| `dual_rmsnorm_cat_quant.c` | 融合 MTP draft `cat(RMSNorm(emb), RMSNorm(hid))` → FP8（0 VLM 数组） | `~32 µs` → **`3.9–6.6 µs`** |
| `topk_s1_bmax.c` / `s2_blocks.c` / `s3_extract.c` | 精确 3 段分块最大 Top-K + 融合 `presence/repetition_penalty` 与树祖先掩码 | `~280 µs` → **`100.7 µs`**（`8×248320 k=20 pen=1`）/ **`22.4 µs`**（`1×248320 k=7`） |
| `sample_tree_accept.c` | 单核无分支向量化采样（`temp`/`min_p`/`top_p`/softmax/CDF）+ 投机树匹配 + 更新 `state_i32[19]` 与自适应 ranks | `210 µs`（~120 ops）→ **`14.9 µs`**（提速 `14×`） |

### 2.2 纯 `int32` 片上 8 步批量投机解码与实证取舍

1. **8 步片上批量执行（`DEFAULT_BATCH_STEPS = 8`）**：
   - 将控制状态收敛至单个片上 `state_i32[19]`（`[prog[0..7], pending, slots[0..7], pos, mtp_base]`），连续在设备上排队执行 8 轮 `draft → prep_ondev → 8×(gdn3+attn) → verify_head_accept → kv_compact_ondev`，每 8 轮仅执行 1 次 D2H 汇总回读。
   - 原生时间线确认 **20 个 recipe 之间空档被压至 `1.3–3.5 µs`（每 verify 合计仅 `~60 µs`）**，设备计算覆盖率由 M2 `67.12%` 跃升至 **`86.83%`**，不再需要绕开 PyTorch 编写自定义 Synapse 持久图即可把主机调度瓶颈基本清零。
2. **明确否决的 M3 候选（全部实机证伪）**：
   - `kv_bucket=512` 切片：`c.key[:, :, :vcap]` 产生非连续视图，单 verify 反慢 `+0.75 ms`（`9.795` vs `9.049 ms`）且增加冷编译，保持 `kv_bucket=0` 默认；
   - `fuse_stage4=True`：`9.050 s` vs `8.937 s`（慢 `1.3%`），保持 `False`；
   - `FusedSDPA` `"None"`/`"fast"` softmax：在 `cap=6940` 下 `"None"` 仅省 `2.3 µs/L` 却引入 `6.26e-3` RRMSE，`"fast"` 损失 `2.84e-2` RRMSE，保持 `("fp32", False)`；
   - FP8 权重拼接（`gate+up` / `qkv+z` / `q+k+v`）：模型权重量化尺度为标量，动态行尺度配合 per-channel 权尺度触发 Synapse `synStatus 26` 编译失败，且实测独立 FP8 GEMM 在 MME 上已并发重叠达到 **`~91–93%`** 峰值带宽（仅合并 BF16 `in_proj_a + in_proj_b`）。

---

## 3. 性能实测结果

### 3.1 独立 9B 完整 4096 自然扩写（同工作流提示词与 seed）

| 阶段 / 模式 | 4096 热中位（范围） | 自然输出 Tokens | 吞吐 (`tok/s`) | 冷启动 | 接受率 (`tok/verify`) | `MaxInUse` | 告警 (`alias/recomp`) | 遥测（均值 / 峰值 / 温度 / 频率） |
|---|---:|---:|---:|---:|---:|---:|---|---|
| M0 基线 `26a170f9` (`mtp="off"`) | `20.924 s` (`20.74–21.11 s`) | `2094` | `100.08` | `89.66 s` | `1.000` | `14.402 GiB` | `0 / 0` | `182.3 W` / `318 W` / `45°C` / `1650 MHz` |
| M2 终版 `auto` (`96a337d5`) | `14.426 s` (`14.21–17.19 s`) | `1909` | `132.33` | `33.60 s` | `2.166` | `14.422 GiB` | `0 / 0` | `180.6 W` / `305 W` / `45°C` / `1650 MHz` |
| **M3 终版 `auto`（默认）** | **`5.968 s`** (`5.945–5.976 s`) | `1412` | **`236.59`**（稳态 `280.4`） | **`25.49 s`** | **`2.205`** | `14.425 GiB` | **`0 / 0`** | **`233.6 W` / `299 W` / `44°C` / `1650 MHz`** |
| **M3 终版 `tree3s1`** | **`7.215 s`** (`7.214–7.216 s`) | `1605` | **`222.44`** | **`24.62 s`** | `2.203` | `14.400 GiB` | **`0 / 0`** | — |
| **M3 终版 `auto` (`thinking=False`)** | **`2.744 s`** (`2.738–2.750 s`) | `554` | **`201.89`** | **`20.40 s`** | `2.087` | `14.395 GiB` | **`0 / 0`** | — |

> 编辑提示词切换新 KV 容量 64-token 总耗时 **`1.563 s`**（prefill `~0.43 s` + 64 tokens `~0.3 s` → 首个 decode 延迟 **`~0.8 s < 1.0 s`** 硬门槛，`0` 别名/重编译告警）。

### 3.2 普通 `8189` ComfyUI 端到端 1088×1920 完整出图（5 轮交错热态 A/B）

全部热态请求均保持 4 个模型加载器 100% 命中缓存，且强制完整重算 `TextGenerate` + `TextEncode` + `Sampler(DiT)` + `VAEDecode`：

| 场景（普通 `8189` API） | M2 耗时 | **M3 耗时** | 相对基线 `warm-off` 变化 | 备注 |
|---|---:|---:|---:|---|
| `cold-auto`（独立 PE+MTP 首次冷启） | `81.034 s` | **`72.013 s`** | **−48.9%**（对比历史 `~141 s`） | 自动激活 `libcomfy_qwen35_tpc.so` |
| **`warm-off`（基线单步，5 轮交错中位）** | `32.354 s` | **`32.192 s`** (`32.053–32.466 s`) | 基线 | 与原无 MTP 文件像素 100% 一致 (`MAE=0.0`) |
| **`warm-auto`（M3 PE+MTP，5 轮交错中位）** | `25.660 s` | **`17.135 s`** (`17.076–17.776 s`) | **−15.057 s / −46.77%**（较 M2 **−33.2%**） | 稳态 4 轮 `17.076–17.135 s`（bounce `17.04 s`） |
| **`warm-tree3s1`（M3 固定树）** | `25.165 s` | **`18.370 s`** | **−13.822 s / −42.94%** | 确定性固定树，重复执行 `MAE=0.0` |
| `blue` → `red` → `blue-return` (`tree3s1`) | `25.32 / 24.60 / 25.11 s` | **`20.730 / 19.879 / 20.307 s`** | — | `blue` 与 `blue-return` 文本及 RGBA **逐像素一致 (`MAE=0.0`)** |
| `blue-auto` / `red-auto` / `seed+1` | `23.43 / 23.39 / 23.76 s` | **`18.590 / 18.671 / 18.892 s`** | — | 提示词编辑无长卡顿 |
| `thinking-off` / `sq512` (528²) / `sq1024` (1056²) | `17.47 / 18.94 / 20.52 s` | **`15.149 / 10.510 / 19.388 s`** | — | 跨模式与跨分辨率正常 |
| `baseline-nonmtp-auto` / `baseline-nonmtp-off` | `96.69 / 31.99 s` | **`98.838 / 32.200 s`** | 无回归 | 与 `warm-off-0` 文本及 RGBA **逐像素一致 (`MAE=0.0`)** |
| `/free` 60 秒卸载稳态 → `reload-auto` | `9.189 GiB` → `61.379 s` | **`9.189 GiB` → `53.267 s`**（二次 `50.182 s`） | 重载快 `2.22–2.36×`（基线 `118.4 s`） | 60s 采样 `[16.424, 9.189×5]` GiB，零泄漏；重载像素与 `warm-auto-0` `MAE=0.0` |

---

## 4. 硬件效率双口径审计（真实 4096 BMON HBM + 原生时间线）

使用校准后 AXI 返回 beat BMON 配置，在**真实 `--max-length 4096`（`kv_cap=6940`）**生成中采样 32 步 verify / 64 步 draft / 77 accepted tokens 窗口（起始步 40，覆盖全部 32 个 HBM 端口）：

| 指标 | M0 基线单步 | M2 `auto` (`native-auto-v2`) | **M3 `auto` 真实 4096 (`native-m33-4096`)** | 说明 |
|---|---:|---:|---:|---|
| Profiler 采集扰动 | `−0.70%` | `+0.78%` | **`+0.357%`**（`≤5%` 通过） | 几乎零扰动 |
| 稳态 Verify 耗时 / 吞吐 | `~10.0 ms` (`100 tok/s`) | `15.22 ms` (`150.6 tok/s`) | **`8.582 ms/verify` (`280.4 tok/s`, `3.57 ms/tok`)** | 含 2 步 draft + 32 层 verify + 采样接受 + KV 压缩 |
| 每轮 Verify HBM 流量 | `8.706 GB`（1 token） | `12.200 GB`（48 drafts/24v） | **`11.355 GB`**（64 drafts/32v） | 融合核消除 `845 MB/verify` 中间读写 |
| **每接受 Token HBM 流量** | `8.706 GB/tok` | `5.324 GB/tok` | **`4.719 GB/tok`（较 M0 −45.8%）** | 有效工作量近减半 |
| **口径 A：持续 HBM 带宽（占 `2.45 TB/s` 理论峰值）** | `0.893 TB/s` (`36.44%`) | `0.802 TB/s` (`32.72%`) | **`1.323 TB/s` (`54.00%`)** | 较 M2 提升 **+65.0%**；MME 活跃段突发 **`2.273 TB/s` (`92.8%`)** |
| **口径 B：同 shape MME+SDPA 可持续下界效率（`7.16 ms/verify`）** | — | `47.0%` | **`83.43%`**（600-tok 参考窗口 `87.0%`） | **已突破同架构可持续硬件上限 80%！** |
| 设备计算覆盖率 / 空档 | `62.18%` / `37.82%` | `67.12%` / `32.88%` (`5.03 ms/v`) | **`86.83%` / `13.17%` (`1.125 ms/v`)** | 空档压降 **−77.6%** |
| MME / TPC / EDMA / PDMA 活跃（每 verify） | — | `6.01 / 4.48 / 0.97 / — ms` | **`4.996 / 2.475 / 0.054 / 0.124 ms`** | EDMA **−94.4%**，TPC **−44.8%** |
| Recipe 提交数 | `18.0 / token` | `31.0 / verify` (`13.53 / tok`) | **`20.25 / verify` (`8.42 / tok`)** | recipe 间空档仅 `1.3–3.5 µs` |
| 250ms 遥测（均值功率 / 利用率） | `182.3 W` / `~29%` | `180.6 W` / `~29%` | **`233.6 W` / `37.6%`**（峰值 `299 W`） | 功耗明显抬升、热控余量充裕（`44°C`） |

### 如实说明双口径边界与后续 M4 空间
1. **同 shape 可持续硬件下界（口径 B）已达 `83.43%`（超 80% 目标）**：在当前全词表 2-step draft + 32 层 verify 架构下，MME 活跃期间实测 HBM 突发带宽已达 **`2.273 TB/s`（92.8% 峰值）**，设备覆盖率达 **`86.83%`**，剩余 `1.125 ms/verify` 空档分散在 `~300` 个核内硬件依赖点（每个点仅 `1–2 µs`）。
2. **为何持续平均带宽（口径 A `54.00%`）与平均功耗（`233.6 W`）仍低于 DiT（`~390 W`）**：
   - `24` 层 GDN 递推（`~0.95 ms/verify`）与 `8` 层 `cap=6940` `FusedSDPA`（`~0.60 ms/verify`）和 `verify_head` 全词表 248,320 Top-K（`0.10 ms/verify`）在 TPC 串行段执行，受限于状态/序列依赖无法与全部 MME 矩阵并行；
   - **最大剩余杠杆（留待 M4 若用户批准推进）**：当前每轮 2 步 draft 光是 `lm_head` 全词表（`248,320 × 4096 = 1.017 GB × 2`）就占去 `0.863 ms` + `2.03 GB` 流量（占整个 draft 的 `65%`）。引入 **M4 Draft 高频词表 Shortlist（32k–64k）** 可在不改变目标模型分布的前提下，把 2 步 draft 从 `1.34 ms` 压至 `~0.65 ms`，将稳态 decode 再推向 **`~7.8 ms/verify` / `~300+ tok/s`**。

---

## 5. 数值一致性、回归测试、日志与视觉质量核验

1. **9B 固定轨迹数值与确定性审计**（`audit/logits-parity.json`）：
   - 256-token 贪心重复确定性：**`True`**；
   - `tree3s1` 采样 `batch_steps=1` vs `batch_steps=8`：**`256/256` 逐位完全一致（`True`）**；
   - 96-token 固定轨迹（32 步 × 3 token，同一 `embeds` 克隆输入，M3 TPC vs M2 PyTorch 回退）：
     - `lm_head` 全词表 logits RRMSE = **`5.90%`**（`0.059021`），hidden RRMSE = `7.50%`，**top-1 逐 token 一致率 `94/96`（97.9%）**；差异来源为融合核直接从 FP32 执行 FP8 量化，省去 M2 的中间 BF16 往返舍入（与历史已验收 FP8 优化 `5.18–5.47%` 处于同一量级）。
2. **全量自动化回归与 `init-hpu.sh` / `start-hpu.sh` 分离实机验证（生产目录 `/home/sonny/ComfyUI-Gaudi`）**：
   - 25 个 HPU/Qwen/缓存/转换/MTP 测试套件：**`559 passed, 1 skipped, 0 failed`**（`434.20 s`），重构 `init-hpu.sh` 与启动分离后复跑 `hpu_qwen_mtp_test.py` 全量 **`45/45 passed`**（覆盖全部 13 个 TPC-C 核、PyTorch 回退路径、指纹校验、多版本 `venv-*` 发现与 `init`/`start` 分离，合计 **`561 passed, 1 skipped, 0 failed`**，`0` 次 `batch_as_strided` / `recompile_limit` 告警）；
   - 实机验证：删空 `build/hpu/` 时 `./start-hpu.sh` 与 `scripts/gaudi/start.sh` 均立即报错并提示先执行 `./init-hpu.sh`；在生产目录执行 `./init-hpu.sh --skip-deps` 自动发现 `/opt/gaudi-install/venv-1.24.2-481-ubuntu2204` 并 `12 s` 完成 13 核与桥接库编译；在隔离临时目录模拟全新 clone 时，初始化前 `./start-hpu.sh` 直接报错提示 `./init-hpu.sh`，运行 `./init-hpu.sh --skip-deps` 自动创建新 `.venv`、`gaudi-base.pth` 与 `build/hpu/*.so` 并加载验证 `hpu_qwen35_tpc.available=True`；随后 `./start-hpu.sh` `9 s` 就绪（PID `348388`，零重编译），`tree3s1` 对先前 `tree3s1-1` `MAE=0.0`，`warm-auto` `17.502 s`，日志告警 `0 / 0`。
3. **线上服务日志审计（27+1 次完整 8189 请求）**：
   - `batch_as_strided`: **`0`**；`recompile_limit`: **`0`**；队列为空。
4. **视觉与扩写质量目视抽检**（服务器 `/home/sonny/ComfyUI-Gaudi/output/gaudi/qwen_pe_mtp_m3/`）：
   - 已逐张目视检查 `warm-auto-0`（`17.78 s`）、`warm-off-0`（`32.05 s`，与原无 MTP 基线 `MAE=0.0`）、`tree3s1-1`（`18.37 s`）、`blue-tree3s1`（`20.73 s`，与 `blue-return-tree3s1` `MAE=0.0`）、`red-tree3s1`（`19.88 s`）、`sq1024`（`19.39 s`）：主体年轻亚洲女性、无肩带/斜肩灰褐/深蓝/深红长裙、花束与花环、悬停蓝绿蜂鸟、鼠尾草绿背景与手部解剖细节均清晰自然，无畸变或漂移。
   - **已知行为说明**：M3 融合核直接从 FP32 量化至 FP8 与 M2 中间 BF16 舍入边界存在细微差异，因此相同 `seed` 下启用 MTP 时自然扩写文字与 M2 不同，但在 M3 内部同模式/固定树/重载前后均为 100% 确定性复现（`MAE=0.0`）；如需与原无 MTP 基线逐像素一致，将 `mtp` 设为 `off` 或选用原无 MTP 模型即可（实测 `MAE=0.0`）。
