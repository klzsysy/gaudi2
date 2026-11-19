# 当前状态与未决问题

截至 2026-11-20，产品分支 `master`（由 `codex/gaudi2-hpu` rebase 至上游 ComfyUI `v0.46.0` / `995456d0` 合并）已验收提交并推送，HEAD 为 `1b096a204dcd04f410801df4148c02584c33197f`（原暂存身份 `c663a324+staged.bca16cb3c3e9`，tree `266cf4c18704b3ecfa1f6489c2acdcd87d92696a`）。普通 8189（PID `363207`）已关联此提交，状态为 **`accepted_committed`**。上一已验收提交为 PE+MTP M3 `ca0d82e0d91da9f73d43dfb79a39d6d8e7bde84f`。

## 2026-11-20 上游 v0.46.0 Rebase 合并与 README 更新（已验收提交）

- 将 `ComfyUI-Gaudi` 的 `master` 快进同步至 `upstream/master`（`995456d0`，ComfyUI `v0.46.0`，较原基线新增 88 个上游提交），将 `codex/gaudi2-hpu` 的 45 个提交完整 rebase 合并至 `master`（`c663a324`），适配上游 `c5751c81`（移除 DirectML）、`6f90ac4f`（greedy penalty 重构，`llama.py`/`qwen35.py`/`hpu_qwen35.py` 及 `hpu_textgen_test.py`）与 `4a9636ac`（`Qwen35ClipModel` 嵌入 tokenizer 导致 `hpu_fp8_linear_test.py` 测试桩补齐 `self.transformer`）。
- 更新 `README.md` 新增 `Intel Gaudi2 (HPU) Support` 章节（非原厂机头推荐 `gaudi2-single-card-driver`、已验证模型列表及 `./init-hpu.sh` / `./start-hpu.sh` 初始化启动说明），与 `hpu_fp8_linear_test.py` 修复一同提交为 `1b096a20` 并推送至 `origin/master`。
- 部署至 `/home/sonny/ComfyUI-Gaudi` 后核对全部 **`1,418/1,418`** 份跟踪文件 SHA256 一致；`./init-hpu.sh` 升级 `comfyui-frontend-package==1.59.6`、`comfyui-workflow-templates==0.11.111`、`comfyui-embedded-docs==0.5.17`（默认 SUSTech 镜像缺 `comfyui-workflow-templates-media-assets-02==0.1.34`，使用服务器已有 `http_proxy_enable` + PyPI 完成）并验证 Habana PyTorch 未被遮蔽与 `build/hpu/*.so` 有效。
- 自动验证：`scripts.gaudi.smoke` `PASS`，27 套 HPU/核心回归 **`564 passed, 2 skipped, 0 failed`**。普通 8189（PID `363207`）端到端 Qwen-Image-2.1 Turbo FP8 PE+MTP：`cold-auto` **`73.439 s`**、`tree3s1` **`21.197 s`**、`warm-auto` **`17.523 s`**，`tree3s1` 与 `warm-auto` 扩写文本及 RGBA 像素相对 M3 基准均 **`MAE=0.0`（逐像素一致）**；日志 `batch_as_strided=0`、`recompile_limit=0`、`Traceback=0`，队列为空。回退备份与部署记录位于 `/home/sonny/gaudi2/comfyui-hpu/deploy-rebase-2026-11-19`。

## 2026-11-19 Qwen PE+MTP M3 验收与提交

产品新增根目录 `init-hpu.sh`（自动发现 `/opt/gaudi-install/venv-*` 创建 `.venv` 与 `gaudi-base.pth`、安装非 torch 依赖并校验 Habana PyTorch 未被遮蔽、调用 `scripts/gaudi/build_qwen35_tpc.py` 并行编译 13 核 Gaudi2 TPC-C 库），与 `start-hpu.sh` / `scripts/gaudi/start.sh` 职责彻底分离——启动脚本固定执行 `.venv/bin/python` 且不做编译，若未执行 `./init-hpu.sh` 导致 `.venv` 或 `build/hpu/*.so` 缺失/过期则直接报错提示先运行 `./init-hpu.sh`（支持 `COMFY_HPU_DISABLE_QWEN35_TPC=1` 显式回退）。13 核 TPC-C 套件覆盖 4×16 打包 FP32 乒乓状态 + 融合 `in_proj_ab` 门控的 GDN 递推（`33.5–42.5 µs/L`，对 CPU FP64 `out_rrmse=1.77e-7`）、因果树卷积+SiLU、`rmsnorm_quant`（含 GDN3 层间残差融合）、`gated_rmsnorm_quant`、`sig_mul_quant`、24 核 1D 负载均衡两段 `silu_mul_quant`、`qk_norm_rope`（`1.8–2.2 µs`，提速 15–21×）、MTP `dual_rmsnorm_cat_quant`、3 段精确罚分 Top-K 与单核向量化采样接受核 `sample_tree_accept`（`14.9 µs`），并接入纯 `int32` 片上 `state_i32[19]` 与默认 `batch_steps=8` 单次回读投机执行环。

独立 9B 完整 4096 自然扩写（`thinking=True`）：`auto` 热中位从 M0 `20.924 s`、M2 `14.426 s` 降至 **`5.968 s`**（`236.6 tok/s`，稳态 `280.4 tok/s`，**较 M0 `3.51×`、较 M2 −58.6%**，冷启 `25.49 s`，`MaxInUse=14.425 GiB`，`0/0` 告警，平均功耗 `180.6→233.6 W` / 峰值 `299 W` / `44°C` / `1650 MHz`）；`tree3s1` **`7.215 s`**（`222.4 tok/s`）；`thinking=False` **`2.744 s`**（`201.9 tok/s`）。新 KV 容量编辑 64-tok 耗时 `1.563 s`（首 decode `<0.9 s`）。96-tok 固定轨迹（M3 TPC vs M2 PyTorch）`lm_head` logits RRMSE `5.90%`、top-1 一致 `94/96`，`tree3s1` `batch_steps=1` vs `8` `256/256` 逐位一致。

真实 4096 窗口低扰动 BMON + 原生时间线（扰动 `+0.357%`，32 verify / 77 tokens）：稳态 **`8.582 ms/verify`** 达到同 shape MME+SDPA 下界（`7.16 ms/verify`）的 **`83.43%`**（突破 80% 线）；持续 HBM 带宽从 M2 `0.802 TB/s` (`32.72%`) 升至 **`1.323 TB/s`（`54.00%` 峰值，MME 突发 `2.273 TB/s` = `92.8%`）**；每接受 token HBM 流量降至 **`4.719 GB/tok`**（较 M0 **−45.8%**）；设备计算覆盖率 **`86.83%`**（空档 `5.03 → 1.125 ms/verify`，−77.6%），EDMA `0.97 → 0.054 ms/verify`（−94.4%），`20.25 recipes/verify`（recipe 间空档 `1.3–3.5 µs`）。

普通 8189 端到端 1088×1920（5 轮交错热态全重算）：`warm-off` 中位 `32.192 s` → `warm-auto` 中位 **`17.135 s`**（稳态 `17.076–17.135 s`，**−15.057 s / −46.77%**，较 M2 `25.660 s` 再降 **−33.2%**），`warm-tree3s1` **`18.370 s`**，`cold-auto` **`72.013 s`**，`/free` 60 秒卸载稳态 **`9.189 GiB`**（零泄漏），重载 **`53.267 s`**（二次 `50.182 s`）。`blue→red→blue-return`、原无 MTP 文件兼容、`warm-auto` 复现与重载全部 `MAE=0.0`，6 张目视样张画质正常。全量 25 套回归 **`559 passed, 1 skipped, 0 failed`**，重构 `init-hpu.sh` 后复跑 `hpu_qwen_mtp_test.py` **`45/45 passed`**（合计 `561 passed`），全新隔离目录 `./init-hpu.sh` 与生产目录重建及重启（`9 s`）均实测通过，`tree3s1` 复验 `MAE=0.0`、`warm-auto` `17.502 s`，`1,093/1,093` 份源码哈希一致，线上日志 `batch_as_strided=0`、`recompile_limit=0`，队列为空。[M3 报告](optimization/QWEN_PE_MTP_M3_2026-11-19.md) · [M3 证据](evidence/qwen-pe-mtp-m3-2026-11-19.json)

## 2026-11-19 Qwen PE+MTP M2 验收与提交

按用户确认执行 [PE+MTP 80% 计划](optimization/QWEN_PE_MTP_80_PLAN.md) 的 M0–M2，新增独立模型文件 `qwen3.5_9b_qwen_image_2.1_pe_t2i_mtp.fp8_from_int8_fp32.safetensors`（`11,133,746,108` bytes，SHA256 `468cdb77...`，209 个 FP8 Linear，保留原无 MTP 文件未动）与独立 UI 工作流 `Qwen-Image-2.1-Turbo-FP8-PE-MTP-HPU.json`。

产品新增 `comfy/text_encoders/hpu_qwen35.py` 及 `tree_decode_sdpa`，实现原始张量元组模块化编译（`8× gdn3 + 8× attn + tree_mask + verify_head + mtp_draft_chain + kv_commit_slot`，规避 Habana 1.24.2 32 层单片退化）、通道末维 `[B, T, C]` 因果卷积与预归一化 FP32 乒乓 GDN 状态、共享 FP8 激活量化、融合逆 CDF 采样、2D 扁平化 RoPE 查表、编译式 8 层 KV sibling 提交以及采样步级 HPU `Event` 计时，同时修复 Habana 1.24.2 采样掩码重叠 `copy_` bug 并支持 MTP 检查点下 `mtp="off"` 单步编译。

独立 9B 完整 4096 自然扩写（`thinking=True`）：基线单步 `20.924 s`（2094 tok，`100.08 tok/s`，冷启 `89.66 s`）→ M2 `auto` **`14.426 s`**（稳态 `14.210 s`，1909 tok，`132.3–134.3 tok/s`，冷启 `33.60 s`）、`tree2s2` **`13.820 s`**（1943 tok，`140.60 tok/s`）、`tree3s1` **`13.795 s`**（1871 tok，`135.63 tok/s`），编辑换新容量首个 decode `<0.9 s`，独立进程 `MaxInUse=14.422 GiB`（基线 `14.402 GiB`）。

普通 8189 端到端 1088×1920（5 轮交错热态全重算）：`warm-off` 中位 `32.354 s` → `warm-auto` 中位 **`25.660 s`**（−6.694 s / **−20.69%**）、`warm-tree3s1` **`25.165 s`**（**−22.22%**），`cold-auto` **`81.034 s`**（历史基线 `~141 s`），`/free` 60 秒卸载稳态 `9.189 GiB`（无泄漏），重载 **`61.379 s`**（快 `1.93×`）。`blue→red→blue-return`（`tree3s1`）及 `warm-off` 对原无 MTP 文件均通过文本与 RGBA 逐像素一致检验（`MAE=0.0`），27 次线上日志 `batch_as_strided=0`、`recompile_limit=0`，454+8 项自动化回归通过（2 项跳过），1074 份 `.py`/`.sh` 源文件哈希与本地暂存树一致，队列为空。

低扰动 BMON 与原生时间线审计（扰动 `+0.78%` / `+1.57%`）：每接受 token HBM 流量从 `8.706 GB` 降至 **`5.324 GB`（−38.9%）**；MME 突发已达 `~2.0 TB/s`，但受限于 24 层 PyTorch 编译态 GDN 的 TPC 开销（`4.48 ms/verify`）和 31 recipes/verify 设备空档（`5.03 ms/verify`），持续 HBM 带宽当前为 `0.802–0.887 TB/s`（`32.7–36.2%` 峰值），尚未达到 80%（`1.96 TB/s`），该差距将在 M2 人工验收通过后由 M3（TPC-C GDN 融合核 + 公共 Synapse `synLaunch` 持久图）和 M4（Draft shortlist）攻克。[执行与部署报告](optimization/QWEN_PE_MTP_M2_2026-11-19.md) · [结构化证据](evidence/qwen-pe-mtp-m2-2026-11-19.json)

## 2026-11-18 D2 方案 B 分段融合部署

在 A 的 RMW attention 上共享 QKV 激活量化，保留 FP32 行尺度与 BF16 投影舍入，以及独立 FFN。在途 block 限制为两个。五轮交错固定 caption DiT 中位 10.375→8.909 秒（−14.14%），完整自然整图 33.497→32.077 秒（−4.24%）。独立进程负条件 TPC-other 独占 191.28→112.20 ms，HBM 731.43→492.09 GB，峰值 40.35→35.25 GiB。8.3–8.6 秒的 DiT 预估仍有差距。

41 次验证请求，20 组配对图像逐像素相同，16 次去噪输出逐位相同。294 项回归通过、1 项跳过，普通服务七次出图匹配验证版，1072 份 Python/shell 源文件与暂存树匹配，普通队列为空。用户确认“验证通过，提交代码”，产品代码及调查记录已提交。[执行记录](optimization/QWEN_D2_SRAM_2026-11-17.md) · [部署证据](evidence/qwen-d2-plan-b-2026-11-18.json)

## 2026-11-18 D2方案A（公共Synapse RMW片上SRAM流水）验证与部署

按用户要求完成首轮D2撤回根因复盘并推进方案A：使用公共Synapse C API在16MiB RMW Section内构建`group_heads=4, slots=2, tile=448/480`双槽位片上SRAM attention流水（`workspace=256B`），128对齐主段在SRAM内执行并与短尾段合并，通过BF16 `q`预缩放实现与D1 100%逐位相等（76/76组真实捕获输入`rrmse=0.0`，单注意力调用`9.411→8.229ms`、−12.56%）。

五轮交错固定caption整图`12.923→12.304`秒（−4.79%）、DiT `11.001→10.386`秒（−5.59%）；五轮交错完整自然整图`34.193→33.707`秒、DiT `10.979→10.359`秒（−5.65%）。20组配对图像（含蓝/红/蓝回切、512²/1024²、KV off）及16步轨迹与D1 100%逐像素/逐位一致（`rgb_mae=0.0`），彻底消除首轮撤回候选的画质漂移。286项自动回归通过、1项CUDA跳过（含36项fast attention与RMW SRAM驻留专项）。

独立硬件进程单次负条件forward `663.565→625.605ms`（−5.72%），HBM写入流量`0.319147→0.187467TB`（`−131.68GB` / **−41.26%**），总流量`0.873142→0.731437TB`（`−141.71GB` / −16.23%），`sdpa-MME + softmax`并发重叠`74.642→154.035ms`（**+106.37%**）。独立进程`InUse` `26.433GiB`、`MaxInUse` `40.389GiB`（D1为`40.278GiB`，仅`+0.111GiB`，消除首轮`+11.31GiB`显存暴涨）。普通8189已部署`22e3972a+staged.3161d0925c58`，cold `140.809s`、warm `33.724s`、蓝/红/回切`33.193/27.269/32.545s`、原图`33.301s`、重载`118.399s`及热复验`34.051/33.450s`全部逐元素匹配，卸载稳定`9.189GiB`，别名及重编译上限告警均为0，队列为空。[执行记录](optimization/QWEN_D2_SRAM_2026-11-17.md) · [交付证据](evidence/qwen-d2-2026-11-17.json)

## 2026-11-17 D2撤回

用户判断本轮副作用与收益不相上下，明确要求“丢弃D2阶段在ComfyUI-Gaudi的代码改动，但保留本repo的工作记录”。撤回两个产品实现文件及新增prefix回归文件，D2补丁归档于`patches/qwen-d2-withdrawn-2026-11-17.patch`，应用检查通过。全部实验脚本、测量证据、原图、失败路径及原部署快照保留，原后续研究设想作为历史资料。

服务器1070份Python/shell源码与D1核对一致，版本标记恢复正式commit。普通cold142.828秒、完整warm34.322秒，caption及RGBA逐元素匹配已验收D1，关键计算节点实际执行。别名及重编译上限告警0次，队列为空。状态为`withdrawn_restored_d1`。

## 2026-11-17 D2首批部署历史

完成当前Qwen图分配审计、编译器/bundle筛选、query/head有限搜索，以及本机SDK1.24.2的直接SCAL与公共Synapse RMW原型。已证明scores/probabilities片上驻留，原型的softmax、准备与同步开销仍抵消收益。选中的产品候选为8160-token prefix整block编译和2048-query切分。

同轮五次warm，固定caption DiT11.003→8.945秒，完整自然图34.343→32.148秒。负条件HBM流量0.873141→0.642825TB，forward0.663503→0.537006秒。流量下降26.38%，候选带宽峰值占比48.86%，80%预算需≤0.327972秒。40%流量下降、DiT6–8秒和80%目标仍未达到。当时的后续研究设想为处理原生计划/同步成本及TPC softmax指令调度。

相关回归去重300项通过、1项CUDA跳过，包含17项新增prefix用例。共享KV存储、prefix offset、长度边界、输入与权重变化、提示词蓝→红→蓝、跨shape及重载通过。独立硬件进程allocator累计峰值40.36→51.67GiB，请求结束InUse26.434→26.516GiB。原图RGB MAE2.039、蓝衣变体7.334，轨迹RRMSE最高7.283%。用户随后决定撤回候选。

普通服务七次完整生成及卸载重载均匹配验证候选caption/RGBA，warm32.028秒、原图恢复32.243秒、重载122.580秒。卸载稳定9.273GiB，普通日志别名和重编译上限警告均为0。三个部署文件与1071份Python/shell源码哈希核对通过，队列为空。成对诊断进程在正式计时完成后的基线KV-off测试出现一次重编译上限，已记录测量边界。[执行记录](optimization/QWEN_D2_SRAM_2026-11-17.md) · [交付证据](evidence/qwen-d2-2026-11-17.json)

## 2026-11-17 D1 attention验收提交

选中Qwen原生FP8 DiT的K中心化、128对齐KV拆分及厂商fast softmax，首层使用原路径。76组真实attention与6组prefix捕获、全部32层补查、283项自动回归及41次普通节点验证完成。连续10种KV长度、128对齐边界和长度回切保持≤4份编译图。普通服务七次完整生成及重载均匹配验证版caption/RGBA，提示词蓝→红→蓝回切通过。连接恢复后又完成两次完整warm，结果一致。

五轮交错完整重算38.508→34.138秒，减少11.35%，固定caption DiT15.243→10.995秒。负条件流量1.386→0.873TB，时间0.919→0.663秒，有效带宽1.318TB/s、峰值53.79%。D1时间预算已达成，按新流量达到80%需≤0.445秒。fast softmax带来图像细节变化，实际轨迹输出RRMSE最高7.73%，用户已确认人工验收通过。

产品暂存tree为`9d01642e8deff0c4cb461a34b808115968ba8729`。普通cold139.686秒、warm34.175秒，蓝/红/回切34.985/28.008/33.362秒，恢复原图34.568秒、重载118.747秒。三份变更文件哈希匹配，1070份Python/shell源码核对通过，队列为空。产品已提交，调查记录随本轮归档。普通日志的别名及重编译上限告警均为0。[D1记录](optimization/QWEN_D1_ATTENTION_2026-11-17.md) · [交付证据](evidence/qwen-d1-2026-11-17.json)


用户明确确认“验证通过，提交代码”。产品提交`22e3972af91d44c7af7a6fa3fb371e4b3dd2e3f4`的tree为`9d01642e8deff0c4cb461a34b808115968ba8729`，与已验证部署快照一致。部署保留`a5308be1+staged.319d9026b5e9`历史身份并关联正式提交，状态为`accepted_committed`。

## 2026-11-17 回归补齐与验收提交

用户确认验收并授权补齐回归后提交。新增14项自动回归覆盖同长度内容变化、A→B→A回切、编译缓存清除后的重算、独立KV/GDN状态，以及prompt/system_prompt/seed/thinking等节点输入变化。专项14项、完整相关集207项通过，1项CUDA专属测试跳过；节点缓存12项CPU独立复验通过，计数存在重叠。

产品提交`a5308be1c7dd56d8ebb694a8a9a630a072089a77`的tree为`f68d0361edcc01559eacb4f68345c30e9349ccfe`，与最终已验证暂存tree一致。服务普通完整图142.826秒恢复预热成功，caption/RGBA匹配已验版本，七个代码/测试文件哈希与提交一致，队列为空。`.run/deployed-version`已关联正式commit，manifest保留原暂存身份和人工验收记录，状态为`accepted_committed`。[提交验收证据](evidence/qwen-cache-acceptance-2026-11-17.json)

BF16合成回归仍出现厂商`batch_as_strided`别名告警，状态隔离及重编译输出逐元素比较通过。已验FP8普通流程该告警为0次。模型首次加载约55秒图编译成本及80%硬件利用率目标继续保留为已知边界。

## 2026-11-17 扩写启动延迟修复部署

将KV容量维标记为动态，并规范RoPE切片的singleton步长。新容量首个decode从52.4–53.6秒降至0.56–0.63秒，三组编辑请求整图78.872/76.789/89.024→26.470/24.531/35.288秒。每侧五次交错warm中位38.503→38.629秒，变化+0.33%。193项回归通过、1项跳过，固定1909-token logits逐元素一致，22次普通节点验证通过，全部对照token及RGBA一致。模型重载仍需一次约55秒初始图编译，后续编辑复用成功。

暂存版本`07fedda3+staged.c8ad74f79c7e`已部署8189。普通服务六次完整计算成功，冷图142.963秒，四个编辑变体27.236/24.754/26.232/36.046秒，warm38.330秒。六张图及caption匹配验证版本，别名警告0次，六个文件哈希匹配、队列为空。两个仓库保持暂存，等待人工验收。[修复与边界](optimization/QWEN_DECODE_STARTUP_2026-11-17.md) · [证据](evidence/qwen-decode-startup-fix-2026-11-17.json)

随后补验七次普通请求，深蓝/深红提示词同为2803 tokens、固定seed，均实际执行TextGenerate并生成对应颜色。蓝→红→蓝回切、分别强制重算的caption/RGBA逐元素匹配各自首轮，上游system_prompt修改正确失效，恢复原提示词匹配部署基准。KV/GDN状态与RoPE表按生成调用持有，节点输出签名包含完整输入及上游值。别名警告0次、哈希匹配、队列为空。[缓存隔离补验](evidence/qwen-prompt-cache-2026-11-17.json)

## 2026-11-17 扩写启动延迟反馈

已定位用户`1/4096 ... 48:05`停顿现场。普通提示词编辑和thinking切换使输入2801–2805 tokens，对应KV容量6897/6899/6900/6901。新容量首次decode等待54–56秒，重复容量恢复约100 token/s。整步静态编译按精确容量专门化，形成频繁冷编译的性能缺陷；伴随的`hpu::batch_as_strided`警告来自厂商编译器插入算子的别名约束问题。调查状态为根因已定位、修复待验证，当前暂存部署仍待人工验收。[调查及边界](optimization/QWEN_DECODE_STARTUP_2026-11-17.md) · [生产证据](evidence/qwen-decode-startup-2026-11-17.json)

## 2026-11-16 P1/P2/P3继续执行

按用户要求保留暂存并继续推进。完整token图及1/2/4/8展开、设备RNG、lazy Graph、共享量化/组合权重、GDN TPC与conv、M1–32布局/K填充、有效长度/GQA/RoPE均已实测。选中FP8生成调用级紧凑频率表、立即状态控制清理和独立Qwen原生RoPE入口。191项回归通过、1项跳过，19次普通节点请求及卸载重载通过。

固定1909-token交错A/B各五次warm，PE26.843→19.070秒，整图44.370→36.593秒，分别减少28.96%/17.53%。两组自然整图减少13.21%/10.27%，扩写与图片改变。有效logits RRMSE5.1793%，人工质量待验收。64-token原生计数扰动−0.70%，HBM0.893TB/s、峰值36.44%，80%目标仍未达到。产品`07fedda3+staged.87ea451fce53`已部署，待人工验收。普通首图141.564秒，完整warm38.635/38.316秒，重载121.408秒，扩写及RGBA均匹配已验自然候选。六个文件哈希匹配、队列为空。[报告](optimization/QWEN80_PE_EXECUTION_2026-11-16.md) · [证据](evidence/qwen80-pe-execution-2026-11-16.json)

## 2026-11-16 硬件80%计划首批执行

首批低扰动早段64-token trace扰动3.96%，后段15.79%。四token回读产品候选完成同进程交错A/B各五次warm，PE27.554→26.254秒，整图45.139→43.819秒，分别减少4.72%和2.92%，十次扩写及RGBA均与原图一致。132项回归通过、1项CUDA跳过，另一组seed的512²成功。RoPE原公式内联和原生候选分别变慢38.8%/35.3%，logits有约4%差异，作为失败路径归档。产品暂存版本`07fedda3+staged.d39241e049d0`已部署、待人工验收。普通冷启动145.016秒，两次完整warm43.856/43.957秒，卸载后重载125.588秒，四次扩写与RGBA均匹配原图，文件哈希匹配、队列为空。80%阶段硬件效率目标仍待后续实现。[首批报告](optimization/QWEN80_FIRST_DELIVERY_2026-11-16.md) · [证据](evidence/qwen80-first-delivery-2026-11-16.json)

## 2026-11-16 硬件80%优化计划

按用户要求完成[深度优化计划](optimization/QWEN21_HARDWARE_80_PLAN.md)，以当前单卡batch1工作流为主场景，目标为主导资源及阶段硬件效率≥80%。固定当前流量时，PE带宽目标1.96TB/s、约4.45ms/token、完整扩写约8.93秒，DiT负条件forward目标≤0.707秒。先建立低扰动FP8执行账本，再推进图外输出头/采样/回读闭合、共享FP8量化和GDN状态融合、有效长度attention及MME/TPC片上流水。第一阶段整图预算22–24秒，深度数据流优化探索预算17–20秒，最终80%达标须逐阶段实测证明。计划及索引已暂存供审阅。

## 2026-11-16 最后一次FP8生成的性能调查

冻结用户13:54:50的`Qwen_image_2.1_00062.png`请求，原执行45.465秒。实际为1088×1920、八步、CFG1.2、thinking开启、FP32来源FP8 PE和DiT KV auto。完整1 cold＋3 warm，cold146.058秒，warm中位44.773秒，PE27.267秒占60.9%、DiT15.144秒占33.8%。2803输入、1909输出tokens，thinking段812 tokens约11.9秒。四张复现图扩写与RGBA均匹配原图。

PE后段16-token计数为8.730GB/token，按plain时间归一的HBM带宽估算0.630TB/s、峰值25.72%，采集扰动71.70%。DiT负条件为1.386TB/forward、1.520TB/s、峰值62.04%，采集扰动0.18%。DiT kernel活动覆盖96.81%，attention BatchGemm及softmax独占合计约59.3%设备跨度。完整warm频率1650MHz、47–55°C、allocator峰值49.955GiB。

文本详细trace触发重编译，已排除定量结论。native DiT进程在成功采集和出图后的退出阶段发生段错误，保留待定位。普通8189已恢复`07fedda`，原图147.813秒冷启动成功，扩写和RGBA一致，队列为空，1067份产品源码哈希匹配。调查文档、工作流及脚本已暂存供审阅。[报告](QWEN21_LATEST_BOTTLENECKS_2026-11-16.md) · [证据](evidence/qwen21-latest-bottlenecks-2026-11-16.json)

## 2026-11-16 提交归档

用户明确要求“提交代码”。产品仓库已提交为`07fedda3278b5c4be8c036fe245dc6991b91dd4a`，覆盖Qwen原生FP8接入、PE整步/DiT FFN编译及两类转换器和回归测试。提交树为`d4264962aec7a96f7318ceca52263edf9b6ee5b7`，与已部署的最终暂存树一致，服务器源文件SHA256核对通过。调查、工作流、失败候选和测量证据在本仓库归档。下方阶段记录保留各次实验的快照、数据和当时验收状态，数值与图像差异继续按原报告记载。

## 2026-11-16 PE从INT8经FP32直接转FP8

按用户要求新增`qwen3.5_9b_qwen_image_2.1_pe_t2i.fp8_from_int8_fp32.safetensors`，10,890,431,404 bytes。201个Linear直接由FP32反量化结果量化FP8，559个保护张量与上一版相同。全量权重对FP32参考的RRMSE从2.65482%降至2.64975%，201层均改善，相对降幅0.1911%。7项转换测试通过。自然扩写420→806 tokens，新模型五次完整warm中位30.248秒，上一版23.943秒；固定432-token为24.478秒，上一版24.129秒，固定图RGBA逐元素一致。512²和卸载重载通过。

新增独立工作流`Qwen-Image-2.1-Turbo-FP8-PE-From-FP32-HPU.json`。普通8189冷图131.448秒成功，RGBA匹配独立新模型流程，源文件哈希匹配、队列为空。暂存部署`b89280c0+staged.d5bda4e78d95`，模型文件和两个仓库相关改动已交付待人工验收。自然图片改变，权重重建精度与使用体验分别记录。[转换与验证](QWEN21_PE_FP32_CONVERSION_2026-11-16.md) · [证据](evidence/qwen21-pe-fp32-2026-11-16.json)

## 2026-11-16 FP8达到BF16无缓存warm水平

按用户要求继续逐项优化并保留暂存。接入原生FP8生成调用级整步编译及DiT FFN区域编译，产品完整5次warm中位23.943秒，原BF16基线25.425秒，同轮BF16确认25.526秒。固定同一432-token轨迹，产品3次warm为24.129秒，BF16确认25.526秒。自然PE稳定420 tokens，PE6.368秒、DiT15.185秒。cold128.631秒，对照BF16 cold104.261秒，首次编译成本仍较高。203项回归通过，1项CUDA专属测试跳过，512²和卸载重载通过。普通8189冷图129.466秒、两次完整重算24.277/24.105秒成功，候选/产品/普通服务扩写及RGBA逐元素一致，源文件哈希匹配、队列为空。

PE窗口HBM 0.214→0.652TB/s，FP8有效算力0.0367%→0.1368%峰值。DiT负条件读写量1.892→1.431TB，时间1.081→0.950秒。FFN三组真实输入的FP64抽样误差更低。PE对首轮FP8的固定logits RRMSE5.465%、top-1为419/432，自然扩写与图片改变，人工质量验收待确认。产品暂存树`b294c6a6d796176956e3f4b0f49d0a9f107ac75a`，状态为已部署、待人工验收。[完整报告](QWEN21_FP8_OPTIMIZATION_2026-11-16.md) · [证据](evidence/qwen21-fp8-optimization-2026-11-16.json)

## 2026-11-15 Qwen Image 2.1 FP8转换

已本地转换192个DiT Linear及201个匹配PE Linear，复用现有Qwen3-VL 8B FP8并接入其HPU原生矩阵入口。53项相关回归通过，2MP完整流程、512²和卸载重载通过。最新请求thinking关闭，2805输入tokens。无节点结果缓存且DiT KV关闭，各5次warm，BF16为25.425秒，FP8自然为71.507秒，扩写432→1061 tokens。固定432-token轨迹后FP8仍为40.149秒，PE20.645对6.529秒，DiT17.205对16.655秒。PE采样HBM约0.214TB/s、8.72%峰值，FP8有效算力0.0367%峰值，逐层提交和量化开销突出。DiT约1.75TB/s、71.42%带宽，FP8 Linear有效算力12.85%峰值，attention/softmax及中间量读写是主要后续方向。容量降低约20GiB，当前FP8完整流程速度退化。产品基线`b89280c0`，本轮改动已暂存并部署，待人工验收。普通8189完整图122.256秒成功，RGBA与独立自然FP8图逐元素一致，源文件哈希匹配暂存快照，队列为空。[完整报告](QWEN21_FP8_2026-11-15.md) · [证据](evidence/qwen21-fp8-2026-11-15.json)

## 2026-11-15 Qwen执行优化上线

用户要求先部署体验，已将候选接入普通产品入口，主实现`28794c4b`、最终版本`92d15ff7`。适用于常驻普通BF16、HPU、batch1、无MTP头的Qwen，生成调用持有独立cache代理，入口兼顾hooks、在线权重变换、低显存、attention配置及中断。155项回归及末次19项专项通过，有17项重叠。固定427-token产品logits与实验候选逐元素一致，三次warm中位6.4555秒。普通8189接口两组自然图103.705/49.905秒、卸载后重载83.915秒、最终提交重启预热107.747秒，扩写及RGBA均匹配对应候选图。MOSS BF16固定tree3s1三组扩写匹配此前基线。最终源文件哈希与提交一致，队列为空，原工作流已预热。新KV容量仍有首次编译成本。[上线记录](optimization/QWEN21_EXECUTION_DEPLOYMENT.md) · [证据](evidence/qwen21-execution-deployment-2026-11-15.json)

## 2026-11-15 Qwen Image 2.1 最新2MP工作流

用户要求逐项推进后，在独立调查进程验证9条GDN/attention/整步执行路径。最终候选为单token GDN立即状态更新、BF16 Q/K/V＋FP32 softmax SDPA和整步编译。固定427-token三次warm为10.228→6.533秒。原2MP同1784-token工作量，每侧1 cold＋2 warm，PE41.693→25.183秒，42.79→70.84 tokens/s，整图60.662→44.178秒，耗时降低27.17%。cold96.804→111.894秒。新HBM计数为16.562 GB/token、1.169TB/s、峰值47.72%，recipe306→84/token。完整logits约0.988% RRMSE，4组自然扩写和图片发生变化，检查页已提供，人工质量接受待用户确认。512²及卸载重载通过，重载前后候选扩写和RGBA一致。调查启动器的模块解析与导入等待、独立probe退出析构告警已修复并复测。普通服务最终恢复已接受的`dfca7cd6`，5份产品源文件哈希一致、队列为空。[逐项验证](optimization/QWEN21_EXECUTION_VALIDATION.md) · [证据](evidence/qwen21-execution-2026-11-15.json)

用户人工确认8组版本对照质量正常，接受`dfca7cd6`并要求保持提交。新增根因分析以真实第33步状态重放核心，未插桩20.902ms、主线程CPU17.276ms。已编译MLP及输出头权重有效带宽约2.005/2.200TB/s。修正后的同轮消融基线20.853ms，跳过GDN核心15.796ms，组合跳过GDN核心与完整attention9.566ms。原生晚位置16-token trace为306 recipes/token，执行事件未覆盖的8.370ms/token中约7.751ms位于recipe区间内，具体等待类型仍待归因。消融改变计算工作和执行重叠，不能视为上线收益。后续优先处理GDN/attention执行链，并将有数值差异的候选推进完整质量验证。两次隔离诊断后普通服务恢复、队列为空、源文件哈希相同。[根因报告](optimization/QWEN21_DECODE_ROOT_CAUSE.md) · [证据](evidence/qwen21-decode-root-cause-2026-11-15.json)

按用户确认，产品`dfca7cd6`与上一提交`ed9552e5`各独立生成8张原提示词2MP图，两类seed逐组配对。16份history确认完整重算扩写和采样，8组扩写文字及RGBA像素均逐元素一致。原图、扩写、8张原尺寸并排图、概览和ZIP已保存，供用户人工检查。当前服务恢复`dfca7cd6`，源文件哈希一致、队列为空。[图组记录](optimization/QWEN21_PRODUCT_AB.md) · [证据](evidence/qwen21-product-ab-2026-11-15.json)

用户要求先推进无MTP的BF16路径。本轮以AXI返回beat完成HBM有效载荷校准，128/512/2048 MiB copy及GEMV读取回算误差约−0.43%至−0.26%。真实解码约17.168 GB读取和1.261 GB写入/token，16轮无profiler为0.36819s，约0.801 TB/s，即峰值32.69%，取整敏感性上沿33.38%。整步eager编译与持久HPUGraph将decoder核心20.871→约17.8ms，存在0.79%–0.93% logits差异。GDN单tokenTPC使固定427-token耗时10.220→9.567s，但完整logits仍有0.986%误差，两个舍入修订也未对齐。普通服务最终运行`dfca7cd6`，原2MP图100.598s成功且像素一致、队列为空。DMA校准首版的地址使用错误曾触发设备自动复位，已恢复并以独立Synapse控制完成补测。[完整校准与验证](optimization/QWEN21_BF16_SINGLE_TOKEN.md) · [证据](evidence/qwen21-bf16-single-token-2026-11-15.json)

后续隔离试验复用已有Qwen3.5的243M参数MTP头。固定512-token窗口三次warm，单步42.03→四行验证58.57 tokens/s，吞吐提高39.35%。完整自然扩写两次warm为1784 tokens/41.507秒→1911 tokens/31.279秒，约61.10 tokens/s。相同前缀首个根token的单行/四行验证logits仍有2.022% RRMSE，完整图片和正常loader接入尚待验收。候选只在调查进程运行，线上使用`dfca7cd6`。实验后原2MP图94.763秒成功且像素一致，队列为空。[候选与边界](optimization/QWEN21_TEXT_NEXT.md) · [证据](evidence/qwen21-text-draft-2026-11-15.json)

已冻结最后一次用户请求，实际1088×1920、8步官方Turbo调度、CFG1.2、thinking开启、匹配PE的BF16文件，2803输入/1784输出tokens。原请求66.274秒。三次完整warm初始基线中位62.250秒，回退原产品三次为61.563秒，最终三次为60.689秒。按回退基线，完整耗时减少1.42%，PE42.575→41.713秒，约41.9→42.8 tokens/s。DiT约16.59秒，文本缓存图约18.47秒。

接入单token普通BF16的conv/delta联合编译及MLP残差融合，GDN两端投影维持原执行边界。16-token cProfile中recipe提交2720→2336、eager add1024→512、Python调用389151→313781。固定427-token全logits逐元素一致，五组不同seed/尺寸/thinking/官方调度的token和RGBA图片逐元素一致。初始组合另有五次完整warm中位60.713秒。157项回归通过、1项跳过，最终含新增单token hooks的39项专项通过，两组测试有重叠。

普通8189冷启动95.048秒，新提示词512²为10.614秒，卸载后原图重载91.124秒。卸载40–60秒稳定约8.984GiB，该2MP尺寸的原版卸载对照未采集。MOSS BF16固定MTP三组新鲜原版/新版扩写文字一致，历史样本漂移在原版中也存在，尚未归因。最终恢复Qwen原2MP图95.721秒成功，cold/reload/final像素与基线一致，队列为空，四份部署文件SHA256与产品提交对应文件一致。[证据](evidence/qwen21-performance-2026-11-15.json)

扩大整GDN编译区仍有约0.98% logits误差，审计确认输入和输出投影均有舍入变化。2MP DiT分组最快局部约5%，尚有数值变化，本轮继续采用原DiT数学路径。单行commit无稳定净收益，RoPE预计算变慢，关闭eager流水线使固定文本10.259→32.620秒。后续HBM校准结果及新的单token候选见本节开头。[完整调查](QWEN21_PERFORMANCE_2026-11-15.md)

## 2026-11-14 原生INT8 MME验证与开发终止

用户明确要求有原生INT8 MME才继续，能力核验阴性后已停止并放弃通用INT8开发。服务器安装库`Gaudi2HalReader::getSupportedMmeTypes()`返回`0x7906`，INT8位为零。直接Synapse的默认、inference、inference＋quantization模式共9组INT8 GEMM均在节点创建阶段失败，3组BF16对照编译成功。

本轮验证范围为已安装HAL与直接SDK建图/编译，INT8在执行前被拒绝。验证前后普通服务PID相同、队列为空。[验证结果](optimization/INT8_NATIVE_MME.md) · [证据](evidence/int8-native-mme-2026-11-14.json) · [历史计划](optimization/INT8_DEVELOPMENT_PLAN.md)

## 2026-11-14 INT8扩写慢调查

已核对普通8189最后一次120.523秒请求，Qwen Image 2.1 PE选择原INT8 ConvRot文件，文本循环约102秒、进度737、平均7.23 tokens/s。此前同提示词/文本seed的同源BF16扩写进度456、平均39.12 tokens/s，中段约42 tokens/s。thinking关闭，两个文件均无MTP权重。

部署源码确认INT8使用`full_precision_mm=True`，每次量化Linear调用先反量化/ConvRot还原为BF16，同时跳过短步MLP及完整attention区域编译。生成长度增长进一步增加耗时。已存在的同源BF16文件提供预先解码及编译路径。本轮完成只读历史、源码和模型header核查，各算子独立耗时尚未profile。[调查](optimization/TEXT_PRECISION.md) · [证据](evidence/qwen21-int8-slow-2026-11-14.json)

## Qwen Image 2.1 全BF16优化

当前组件为Turbo BF16、Qwen3-VL 8B BF16、匹配文生图PE的BF16解码文件和Qwen Image 2.1 VAE BF16。`17b4b231`接入原生RoPE与无mask attention编译，`ed9552e5`编译temperature=1.0的HPU采样概率。投影/RMSNorm保留原舍入边界，有mask前缀采用原数学执行。

用户832×1504、8步Euler/simple、CFG1、PE开启且thinking关闭，五次完整warm中位21.028→16.079s，PE11.416→10.593s，DiT8.089→4.006s。复用文本结果后的整图9.2185→5.130s，进程峰值76.645→60.335GiB。回退原数学路径复验为20.790s，收益可复现。

12组真实block逐元素一致，六组完整扩写token与RGBA PNG逐元素一致，含不同seed、512²/1024²、thinking开启与官方Turbo调度。159项回归通过、1项跳过。普通8189首次49.833s、换seed18.267s、重载42.926s成功，清理稳定占用约3.15GiB。当前验证范围为batch1文生图，2K及编辑另验。[深入记录](QWEN21_BF16.md) · [证据](evidence/qwen21-bf16-2026-11-13.json)

## Qwen Image 2.1 Turbo 首版

`0c764d59`修复官方增强器 presence penalty 的 HPU 布尔 mask 更新，`9f2ff96e`接入普通 BF16 FFN 区域编译。使用官方基础文生图结构与 Turbo 8 步调度，复用已有 BF16 主模型、Qwen3-VL 8B、Qwen Image 2.1 VAE。新文生图增强器已下载并校验，默认 HPU 图使用它的本地 BF16 解码文件。

1024²直接图五次 warm 中位 7.726s，基线8.262s。开启 thinking 增强器的完整图为42.340/42.229s，2548输入/1316输出tokens。原 INT8 同提示词自然生成1124 tokens，完整热运行161.042s。512²及卸载重载通过，清理稳定后约0.849GiB。80项回归通过、1项跳过。九组真实 block 相对 eager RRMSE 为0.00180%–0.01142%，九组 FFN 抽样对CPU FP64的误差比0.99988–1.00228。相同seed图像细节存在变化，本轮覆盖512²和1024²。普通8189最终直接图17.389/7.712/7.733s、首次增强图69.834s成功，工作流已安装，队列为空。[首版记录](QWEN21_FIRST.md) · [证据](evidence/qwen21-first-2026-11-13.json)

## 当前接受的实现

| 方向 | 现行实现与主要提交 | 详细记录 |
|---|---|---|
| 设备与模型 | HPU后端、E4M3解码表、VAE非原地残差 | [适配](ADAPTATION.md) |
| BF16 DiT | 两段编译/静态recipe、条件预计算、native norm、验证后BLOCKWISE | [BF16](optimization/DIT_BF16.md) |
| FP8 DiT | `2227f7d3`原生Linear、`0995638a`FP32 Q/K norm及block扩区 | [FP8](optimization/DIT_FP8.md) |
| Attention | `700bdb6d`单head、`f67e7f3f`混合AV、`69fe4a54`移除8192上限、`ef16c195`长序列分批 | [Attention](optimization/ATTENTION.md) |
| 文本精度 | INT8解码BF16、`148b252a`原生FP8文本 | [量化](optimization/TEXT_PRECISION.md) |
| 文本执行 | KV行clone、MLP/norm/attention编译、`34e2649e`短步RoPE | [decode](optimization/TEXT_DECODE.md) |
| Qwen单token上线试用 | `28794c4b`直接GDN状态、SDPA与整步编译，`92d15ff7`可选norm兼容 | [上线](optimization/QWEN21_EXECUTION_DEPLOYMENT.md) |
| GDN | `60da4e33`区域编译、prefill分桶、`41f01e68`并行递推、`a8f42df2`八行扩区 | [GDN](optimization/TEXT_GDN.md) |
| 运行时 | `441d459d`初始化后恢复SIGPIPE忽略策略，独立eager环境 | [运行时](optimization/RUNTIME.md) |

## 最近一轮 MOSS 验收

各项为自己的同轮五次warm中位数。

| 场景 | 基线 | 最终候选 |
|---|---:|---:|
| 1088×1936固定caption整图 | 29.645s | 28.892s |
| 1024² Moody FP8＋FP8文本固定MTP完整流程 | 16.092s | 15.902s |
| 上述TextGenerate | 5.227s | 4.985s |
| 1024² SFT BF16＋BF16文本固定MTP完整流程 | 19.521s | 19.396s |
| 上述TextGenerate | 5.314s | 5.156s |

18组真实block、两种文本固定logits逐元素一致，六组固定tree3s1自然扩写token相同。BF16整block共享候选保留舍入后无稳定收益。341项广泛回归通过、1项CUDA跳过，动态padding整理后最终37项专项通过，源码树为`4d9cc16d8fc372dc580511bed073bb0ce606672e`。临时原文本回退另过335项，测试范围存在重叠。

最后一次普通8189验收原始1088×1936工作流三次成功，cold100.467秒，文本缓存29.113/28.887秒，四份产品文件哈希核对一致、队列为空。该状态记录来自本轮优化验收。[结构化证据](evidence/moss-three-stage-2026-11-13.json)

## 质量与调查边界

- 默认Auto额外文字在优化版和原始文本回退版均出现，已修正最初的单项归因。固定树保值对照通过，两侧样本仍有BF16茶壶把手不可见、FP8海报低对比度等表现。
- TPC GDN候选因完整logits变化和双把手样本撤回。32MiB SRAM设置、额外小M FP8编译及BF16扩区未达到各自接入条件，原因保留在主题候选表。
- QK scores仍落DRAM，概率tile已在SRAM。文本HBM访问已用AXI返回beat校准，DiT尚待按新口径重测。约0.015秒kernel归属和文本空档因果归因仍需补齐。
- 调查进程的若干段错误、profiler退出tcache异常、部分vendor异步输入问题仍缺底层根因。已验产品路径和各故障触发条件分别记录。
- 质量范围限于已有模型/样本/参数组合。完整批量、多卡、编辑/控制LoRA、SLG、其它VAE/custom nodes及超大尺寸需要独立验证。

后续投入依据见 [优化方向](optimization/STRATEGY.md)，测量边界见 [验证方法](optimization/MEASUREMENT.md)。
