# MOSS BF16 DiT 执行优化

日期：2026-11-09。起始产品版本 `60da4e33`。用户授权按路线逐步实施，当前不自行编写 kernel，优先区域编译与已有厂商/Kitchen 算子。

## 测量契约

- 独立服务器 checkout `/home/sonny/ComfyUI-Gaudi`，HPU 串行独占；原始 `/home/sonny/ComfyUI` 不修改。
- 主对照固定前一轮 compiled BF16 的 caption、扩散 seed `649029532689846`、1024²、25 步 `er_sde/simple`、CFG 2.5、SLG 关闭。扩写在本对照中关闭，文本编码、采样、VAE 每次真实执行，保留 loader 缓存。
- 真实输入采集确认：正条件 context `[1,312,4096]`，DiT block `[1,4408,3072]`；负条件 context `[1,23,4096]`，block `[1,4119,3072]`。CFG 分两次执行。
- 基线与候选各一次 cold、五次 warm；cold 包含首次加载/编译，不混入 warm。候选先做真实 block 输入重放、浮点误差及 trace 检查。
- attention 比较原 PyTorch SDPA、厂商 FusedSDPA 默认与 FP32 softmax、recompute/non-recompute；不以降精度换来的时间直接视为无损优化。
- `hl-smi` 100 ms 轮询的 AIP 实际刷新窗口约 1 秒；`power.draw` 仅 54V rail。原始数据目录 `/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-09-dit`。

## 复现脚本

- `probes/run_moss_dit_trials.py`：固定 caption 完整工作流、计算节点缓存失效、耗时与 telemetry。
- `probes/moss_dit_server.py`：仅调查进程中应用候选、捕获真实 DiT/block 输入。
- `probes/moss_dit_replay.py`：同一权重和输入重放现有 attention/compile 路径，记录误差与 trace。

## 第一轮结果与取舍

| 路径 | 固定 caption 整图 warm median | DiT warm median | DiT AIP 平均 | 图像 MAE /255 |
|---|---:|---:|---:|---:|
| 原始 | 40.093 s | 37.916 s | 47.71% | 自身重复约 0.12 |
| attention 区域静态编译 | 27.454 s | 25.299 s | 56.76% | 0.122 |
| block 静态编译 | 21.310 s | 19.211 s | 57.16% | 7.973 |

三组各 1 cold + 5 warm。所有成功请求 fallback 为空。attention 区域编译的 allocator 峰值由 44.35 降至 39.20 GiB。图像 MAE 以原始 warm-1 为参照，重复生成本身非逐像素确定。

真实 block 重放（首层、N=4408）：原始 20.31 ms；attention 区域编译 12.01 ms、逐值一致；block 编译 8.99 ms、相对 RMS 误差 0.326%。原始 SDPA 单独为 11.81 ms，编译后 3.39 ms、逐值一致；FusedSDPA 默认为约 3.60 ms，但 attention 相对 RMS 误差约 0.606%。FusedSDPA FP32 softmax 约 11.4 ms，没有相同收益。

trace 确认原始走 `aten::_scaled_dot_product_attention_math`。原始/attention 编译/block 编译的 runtime `launch` 事件分别为 72/67/49；kernel 类事件横跨多个硬件 lane，不能当作启动次数。单次 profile 的设备 kernel 时间区间并集分别约 19.73/11.33/8.44 ms，span 21.54/13.04/9.47 ms；这是受 profiler 扰动的单 block 证据，不是全流程 MFU。

## 已定位限制

- 整块 `fullgraph=True` 最初被 `_fp16_linear_wanted` 阻断：它先读取 CUDA `allow_fp16_accumulation` 再检查设备。调查进程将设备/类型条件移到前面后可编译；尚未因此单独修改产品。
- Kitchen RMS/AdaLN/RoPE 是 custom op，HPU 选 eager tensor 实现。直接向 Habana decomposition 表加入其实现会出现 AOT functional graph 对 `add_.Scalar` 的断言；该路径拒绝。
- 内联现有 Kitchen eager 实现可降到约 7.9 ms/block，但存在同级舍入差异，尚未做完整工作流验收。没有自写 kernel。
- 把 modulation、SiTU 单独移出整块编译没有消除误差；当前不能把差异归因于某一项，可能涉及跨算子融合/累加，须进一步定位。
- 全动态 attention 编译触发 Habana `get_input_symbolic: Inputs and shapes don't match`；固定 head 数与 head_dim 后，单进程十种长度重放成功且这组输入逐值一致。
- **共享动态 SDPA 产品候选仍未通过完整图：**19 项新测试加既有回归合计 78 passed/1 skipped，但第一个真实 1024² 请求第 6 步出现 SIGSEGV。保持为失败候选，不能据单测宣称可部署。原始日志保存在 `product/server.log`，下一实验仅检查连续布局是否影响失败。
- 额外查到 HPU eager FP32 SDPA 对整行被遮挡的 mask 本身会产生 NaN，compiled FP32 也同样如此；不是本次新引入。当前候选仅覆盖 BF16 推理，FP32 保持原路径；BF16 全遮挡 bool/加性 mask 已验证返回零。

共享 SDPA 后续加连续布局后完成 1 cold + 5 warm，但 DiT 为 28.995 s、图片 MAE 5.656/255；它同时改变文本编码等调用，复用范围过大。本轮完整撤销共享 `comfy/ops.py`、helper 及其测试，保存失败候选 patch 到 `patches/rejected-shared-dynamic-sdpa.patch`。这些测试成绩不作为最终产品的回归成绩。

## 最终产品实现及复测

修改仅为 `comfy/ldm/moss_image/model.py` 的 attention 区域：BF16 HPU 推理通过 `hpu_backend` 编译，`dynamic=None` 先特化、再自动适配长度；保留 Kitchen 操作和既有 attention dispatcher。编译器管理 callable/recipe，无新增模型权重缓存。CPU、训练、用户 attention override 和 `--disable-comfy-compiler` 走原函数。

没有合入整块编译、共享 SDPA、内联 Kitchen 实现、FusedSDPA 替换或自写 kernel。最终代码专项 **7 passed**，覆盖 10 个不同序列长度、同 shape 更换输入/norm 参数、加性 mask、参考 KV 读写、CPU、训练、自定义 override 与关闭编译。

最终产品代码（非 monkeypatch）固定 caption、1 cold + 5 warm：

| 指标 | 原始 | 最终产品 |
|---|---:|---:|
| 整图 warm median（扩写关闭） | 40.092826 s | **30.257719 s** |
| DiT warm median | 37.915529 s | **28.216700 s** |
| DiT warm 范围 | 37.865882–37.973207 s | 28.208581–28.272458 s |
| cold 整图（包含加载/首次编译） | 51.171390 s | 42.644502 s |
| DiT AIP 平均 / 峰值 | 47.71% / 59% | 52.78% / 64% |
| HPU allocator 峰值 | 44.3546 GiB | 40.1147 GiB |
| DiT 54V rail 平均功耗 | 292.02 W | 326.15 W |

固定工作量下 DiT 延迟减少 **25.58%**、固定 caption 整图减少 **24.53%**。产品图像 MAE **0.12055/255**、PSNR **57.27 dB**；原始和产品各自 warm 重复的 MAE 都约 0.12/255。只能说图像差异与重复波动相当，不能说全流程逐位等价。

原始用户图重新开启 BF16 TextGenerate、1 cold + 5 warm：cold **63.3561 s**；warm `[55.5298,52.9965,52.5295,52.5911,50.3233]`，median **52.5911 s**。上一轮为 60.0625 s，描述性差异约 12.44%；本轮自然结束 tokens `[333,303,303,287,290]`，不作相同解码轨迹的严格 A/B。TextGenerate median **19.6479 s / 15.241 token/s**，DiT median **31.3022 s**，均无 fallback。

该完整用户图的 warm AIP：全请求 **35.85%**、文本 **15.20%**、DiT **50.33%**，峰值 64%。**没有达到用户提出的 80% 目标**；本轮证明了减少延迟和临时内存，不能视为 BF16 已到硬件上限。

最终证据：`final/product-final/{runs.json,events.jsonl,hpu.csv}`、`user/{workflow-summary.json,phase-telemetry.json}`。额外验收 **5/5 通过**：512 短提示词、768 加权提示词、1024 卸载重载、原 INT8 用户图两次；所有 latent/VAE tensor 均 finite，fallback 为空。INT8 两次均生成 363 tokens，诊断请求约 70.22/64.84 s（含同步和检查，不作为正式性能中位数）。复核 512 红茶壶、768 蓝花瓶海报、1024 猫及 INT8 MOSS 街景，无异常纹理；海报有模型额外生成的小字，不将该 smoke 当作严格文字质量基准。

验收脚本首次把 `/free` 的空 HTTP 200 响应当 JSON 解析失败，这是调查客户端错误。修复后在新进程完整重跑上述五例，等待空闲显存超过 80 GiB 再提交重载请求，并确认 loader 重新加载；没有把首次中断当作通过。

已提交并部署 **`f739cb93145c34d1d9c0661f23055269f986165c`**，本地 `codex/gaudi2-hpu` 未 push。服务器产品文件 SHA256 与本地一致，失败的共享 SDPA 文件已删除，`comfy/ops.py` 恢复原版本。8189 日常服务已恢复、队列空闲；不修改默认模型选择或原始 `/home/sonny/ComfyUI`。

## 后续方向

优先继续复用现有算子：定位整块编译舍入差异的具体来源，验证 Kitchen norm/RoPE 分区的有效编译边界，再推进文本小矩阵/状态与提交开销。保留静态 block 约 19.2 s 的实验线索，但不将一次外观正常替代系统质量验收。原生 FP8 仍在 BF16 稳定基线之后；按用户最新要求，本阶段不自行编写 kernel。

### 下一阶段执行顺序

当前完整用户图约为 DiT 31.30 s、扩写 19.65 s、其它约 1.6 s；按阶段中位数近似，DiT 占约 60%、扩写约 37%。阶段中位数不要求严格相加。

1. **定位整块编译差异，扩大可验收的 DiT 编译区域。** 固定真实输入，在早/中/晚 timestep 和不同深度 block 中比较 norm/modulation、QKV、RoPE、attention、输出投影、FFN/SiTU、残差的中间结果，找首个偏离点。对有分歧的小区域增加 FP32 参考，区分中间舍入、累加顺序与实际语义错误；图片差异本身不证明质量下降。敏感边界保留原精度和执行方式，其余部分继续复用 Kitchen/torch 操作编译。当前 28.22 s 与整块原型 19.21 s 之间约 9 s 是优先调查的空间，需完成数值、不同提示词及完整图稳定性验收才算收益。
2. **减少 DiT 动态形状的成本。** 现有 attention 静态原型为 25.30 s，自动适配长度的产品为 28.22 s，约 3 s 差距值得单独验证。评估有限形状特化或长度分桶，严格保留有效 token、mask 和位置编码语义；覆盖连续改变 caption 长度/分辨率、返回旧 shape、卸载重载和内存增长。此项与整块编译的收益重叠，不将两个差值相加作为预期。
3. **优化文本扩写的剩余执行开销。** 先对当前 GDN 已编译的版本重新采 trace，固定解码 token 和 MTP 树，分别测 prefill 与 decode；候选区域为 MLP/Linear/norm、KV attention 与 tree merge、采样/接受判定的 host/device 同步。它们是待归因候选，尚无当前占比证据。仅扩展被证实的热点编译区域；shape 与状态地址稳定、且 host 提交确为主要成本后再评估 HPU Graph。另用自然结束用户图验收真实收益。

上述工作均沿用 BF16 和现有算子。验收继续采用固定工作量五次 warm、cold 单列、质量/状态语义、fallback、内存及 AIP；80% AIP 保留目标。现有测量不支持给出下一阶段必达的整图秒数或利用率承诺。
