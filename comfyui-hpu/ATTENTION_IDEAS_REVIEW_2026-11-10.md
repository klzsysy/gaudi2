# 讨论组 attention 优化建议复核

日期：2026-11-10（Asia/Shanghai）。对照当前 ComfyUI MOSS/Qwen3.5 9B、约 1970 输入 tokens、6073 KV 容量、batch=1；不是对讨论组的长上下文 vLLM 实测作否定。仅有建议描述，没有其具体 patch、基线、测量和质量数据，因此不能验证其宣称的实际收益幅度。

## 适用性

| 建议 | 当前路径与判断 |
|---|---|
| 8 个 verify tokens 共用历史 KV | 方向正确，但当前已按 `[1,16,8,256]` 联合计算：GQA 将 Q 重排为 `[1,4,32,256]`，与同一份 K 做 batched matmul，没有按 8 个 token 循环调用 attention。代码上的复用不能直接证明物理 HBM 恰好只读一次；不能再计一笔 8× 收益。 |
| 全上下文一次 QK/AV、一次 softmax，使用 FP8 | 数学上可行，必须保留 MTP 树的祖先可见性，而非普通下三角 mask。已经补测厂商 BF16/FP8 FusedSDPA，见下表。该输入上 FP8 未表现出超过 BF16 的净优势，未进入产品。 |
| FP8 KV cache | KV payload 的字节数可减半，但不是总权重/总流量减半。按 8 个主干 attention + 1 个 MTP、每份 K/V `[1,4,6073,256]` 推算，当前 BF16 KV 约 **213.50 MiB**，FP8 节省约 **106.75 MiB**，不含 scale。相比约 15.87 GB 主干+lm_head 参数，短上下文下应先确认 KV 流量是否主导。GDN recurrent state 仍是另一类状态，不能一起按半精度节省计算。 |
| 连续 KV 镜像、每轮补拷末尾页 | 当前 `FixedKVCache` 本来就是连续 tensor，直接按位置写入，没有 vLLM 分页收集过程。再次建镜像没有对应的已知收益点。树形接受分支的行搬移是另一问题，本轮已修复 HPU 同 storage 视图复制失效。 |
| attention 组共享 mask | `VerifyTree.mask` 已共享给各层/头使用，拓扑改变时才更新；公共 position/seqlen/tracker 也在层间共享。参考实现的设备端 range/可见性计算仍可能重复，本轮区域编译已减少其提交开销。可继续测试按步共享完整 mask，但收益未测，不能把它等同于目前存在大量主机逐头建 mask。 |
| cached prefix 按 8K keys 切片、批量 FusedSDPA | 适合很长的已缓存历史；当前有效上下文约 2K、分配容量约 6K，不足一个 8K 切片，也没有跨请求 prefix cache。不能直接搬用其长上下文收益。若使用多个切片，必须做正确的跨片 softmax 归一化合并。 |
| 更细的 context buckets | 减少补齐计算的原则成立；当前没有把 131K 补齐到 262K 的场景。GDN prefill 按 64 对齐，1970 仅补到 1984；DiT 已按实际 shape 选择静态 recipe。对 KV 另做的 256-token 前缀分桶在交替文本测试中只差约 1.4%，第一组反而更慢，故暂不合入。 |
| vLLM 前缀缓存 + GDN 设备快照 | 对重复长前缀有潜力，但当前执行器是 ComfyUI，没有可直接启用的 vLLM prefix-cache 开关。还需支持 GDN 连续 prefill、conv history、位置与模型/LoRA 失效规则；不能只存 attention KV。属于缓存架构工作，暂排在已量化的执行热点之后。 |

## 厂商 SDPA 的真实输入对照

使用本轮采集的第一层 verify 输入，Q `[1,16,8,256]`、K/V `[1,4,6073,256]`、有效前缀 1970，保留 tree mask；各实现均区域编译，5 组×20 次、HPU 组尾同步。CPU FP64 reference 使用同一组已量化为 BF16 的输入。

| 实现 | median ms/call | 对 FP64 相对 RMS 误差 |
|---|---:|---:|
| 当前编译 GQA + tree merge | 0.1544 | 0.2235% |
| 厂商 BF16 FusedSDPA，默认 softmax | **0.0848** | 0.5970% |
| 厂商 BF16 FusedSDPA，FP32 softmax | 0.1085 | **0.1916%** |
| 厂商 FP8 FusedSDPA，FP8 KV 常驻 | 0.1064 | 7.5229% |

全部 finite、fallback 为空。FP8 使用 `cast_to_fp8_v2` 与 `Fp8FusedSDPA`，并非字节 reinterpret 或自写 kernel。FP8 只做初始逐 tensor maxabs scale，**没有完整模型校准**；KV 一次量化不计入每次执行，Q 的动态量化计入。尚未计增量 KV 更新/量化、实际模型质量或更长上下文。因此 7.52% 不能被解读为 FP8 的最终质量上限，0.1064 ms 也不能作为全模型交付性能。

在当前 shape 上，FP8 与 BF16/FP32-softmax 的约 2% 时间差不足以证明收益，且默认 BF16 FusedSDPA 更快。更合理的下一候选是先验证 **BF16 FusedSDPA + FP32 softmax** 的完整模型表现；当前每次最多只省约 0.046 ms，不能把单算子相对百分比外推到 30 秒整图。现在的完整 attention 编译还包含投影/norm/RoPE/cache write，这一独立算子测量不代替其中的集成测量。

此前 DiT 的 N=4408 对照也已测试厂商 FusedSDPA：现有 SDPA 编译后约 3.39 ms，厂商默认约 3.60 ms，FP32 softmax 约 11.4 ms。不同 shape 的相对排名不同，不能把文本 decode 的结果直接外推到 DiT；见 `MOSS_DIT_OPTIMIZATION_2026-11-09.md`。

## 长上下文分片与前缀复用的正确性条件

历史前缀对所有新 query 可见时，历史切片内可以不加因果 mask；但最后一片的 padding、当前新 tokens 的因果约束或 MTP tree mask 仍需处理。若第 i 片返回局部 attention 输出 O_i 和 log-sum-exp L_i，完整输出应为 `sum(exp(L_i - logsumexp(L)) * O_i)`，不能简单平均各片输出。当前 Qwen 路径的 prefix/tree merge 正是在处理分区归一化问题。

GDN 除 attention KV 外，还需要卷积历史和 recurrent state。现有 `GatedDeltaNet.forward` 的长段 prefill 调 `torch_chunk_gated_delta_rule(..., initial_state=None)`，卷积也按该段重新开始；因此恢复快照后直接调用此路径并不能自动连续计算。还要正确管理已接受的 prefix 位置、末尾 hidden/logits、权重/LoRA/位置编码等缓存键以及卸载失效。产品仓库要求模型不持久保存跨执行的大 tensor，若做跨请求复用，应先在执行/缓存所有者层设计生命周期。

## 依据

- 实测：`performance-2026-11-10-kv/fused-decode-results.json`、`fused-decode.log`；复现脚本 `probes/moss_fused_decode_probe.py`。
- 当前源码：`llama.py::FixedKVCache`、`qwen35.py::VerifyTree/GatedDeltaNet/_generate_mtp`；本机 Kitchen `flash_attention.py`。
- [Gaudi FusedSDPA API](https://docs.habana.ai/en/latest/PyTorch/Reference/Python_Packages.html#hpex-kernels-fusedsdpa)：自动识别 GQA，并提供 softmax 精度与执行模式选择。实际可用接口以本机 1.24.2 的 `FusedSDPA.py` / `Fp8FusedSDPA.py` 为准。
- [Gaudi FP8 推理文档](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html)：FP8 有现成厂商路径，仍需测量和校准；本次没有据此承诺模型级加速。
