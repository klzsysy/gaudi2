# Intel Gaudi 2（HL-225B/225D）原生支持的数据精度


官方架构文档的原话是：Gaudi 2 支持深度学习所需的全部主流数据类型：FP32、TF32、BF16、FP16、FP8（E4M3 和 E5M2），并且 MME（矩阵乘法引擎）中所有类型都累加到 FP32 累加器。来源

精度	是否原生	说明
FP32	✅	训练/推理基准精度，也用作累加精度
TF32	✅	MME 原生，替代 FP32 做矩阵乘加速
BF16	✅	主力训练精度（432 TFLOPS）
FP16	✅	支持，算子覆盖少于 BF16
FP8 E4M3	✅	Gaudi2 新增，原生格式是 E4M3 FNUZ（最大值约 240，非 OCP 的 448）
FP8 E5M2	✅	主要用于反向传播/梯度
INT8	✅	量化推理；PyTorch 支持矩阵中 Int8 为 Yes（算子有限）来源

## 几个容易踩坑的点

- FP8 是 Gaudi2 的关键增量：Gaudi1 没有 FP8，FP8 训练通过 Intel Gaudi Transformer Engine 使用，支持 E5M2 和 HYBRID（前向 E4M3、反向 E5M2）两种 recipe。[来源](https://docs.habana.ai/en/latest/PyTorch/PyTorch_FP8_Training/index.html)
- Gaudi2 的 E4M3 是 FNUZ 变体（max≈240），而 Gaudi3/其他 GPU 用 E4M3 FN（max≈448）。跨平台部署量化模型时需要做 scale 调整，否则精度会掉。[vLLM-Gaudi 实现](https://github.com/vllm-project/vllm-gaudi/blob/57ef5ce97948228b5367c53aa72789aa804f0a55/vllm_gaudi/ops/hpu_compressed_tensors.py#L1148)
- INT4/UINT4/NF4 不是原生算术精度：官方提供的 UINT4/NF4 推理是权重量化 + 反量化（软件路径），不是硬件 INT4 MAC。
- 不支持：FP64、Complex32/64（PyTorch 支持矩阵明确为 No），也没有 FP4。
- 算力参考：BF16 ≈ 432 TFLOPS，FP8 ≈ 865 TFLOPS（INT8 同量级），96 GB HBM2e / 2.45 TB/s。[规格对比](https://dstack.ai/blog/intel-gaudi/)

一句话总结：FP32 / TF32 / BF16 / FP16 / FP8(E4M3+E5M2) / INT8 都是原生支持，累加统一 FP32；FP8 是 Gaudi2 相对上代最大的精度升级。
