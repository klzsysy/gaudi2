# Qwen21 PE从INT8经FP32直接生成FP8

2026-11-16。用户要求从原始INT8 ConvRot文件，经FP32反量化后重新生成FP8 PE。本轮复用已有FP8执行优化，新增独立模型和工作流。

## 转换产物

- 源文件为`qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors`。
- 新文件为`qwen3.5_9b_qwen_image_2.1_pe_t2i.fp8_from_int8_fp32.safetensors`，大小10,890,431,404 bytes。
- 服务器目录为`/home/sonny/ComfyUI-Models/models/text_encoders`，SHA256为`461552bb50894a88b5072d4667870c2f4980793cfc373120b4a0e48b9dd59ccf`。

产品转换器`scripts/gaudi/convert_qwen35_fp8.py`新增`--from-int8-convrot`，沿用浮点源文件的默认接口及原子发布。读取inline量化metadata，使用已有Comfy Kitchen eager反量化和ConvRot逆变换，显式指定FP32，随后以FP32计算weight scale并量化为E4M3，安全范围224。

201个目标语言Linear及输出头直接由FP32进入FP8。其它已量化保护层解码为BF16，其余源张量按原dtype复制。输出的559个保护张量与上一版逐元素相同，全部张量名称、shape、dtype相同。metadata的`source_conversion`明确记录`int8_convrot_to_fp32`。

在独立产品checkout执行以下命令可复现，转换器对已存在的目标文件有覆盖保护。

```bash
.venv/bin/python -m scripts.gaudi.convert_qwen35_fp8 --from-int8-convrot \
  models/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors \
  models/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.fp8_from_int8_fp32.safetensors
```

## 量化误差

对全部201层逐层重新执行FP32反量化，核对新FP8编码与scale严格等于直接量化公式。以相同INT8文件的FP32反量化结果为参考，跨全部7,929,331,712个FP8参数累积平方误差。

| 指标 | BF16中间版本 | FP32直接版本 |
|---|---:|---:|
| 全部目标权重RRMSE | 2.6548197% | **2.6497462%** |
| 相对RRMSE下降 | — | **0.1911%** |
| 改善层数 | — | **201/201** |

FP8原始编码变化130,108,380个，占1.64085%。scale也有变化，因此编码相同的元素仍可能具有不同的反量化数值。上述结果确认FP32路线更接近所用权重参考，幅度较小。原始INT8量化误差继续存在，实际扩写和画面质量需要独立评价。

7项转换相关测试通过，覆盖浮点源兼容、直接FP32量化、敏感层保留、错误格式拒绝及原有原子发布/覆盖保护。

## 工作流验证与性能

沿用1088×1920、8步、CFG1.2、thinking关闭及相同提示词/两类seed，完整节点强制重算、DiT KV off。warm保留模型和编译recipe。

| 阶段 | 新模型自然，806 tokens | 新模型固定432-token轨迹 |
|---|---:|---:|
| PE | 12.126s | 6.869s |
| CLIP | 0.501s | 0.536s |
| DiT | 15.750s | 15.205s |
| VAE | 1.100s | 1.099s |
| PNG保存 | 0.752s | 0.762s |
| 完整warm中位 | **30.248s，5次** | **24.478s，3次** |
| cold | 166.792s | 126.200s |

上一版相同输入的自然扩写为420 tokens、完整23.943秒，新模型稳定生成806 tokens，完整等待增加。固定432-token轨迹的上一版为24.129秒，本轮24.478秒，工作量相同时耗时接近。自然耗时变化主要伴随生成长度和caption长度变化。本轮权重重建误差改善与自然流程延迟分别记录。

512²另一组seed成功，12.090秒。卸载后原2MP图重载成功，113.077秒。正式完整请求fallback计数均为空。已目视2MP自然图，主体、花卉和蜂鸟呈现正常，人物姿态、手部位置和构图与上一版变化明显，人工质量验收待确认。

## 交付

新增独立UI工作流`Qwen-Image-2.1-Turbo-FP8-PE-From-FP32-HPU.json`，本仓库文件为[UI](workflows/qwen21-fp8-pe-fp32.json)与[API](workflows/qwen21-fp8-pe-fp32-api.json)。该选择使新旧PE均可在已有服务中体验。

产品基线`b89280c0`，暂存树`d4264962aec7a96f7318ceca52263edf9b6ee5b7`，完整暂存patch SHA256为`d5bda4e78d95c3d0787e4c91335bfa0a8bd2ada07b1afa91306a291bc2c7c691`。普通8189已部署`b89280c0+staged.d5bda4e78d95`，新模型普通API冷启动131.448秒成功，prompt ID为`7dfc863f-eda4-4a8c-ba18-150febb3799e`。普通服务RGBA与独立自然流程逐元素一致，固定432-token新旧PE流程的RGBA也逐元素一致。源文件哈希匹配暂存快照，队列为空。用户于2026-11-16明确要求提交代码，产品已提交为`07fedda3278b5c4be8c036fe245dc6991b91dd4a`，提交tree与上述已验证暂存树一致。当前状态为**已部署，已按用户指示提交**。代码、测试、工作流和调查记录随本轮提交归档。

[结构化证据](evidence/qwen21-pe-fp32-2026-11-16.json)保存完整模型哈希、201层误差、各次耗时与部署身份。服务器原始资料为`/home/sonny/gaudi2/comfyui-hpu/performance-qwen21-pe-fp32-2026-11-16`，审计入口为`probes/qwen21_pe_fp32_audit.py`。
