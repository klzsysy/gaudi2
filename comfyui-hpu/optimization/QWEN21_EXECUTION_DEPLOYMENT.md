# Qwen无MTP执行优化上线

2026-11-15，用户要求将已验证的优化候选部署到普通服务，供实际体验。当前产品为`92d15ff701f932e7d349387f8f46ca960baf83d8`，包含主实现`28794c4b`及可选末尾norm兼容修正。此前线上产品为`dfca7cd6`。

## 产品接入

将GDN单token直接状态更新、BF16 Q/K/V＋FP32 softmax SDPA以及整步编译接入普通生成入口。BaseGenerate使用生成调用持有的forward callable，Qwen为符合条件的请求创建独立cache代理，共享当前请求的状态张量。

快路径适用于HPU、普通BF16、batch1、无MTP头、权重常驻、编译开启的Qwen。入口检查模型和子模块hooks、在线权重变换、低显存/VBAR状态、指定attention实现及梯度/训练状态，覆盖原执行入口的选择。单token边界检查用户中断，末尾norm支持模型已有的可选配置。

产品改动为4份文本实现文件及1份专项测试。普通`main.py`和既有启动脚本直接运行该产品代码。调查脚本、服务器路径、测试结果和部署记录保存在本研究仓库。

## 回归与候选一致性

- 155项回归通过，覆盖GDN、MLP、norm、KV attention、RoPE、采样及生成入口。末次19项专项通过，其中17项与前述回归重叠，另覆盖中断和可选末尾norm。
- 产品固定427-token完整logits与已验证实验组合逐元素一致，RRMSE为0，top-1为427/427。三次无profiler warm中位6.4555秒，约66.15 tokens/s。
- 数值对照在增加中断检查和可选末尾norm分支前完成，两项都保留当前模型的计算。最终提交另通过专项测试和普通接口图片对照。
- 普通8189入口的MOSS BF16 `tree3s1`三组teapot/poster/person扩写文字均匹配此前普通服务基线。

## 普通接口验收

使用原1088×1920工作流、8步、CFG1.2、thinking开启，执行自然扩写与完整采样。两类seed逐组对应已验证候选的自然出图，history确认扩写和采样实际重算。

| 用例 | 普通接口秒 | 对应候选扩写 | 对应候选RGBA |
|---|---:|---|---|
| 原seed | 103.705 | 一致 | 一致 |
| seed＋1 | 49.905 | 一致 | 一致 |
| 卸载40秒后重载原seed | 83.915 | 一致 | 一致 |
| 最终提交重启后预热原seed | 107.747 | 一致 | 一致 |

首轮Qwen验收前，进程已经执行MTP兼容回归。上表用于上线功能和一致性验证。严格相同1784-token工作量的性能对照为PE41.69→25.18秒、完整图60.66→44.18秒，条件与候选数值变化见[逐项验证](QWEN21_EXECUTION_VALIDATION.md)。

最终服务为`http://192.168.10.21:8189`，源文件SHA256与产品提交一致，队列为空，当前原提示词工作流已预热。静态KV容量首次遇到新shape时仍有编译成本。

## 部署与恢复资料

- 服务器独立产品目录为`/home/sonny/ComfyUI-Gaudi`，原始`/home/sonny/ComfyUI`遵循只读约束。
- 原始测试和history位于`/home/sonny/gaudi2/comfyui-hpu/deploy-qwen-execution-2026-11-15`。`baseline/`保留此前`dfca7cd6`的4份源码及部署标记，`version.json`保存最终提交及5份文件哈希。
- `qwen_execution_release.py`负责空队列检查、源码备份、测试、固定文本验收和失败恢复。`qwen_execution_online.py`通过普通API校验图片，支持卸载重载和最终预热。
- MTP兼容验收保存在`performance-qwen21-2026-11-15/deployed-execution/`及相邻`deployed-execution-regression.json`。
- 结构化结果及完整request ID见[上线证据](../evidence/qwen21-execution-deployment-2026-11-15.json)。
