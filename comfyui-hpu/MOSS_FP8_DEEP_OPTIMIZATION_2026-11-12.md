# MOSS FP8进一步优化

## 目标与基线

从产品`148b252a`继续，固定298-token扩写、1024²、25步、CFG2.5，FP8文本与Moody FP8 DiT。上一轮完整warm18.933s，DiT13.074s，allocator峰值27.267GiB。

以端到端耗时、原生FP8覆盖的运算量和有效吞吐共同验收。FP8矩阵时间占比随算子加速及其它耗时减少而变化，需同时报告绝对耗时。采样、步数、分辨率与固定工作量沿用基线。

## 顺序与验证

1. Attention的QK/AV原生FP8与softmax输出量化融合。先复用已有真实激活筛选head分组和query分块，再验证Moody多层、多timestep实际输入及完整生成。
2. 细分激活动态量化、norm/modulation、situ和RoPE成本。评估既有厂商算子及区域融合，单独计量量化和GEMM，关注临时张量读写。
3. 文本短步的投影与提交粒度。沿用原有MTP状态与重载语义，新增编译路径需覆盖默认异步执行和MTP off。

局部候选通过后进行同工作量5 warm比较、自然提示词质量、多尺寸与卸载重载回归。详细原始证据放`performance-2026-11-12-fp8-deep`。

## 环境恢复

启动probe发现测试checkout的`.venv`中出现CUDA版`torch 2.14.1+cu130`，遮蔽已有Habana PyTorch，bridge报版本不兼容。原8189进程仍持有此前加载的环境，空队列停止后发现该问题。

将测试venv的`torch`、`torchgen`和对应dist-info移动备份到`.run/torch-cu130-backup-2026-11-12`，恢复已有`gaudi-base.pth`引用的Habana`2.13.0a0+git2469d18`。HPU BF16矩阵smoke通过。包安装来源尚未确认。

## Attention候选筛选

既有SFT第0层真实输入用于首轮筛选。BF16按head3.197ms，直接FP8 QK/softmax/cast/AV为3.462ms。厂商`softmax_fp8`合并概率量化后为2.867ms，RRMSE2.43%。概率scale从240改为128后消除一部分额外误差，符合Gaudi2指数偏置缩放的使用方式。较大的head分组和query分块未带来进一步收益。

最初scale240扫描的最后两项触发Dynamo重编译上限，相关耗时排除。后续每个候选独立reset，判断采用`attention-fusion-scale128.json`及Moody重放数据。

随后在当前Moody流程捕获正/负条件、0/12/24步、0/20/41层，共18组实际输入。全QK/AV FP8候选末层RRMSE18.5%–48.1%。逐head/逐行scale与key中心化降低部分误差，敏感样本最佳约8.57%，局部时间3.49ms，超过BF16参考约3.25ms。

保留BF16 QK，采用融合softmax量化与FP8 AV后，代表三层局部2.62ms，RRMSE1.90%–2.58%。该候选进入18组与完整流程验证。

18组完整验证的RRMSE范围为1.76%–3.07%，局部耗时比中位数约0.782。另测未归一化exp→FP8 AV→除以分母，BF16/FP32 exp版本约3.70/6.62ms，局部误差约1.0%–2.4%，耗时超过BF16参考，未采用。

静态Linear activation scale局部筛选使用Moody第0层实际输入，比较逐行动态、静态tensor及静态scalar。静态tensor在部分Linear上节省0.02–0.14ms，但相对现有逐行量化产生1.4%–3.8%输出差异。静态scalar在较大的up/gate上更慢。该实验使用本输入的amax，尚未建立跨prompt/timestep校准模型。

厂商接口依据为[softmax_fp8实现](https://github.com/HabanaAI/gaudi-pytorch-bridge/blob/27eccbcffeab10842c9f29f64c6bf50d8f65d78f/hpu_ops/backend/softmax_fp8_gen.cpp)及本机1.24.2实际schema。

## 精度与产品接入

标准高斯Q/K/V、N4403的压力测试中，厂商融合概率相对FP32 softmax误差6.63%，显式CPU FP8量化参考为2.75%。整体attention输出约7.31%，与厂商概率输出及CPU量化V的FP32乘法参考差0.236%。厂商softmax在分散概率分布上的额外误差来源尚未完全确认。

回归分别检查总体量化误差和FP8 AV执行误差。总体阈值按该压力测试设为8%，AV与独立量化参考阈值为0.5%。最初5%的总体阈值未通过，随后通过上述分解定位到厂商softmax。HPU eager与compiled激活量化舍入也有区别，AV参考使用CPU量化V，以避免将两种量化路径的差异混入矩阵执行误差。

MOSS原生HPU FP8 operations声明量化能力，由模型在本次forward的`transformer_options`中请求`fp8_attention`。实际执行由已有`attention_pytorch` dispatcher选择，使用BF16 QK、融合FP8 softmax输出和原生FP8 AV。当前选择范围为BF16、单batch/单head、head_dim128、序列4096–8192、无mask及额外SDPA参数的编译推理。显式`fp8_attention=False`由模型保留。

普通工作流的三组配对图采用相同扩写结果、seed、采样器和25步。茶壶、MOSS文字与三朵黄花、红围巾白杯人物均通过目视检查，花瓣、器型、手部等细节有变化。首次配对脚本切换计算节点失效策略时重新执行了文本，相关结果独立保存在`quality-av`。修正后两次请求均仅使KSampler/VAE失效，`quality-product`已断言候选请求复用文本。

## 完整流程验收

产品`f67e7f3f`已部署8189，七份产品/测试文件哈希核对一致，266项回归通过。自然1024²配对图片、卸载后MTP off/512²、原INT8/MTP auto均通过，检查到的latent/VAE有限，fallback为空。正常服务cold76.278s、缓存11.881s，额外768²/MTP off请求13.676s，队列为空。

| 固定298-token、1024²、25步 | 本轮基线 | 最终产品 |
|---|---:|---:|
| 完整流程，5 warm中位数 | 18.935852s | 17.190751s |
| TextGenerate | 5.142746s | 5.149400s |
| CLIPTextEncode | 0.416168s | 0.421670s |
| DiT | 13.068357s | 11.364063s |
| VAE | 0.132571s | 0.131917s |
| cold完整流程 | 74.551791s | 73.528932s |
| warm阶段已达allocator峰值 | 27.443489GiB | 27.392476GiB |
| 请求结束常驻 | 17.987434GiB | 17.987203GiB |

完整耗时下降9.2%，DiT下降13.0%。最终warm范围17.121–17.474s。卸载回到2,685,928,832 bytes。自然配对图片MAE为8.986/3.069/8.561（0–255），该值描述像素变化，质量结论来自有限样本的语义和目视检查。

## FP8覆盖与设备预算

新产品第12步的正负条件两段trace按25步外推，按FP8 MME、其它MME、softmax、其它kernel、间隙的优先级计算互斥区间。参考使用前轮`native-cast`同一步、同条件、相同DiT实现的trace。正式端到端时间采用上表本轮无profiler的独立5 warm测量。

| 互斥设备预算 | 改动前 | 改动后 |
|---|---:|---:|
| FP8 MME | 2.5153s | 3.0464s |
| 其它MME | 3.5068s | 2.3919s |
| softmax | 2.7012s | 1.3562s |
| 其它kernel | 3.6074s | 3.6677s |
| kernel间隙 | 0.9163s | 1.0786s |
| 合计 | 13.2470s | 11.5408s |
| FP8 MME时间占比 | 19.0% | 26.4% |

42层主干的四类Linear约2.0267PFLOP，QK与AV各约0.2345PFLOP。按这部分矩阵运算量计算，FP8覆盖从81.2%提高到90.6%。该比例的分母为主干矩阵运算量，时间占比的分母为采样设备区间。

warm阶段去掉节点首尾各1秒后，DiT AIP均值56.36%→55.80%，功率344.65→343.43W。AIP粗采样读数与矩阵活动时间的变化并不同步，当前分析继续使用分阶段耗时和device trace。

## 文本共享量化候选

通过已有`CombinedLinear`成员共享短步activation量化，权重仍由原cast context持有。固定298-token流程17.166s、文本5.157s，对应产品基线17.191/5.149s，未观察到稳定净收益，候选只保留在调查脚本。短步调用受已有区域编译管理，本轮没有逐kernel确认该候选减少了多少量化操作。

候选的第7次diagnostic单独保存logits，卸载后的3,711,490,432 bytes包含调查脚本保留的logits。产品无logits保留的卸载值见上文。独立审计launcher首次因`runpy`覆盖`__main__`使Dynamo无法解析模块变量，改为从具名模块安装候选后完成8-token运行。

## 接续方向

1. 其它kernel的3.67s成为首要目标。继续映射融合TPC节点，重点评估norm/modulation、situ与FP8动态量化的联合实现，以及FP32 RoPE的中间读写。逐行动态量化目前仍有精度优势，静态tensor scale局部预算约3.299→2.901s，收益需要跨prompt/timestep校准与完整流程确认。
2. Attention继续围绕QK数据流和softmax精度推进。已测试全FP8 QK、逐head/逐行缩放、key中心化及未归一化exp路径。新方案需要同时改善当前2.39s其它MME预算及概率转换误差，沿用18组多层/timestep输入。
3. 文本继续关注投影、GDN状态处理与采样之间的提交粒度。简单共享量化和上一轮额外小M编译均缺少可部署收益，后续以recipe/launch变化和相同MTP轨迹建立归因。

[结构化证据](evidence/moss-fp8-deep-2026-11-12.json)保存正式计时、预算来源、局部候选、精度、质量、回归及部署信息。`probes/moss_fp8_deep_jobs.py`与`moss_fp8_deep_server.py`复现本轮流程，`moss_fp8_system_summary.py`继续用于时间区间分析。
