# 独立环境与复现

本文件只保留可操作的环境约束、启动方式和模型路径。验证结论见 `PROGRESS.md`；性能调查的测量契约见 `MOSS_HPU_PERFORMANCE_PLAN_2026-11-06.md`。

## 服务器与安全边界

- 服务器：`sonny@192.168.10.21`；产品测试 checkout：`/home/sonny/ComfyUI-Gaudi`。
- **禁止修改 `/home/sonny/ComfyUI`**，包括 venv、代码、配置、输出和模型；只允许通过独立配置只读复用模型。
- 基础运行时：`/opt/gaudi-install/venv-1.24.2-481-ubuntu2204`；不要用普通 pip resolver 替换 Habana torch。测试依赖安装到独立 `.venv`。
- 当前验证环境：Gaudi2 HL-225、Habana 1.24.2、Python 3.10、Habana torch `2.13.0a0+git2469d18`、eager（`PT_HPU_LAZY_MODE=0`）。
- 模型下载先使用服务器已有 `HF_ENDPOINT`、`HF_TOKEN` 和 shell 代理；不把令牌或代理凭据写入命令、脚本、日志或文档。已有 VAE 优先复用。

## 内核、驱动和设备检查

当前内核为 `5.15.0-202-generic`；Habana DKMS 已覆盖 `5.15.0-186-generic` 和 `5.15.0-202-generic`。重启前检查目标内核的 DKMS，而不是只看 `lspci` 或 `lsmod`：

```bash
sudo bash /home/sonny/gaudi2/gaudi2-single-card-driver/scripts/install-gaudi2.sh \
  --verify-only --skip-pytorch
hl-smi
```

驱动脚本现在枚举当前内核和 `/boot` 中的有效内核镜像，校验五个 Habana 模块、vermagic、Gaudi2 PCI alias、DKMS 路径和补丁字符串；审查补丁保存在 `patches/install-gaudi2-all-boot-kernels.patch`。完整重启、BF16 matmul、双内核 verify 和 19 项模拟回归已通过。未配置开机自启。

## Python 环境

在 `/home/sonny/ComfyUI-Gaudi` 执行，禁止在原始 checkout 中安装：

```bash
/opt/gaudi-install/venv-1.24.2-481-ubuntu2204/bin/python -m venv .venv
printf '%s\n' /opt/gaudi-install/venv-1.24.2-481-ubuntu2204/lib/python3.10/site-packages \
  > .venv/lib/python3.10/site-packages/gaudi-base.pth
.venv/bin/python -m pip install --no-deps -r requirements.txt
.venv/bin/python -m pip install multidict propcache trampoline greenlet python-dotenv \
  kornia-rs mako aiohappyeyeballs aiosignal async-timeout attrs frozenlist \
  'setuptools>=77.0.3' pytest
http_proxy_enable
.venv/bin/python -m pip install --index-url https://pypi.org/simple \
  comfyui-workflow-templates==0.11.102
.venv/bin/python -m pip check
```

`pip check` 预期保留继承的 torchvision 0.16.2 / Habana torch 声明冲突；不要为消除该 warning 替换 Habana torch。

## Smoke、回归和服务

```bash
export PT_HPU_LAZY_MODE=0
export GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so
.venv/bin/python -m scripts.gaudi.smoke
.venv/bin/python -m pytest tests-unit/comfy_test/model_management_test.py -q
```

HPU 服务与独立 smoke/probe 必须串行占卡。用户试用服务：

```bash
ssh sonny@192.168.10.21
cd ~/ComfyUI-Gaudi
./start-hpu.sh
./stop-hpu.sh
tail -f logs/hpu.log
cat .run/deployed-version
```

当前部署提交为 `148b252a`，监听 `http://192.168.10.21:8189`。根目录脚本负责 PID、端口和同目录进程校验，`scripts/gaudi/start.sh` 仍是前台开发入口。停止服务会终止其正在执行的任务。

应用入口在 HPU 初始化后恢复 Python 的 SIGPIPE 忽略策略，避免浏览器取消文件下载使 Synapse 进入 fatal 状态。遇到 `synStatus 31` 时先检查最早的 DFA 原因，不能仅凭后续 `Host malloc failed` 判定主机内存不足。详见 `HPU_SIGPIPE_RECOVERY_2026-11-10.md`。

当前优化由`--disable-comfy-compiler`统一关闭区域编译。首次shape/recipe有编译成本，自然caption和图像细节会受舍入及MTP调度影响。

- Qwen3.5 deferred conv/GDN、KV attention/tree merge使用HPU编译。普通BF16模型另外启用短步MLP、完整attention、norm及八行GDN投影扩区。GDN prefill覆盖BF16扩写和原有FP32普通编码，按至少64及2的幂分桶。
- 八行GDN扩区要求标准LinearKV、普通operations、decode、无hooks/梯度。四行激活在编译递推外clone。KV分支行搬移先clone源行，相关状态回归已验收。
- MOSS采用attention/投影与FFN分区及局部静态recipe。BF16、batch1、24×128 heads、N≥4096、默认无mask/参考KV的推理编译路径按head独立调用原dispatcher。
- 条件text_embedder和FP32 RoPE由本次conditioning持有。BLOCKWISE经四系数语义检查启用，一般矩阵使用PAIRWISE。CPU/训练/梯度/动态hooks/不适用options按相应门控分流。
- 参数、状态及临时tensor由既有模型/执行生命周期管理。完整pipeline卸载稳定约2.50GiB，文本单独卸载约94MiB，各用同工作流基线验证。

当前产品条件和验收范围见 [最新报告](MOSS_BF16_ATTENTION_AND_SUBMISSION_2026-11-11.md)，算法及历史失败见 [复盘索引](MOSS_OPTIMIZATION_STRATEGY_2026-11-09.md)。

## 共享模型路径

两个 checkout 的 `extra_model_paths.yaml` 使用同一配置：

```yaml
shared_models:
  base_path: /home/sonny/ComfyUI-Models
  checkpoints: models/checkpoints
  configs: models/configs
  clip: models/clip
  diffusion_models: |
    models/diffusion_models
    models/unet
  loras: models/loras
  text_encoders: |
    models/text_encoders
    models/clip
  vae: models/vae
```

`models/<分类>` symlink 保留旧式直接路径兼容性。Qwen 2.1 Turbo、MOSS、Qwen3.5/Qwen3VL 和 VAE 已在共享目录；迁移验证和旧配置备份见 `MODEL_STORAGE_PLAN_2026-11-06.md`。

## API 工作流与验证边界

服务器独立 checkout 可提交：

```bash
.venv/bin/python -m scripts.gaudi.submit scripts/gaudi/krea2-api.json \
  --url http://192.168.10.21:8189 --result test-results/krea2.json
.venv/bin/python -m scripts.gaudi.submit scripts/gaudi/qwen-image21-turbo-api.json \
  --url http://192.168.10.21:8189 --result test-results/qwen21.json
.venv/bin/python -m scripts.gaudi.submit scripts/gaudi/moss-image-sft-api.json \
  --url http://192.168.10.21:8189 --result test-results/moss.json
```

已验证的 MOSS 基础工作流为 25 步 `er_sde/simple`、cfg2.5；基础图关闭 prompt enhancer、SLG 和编辑 LoRA。官方 TextGenerate 扩写图也已验证，但自回归仍明显慢于直接文本编码。编辑/控制 LoRA、SLG、完整批量/多卡、lazy/通用全模型编译/HPU Graph、超大分辨率和其它 custom nodes 未验证。

## 性能复现资料

当前正式基准为固定298-token完整MTP19.978s、独立文本4.867s、固定caption缓存14.386s，各5 warm中位数。完整cold64.665s，详情见最新报告与`evidence/moss-bf16-next-2026-11-11.json`。

最新大型数据位于`/home/sonny/gaudi2/comfyui-hpu/performance-2026-11-11-bf16-next`。固定MTP程序位于前轮`performance-2026-11-11-completion/mtp-record-298/program.pt`。该轮六段DiT预算也在`performance-2026-11-11-completion`。

服务使用既有启动脚本和默认32个PyTorch CPU线程，改16线程的完整流程复验无收益。当前lazy/Graph单独试验成本高于eager，lazy＋`hpu_backend`出现`GraphStorage::get()`符号缺失，诊断见预算报告。

## FP8主模型

`Moody-Moss-Mix-V4.1_00001__clean-fp8.safetensors`通过普通UNETLoader/default加载，HPU MixedPrecision Linear自动使用原生E4M3量化/GEMM。FP8安全范围重编码由加载器持有，原模型文件保持原始编码。BF16/FP32激活使用原生入口，FP16/梯度/动态patch沿用浮点路径。已验证文本组合包括现有BF16与新转换FP8。性能和质量范围见 [FP8适配](MOSS_FP8_ADAPTATION_2026-11-11.md)。

## FP8文本模型

CLIPLoader可选择`qwen3.5_9b_fp8_from_int8.safetensors`，位置为共享模型目录`models/text_encoders`。产品自动识别E4M3量化格式并启用HPU原生计算和已有Qwen区域编译。文件为混合精度，201个语言Linear为FP8，MTP、embedding、vision、GDN门控/卷积/norm保留源精度。转换脚本为产品仓库`scripts/gaudi/convert_qwen35_fp8.py`。

完整同工作量warm18.933s、峰值27.267GiB。该文件主要节省显存，自然扩写长度与图像细节会变化，cold比BF16文本基线更长。当前采用默认异步eager pipeline，动态FP8 Linear额外编译候选已撤回。验收与复现见 [全流程记录](MOSS_FP8_SYSTEM_OPTIMIZATION_2026-11-12.md)。
