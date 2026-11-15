# 独立环境与复现

当前验证环境为Gaudi2 HL-225、Habana1.24.2、Python3.10、Habana torch `2.13.0a0+git2469d18`、eager。最近验收版本与质量范围见 [状态](PROGRESS.md)，实验方法见 [测量](optimization/MEASUREMENT.md)。

## 服务器与目录

- 服务器为`sonny@192.168.10.21`，独立产品checkout为`/home/sonny/ComfyUI-Gaudi`，调查目录为`/home/sonny/gaudi2/comfyui-hpu`。
- 原始`/home/sonny/ComfyUI`只读，代码、venv、配置、输出和模型操作均遵守AGENTS约束。
- 基础运行时为`/opt/gaudi-install/venv-1.24.2-481-ubuntu2204`。测试依赖安装到独立`.venv`并保留厂商torch。
- 下载先使用已有HF_ENDPOINT、HF_TOKEN及shell代理，记录必要的实际失败后再改来源，凭据不进入日志。已有模型尤其VAE先核对再复用。

## 内核与依赖

2026-11-12实测内核为`5.15.0-203-generic`，驱动`1.24.2-52a555d`。此前186/202双内核DKMS、重启及HPU算子已通过。重启前可用既有安装脚本只读verify，脚本检查当前和全部有效启动内核。

```bash
sudo bash /home/sonny/gaudi2/gaudi2-single-card-driver/scripts/install-gaudi2.sh \
  --verify-only --skip-pytorch
hl-smi
```

独立venv的建立与已验证依赖安装方式如下，在`/home/sonny/ComfyUI-Gaudi`执行。

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

`pip check`保留继承的torchvision0.16.2/Habana torch声明冲突。曾出现CUDA torch2.14.1遮蔽厂商版本，冲突包备份于独立checkout的`.run/torch-cu130-backup-2026-11-12`，恢复通过`gaudi-base.pth`引用厂商包。故障根因与验证见 [运行时](optimization/RUNTIME.md)。

## 服务与单卡执行

```bash
ssh sonny@192.168.10.21
cd ~/ComfyUI-Gaudi
./start-hpu.sh
./stop-hpu.sh
tail -f logs/hpu.log
cat .run/deployed-version
```

最近部署为`92d15ff7`，绑定`http://192.168.10.21:8189`。根目录脚本校验PID、端口和同目录进程，`scripts/gaudi/start.sh`为前台入口。停止会终止正在执行的请求。独立probe、smoke和服务串行占卡，等待持卡进程实际退出再启动下一项。

```bash
export PT_HPU_LAZY_MODE=0
export GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so
.venv/bin/python -m scripts.gaudi.smoke
.venv/bin/python -m pytest tests-unit/comfy_test/model_management_test.py -q
```

默认32个PyTorch CPU线程。`--disable-comfy-compiler`关闭区域编译，新shape/recipe有首次成本。普通服务使用厂商TPC库和已验配置，32MiB SRAM及自写GDN TPC为历史调查候选。应用初始化后恢复SIGPIPE忽略策略，`synStatus31`先查最早DFA原因。

## 模型与工作流

共享根目录为`/home/sonny/ComfyUI-Models/models`，两个checkout使用分类symlink及相同base_path配置。历史布局、验证与备份见 [模型存储](MODEL_STORAGE.md)。

| 组件 | 已验证文件/来源 |
|---|---|
| BF16 DiT | MOSS SFT BF16 |
| FP8 DiT | `Moody-Moss-Mix-V4.1_00001__clean-fp8.safetensors`，UNETLoader/default自动选择原生E4M3 |
| 文本 | `qwen3.5_9b_int8_convrot.safetensors`、`qwen3.5_9b_bf16_from_int8.safetensors`、`qwen3.5_9b_fp8_from_int8.safetensors` |
| VAE | 已有Flux2 VAE，251个张量核对后复用 |

文本衍生文件在`models/text_encoders`，FP8转换入口为产品`scripts/gaudi/convert_qwen35_fp8.py`。普通CLIPLoader识别量化metadata，FP8来源和保留精度见 [文本精度](optimization/TEXT_PRECISION.md)。模型forward的`transformer_options["fp8_attention"] = False`可关闭混合FP8 attention请求。

```bash
.venv/bin/python -m scripts.gaudi.submit scripts/gaudi/krea2-api.json \
  --url http://192.168.10.21:8189 --result test-results/krea2.json
.venv/bin/python -m scripts.gaudi.submit scripts/gaudi/qwen-image21-turbo-api.json \
  --url http://192.168.10.21:8189 --result test-results/qwen21.json
.venv/bin/python -m scripts.gaudi.submit scripts/gaudi/moss-image-sft-api.json \
  --url http://192.168.10.21:8189 --result test-results/moss.json
```

基础MOSS流程为25步er_sde/simple、CFG2.5。官方TextGenerate扩写图另有验收，Auto/固定树的复现条件见测量专题。各优化的dtype、shape、hooks、训练/梯度和回退门控由对应主题记录。

## 原始性能资料

固定MTP程序位于`performance-2026-11-11-completion/mtp-record-298/program.pt`，最新三阶段资料在`performance-2026-11-12-three-stage`，长序列修复在`performance-2026-11-12-resolution-fix`。其它目录、probe与证据入口见各主题末尾。完整流程、文本单独、混合shape和带logits诊断的卸载值分别对照，避免把保留诊断对象计为产品泄漏。


## Qwen Image 2.1 Turbo 首版

匹配文生图增强器已下载到共享 text_encoders，文件名为 `qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors`。HPU 工作流使用同源本地解码文件 `qwen3.5_9b_qwen_image_2.1_pe_t2i.bf16_from_int8.safetensors`。主模型实际为 `qwen_image_2.1_turbo_bf16.safetensors`，复用已有 Qwen3-VL 8B 和 Qwen Image 2.1 VAE。

导入本仓库 `workflows/qwen21-turbo-hpu.json`，默认 1024²/8 步/CFG1，启用 refine_prompt 即使用 thinking 增强器。API 版本为 `qwen21-turbo-direct-api.json` 和 `qwen21-turbo-enhanced-api.json`，原 INT8 增强器复验使用 `qwen21-turbo-enhanced-int8-api.json`。详细耗时、精度变化与候选取舍见 [首版记录](QWEN21_FIRST.md)。


## 当前全BF16工作流

用户当前图保存为 `workflows/qwen21-user-bf16.json` 和对应API图，基础编码器为已有 `qwen3vl_8b_bf16.safetensors`。832×1504、8步Euler/simple、CFG1、thinking关闭、PE开启作为本轮基线。优化自动通过原节点入口生效，详情见 [全BF16记录](QWEN21_BF16.md)。

调查入口 `probes/qwen21_deep_jobs.py` 使用独立8190服务和强制计算fingerprint，普通服务验收使用 `probes/qwen21_deep_normal.py`。HPU任务继续串行，`/free`后观察稳定内存再确认卸载，本轮全BF16模型约40秒后稳定。服务现为 `dfca7cd6`，原始ComfyUI目录遵循只读约束。


## 2026-11-15 最新2MP工作流

最后一次用户API图为`workflows/qwen21-latest-2026-11-15-api.json`，1088×1920、CFG1.2、8步官方Turbo调度、thinking开启。产品`dfca7cd6`启用单token普通BF16的GDN核心和MLP残差联合编译。

新调查目录为`/home/sonny/gaudi2/comfyui-hpu/performance-qwen21-2026-11-15`。`qwen21_deep_jobs.py --workflow <API文件> --output-root <调查根目录>`支持保存本轮完整/缓存工作流。普通服务重载验证入口为`probes/qwen21_normal_next.py`。运行任何HPU probe或回归均需显式设置`PT_HPU_LAZY_MODE=0`和`GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so`。本轮对照确认eager流水线关闭会显著变慢，普通服务继续使用默认开启状态。完整条件、候选和数值边界见[本轮记录](QWEN21_PERFORMANCE_2026-11-15.md)。

后续无MTP执行候选保存在`performance-qwen-execution-2026-11-15`，采用相同模型、驱动和eager环境。`qwen_execution_jobs.py`在空队列时串行管理文本/完整工作流/原生计数，并在finally恢复8189。调查API启动器采用稳定模块导入后显式调用`qwen21_deep_server.main()`，避免Dynamo解析main模块和导入锁等待。该调查轮恢复了`dfca7cd6`。[验证结果与复现参数](optimization/QWEN21_EXECUTION_VALIDATION.md)

随后按用户要求上线为`92d15ff7`。现有BF16 PE工作流自动选择符合条件的单token快路径，普通main.py入口运行。产品实现与已验候选427-token logits相同，普通API图片和重载验收通过。部署、回退备份及源文件哈希位于`deploy-qwen-execution-2026-11-15`，最后服务队列为空且原工作流已预热。[上线记录](optimization/QWEN21_EXECUTION_DEPLOYMENT.md)
