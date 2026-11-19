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

当前部署为已验收的 PE+MTP M3 `ca0d82e0d91da9f73d43dfb79a39d6d8e7bde84f`（原暂存身份 `96a337d5+staged.be6bf7119839`，tree `fd2d6036a94cd1093d09392f99d2827f31eaad98`，状态 `accepted_committed`），绑定`http://192.168.10.21:8189`（PID `348388`）。根目录 `./init-hpu.sh` 统一负责从 `/opt/gaudi-install/venv-*` 创建 `.venv`、`gaudi-base.pth`、安装依赖及编译 13 核 TPC-C 库；`./start-hpu.sh` 与 `scripts/gaudi/start.sh` 固定使用 `.venv/bin/python` 启动且不执行编译，未运行 `./init-hpu.sh` 时直接报错提示。停止会终止正在执行的请求。独立probe、smoke和服务串行占卡，等待持卡进程实际退出再启动下一项。

后续优化默认在验证后暂存两个仓库的相关改动，再部署环境并交付人工验收。验收前以基线提交、暂存差异或快照及部署文件SHA256标识运行版本，记录“已部署，待人工验收”，用户明确验收通过后提交并关联最终commit。用户特别说明优先。[交付流程](optimization/MEASUREMENT.md#默认交付与人工验收)

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

## 2026-11-15 Qwen21 FP8待验收部署

普通8189运行`b89280c0+staged.10ed4fb5415a`，状态为已部署、待人工验收。源文件及模型哈希、暂存树和普通API验收记录见[FP8报告](QWEN21_FP8_2026-11-15.md)及[证据](evidence/qwen21-fp8-2026-11-15.json)。人工验收通过后提交。

UI已安装`user/default/workflows/Qwen-Image-2.1-Turbo-FP8-HPU.json`。主模型为`qwen_image_2.1_turbo_fp8.safetensors`，PE为`qwen3.5_9b_qwen_image_2.1_pe_t2i.fp8_from_bf16.safetensors`，CLIP复用`qwen3vl_8b_fp8_scaled.safetensors`，VAE复用已有BF16。PE的源BF16来自此前INT8解码。转换命令在独立产品checkout执行。

```bash
.venv/bin/python -m scripts.gaudi.convert_qwen21_fp8 models/diffusion_models/qwen_image_2.1_turbo_bf16.safetensors models/diffusion_models/qwen_image_2.1_turbo_fp8.safetensors
.venv/bin/python -m scripts.gaudi.convert_qwen35_fp8 models/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.bf16_from_int8.safetensors models/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.fp8_from_bf16.safetensors
```

文件已存在时转换器会拒绝覆盖。完整无缓存计时使用强制重算compute nodes和DiT KV off，warm保留常驻模型及recipe。当前全FP8自然流程71.507秒，BF1625.425秒，主要退化来自PE逐层执行和更长自然扩写。512²、卸载重载及普通服务完整图均已通过。

## 2026-11-16 FP8优化后部署

普通8189当前运行`b89280c0+staged.988121fee669`。已有FP8工作流自动启用符合条件的Qwen3.5原生FP8整步解码和Qwen21 FP8 FFN区域编译。三个模型文件及工作流参数沿用首轮转换版本，UI文件仍为`Qwen-Image-2.1-Turbo-FP8-HPU.json`。

完整无缓存warm产品中位23.943秒，同轮BF16为25.526秒，固定432-token轨迹24.129秒。普通API冷启动129.466秒，两次完整重算24.277/24.105秒成功，最后维持原MTP auto设置，队列为空。源文件SHA256、暂存快照和质量边界见[优化报告](QWEN21_FP8_OPTIMIZATION_2026-11-16.md)。两个仓库的改动保持暂存，等待人工验收。

## 2026-11-16 FP32来源PE

普通8189当前暂存部署`b89280c0+staged.d5bda4e78d95`，待人工验收。新增共享模型`text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.fp8_from_int8_fp32.safetensors`及独立UI工作流`Qwen-Image-2.1-Turbo-FP8-PE-From-FP32-HPU.json`。转换器使用`--from-int8-convrot`选择FP32反量化路线，具体命令和模型SHA256见[转换记录](QWEN21_PE_FP32_CONVERSION_2026-11-16.md)。

新模型相对FP32反量化参考的权重误差小幅降低。该提示词自然扩写806 tokens，完整warm30.248秒；固定432-token轨迹24.478秒。普通服务冷启动131.448秒、512²、卸载重载均通过，队列为空。运行时沿用前轮原生FP8整步和FFN优化。

## 2026-11-16 提交关联

用户明确授权提交代码，最终产品提交为`07fedda3278b5c4be8c036fe245dc6991b91dd4a`。该提交的tree与已验证暂存树`d4264962aec7a96f7318ceca52263edf9b6ee5b7`相同，服务器源文件哈希再次核对通过。`.run/deployed-version`使用最终产品commit，`.run/deployed-staged.json`及调查目录的`deployment.json`保留原暂存身份并补充commit关联。

## 2026-11-16 硬件80%计划首批部署

普通8189已运行`07fedda3+staged.d39241e049d0`，待人工验收。符合现有Qwen整步生成准入条件的请求采用四token回读，普通工作流自动进入该路径。源码快照、五轮交错与部署验证见[首批报告](optimization/QWEN80_FIRST_DELIVERY_2026-11-16.md)。服务器资料位于`performance-qwen80-2026-11-16`，部署、回退源文件及普通history位于其`deployment/`目录。

普通冷图145.016秒、两次完整warm43.856/43.957秒、卸载重载125.588秒通过，扩写和RGBA均与原图一致。最后保留原MTP auto参数、队列为空，四个部署文件SHA256匹配产品暂存树`0f4df163a296fd3829d2344ab7385158510797dd`。

## 2026-11-16 P1/P2/P3产品部署

普通8189已运行`07fedda3+staged.87ea451fce53`，待人工验收。产品tree为`7f6b735fa3cd5350e9d6cbfdebb9fe8b79d9eb2d`，六个源文件/测试文件哈希与验证快照一致。原始试验、191项回归、普通节点A/B、跨shape及重载记录位于`/home/sonny/gaudi2/comfyui-hpu/performance-qwen80-pe-2026-11-16/product-validation4`。部署身份、回退源文件与普通8189验收记录进入同根目录的`deployment`。

符合准入条件的原生FP8 Qwen生成采用调用级紧凑RoPE表和专用标准旋转入口。自然扩写与图片会改变，固定1909-token整图中位44.370→36.593秒。当前HBM0.893TB/s、峰值36.44%，80%目标尚未达成。[执行与质量记录](optimization/QWEN80_PE_EXECUTION_2026-11-16.md)

普通首图141.564秒，两次完整warm38.635/38.316秒，卸载重载121.408秒。四次扩写及RGBA匹配验证候选，六个文件哈希匹配，验收时队列为空。部署标记保留baseline、暂存tree/patch及文件SHA256，代码与记录均保持暂存。

## 2026-11-17 扩写编译复用修复

普通8189当前运行`07fedda3+staged.c8ad74f79c7e`，产品暂存tree为`cfdce5bf049c851dd042a84ab7b85248f535b3cd`，已部署、待人工验收。符合准入条件的Qwen整步解码采用动态KV容量标注及规范RoPE步长。提示词变化或thinking切换后的新容量首个decode约0.6秒，首次加载模型仍有初始图编译成本。

源码快照、193项回归、22次普通节点对照、1909-token logits逐元素一致检查及普通服务六次验收位于`/home/sonny/gaudi2/comfyui-hpu/performance-qwen-startup-fix-2026-11-17`。`deployment`保存manifest、暂存patch、回退源文件、实际部署哈希和普通history，状态为`deployed_awaiting_human_acceptance`。普通服务cold142.963秒，四个编辑变体27.236/24.754/26.232/36.046秒，warm38.330秒。六个文件哈希匹配、队列为空。[完整记录](optimization/QWEN_DECODE_STARTUP_2026-11-17.md)

## 2026-11-17 验收与提交关联

用户验收后，缓存case已固化为节点输入签名回归和BF16/FP8实际生成入口回归。完整相关集207项通过、1项跳过，普通服务恢复出图142.826秒、caption/RGBA匹配已验版本。七个代码/测试文件哈希与产品提交`a5308be1c7dd56d8ebb694a8a9a630a072089a77`一致，提交tree为`f68d0361edcc01559eacb4f68345c30e9349ccfe`。

`.run/deployed-version`使用正式commit，`.run/deployed-staged.json`保留`07fedda3+staged.44b0b24b6fca`快照、验收指令与commit关联，状态为`accepted_committed`。原始回归日志、验证哈希、恢复图和部署记录位于`/home/sonny/gaudi2/comfyui-hpu/performance-qwen-cache-regression-2026-11-17`。独立补验入口为`probes/qwen_cache_regression_jobs.py`。

## 2026-11-17 D1验收提交

暂存tree为`9d01642e8deff0c4cb461a34b808115968ba8729`，普通`main.py`入口运行Qwen原生FP8 DiT attention优化。显式`transformer_options["hpu_fast_attention"] = False`使用原attention路径。当前验证范围包含1088×1920、512²/1024²、KV auto/off、提示词回切及卸载重载。

资料根目录为`/home/sonny/gaudi2/comfyui-hpu/performance-qwen-d1-2026-11-17`。`product3`保存完整流程、硬件计数及主要回归，`followup`保存最终33项专项和prefix实验，`deployment`保存暂存patch、manifest、回退源文件与七次部署验证及两次恢复后普通history。三份变更文件及1070份Python/shell源码哈希验证通过，普通服务已预热、队列为空。当前status为`accepted_committed`，正式提交为`22e3972af91d44c7af7a6fa3fb371e4b3dd2e3f4`，manifest保留暂存身份与人工验收指令。[D1记录](optimization/QWEN_D1_ATTENTION_2026-11-17.md)

## 2026-11-17 D2未暂存快照部署历史

普通`http://192.168.10.21:8189`曾运行`22e3972a+worktree.54b8266a1348`，基线为D1正式提交`22e3972af91d44c7af7a6fa3fb371e4b3dd2e3f4`。用户要求本轮先不暂存，manifest明确`staged=false`，当时状态为`deployed_awaiting_human_acceptance`。部署期间两个本地仓库的index保持空差异。用户随后决定撤回D2产品改动，恢复D1，保留调查记录。

原始资料根目录为`/home/sonny/gaudi2/comfyui-hpu/performance-qwen-d2-2026-11-17`。`product-validation2`保存最终共享存储回归、41次成对工作流和独立硬件计数，`deployment`保存未暂存patch、候选源码、manifest、D1回退文件、七次普通history和普通日志。`probe-source.tar`及`probe-source/`保存调查工具快照，`product-source-hashes.json/source-audit.json`核对1071份Python/shell源码。代码与测试的三份部署SHA256均匹配验证快照。

普通cold146.915秒、warm32.028秒、蓝/红/蓝回切32.824/26.298/31.416秒、恢复原图32.243秒、卸载重载122.580秒。全部caption/RGBA匹配验证候选，关键计算节点实际执行。卸载稳定9.273GiB，别名及重编译上限警告0次，服务已预热、队列为空。普通进程沿用原有启动环境，SCAL隔离配置和TPC实验库仅供独立研究进程使用。[D2执行与剩余目标](optimization/QWEN_D2_SRAM_2026-11-17.md)

## 2026-11-17 D2撤回与D1恢复

用户明确决定丢弃D2产品改动并保留调查记录。普通8189已恢复D1正式提交`22e3972af91d44c7af7a6fa3fb371e4b3dd2e3f4`，恢复原有`.run/deployed-version`及已验收manifest。1070份Python/shell源码与本地D1一致，D2新增prefix回归文件已移除。

原D2资料根目录的`rollback/rollback.json`记录状态`withdrawn_restored_d1`，同目录保存冷/热history和普通日志。cold142.828秒、完整warm34.322秒，扩写及RGBA匹配已验D1，别名及重编译上限告警0次，服务已预热、队列为空。本地产品仓库工作区干净，本仓库调查与归档改动保持未暂存。完整撤回补丁保存在`patches/qwen-d2-withdrawn-2026-11-17.patch`。

## 2026-11-18 D2方案A片上SRAM流水部署

普通`http://192.168.10.21:8189`曾运行方案 A 正式提交`0818699bc7bee8c5ff9e3f13eef87c20153f4e66`（原暂存身份`22e3972a+staged.3161d0925c58`，tree `a6ebfc0a3d59ac8a8fcdddb0b43922d339827e4f`，状态`accepted_committed`）。产品新增`comfy/ldm/modules/hpu_attention.py`并在首次调用时自动编译`/tmp/comfy_hpu_rmw_attn_<hash>.so`（链接系统`/usr/lib/habanalabs/libSynapse.so`），无需修改服务启动环境变量。

验证资料位于`/home/sonny/gaudi2/comfyui-hpu/performance-qwen-d2-2026-11-17/plan-a-validate76`、`plan-a-product-validation`与`plan-a-deployment`。三份部署文件（`comfy/ldm/modules/hpu_attention.py`、`comfy/ldm/modules/attention.py`、`tests-unit/comfy_test/hpu_fast_attention_test.py`）SHA256匹配验证manifest，普通cold 140.809秒、warm 33.724秒、蓝/红/回切33.193/27.269/32.545秒、原图33.301秒、重载118.399秒及热复验34.051/33.450秒全部匹配D1基准，别名及重编译上限告警0次，队列为空。用户已确认验收并提交。

## 2026-11-18 D2 方案 B 部署

普通服务 `http://192.168.10.21:8189` 曾运行已验收的 `26a170f97b360a61cc597c0908df294eea03dfa5`（原暂存身份 `0818699b+staged.8183faca48ab`，tree `58081bb0060c43c3ee670fc8a017d23d204565f5`）。服务器资料根目录为 `/home/sonny/gaudi2/comfyui-hpu/performance-qwen-d2-2026-11-17/plan-b-resume`，其中 `final/` 保存交错工作流、原生 trace 和回归日志，`deployment/` 保存普通 history、源码核对、暂存 patch、manifest 及回退文件。部署标记与 manifest 已关联正式提交，状态为 `accepted_committed`。[方案 B 记录](optimization/QWEN_D2_SRAM_2026-11-17.md)

## 2026-11-19 Qwen PE+MTP M2 验收部署

普通服务 `http://192.168.10.21:8189`（PID `145830`）运行已验收的 `96a337d58b3674593c252d804755a6cb7c5d8d36`（原暂存身份 `26a170f9+staged.dfe536e987e3`，基线 `26a170f97b360a61cc597c0908df294eea03dfa5`，暂存 tree `0848de81178693597c3ba5542d4b272693489166`，patch SHA256 `dfe536e987e3b92a68d0b91f0d5aa2a233b20ce1523871dc8a43c806dd325b34`），状态为 `accepted_committed`。

新增独立共享模型 `/home/sonny/ComfyUI-Models/models/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i_mtp.fp8_from_int8_fp32.safetensors`（`11,133,746,108` bytes，SHA256 `468cdb77ae8ba0bc0ad906af4282755f90bf12358f1113e935cce36fc4c864bd`）与独立 UI 工作流 `user/default/workflows/Qwen-Image-2.1-Turbo-FP8-PE-MTP-HPU.json`（保留原无 MTP 模型及原工作流不覆盖）。复现转换命令：

```bash
.venv/bin/python -m scripts.gaudi.convert_qwen35_fp8 \
  --from-int8-convrot \
  --attach-mtp /home/sonny/ComfyUI-Models/models/text_encoders/qwen3.5_9b_int8_convrot.safetensors \
  --quantize-mtp \
  /home/sonny/ComfyUI-Models/models/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors \
  /home/sonny/ComfyUI-Models/models/text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i_mtp.fp8_from_int8_fp32.safetensors
```

服务器资料根目录为 `/home/sonny/gaudi2/comfyui-hpu/performance-qwen-pe-mtp-2026-11-19`，包含 `m0/`（微基准与未优化 MTP 基线）、`m2_mod/` 与 `m2_v2/`（消融）、`m2_audit/`（完整 4096 自然扩写、遥测、BMON/原生时间线及 454 项回归日志）以及 `deployment/`（暂存 patch、manifest、回退基线备份、`e2e/` 27 次普通服务 history/caption 及日志审计）。1074 份 `.py`/`.sh` 源码核对通过，普通服务已预热、队列为空，已验收提交。[M2 报告](optimization/QWEN_PE_MTP_M2_2026-11-19.md) · [证据](evidence/qwen-pe-mtp-m2-2026-11-19.json)

## 2026-11-19 Qwen PE+MTP M3 验收部署

普通服务 `http://192.168.10.21:8189`（PID `348388`）运行已验收的 `ca0d82e0d91da9f73d43dfb79a39d6d8e7bde84f`（原暂存身份 `96a337d5+staged.be6bf7119839`，基线 `96a337d58b3674593c252d804755a6cb7c5d8d36`，tree `fd2d6036a94cd1093d09392f99d2827f31eaad98`，补丁 SHA256 `be6bf7119839c2f972e77f1ef3e622c7d3081faa51418f793fb9d4aff451a640`），状态为 **`accepted_committed`**（按用户指令“提交代码，不推进M4”在 M3 收口）。

环境准备与启动已分离：全新 clone 后先执行 `./init-hpu.sh`（自动从 `/opt/gaudi-install/venv-*` 创建 `.venv`、写入 `gaudi-base.pth`、安装非 torch 依赖并编译 13 核 TPC-C 库；已有 `.venv` 仅需重编内核时可加 `--skip-deps`），再执行 `./start-hpu.sh` 或 `bash scripts/gaudi/start.sh`（固定 `.venv/bin/python`，不做编译；未执行 `./init-hpu.sh` 时直接报错提示；设置 `COMFY_HPU_DISABLE_QWEN35_TPC=1` 可显式回退 PyTorch）：

```bash
cd /home/sonny/ComfyUI-Gaudi
./init-hpu.sh          # 全新 clone：创建 .venv + 装依赖 + 编译 build/hpu/*.so
./start-hpu.sh         # 启动服务（固定 .venv/bin/python）
```

服务器资料根目录为 `/home/sonny/gaudi2/comfyui-hpu/performance-qwen-pe-mtp-m3-2026-11-19`，包含 `audit/`（4096 自然扩写、遥测、固定轨迹 logits 一致性）、`native-m33-4096/`（真实 4096 窗口低扰动 BMON 与原生时间线）及 `deployment/`（暂存 patch、manifest、M2 基线回退备份、`source-audit.json` `1,093/1,093` 核对、`regression.log` `559 passed, 1 skipped`、`mtp-45-test.log` `45 passed`、`rev3-verify.json` `init-hpu.sh`/`start-hpu.sh` 分离实测、`e2e/` 27+1 次普通服务 history/caption 与 `free-vram-audit.json`）。服务已热态留存、队列为空。[M3 报告](optimization/QWEN_PE_MTP_M3_2026-11-19.md) · [M3 证据](evidence/qwen-pe-mtp-m3-2026-11-19.json)
