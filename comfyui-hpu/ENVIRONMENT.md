# 独立环境与复现

以下在服务器独立 checkout `~/ComfyUI-Gaudi` 中执行；不要在原 `~/ComfyUI` 中运行安装命令。

```bash
/opt/gaudi-install/venv-1.24.2-481-ubuntu2204/bin/python -m venv .venv
printf '%s\n' /opt/gaudi-install/venv-1.24.2-481-ubuntu2204/lib/python3.10/site-packages > .venv/lib/python3.10/site-packages/gaudi-base.pth
.venv/bin/python -m pip install --no-deps -r requirements.txt
.venv/bin/python -m pip install multidict propcache trampoline greenlet python-dotenv kornia-rs mako aiohappyeyeballs aiosignal async-timeout attrs frozenlist 'setuptools>=77.0.3' pytest
# shell 中启用已有代理；不要打印 HF_TOKEN 或代理凭据
http_proxy_enable
.venv/bin/python -m pip install --index-url https://pypi.org/simple comfyui-workflow-templates==0.11.102
.venv/bin/python -m pip check
```

最后一项预期存在继承的 torchvision 0.16.2 / Habana torch 版本声明冲突，详见 PROGRESS.md。不要执行无约束的普通 `pip install -r requirements.txt`。

```bash
export PT_HPU_LAZY_MODE=0
export GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so
.venv/bin/python -m scripts.gaudi.smoke
.venv/bin/python -m pytest tests-unit/comfy_test/model_management_test.py -q
bash scripts/gaudi/start.sh
```

服务默认仅绑定服务器 `127.0.0.1:8189`，本地访问使用：

```bash
ssh -N -L 8189:127.0.0.1:8189 sonny@192.168.10.21
```

浏览器打开 `http://127.0.0.1:8189`。独立 smoke 与服务必须串行占用 HPU。

## 模型下载优先级

模型下载先使用服务器 shell 已有的 `HF_ENDPOINT` 和 `HF_TOKEN`，不要显式覆盖端点。`huggingface_hub` 可自动读取；需要显式传参时从 `os.environ` 取值而不是写入脚本。先检查已有文件，尤其是 VAE，避免重复下载。代理或备用站点仅在必要时使用并记录失败证据。

## 模型路径与工作流

服务器独立 checkout 的 `extra_model_paths.yaml` 只读引用已有 Krea2 组件：

```yaml
existing_readonly:
  base_path: /home/sonny/ComfyUI
  diffusion_models: models/diffusion_models
  text_encoders: models/text_encoders
  vae: models/vae
```

Qwen 2.1 Turbo 的新权重在独立 checkout 自己的 `models/{diffusion_models,text_encoders,vae}`，文件和固定 revision 见 PROGRESS.md。加载不需要 HF 访问或令牌；只有重新下载才需要已有 shell 代理/环境。

服务运行后，在服务器独立 checkout 执行：

```bash
.venv/bin/python -m scripts.gaudi.submit scripts/gaudi/krea2-api.json --result test-results/krea2.json
.venv/bin/python -m scripts.gaudi.submit scripts/gaudi/qwen-image21-turbo-api.json --result test-results/qwen21.json
.venv/bin/python -m scripts.gaudi.submit scripts/gaudi/moss-image-sft-api.json --result test-results/moss.json
```

三份 JSON 是 API 工作流，使用原生节点，不需要自定义节点。额外 1024/重载变体保存在本调查仓库 `workflows/`。图片写入独立 `output/gaudi/`。

新增回归测试（停止服务后运行）：

```bash
.venv/bin/python -m pytest tests-unit/comfy_test/model_management_test.py tests-unit/comfy_test/hpu_quant_test.py tests-unit/comfy_test/hpu_vae_test.py -q
```

服务器目录通过 tar/scp 同步源码，**没有 `.git` 元数据**；本地代码仓库才是版本来源。同步时仅从已审查代码提交打包，例如 `git archive codex/gaudi2-hpu | ssh sonny@192.168.10.21 'tar -xf - -C ~/ComfyUI-Gaudi'`，不要复制本调查仓库、过程文档、令牌或配置。修改后重启服务，先等待旧进程彻底退出。当前验证版本为 `21e98390`。

MOSS SFT 与 Qwen3.5 9B 的新权重位于独立 models；Flux2 VAE 通过上述只读路径复用。MOSS 固定验证参数是 25 步 er_sde/simple、cfg2.5；没有启用可选 prompt enhancer、SLG 或编辑 LoRA。
