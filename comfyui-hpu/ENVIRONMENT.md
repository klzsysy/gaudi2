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
