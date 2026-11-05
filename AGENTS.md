# ComfyUI Gaudi2 adaptation

Read `comfyui-hpu/README.md`, `comfyui-hpu/PROGRESS.md`, and `comfyui-hpu/ENVIRONMENT.md` before continuing this task.

All adaptation plans, investigation diaries, failed/successful paths, server details, and progress belong in this repository. Do not put them in ComfyUI-Gaudi or its commit history. The sibling ComfyUI-Gaudi repository should contain only product code, portable scripts and tests.

Never modify the server's original `/home/sonny/ComfyUI` directory. Use the independent `/home/sonny/ComfyUI-Gaudi` test checkout. Never record tokens or credentials.

For model downloads, first use the server's existing `HF_ENDPOINT` and `HF_TOKEN`. Do not override the endpoint by default. Use the existing shell proxy or an alternate source only when needed and record the observed failure that justified it. Reuse existing model components, especially VAEs, instead of downloading duplicates.
