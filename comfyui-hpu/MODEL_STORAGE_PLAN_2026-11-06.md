# ComfyUI 模型集中存储结果

更新时间：2026-11-06（Asia/Shanghai）。迁移已完成；本文件保留最终布局、验证和回退信息，不再保留已完成的迁移计划流水账。

## 最终布局

共享根目录为 `/home/sonny/ComfyUI-Models/models`，只存模型、配置和 tokenizer，不存代码、venv、output、user 或 custom nodes。两个 checkout 的 `models/<分类>` 通过 symlink 兼容旧式直接路径；两份 `extra_model_paths.yaml` 都以 `/home/sonny/ComfyUI-Models` 为 `base_path`。

保留的类别包括 `checkpoints`、`configs`、`clip`、`clip_vision`、`diffusion_models`、`embeddings`、`loras`、`text_encoders`、`unet`、`upscale_models` 和 `vae`。`diffusion_models`/`text_encoders` 同时列出新旧目录名，以兼容 ComfyUI 的历史布局。

## 迁移结果

- 两个 checkout 的模型内容已合并；移动 38 个目录/文件项，36 个相同内容项去重，0 个冲突，共享目录约 399 GB。
- `/home/sonny/ComfyUI/models` 和 `/home/sonny/ComfyUI-Gaudi/models` 均不再保留普通模型文件，分类 symlink 可用。
- 配置备份保留在服务器：
  - `/home/sonny/ComfyUI/extra_model_paths.yaml.pre-shared-20261106-142147`
  - `/home/sonny/ComfyUI-Gaudi/extra_model_paths.yaml.pre-shared-20261106-142147`

## 验证与边界

- 只启动并验证了 `/home/sonny/ComfyUI-Gaudi`；`/system_stats` 返回 `hpu:0 GAUDI2`，模型列表发现 Qwen 2.1、MOSS、Qwen3、T5 和两个 VAE。
- Qwen 2.1 Turbo 512×512 原生工作流实际出图成功；原始 `/home/sonny/ComfyUI` 没有启动，也没有修改其代码、环境或输出。
- 共享目录会同时影响两个 checkout 的模型读取、下载和删除；代码、依赖、编译缓存、服务端口和模型内存仍各自独立。
