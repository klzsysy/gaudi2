# ComfyUI 模型集中存储方案

更新时间：2026-11-06（Asia/Shanghai）。本记录只保存调查结果和迁移方案，不保存令牌或凭据。

## 只读核对结果

在 `192.168.10.21` 上只读检查了两套 checkout：

- `/home/sonny/ComfyUI/models`：约 358 GB。
- `/home/sonny/ComfyUI-Gaudi/models`：约 42 GB。
- 两者位于同一个 `/home` 文件系统，剩余空间约 1.1 TB。
- 采样时没有发现 ComfyUI 进程在运行。
- 原 `/home/sonny/ComfyUI/extra_model_paths.yaml` 仍是上游示例模板，没有集中模型目录配置。
- `/home/sonny/ComfyUI-Gaudi/extra_model_paths.yaml` 当前只读引用原 checkout 的 `diffusion_models`、`text_encoders` 和 `vae`。
- 两棵目录共有 233/53 个文件；按相对路径和文件大小比较，重合项是空占位文件和 `models/configs` 中的默认配置，没有发现实际模型文件的同相对路径重合。这个比较没有替代最终迁移前的 SHA256 校验。

ComfyUI 的 `main.py` 会自动读取 checkout 根目录的 `extra_model_paths.yaml`，配置中的 `base_path` 会作为相对模型路径的基准目录。模型搜索路径在启动时建立，改完配置必须重启对应服务。

## 建议的目标布局

建立独立的模型仓库，例如：

```text
/home/sonny/ComfyUI-Models/
└── models/
    ├── checkpoints/
    ├── clip/
    ├── clip_vision/
    ├── configs/
    ├── controlnet/
    ├── diffusion_models/
    ├── embeddings/
    ├── loras/
    ├── text_encoders/
    ├── unet/
    ├── upscale_models/
    └── vae/
```

保留 `models/` 这一层可以继续使用 ComfyUI 约定的相对路径，也方便以后增加其它模型类别。模型仓库只放模型、配置和 tokenizer 等共享数据，不放代码、虚拟环境、`output`、`user` 或 `custom_nodes`。

两个 checkout 的 `extra_model_paths.yaml` 都可以使用同一段配置，按实际需要补齐类别：

```yaml
shared_models:
  base_path: /home/sonny/ComfyUI-Models
  checkpoints: models/checkpoints
  configs: models/configs
  clip: models/clip
  clip_vision: models/clip_vision
  diffusion_models: |
    models/diffusion_models
    models/unet
  embeddings: models/embeddings
  loras: models/loras
  text_encoders: |
    models/text_encoders
    models/clip
  vae: models/vae
  upscale_models: models/upscale_models
```

`text_encoders` 和 `diffusion_models` 同时列出新旧目录名，是为了兼容 ComfyUI 的 `clip`/`unet` 历史布局。不要把 `custom_nodes` 写入共享模型配置；两个 checkout 的代码和依赖仍应独立。

## 迁移顺序

1. 先保存两份 `extra_model_paths.yaml`，生成包含相对路径、大小和 SHA256 的清单。
2. 停止两个服务，确认队列为空、没有遗留的 `main.py`/模型加载进程；单卡 Gaudi2 不能在迁移期间被另一个进程占用。
3. 在同一个 `/home` 文件系统创建目标目录。对不冲突的文件可以用同文件系统的移动操作；相同相对路径必须先比较 SHA256。相同内容只保留一份，不同内容保留两份并使用明确的子目录或新文件名，不能静默覆盖。
4. 先更新独立 `ComfyUI-Gaudi` 的配置并启动，确认 `/system_stats`、模型下拉列表和一条已验证工作流都能加载。
5. 再更新原 ComfyUI 的配置并启动，确认至少一个文本编码器、扩散模型和 VAE 的列表和加载结果。
6. 两边都验证通过后，才删除旧目录中的重复文件；迁移期间不要用“目录大小变小”代替文件清单和哈希验证。

共享磁盘目录只共享文件，不共享 Python 环境、模型内存、编译缓存或服务端口。两个服务同时加载大模型仍会分别占用资源；模型下载和删除也会作用于同一个目录，因此权限和备份策略需要一起调整。

## 实际迁移结果（2026-11-06）

用户明确授权后已完成迁移。两个服务均确认未运行，目标目录此前不存在；在同一个 `/home` 文件系统内创建并合并到 `/home/sonny/ComfyUI-Models/models`。

- `/home/sonny/ComfyUI/models` 和 `/home/sonny/ComfyUI-Gaudi/models` 的模型内容已移动到共享目录。
- 迁移统计：38 个目录/文件项移动，36 个相同内容项去重，0 个冲突；迁移后共享目录约 399 GB。
- 两个 checkout 的 `models/<分类>` 均建立到共享目录的兼容 symlink，包含 `diffusion_models`、`unet`、`text_encoders`、`vae` 以及 `SEEDVR2`、`prompt_generator`、`ultralytics` 等旧式直接访问目录；checkout 内没有遗留普通模型文件。
- 两份 `extra_model_paths.yaml` 已写成相同的共享配置，根目录为 `/home/sonny/ComfyUI-Models`。旧配置备份为：
  - `/home/sonny/ComfyUI/extra_model_paths.yaml.pre-shared-20261106-142147`
  - `/home/sonny/ComfyUI-Gaudi/extra_model_paths.yaml.pre-shared-20261106-142147`
- 只启动并验证了 `/home/sonny/ComfyUI-Gaudi`（PID 29439）。`/system_stats` 返回 `hpu:0 GAUDI2`，模型 API 列表发现 Qwen 2.1、MOSS、Qwen3、T5 和两个 VAE；Qwen 2.1 Turbo 512×512 原生工作流实际出图成功，结果为 `output/gaudi/qwen21_turbo_512_seed42_00003_.png`。
- 原 `/home/sonny/ComfyUI` 没有启动。调查结论和迁移记录保留在本仓库，不同步到 `ComfyUI-Gaudi` 产品 checkout。
