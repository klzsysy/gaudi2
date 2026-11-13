# 模型共享与生命周期

2026-11-06完成存储整理，运行配置见 [环境](ENVIRONMENT.md)。共享根目录为`/home/sonny/ComfyUI-Models/models`，保存模型、配置和tokenizer。两个checkout各自管理代码、venv、output、user、custom nodes、服务和设备内存。

## 布局与验证

两份`extra_model_paths.yaml`以`/home/sonny/ComfyUI-Models`为base_path，`models/<分类>`symlink兼容旧式直接读取。类别包括checkpoints、configs、clip、clip_vision、diffusion_models、embeddings、loras、text_encoders、unet、upscale_models和vae，diffusion_models/text_encoders同时配置新旧名称。

迁移移动38项、去重36项、冲突0，共享目录当时约399GB。仅启动Gaudi checkout验证，模型列表发现Qwen2.1、MOSS、Qwen3、T5和VAE，Qwen2.1 Turbo512²实际出图成功。共享目录内的下载、修改和删除会影响两个checkout的模型读取。

两份历史配置备份为`/home/sonny/ComfyUI/extra_model_paths.yaml.pre-shared-20261106-142147`与`/home/sonny/ComfyUI-Gaudi/extra_model_paths.yaml.pre-shared-20261106-142147`。后续操作遵循仓库约束，原始`/home/sonny/ComfyUI`只读。

## 优化相关经验

- 下载使用已有HF_ENDPOINT/HF_TOKEN，已有VAE先核对header/shape/dtype再复用。MOSS的251个VAE张量已与官方组件核对。
- BF16和FP8文本衍生文件记录INT8来源，分别原子发布为新文件，完整转换和精度结果见 [文本精度](optimization/TEXT_PRECISION.md)。
- 权重量化/转换由加载、patch和卸载持有，重复加载避免二次范围缩放。执行期静态条件由conditioning持有，跨请求缓存需单独设计失效。
- allocator峰值、模型常驻、诊断对象和卸载稳定值分别记录，具体比较方法见 [测量](optimization/MEASUREMENT.md)。
