# 浏览器断连导致 HPU 运行时失效

日期：2026-11-10（Asia/Shanghai）。用户在日常 8189 服务运行普通 BF16 MOSS 图时报 `Host malloc failed`。产品基线为 `3fec1143`，修复提交为 **`441d459d14e8a9bf49a7aa2ac678750b92c6fe00`**。

## 原因与证据

- 部署版本为 `3fec1143`，`llama.py`、`qwen35.py`、`hpu_kv_attention.py` 和修改前 `main.py` 的服务器 SHA256 均与本地提交一致，不是代码漏同步。
- 故障后主机约 89 GiB 可用，HPU 仅约 773 MiB 占用；当前进程 RSS 约 2.3 GiB。模型与 checkpoint 中列出的尺寸相同。
- **12:25:15**，Synapse DFA 日志记录 `exception 13 Broken pipe. severity: high`，用户栈入口为 libc `sendfile64`，故障原因明确为 signal 13。设备此时进入 fatal 状态。
- **12:25:48**，加载文本模型第一份 embedding 权重时开始报 `synStatus 31 ... Host malloc failed`，此后其它权重全部连带失败。该消息是失效运行时拒绝后续操作的表现，不是本次根因。
- 后续 worker 在 HPU synchronize 上再次抛错退出，导致一个用户请求留在 pending 队列。

Python 默认忽略 SIGPIPE，让关闭管道/套接字上的写操作以普通异常报告。Habana 1.24.2 初始化安装了自己的原生信号处理器，导致浏览器取消文件响应也被当作致命设备故障。[Python signal 文档](https://docs.python.org/3/library/signal.html#general-rules)

前轮性能验收走工作流 JSON API 和完整响应读取，未覆盖浏览器取消静态文件传输。本次补测该入口。

## 最小复现

`probes/hpu_sigpipe.py` 在独立进程中初始化 HPU，然后向已关闭另一端的 socket 使用 `os.sendfile`：

| 路径 | 初始化后的 SIGPIPE | 断连后的 HPU 运算 |
|---|---|---|
| 原生运行时策略 | ignored=false，caught=true | DMA 失败；设备释放时再次抛错，最终 SIGABRT，exit=-6 |
| 初始化后恢复 `SIG_IGN` | ignored=true，caught=false | BrokenPipeError 正常返回；后续 HPU 加法得到 5，exit=0 |

初版调查 runner 预期失败进程 exit=2，但 native destructor 的再次异常使实际退出为 -6；因此最初 runner 的退出码断言失败。失败日志和成功路径均保留，未把断言失败宣称为测试通过。实际证据是原路径设备被毒化、修复路径仍可计算。

## 修复与验收

仅在 `main.py` 独立应用启动、HPU/节点初始化完成后、HTTP 服务开始前，针对 HPU 恢复 `signal.signal(signal.SIGPIPE, signal.SIG_IGN)`。没有更改模型、算子、权重、显存策略或其它信号处理。

1. 停止已失效进程前，保存唯一 pending 请求的图、原 prompt ID 和非敏感 UI metadata；不保存队列中的 sensitive 字段。
2. 同步修复并正常启动 8189，保留现有 manager/custom nodes 配置。
3. `probes/hpu_server_recovery.py` 从真实前端静态文件路由发起 **8 次下载后立即断开**，再向服务进程注入一次 SIGPIPE；每次检查服务可用，OS 的 SIGPIPE ignore bit 正确。
4. 重放用户原 pending 图，不修改模型、提示词、seed 或尺寸。原 ID `289d6480-7246-4534-98b2-ca3f39887dae`，新 ID `91c58557-05ee-41c3-8a12-f28b76d48736`，history=success，生成 **1024×1024** PNG。客户端记录 **48.152 s**，包含重启后的模型加载/编译，属于 cold，不与此前 29.925 s warm 混比。
5. 推理结束后 SIGPIPE 仍为忽略状态，证明首次模型加载/编译没有再次覆盖策略。

服务器调查目录 `/home/sonny/gaudi2/comfyui-hpu/incident-2026-11-10-sigpipe`，保留信号/DFA 摘要、重放和断连结果；本地结果在被 git 忽略的 `comfyui-hpu/results/sigpipe-2026-11-10/`。没有触碰原始 `/home/sonny/ComfyUI`，没有下载或改写模型。
