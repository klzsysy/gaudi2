# 运行时、执行模式与故障定位

当前验证栈为Gaudi2、Habana1.24.2、eager＋局部`hpu_backend`。本篇保留影响性能解释、完整入口和可移植实现的故障经验，运行命令见 [环境](../ENVIRONMENT.md)。

## 执行模式和主机配置

本机`hpu/graphs.py`在eager模式拒绝HPUGraph。隔离lazy/Graph实验中，单block graph正/负6.880/6.326ms，lazy7.092/6.482ms，单shape capture约0.27秒。42层每请求预热/capture/replay/destroy的25次正负forward预算约15.24秒，eager14.141秒，首次35.54秒，链输出RRMSE正3.44%、负2.30%。lazy＋`hpu_backend`另报`habana::graph::GraphStorage::get()`缺失。

已有实验限制的是逐请求capture生命周期。固定shape长期buffer、一次capture多次replay需另验模型patch、卸载、动态控制和设备原地view行为。32→16 CPU线程完整流程21.060→21.168秒，无收益。NUMA/绑核/迁移尚无受控收益证据，单NUMA硬件事实见 [预算](BOTTLENECKS.md)。

早期源码复核发现HPU的异步prefetch和Comfy graph门控关闭，Kitchen deferred GDN走Torch参考实现。后来区域编译取得大幅收益，说明应沿实际调用链确认优化是否执行。`PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT`在已读bridge代码的调用位于lazy路径，当前eager收益还需独立验证。

## 异步输入、注册与编译故障

| 现象 | 已验证定位/处理 | 证据边界 |
|---|---|---|
| KV同storage行复制丢写 | 源行clone修复，关闭编译仍可复现 | 完整分支回滚验证见文本decode |
| MTP3四行GDN `synStatus26` | 逐段同步及逐组重建输入定位激活x，编译区外clone有效 | vendor内部原因未知，补八行只过离线不足以修复完整图 |
| 小M FP8 Linear编译重载失败 | 同步模式可运行，另发现额外舍入，整个候选撤回 | 栈位置随异步完成点变化，不能据最后一层定根因 |
| 共享动态SDPA段错误 | 单测通过、真实第6步失败，撤回共享范围 | 使用MOSS所属区域编译 |
| RoPE原地临时view测量崩溃 | 最终采用非原地公式、原生addcmul及单次注册 | 未改基线也曾段错误，底层原因尚未确定 |
| 运行中热换custom-op数值失真 | 每进程固定实现后logits逐元素一致 | marker证明回调执行仍不足以保证热替换对照有效 |
| profiler导出后退出`tcache`异常 | 数据已导出、卸载到基线，普通入口另验 | 两轮诊断均出现，根因未知 |
| 首个warm在profiler前段错误 | 改用已验deep wrapper后五轮/trace完成 | 两个launcher和失败日志均保留 |
| Device acquire failed | 等待持卡进程实际退出后再占卡 | 单卡任务串行，停止命令返回不代表资源已释放 |

这些失败覆盖不同触发条件，现有证据尚不足以合并为同一个底层故障。最小输入、版本、同步点及完整入口均需保留。

## SIGPIPE导致运行时失效

`3fec1143`服务浏览器取消文件下载后，DFA先记`exception 13 Broken pipe`，栈为`sendfile64`。随后加载embedding报`synStatus31 Host malloc failed`，当时主机约89GiB可用、HPU约773MiB，内存不足解释与现场不符。Habana初始化覆盖了Python的SIGPIPE忽略策略，断连使运行时进入fatal状态。

独立初始化HPU后对关闭socket执行`os.sendfile`，原策略使后续DMA失败、析构再抛错最终SIGABRT；恢复`SIG_IGN`后得到正常BrokenPipeError，HPU加法输出5并正常退出。`441d459d`在应用HPU/节点初始化后、HTTP启动前恢复SIGPIPE忽略策略。

真实静态文件路由8次下载后断连、一次注入SIGPIPE、原pending用户图重放均通过，首次模型加载后策略仍正确。cold48.152秒只作恢复验收。原始资料为`incident-2026-11-10-sigpipe`，复现入口为`hpu_sigpipe.py`和`hpu_server_recovery.py`。故障处理应先看最早DFA原因，再解释后续分配错误。

## 驱动和依赖恢复

- 重启进入内核5.15.0-202时只有186安装Habana DKMS，内核自带模块缺Gaudi2 PCI alias，`hl-smi`无设备。修复安装脚本枚举当前及全部有效`/boot/vmlinuz-*`，检查五个模块、vermagic、路径、alias、补丁字符串并刷新initramfs。19项模拟回归、186/202双内核verify、重启、HPU matmul及HTTP通过。[脚本补丁](../patches/install-gaudi2-all-boot-kernels.patch)
- 测试venv曾被CUDA torch2.14.1遮蔽，原服务仍持有旧环境，重启probe才暴露。移走冲突包备份后恢复`gaudi-base.pth`引用的Habana2.13.0a0，HPU smoke、266项回归和普通服务通过。安装来源未知，备份位置见环境。
- 普通pip resolver可能替换定制torch，独立venv继承厂商site-packages，按已有兼容依赖安装。启动成功后仍需核实实际torch路径、版本和HPU算子。

## 复用与证据

用户入口验收包括普通HTTP工作流、浏览器断连、自然MTP和卸载重载。特定运行时故障的重测需要触发条件、版本或实现变化，重复切换开关应由新的trace定位支持。

[Graph证据](../evidence/moss-bf16-completion-2026-11-11.json) · [四行GDN](../evidence/moss-bf16-next-2026-11-11.json) · [FP8重载](../evidence/moss-fp8-system-2026-11-12.json) · [profiler与首次warm故障](../evidence/moss-evidence-boundaries-2026-11-12.json) · [RoPE完整验收](../evidence/moss-three-stage-2026-11-13.json)
