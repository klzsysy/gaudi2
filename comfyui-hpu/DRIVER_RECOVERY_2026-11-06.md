# 重启后 Gaudi2 驱动不可用：2026-11-06

服务器：`sonny@192.168.10.21`。时间均为 Asia/Shanghai。

## 原因与证据

本次重启从 `5.15.0-186-generic` 切换到了 `5.15.0-202-generic`，但 Habana DKMS 驱动此前只为旧内核安装。PCI 总线仍枚举到 `1da3:1020`，并不代表驱动已绑定设备。

故障时实际状态：

- `uname -r`：`5.15.0-202-generic`。
- `dkms status -m habanalabs`：只有 `1.24.2-481, 5.15.0-186-generic: installed`。
- `lsmod` 虽然有 `habanalabs`，但新内核 `modinfo` 指向 `/lib/modules/5.15.0-202-generic/kernel/drivers/misc/habanalabs/habanalabs.ko`；这是 `intree: Y` 的内核自带驱动，PCI alias 只有 `0001/1000/1010`，没有 Gaudi2 的 `1020`。
- `lspci -nnk -s 01:00.0` 没有 `Kernel driver in use`，没有 `/dev/accel` 设备节点。
- 启动日志显示 `habanalabs: unknown parameter 'nic_ports_ext_mask' ignored`，进一步说明加载的不是原先支持该选项的厂商驱动。
- `hl-smi` 报 `habanalabs driver is not loaded or no AIPs available, aborting...`。

包管理日志与安装脚本说明了为何漏掉新内核：

1. 11 月 5 日 09:21 安装了 `5.15.0-202` 内核和 headers，当时仍运行 `186` 内核。
2. 同日 14:34–14:35 删除并重装 `habanalabs-dkms 1.24.2-481`。
3. `/var/lib/dpkg/info/habanalabs-dkms.postinst` 调用不带 `-k` 的 `dkms build/install`，只重新安装当时运行内核的驱动。当前 DKMS 清单证实新内核条目缺失。
4. 11 月 6 日 08:39 重启进入 `202` 后，只有不支持 Gaudi2 的内核自带驱动可供加载。

不是 ComfyUI/Python venv 问题，也没有证据表明卡从 PCI 总线消失。`iommu=pt` 仍有效，Secure Boot 已关闭；新内核 headers 和驱动源代码都完整存在。

## 已执行修复

未重启主机，未修改原 `/home/sonny/ComfyUI`、任何 Python 环境或模型文件。没有正在使用 HPU 的进程，原内核自带模块引用计数为 0。

为当前内核单独编译安装已存在的驱动版本：

```bash
sudo dkms install -m habanalabs -v 1.24.2-481 -k 5.15.0-202-generic
```

编译成功，五个模块进入 `/lib/modules/5.15.0-202-generic/updates/dkms/`，DKMS 已执行 `depmod`。没有重新安装整套软件栈，也没有改 NVIDIA DKMS 条目。

卸载未绑定设备的旧模块，按现有单卡服务器配置加载厂商模块：

```bash
sudo modprobe -r habanalabs
sudo modprobe habanalabs_compat
sudo modprobe habanalabs_cn
sudo modprobe habanalabs_en
sudo modprobe habanalabs_ib
sudo modprobe habanalabs
```

09:59:33 开始加载，10:00:05 完成固件加载并绑定 Gaudi2。加载过程约 32 秒，不应在此期间重复启动持卡进程。

备份并刷新当前内核的 initramfs：

```bash
# 本次实际备份，保留在服务器：
# /var/backups/gaudi2-kernel-202.QkkZ2t/initrd.img-5.15.0-202-generic
sudo update-initramfs -u -k 5.15.0-202-generic
```

刷新前的 initramfs 中没有 Habana `.ko`，因此本次故障不是启动镜像内旧模块残留所致。刷新完成，仅有 ASPEED 显示模块缺少 `ast_dp501_fw.bin` 的警告，与本次 HPU 恢复无关。

现有 `/etc/modules-load.d/habanalabs.conf` 已配置上述五模块加载顺序；`/etc/modprobe.d/habanalabs-options.conf` 保持 `nic_ports_ext_mask=0`，未更改。

## 验证与边界

- `dkms status -m habanalabs`：`186` 和 `202` 两个内核均为 installed。
- 当前 `modinfo` 指向 `updates/dkms/habanalabs.ko`，版本 `1.24.2-52a555d`，alias 包含 `1da3:1020`。
- `lspci` 显示 `Kernel driver in use: habanalabs`。
- `/dev/accel/accel0`、`accel_controlD0` 已出现，设备状态为 `Operational`。
- `hl-smi` 正常返回 `HL-225`、`98304 MiB` 显存、驱动和 NIC 驱动 `1.24.2-52a555d`，无计算进程。
- 使用原 Habana venv 验证小型 BF16 HPU 矩阵乘法，与 CPU 参考精确一致；没有加载生成模型。
- 没有再次重启验证完整开机链路；已完成磁盘模块安装、依赖解析、启动配置检查和 initramfs 刷新。
- 本轮没有启动 ComfyUI 服务。已有服务未配置开机自启，需要使用独立目录的 `./start-hpu.sh` 启动。

编译命令日志：服务器 `/tmp/gaudi2-dkms-202-recovery.log`（临时文件可能被清理）；DKMS 构建日志位于 `/var/lib/dkms/habanalabs/1.24.2-481/5.15.0-202-generic/x86_64/log/`。本调查仓库的 `results/driver-recovery-2026-11-06/` 保存诊断摘录，按仓库规则不进入 Git。

## 防止再次发生

现有 `AUTOINSTALL=yes` 和内核/header 的 DKMS post-install hook 均已存在且可执行；`/etc/dkms/framework.conf.d/autoinstall_all_kernels.conf` 也已设置 `autoinstall_all_kernels="yes"`。这些配置不意味着厂商驱动重装脚本会自动补装所有已经安装的内核。

重装 Habana 驱动、更新内核之后，重启之前，检查**下一次实际要启动的内核**，不要只检查 `uname -r`：

```bash
dkms status -m habanalabs
modinfo -k 5.15.0-202-generic -F filename habanalabs
modinfo -k 5.15.0-202-generic -F version habanalabs
modinfo -k 5.15.0-202-generic -F alias habanalabs
```

将上述版本替换成目标内核。如果缺失对应 installed 条目，确保该内核 headers 已安装，再执行针对它的 `dkms install -m habanalabs -v 1.24.2-481 -k <目标内核>` 并刷新对应 initramfs。必须看到厂商模块路径和 Gaudi2 alias，再安排重启。本次保留旧 `186` 内核及其 Habana 驱动作为回退条件，没有锁定内核或禁止系统更新。

## 安装脚本缺陷与修复（同日后续）

用户要求检查并修复 `/home/sonny/gaudi2/gaudi2-single-card-driver/scripts/install-gaudi2.sh`。已确认脚本存在直接相关的缺陷，基线为该仓库 `ecbc579218cd2a5c454bf682c322cd4c0e176c30`：

- 阶段 4 使用 `KVER=$(uname -r)`，仅针对当前内核构建和验证。
- `GAUDI_REUSE_DRIVER=1` 时完全跳过 DKMS 阶段，当前设备健康会掩盖其它可启动内核缺驱动的问题。
- 阶段 5 的 `update-initramfs -u -k "$KVER"` 同样只覆盖当前内核。
- 最终成功条件也只检查活动设备，未证明重启到新内核后仍可用。前一节描述的厂商 postinst 行为没有被此封装脚本补齐。

已直接修改指定服务器 checkout，保留原有不相关未跟踪文件：

1. `scripts/install-gaudi2.sh`：枚举当前内核与 `/boot/vmlinuz-*` 的有效镜像，安装模式先检查全部 headers，再进入包卸载/重装阶段；将活动模块是否重载与各内核磁盘模块是否完整分开判断。
2. `scripts/lib/gaudi-dkms.sh`：按内核检查精确 DKMS 包版本的 installed 状态、五个模块、非空文件、vermagic、DKMS 路径、Gaudi2 PCI alias，以及主模块两个补丁字符串。只对不合格内核按 build→install→depmod 顺序修复，安装后再次校验；逐个更新 initramfs。任一步失败返回非零。忽略没有 `/boot` 镜像的残留模块目录。
3. `tests/test-gaudi-dkms.sh`：19 项 Linux/Bash 回归测试，用临时文件和模拟系统命令，不修改实际驱动。
4. `README.md`：补充多内核检查、重启前命令及测试说明，同时更正旧 README 中与现有参数行为不符的说明。

验证：Bash 语法检查、`git diff --check` 和 19 项回归全部通过；覆盖旧内核健康/新内核缺失、全部健康不重建、verify-only 无写操作、缺少 headers/辅助模块、空模块、内核自带模块、错误 vermagic/alias/补丁/包版本、DKMS 查询/构建/安装/安装后校验失败、多个 initramfs 与失败中止。已在实际部署路径执行：

```bash
sudo bash /home/sonny/gaudi2/gaudi2-single-card-driver/scripts/install-gaudi2.sh \
  --verify-only --skip-pytorch
```

退出码 0，两个实际内核都通过，设备 Operational、运行库链接正常。未为测试故意卸载正确驱动；此次新内核缺失场景由模拟命令复现，上一轮真实 DKMS 补装已经验证编译安装可成功。本轮没有再次执行完整安装模式、生成模型或重启。

备份：`/home/sonny/gaudi2/installer-backup.5PMyzd/original-files.tar`，含原安装脚本与 README；同目录 `base-commit.txt` 记录基线。恢复时在驱动仓库根目录解包该 tar 即恢复原入口及 README，新 helper/tests 可保留但旧入口不会调用它们。

本地可审查补丁：[install-gaudi2-all-boot-kernels.patch](patches/install-gaudi2-all-boot-kernels.patch)。补丁适用于上述基线，包含四个文件；服务器已经应用，不要重复应用。代码及 README 已于同日提交为 `8d291ee568e84291a7eee48b2515e345f076ebbb`，未 push。验证日志位于本调查仓库忽略的 `results/driver-recovery-2026-11-06/installer-{regression,verify}.log`。


## README 硬件补充与提交

用户要求补充驱动仓库 README 的 `Validated setup` 和 `Before installation` 后 commit。通过只读 DMI、lscpu、lspci 和驱动查询确认：

- Supermicro H13SSL-N rev 1.01；AMI BIOS 3.8，DMI 日期 01/06/2026。
- AMD 工程样品标识 `100-000000897-03`；单路 32 核 / 64 线程。不推断其对应零售型号。
- 4 × 24 GB DDR5 registered DIMM，配置速率 4800 MT/s，总安装 96 GB，Linux 可见约 94 GiB。
- Gaudi2 HL-225，PCI ID `1da3:1020` rev 01；实际链路 16 GT/s ×16（Gen4 x16），显存 98304 MiB。
- Ubuntu 22.04.5 LTS，当前内核 5.15.0-202-generic；Python 3.10.12、Habana torch 2.13.0a0+git2469d18、plugin 1.24.2.481。
- iommu=pt / identity、nic_ports_ext_mask=0、Secure Boot disabled；两个目标内核的 headers/DKMS 均已就绪。

README 保留原 GIGABYTE 平台验证，新增独立本机配置表及安装前检查的本机结果列；明确电源电压/转接板接线未由软件测量、Secure Boot 签名启动未验证、修复后尚未再次重启。原 186 内核的模型结果来自此前独立 ComfyUI HPU 适配；本轮没有重新运行模型。

服务器驱动仓库提交：`8d291ee568e84291a7eee48b2515e345f076ebbb`，标题 `Fix DKMS coverage for all bootable kernels`。仅包含 README、安装入口、DKMS helper、回归测试四个文件；用户原有未跟踪 Qwen 文档/脚本/图片未加入。提交后 main 比 origin/main 领先 1，未 push。本地补丁已更新到该提交的完整差异。
