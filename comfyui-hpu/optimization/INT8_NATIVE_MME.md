# Gaudi2 原生 INT8 MME 能力验证

2026-11-14（Asia/Shanghai）。结论为当前Gaudi2 HL-225／Synapse1.24.2不支持原生INT8 MME。用户要求有原生能力才继续，阴性结果触发通用INT8开发计划整体关闭。

## 已安装厂商库的能力声明

只读反汇编服务器`/usr/lib/habanalabs/libSynapse.so`的两个HAL方法，返回值均为固定立即数。

| 方法 | 返回掩码 | INT8位 |
|---|---|---|
| `Gaudi2HalReader::getSupportedTypes()` | `0x7f3f` | 有 |
| `Gaudi2HalReader::getSupportedMmeTypes()` | `0x7906` | 无 |

安装头文件`synapse_common_types.h`定义`syn_type_int8 = 1`。MME掩码包含BF16、FP32、FP16、TF32、HB_FLOAT、FP8_143和FP8_152，INT8、UINT8、INT16和INT32位均为零。

库SHA256为`b2cc69367001ac417a2f7b213aba46168e78492df902870b78fc34c73894bf50`。运行时版本为Synapse1.24.2，Synapse commit `79e284b`、MME `e90bf0c`、SCAL `d93c3e2`。函数地址和完整反汇编保存在结构化证据中，地址仅适用于这一库版本。

## 直接Synapse接口验证

probe以`synDeviceGaudi2`建立128×128矩阵的单个`gemm`图，使用persistent输入输出。调用层级为`synTensorHandleCreate`、`synTensorSetDeviceDataType`、`synNodeCreate`及`synGraphCompile`。

| 输入与输出 | 默认图 | inference图 | inference＋quantization图 |
|---|---|---|---|
| BF16×BF16→BF16 | 建图和编译成功 | 建图和编译成功 | 建图和编译成功 |
| INT8×INT8→INT32 | 建立GEMM节点失败 | 建立GEMM节点失败 | 建立GEMM节点失败 |
| INT8×INT8→BF16 | 建立GEMM节点失败 | 建立GEMM节点失败 | 建立GEMM节点失败 |
| INT8×INT8→INT8 | 建立GEMM节点失败 | 建立GEMM节点失败 | 建立GEMM节点失败 |

9个INT8用例均在`synNodeCreate`返回26／`synFail`，运行时日志记录`Can not add gemm generic node`。此前tensor创建、shape、dtype和section设置均成功。3个BF16对照均编译成功，确认测试图和SDK入口有效。

验证在目标设备的图创建/编译层完成，测试程序没有申请设备或launch recipe。节点阶段已拒绝INT8，因此本轮设备执行计时与输出数值比较不适用。服务检查前后PID均为1701354，队列为空。

## 官方资料交叉核对

官方Gaudi2架构列出MME浮点类型及FP32累加，白皮书TPC章节另列整数类型。PyTorch算子表Integer Types中`mm`、`matmul`的INT8栏为No。结合安装库HAL及直接SDK验证，当前原生INT8 MME的开发前提不成立。[官方架构](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html) · [Gaudi2白皮书](https://habana.ai/wp-content/uploads/pdf/2021/gaudi2-whitepaper.pdf) · [PyTorch算子表](https://docs.habana.ai/en/latest/PyTorch/Reference/Pytorch_Operators/Pytorch_Operators.html)

## 复现与归档

probe位于本仓库`probes/int8_synapse_capability.cpp`，构建时使用服务器已安装的Synapse头文件和库。示例命令在服务器调查目录执行。

```bash
g++ -std=c++17 -O2 -I/usr/include/habanalabs int8_synapse_capability.cpp \
  -L/usr/lib/habanalabs -Wl,-rpath,/usr/lib/habanalabs -lSynapse \
  -o int8_synapse_capability
GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so \
  ./int8_synapse_capability bf16 default ./bf16-control
GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so \
  ./int8_synapse_capability int8_int32 inference ./int8-test
objdump -d -C --start-address=0x1804ad0 --stop-address=0x1804af0 \
  /usr/lib/habanalabs/libSynapse.so
```

服务器资料目录为`/home/sonny/gaudi2/comfyui-hpu/performance-int8-native-2026-11-14`，包含源码、二进制、12组stdout/stderr、HAL反汇编及Synapse日志。结构化证据保存全部API状态、版本、hash和类型位解析。[验证源码](../probes/int8_synapse_capability.cpp) · [结构化证据](../evidence/int8-native-mme-2026-11-14.json)

开发状态为已停止、已放弃。原[开发计划](INT8_DEVELOPMENT_PLAN.md)保留为历史设计归档。
