#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <cstdint>
#include <cstdio>

using namespace tpc_lib_api;
extern "C" const unsigned char situ_start[], situ_end[], situ_quant_start[], situ_quant_end[];
const char* names[] = {"moss_situ_fast_bf16", "moss_situ_quant_fast_bf16"};

extern "C" uint64_t GetLibVersion() { return 1; }

extern "C" GlueCodeReturn GetKernelGuids(DeviceId device, uint32_t* count, GuidInfo* guids) {
    const uint32_t available = device == DEVICE_ID_GAUDI2 ? 2 : 0;
    const uint32_t capacity = *count;
    *count = available;
    if (guids) {
        for (uint32_t i = 0; i < available && i < capacity; ++i) {
            std::memset(&guids[i], 0, sizeof(GuidInfo));
            std::strcpy(guids[i].name, names[i]);
        }
    }
    return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn GetShapeInference(DeviceId device, const ShapeInferenceParams* params, ShapeInferenceOutput* result) {
    const char* name = (params->pGuid ? params->pGuid : &params->guid)->name;
    const bool quantized = std::strcmp(name, names[1]) == 0;
    if (device != DEVICE_ID_GAUDI2 || (!quantized && std::strcmp(name, names[0]) != 0)) return GLUE_NODE_NOT_FOUND;
    result->outputTensors[0]->geometry = params->inputTensors[0]->geometry;
    result->outputTensors[0]->geometry.maxSizes[0] /= 2;
    result->outputTensors[0]->geometry.minSizes[0] /= 2;
    if (quantized) {
        result->outputTensors[0]->geometry.dataType = DATA_F8_143;
        result->outputTensors[1]->geometry = params->inputTensors[0]->geometry;
        result->outputTensors[1]->geometry.dataType = DATA_F32;
        result->outputTensors[1]->geometry.maxSizes[0] = 1;
        result->outputTensors[1]->geometry.minSizes[0] = 1;
    }
    return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn InstantiateTpcKernel(const HabanaKernelParams* params, HabanaKernelInstantiation* result) {
    int kernel = std::strcmp(params->guid.name, names[0]) == 0 ? 0 :
                 std::strcmp(params->guid.name, names[1]) == 0 ? 1 : -1;
    if (kernel < 0 || params->deviceId != DEVICE_ID_GAUDI2) return GLUE_NODE_NOT_FOUND;
    if (params->inputTensorNr != 1) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorNr != unsigned(kernel + 1)) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto& input = params->inputTensors[0].geometry;
    const auto& output = params->outputTensors[0].geometry;
    const int columns = input.maxSizes[0] / 2;
    const int rows = input.maxSizes[1];
    std::fprintf(stderr, "MOSS_TPC_GLUE %s dims=%u width=%d rows=%d dtype=%d\n", params->guid.name, input.dims, columns, rows, int(input.dataType));
    if (input.dims != 2 || input.dataType != DATA_BF16 || columns <= 0 || columns % 256 || columns > 8192)
        return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    if (output.maxSizes[0] != unsigned(columns) || output.maxSizes[1] != unsigned(rows))
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    result->indexSpaceRank = 1;
    result->indexSpaceGeometry[0] = rows;
    result->auxiliaryTensorNr = 0;
    for (unsigned i = 0; i < params->inputTensorNr; ++i) {
        std::memset(&result->inputTensorAccessPattern[i], 0, sizeof(TensorAccessPattern));
        result->inputTensorAccessPattern[i].allRequired = 1;
    }
    for (unsigned i = 0; i < params->outputTensorNr; ++i) {
        std::memset(&result->outputTensorAccessPattern[i], 0, sizeof(TensorAccessPattern));
        result->outputTensorAccessPattern[i].allRequired = 1;
        result->outputTensorAccessPattern[i].noRmwAccess = 1;
    }
    const int scalars[] = {columns, rows};
    std::memcpy(result->kernel.scalarParams, scalars, sizeof(scalars));
    result->kernel.paramsNr = 2;
    const unsigned char* begin = kernel ? situ_quant_start : situ_start;
    const unsigned char* end = kernel ? situ_quant_end : situ_end;
    const uint32_t capacity = result->kernel.elfSize;
    result->kernel.elfSize = end - begin;
    if (!result->kernel.kernelElf || capacity < result->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(result->kernel.kernelElf, begin, result->kernel.elfSize);
    return GLUE_SUCCESS;
}
