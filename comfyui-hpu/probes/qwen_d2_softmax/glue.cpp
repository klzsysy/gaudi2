#include <tpc_kernel_lib_interface.h>
#include <cstring>

using namespace tpc_lib_api;
extern "C" const unsigned char softmax_start[], softmax_end[];
extern "C" const unsigned char softmax_exp_start[], softmax_exp_end[];
extern "C" const unsigned char softmax_u4_start[], softmax_u4_end[];
extern "C" const unsigned char softmax_exp_u4_start[], softmax_exp_u4_end[];
extern "C" const unsigned char softmax_core_start[], softmax_core_end[];
extern "C" const unsigned char softmax_core_u4_start[], softmax_core_u4_end[];
static const char* guids[] = {"qwen_d2_softmax_f16", "qwen_d2_exp_f16", "qwen_d2_softmax_u4_f16", "qwen_d2_exp_u4_f16",
                             "qwen_d2_core_f16", "qwen_d2_core_u4_f16"};

static int variant(const char* name) {
    for (int index = 0; index < 6; ++index) if (!std::strcmp(name, guids[index])) return index;
    return -1;
}

extern "C" uint64_t GetLibVersion() { return 1; }
extern "C" GlueCodeReturn GetKernelGuids(DeviceId device, uint32_t* count, GuidInfo* output) {
    const uint32_t capacity = *count;
    *count = device == DEVICE_ID_GAUDI2 ? 6 : 0;
    if (output && capacity >= *count && *count) {
        std::memset(output, 0, sizeof(GuidInfo) * 6);
        for (unsigned index = 0; index < 6; ++index) std::strcpy(output[index].name, guids[index]);
    }
    return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn GetShapeInference(DeviceId device, const ShapeInferenceParams* p, ShapeInferenceOutput* out) {
    if (device != DEVICE_ID_GAUDI2 || variant((p->pGuid ? p->pGuid : &p->guid)->name) < 0) return GLUE_NODE_NOT_FOUND;
    out->outputTensors[0]->geometry = p->inputTensors[0]->geometry;
    out->outputTensors[0]->geometry.dataType = DATA_BF16;
    auto& shape = out->outputTensors[1]->geometry;
    shape = p->inputTensors[0]->geometry;
    shape.maxSizes[0] = shape.minSizes[0] = 1;
    shape.dataType = DATA_F32;
    return GLUE_SUCCESS;
}

extern "C" GlueCodeReturn InstantiateTpcKernel(const HabanaKernelParams* p, HabanaKernelInstantiation* out) {
    const int selected = variant(p->guid.name);
    if (p->deviceId != DEVICE_ID_GAUDI2 || selected < 0) return GLUE_NODE_NOT_FOUND;
    if (p->inputTensorNr != 1 || p->outputTensorNr != 2) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    const auto& shape = p->inputTensors[0].geometry;
    if (shape.dims != 2 || shape.dataType != DATA_F16 || p->outputTensors[0].geometry.dataType != DATA_BF16 ||
        p->outputTensors[1].geometry.dataType != DATA_F32) return GLUE_INCOMPATIBLE_DATA_TYPE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = shape.maxSizes[1];
    out->auxiliaryTensorNr = 0;
    std::memset(&out->inputTensorAccessPattern[0], 0, sizeof(TensorAccessPattern));
    auto& input = out->inputTensorAccessPattern[0];
    input.mapping[0] = {0, 0, 0, static_cast<float>(shape.maxSizes[0]-1)};
    input.mapping[1] = {0, 1, 0, 0};
    for (unsigned index = 0; index < 2; ++index) {
        std::memset(&out->outputTensorAccessPattern[index], 0, sizeof(TensorAccessPattern));
        auto& output = out->outputTensorAccessPattern[index];
        output.mapping[0] = {0, 0, 0, index ? 0.0f : static_cast<float>(shape.maxSizes[0]-1)};
        output.mapping[1] = {0, 1, 0, 0};
        output.noRmwAccess = 1;
        output.fullyWritten = 1;
    }
    out->outputTensorAccessPattern[0].inputsReusability = 1;
    out->outputTensorAccessPattern[0].inputReusabilityBinding = 1;
    out->kernel.paramsNr = 0;
    const uint32_t capacity = out->kernel.elfSize;
    const unsigned char* starts[] = {softmax_start, softmax_exp_start, softmax_u4_start, softmax_exp_u4_start,
                                    softmax_core_start, softmax_core_u4_start};
    const unsigned char* ends[] = {softmax_end, softmax_exp_end, softmax_u4_end, softmax_exp_u4_end,
                                  softmax_core_end, softmax_core_u4_end};
    out->kernel.elfSize = ends[selected]-starts[selected];
    if (!out->kernel.kernelElf || capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, starts[selected], out->kernel.elfSize);
    return GLUE_SUCCESS;
}
