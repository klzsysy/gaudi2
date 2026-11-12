#include <tpc_kernel_lib_interface.h>
#include <cstring>
#include <cstdint>

using namespace tpc_lib_api;
extern "C" const unsigned char gdn_start[], gdn_end[];
const char* guid = "moss_gdn_core_f32";

extern "C" uint64_t GetLibVersion() { return 1; }
extern "C" GlueCodeReturn GetKernelGuids(DeviceId device, uint32_t* count, GuidInfo* guids) {
    const uint32_t capacity = *count;
    *count = device == DEVICE_ID_GAUDI2 ? 1 : 0;
    if (guids && capacity && *count) {
        std::memset(guids, 0, sizeof(GuidInfo));
        std::strcpy(guids[0].name, guid);
    }
    return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn GetShapeInference(DeviceId device, const ShapeInferenceParams* p, ShapeInferenceOutput* out) {
    if (device != DEVICE_ID_GAUDI2 || std::strcmp((p->pGuid ? p->pGuid : &p->guid)->name, guid)) return GLUE_NODE_NOT_FOUND;
    out->outputTensors[0]->geometry = p->inputTensors[0]->geometry;
    auto& shape = out->outputTensors[1]->geometry;
    shape = p->inputTensors[0]->geometry;
    shape.maxSizes[0] = shape.minSizes[0] = p->inputTensors[0]->geometry.maxSizes[1];
    shape.maxSizes[1] = shape.minSizes[1] = p->inputTensors[0]->geometry.maxSizes[2];
    shape.maxSizes[2] = shape.minSizes[2] = p->inputTensors[5]->geometry.maxSizes[1];
    return GLUE_SUCCESS;
}
extern "C" GlueCodeReturn InstantiateTpcKernel(const HabanaKernelParams* p, HabanaKernelInstantiation* out) {
    if (p->deviceId != DEVICE_ID_GAUDI2 || std::strcmp(p->guid.name, guid)) return GLUE_NODE_NOT_FOUND;
    if (p->inputTensorNr != 11 || p->outputTensorNr != 2) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    const auto& state = p->inputTensors[0].geometry;
    if (state.dataType != DATA_F32 || state.dims != 4 || state.maxSizes[0] != 128) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const int parameters[] = {int(state.maxSizes[1]), int(state.maxSizes[2]), int(state.maxSizes[3]),
                              int(p->inputTensors[1].geometry.maxSizes[1]), int(p->inputTensors[5].geometry.maxSizes[1])};
    if (parameters[3] > 8 || parameters[4] > 8) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    out->indexSpaceRank = 3;
    for (unsigned i = 0; i < 3; ++i) out->indexSpaceGeometry[i] = parameters[i];
    out->indexSpaceGeometry[0] = (parameters[0] + 3) / 4;
    out->auxiliaryTensorNr = 0;
    for (unsigned i = 0; i < p->inputTensorNr; ++i) {
        std::memset(&out->inputTensorAccessPattern[i], 0, sizeof(TensorAccessPattern));
        out->inputTensorAccessPattern[i].allRequired = 1;
    }
    for (unsigned i = 0; i < p->outputTensorNr; ++i) {
        std::memset(&out->outputTensorAccessPattern[i], 0, sizeof(TensorAccessPattern));
        out->outputTensorAccessPattern[i].allRequired = 1;
        out->outputTensorAccessPattern[i].noRmwAccess = 1;
    }
    std::memcpy(out->kernel.scalarParams, parameters, sizeof(parameters));
    out->kernel.paramsNr = 5;
    const uint32_t capacity = out->kernel.elfSize;
    out->kernel.elfSize = gdn_end - gdn_start;
    if (!out->kernel.kernelElf || capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, gdn_start, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
