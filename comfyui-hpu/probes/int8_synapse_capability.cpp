// Compile a single Gaudi2 GEMM without acquiring or launching on a device.
// This probe checks the installed Synapse compiler below the PyTorch bridge.
#include <synapse_api.h>
#include <synapse_common_types.hpp>

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(const char* operation, synStatus status) {
    char description[256] = {};
    synStatusGetBriefDescription(status, description, sizeof(description));
    std::cout << "{\"operation\":" << std::quoted(operation)
              << ",\"status\":" << static_cast<int>(status)
              << ",\"description\":" << std::quoted(description) << "}" << std::endl;
    if (status != synSuccess) {
        throw std::runtime_error(operation);
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: probe bf16|int8_int32|int8_bf16|int8_int8 default|inference|quantized recipe_path\n";
        return 2;
    }
    const std::string kind = argv[1];
    const std::string mode = argv[2];
    if (kind != "bf16" && kind != "int8_int32" && kind != "int8_bf16" && kind != "int8_int8") {
        return 2;
    }
    if (mode != "default" && mode != "inference" && mode != "quantized") {
        return 2;
    }
    const synDataType input_type = kind == "bf16" ? syn_type_bf16 : syn_type_int8;
    synDataType output_type = syn_type_bf16;
    if (kind == "int8_int32") output_type = syn_type_int32;
    if (kind == "int8_int8") output_type = syn_type_int8;
    synGraphHandle graph = nullptr;
    synRecipeHandle recipe = nullptr;
    std::vector<synSectionHandle> sections;
    bool initialized = false;
    int result = 0;
    try {
        check("initialize", synInitialize());
        initialized = true;
        check("graph_create_gaudi2", synGraphCreate(&graph, synDeviceGaudi2));
        if (mode != "default") {
            const synGraphAttribute attributes[] = {GRAPH_ATTRIBUTE_INFERENCE, GRAPH_ATTRIBUTE_QUANTIZATION};
            synGraphAttributeVal values[2] = {};
            values[0].iAttrVal = 1;
            values[1].iAttrVal = mode == "quantized";
            check("set_graph_attributes", synGraphSetAttributes(graph, attributes, values, 2));
        }
        synTensor tensors[3] = {};
        const char* names[] = {"input_a", "input_b", "output_c"};
        for (int index = 0; index < 3; ++index) {
            synSectionHandle section = nullptr;
            check("section_create", synSectionCreate(&section, 0, graph));
            sections.push_back(section);
            check("section_persistent", synSectionSetPersistent(section, true));
            check("tensor_create", synTensorHandleCreate(&tensors[index], graph, DATA_TENSOR, names[index]));
            synTensorGeometry geometry = {};
            geometry.dims = 2;
            geometry.sizes[0] = 128;
            geometry.sizes[1] = 128;
            check("tensor_geometry", synTensorSetGeometry(tensors[index], &geometry, synGeometryMaxSizes));
            check("tensor_dtype", synTensorSetDeviceDataType(tensors[index], index == 2 ? output_type : input_type));
            check("tensor_section", synTensorAssignToSection(tensors[index], section, 0));
        }
        const synGEMMParams parameters(false, false);
        check("node_create_gemm", synNodeCreate(graph, tensors, tensors + 2, 2, 1,
              &parameters, sizeof(parameters), "gemm", "native_type_probe", nullptr, nullptr));
        check("graph_compile", synGraphCompile(&recipe, graph, argv[3], nullptr));
    } catch (const std::exception& error) {
        std::cout << "{\"failed_at\":" << std::quoted(error.what()) << "}" << std::endl;
        result = 1;
    }
    if (recipe) synRecipeDestroy(recipe);
    if (graph) synGraphDestroy(graph);
    for (auto section : sections) synSectionDestroy(section);
    if (initialized) synDestroy();
    return result;
}
