// Compile attention placement experiments without acquiring or launching a device.
#include <synapse_api.h>
#include <synapse_common_types.hpp>
#include <perf_lib_layer_params.h>

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(const char* operation, synStatus status) {
    std::cout << "{\"operation\":" << std::quoted(operation)
              << ",\"status\":" << static_cast<int>(status) << "}" << std::endl;
    if (status != synSuccess) throw std::runtime_error(operation);
}

int main(int argc, char** argv) {
    if (argc != 5) return 2;
    const std::string mode = argv[1];
    const uint64_t queries = std::stoul(argv[2]);
    const uint64_t keys = std::stoul(argv[3]);
    synGraphHandle graph = nullptr;
    synRecipeHandle recipe = nullptr;
    std::vector<synSectionHandle> sections;
    int result = 0;
    try {
        check("initialize", synInitialize());
        const synDeviceAttribute attribute = DEVICE_ATTRIBUTE_MAX_RMW_SIZE;
        uint64_t maximum = 0;
        check("rmw_capacity", synDeviceTypeGetAttribute(&maximum, &attribute, 1, synDeviceGaudi2));
        std::cout << "{\"max_rmw_bytes\":" << maximum << "}" << std::endl;
        check("create_graph", synGraphCreate(&graph, synDeviceGaudi2));
        synTensor tensors[6] = {};
        synSectionHandle shared = nullptr;
        const char* names[] = {"query", "key", "value", "scores", "probabilities", "output"};
        const uint64_t shapes[6][2] = {{128, queries}, {128, keys}, {128, keys},
                                       {keys, queries}, {keys, queries}, {128, queries}};
        for (unsigned index = 0; index < 6; ++index) {
            check("tensor", synTensorHandleCreate(&tensors[index], graph, DATA_TENSOR, names[index]));
            synTensorGeometry geometry = {};
            geometry.dims = 2;
            geometry.sizes[0] = shapes[index][0];
            geometry.sizes[1] = shapes[index][1];
            check("geometry", synTensorSetGeometry(tensors[index], &geometry, synGeometryMaxSizes));
            check("dtype", synTensorSetDeviceDataType(tensors[index], syn_type_bf16));
            const bool persistent = index < 3 || index == 5;
            const bool rmw = (index == 3 && mode != "baseline") ||
                             (index == 4 && (mode == "both" || mode == "shared" || mode == "alias"));
            if (persistent || rmw) {
                synSectionHandle section = nullptr;
                if (index == 4 && shared) {
                    section = shared;
                } else {
                    check("section", synSectionCreate(&section, 0, graph));
                    sections.push_back(section);
                    check("persistent", synSectionSetPersistent(section, persistent));
                    if (rmw) check("set_rmw", synSectionSetRMW(section, true));
                    if (index == 3 && (mode == "shared" || mode == "alias")) shared = section;
                }
                const uint64_t offset = index == 4 && mode == "shared" ? keys * queries * 2 : 0;
                check("assign", synTensorAssignToSection(tensors[index], section, offset));
            }
        }
        const synGEMMParams qk(false, true), pv(false, false);
        check("qk", synNodeCreate(graph, tensors, tensors + 3, 2, 1, &qk, sizeof(qk),
                                  "gemm", "query_key", nullptr, nullptr));
        const ns_Softmax::Params softmax{0};
        check("softmax", synNodeCreate(graph, tensors + 3, tensors + 4, 1, 1, &softmax, sizeof(softmax),
                                       "softmax_fwd_bf16", "probabilities", nullptr, nullptr));
        const synTensor inputs[] = {tensors[4], tensors[2]};
        check("pv", synNodeCreate(graph, inputs, tensors + 5, 2, 1, &pv, sizeof(pv),
                                  "gemm", "probability_value", nullptr, nullptr));
        check("compile", synGraphCompile(&recipe, graph, argv[4], nullptr));
    } catch (const std::exception& error) {
        std::cout << "{\"failed_at\":" << std::quoted(error.what()) << "}" << std::endl;
        result = 1;
    }
    if (recipe) synRecipeDestroy(recipe);
    if (graph) synGraphDestroy(graph);
    for (auto section : sections) synSectionDestroy(section);
    synDestroy();
    return result;
}
