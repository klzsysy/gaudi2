// Build and execute attention recipes using the installed Synapse runtime.
#include <synapse_api.h>
#include <synapse_common_types.hpp>
#include <perf_lib_layer_params.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Binding {
    std::string name;
    unsigned buffer;
    uint64_t offset;
    uint64_t id = 0;
    unsigned dims;
    uint64_t sizes[2];
};

struct Plan {
    synGraphHandle graph = nullptr;
    synRecipeHandle recipe = nullptr;
    std::vector<synSectionHandle> sections;
    std::vector<Binding> bindings;
    uint64_t workspace = 0;
    ~Plan() {
        if (recipe) synRecipeDestroy(recipe);
        if (graph) synGraphDestroy(graph);
        for (auto section : sections) synSectionDestroy(section);
    }
};

static void check(synStatus status, const char* operation) {
    if (status != synSuccess) {
        throw std::runtime_error(std::string(operation) + " returned " + std::to_string(status));
    }
}

static synTensor tensor(Plan& plan, const std::string& name, uint64_t columns, uint64_t rows,
                        synDataType dtype, synSectionHandle section, uint64_t offset) {
    synTensor value = nullptr;
    check(synTensorHandleCreate(&value, plan.graph, DATA_TENSOR, name.c_str()), "tensor");
    synTensorGeometry geometry = {};
    geometry.dims = rows ? 2 : 1;
    geometry.sizes[0] = columns;
    if (rows) geometry.sizes[1] = rows;
    check(synTensorSetGeometry(value, &geometry, synGeometryMaxSizes), "geometry");
    check(synTensorSetDeviceDataType(value, dtype), "dtype");
    if (section) check(synTensorAssignToSection(value, section, offset), "assign");
    return value;
}

static synSectionHandle section(Plan& plan, bool persistent, bool rmw) {
    synSectionHandle value = nullptr;
    check(synSectionCreate(&value, 0, plan.graph), "section");
    plan.sections.push_back(value);
    check(synSectionSetPersistent(value, persistent), "persistent");
    if (rmw) check(synSectionSetRMW(value, true), "rmw");
    return value;
}

static synTensor persistent(Plan& plan, const std::string& name, unsigned buffer, uint64_t offset,
                            uint64_t columns, uint64_t rows, synSectionHandle memory) {
    auto value = tensor(plan, name, columns, rows, syn_type_bf16, memory, offset);
    Binding binding{name, buffer, offset, 0, rows ? 2u : 1u, {columns, rows}};
    plan.bindings.push_back(binding);
    return value;
}

extern "C" void* build_attention(unsigned queries, unsigned keys, unsigned heads,
                                  unsigned tile, unsigned slots, int alias, int fast, int retained,
                                  const char* recipe_name) {
    try {
        if (!queries || !keys || !heads || !tile || !slots) throw std::runtime_error("empty geometry");
        const uint64_t score_bytes = static_cast<uint64_t>(keys) * tile * 2;
        const uint64_t slot_bytes = score_bytes * (alias ? 1 : 2);
        if (slot_bytes * slots > 16 * 1024 * 1024) throw std::runtime_error("RMW capacity exceeded");
        auto plan = std::make_unique<Plan>();
        check(synGraphCreate(&plan->graph, synDeviceGaudi2), "graph");
        std::vector<synSectionHandle> inputs;
        for (unsigned index = 0; index < 6; ++index) inputs.push_back(section(*plan, true, false));
        std::vector<synSectionHandle> scratch;
        for (unsigned index = 0; index < slots; ++index) scratch.push_back(section(*plan, false, true));
        std::vector<synNodeId> previous(slots, 0);
        synNodeId last_qk = 0, last_pv = 0;
        const bool pipeline = std::getenv("QWEN_RMW_PIPELINE") != nullptr;
        if (pipeline && slots < 2) throw std::runtime_error("pipeline requires two slots");
        unsigned operation = 0;
        const synGEMMParams qk_params(false, true), pv_params(false, false);
        const ns_Softmax::Params ordinary{0};
        ns_Softmax::ParamsV8 accelerated{};
        accelerated.dim = 0;
        accelerated.dataTypeSize = DATA_32BIT;
        accelerated.validCountAxis = -1;
        accelerated.mode = static_cast<SoftmaxMode_t>(SOFTMAX_HF8_1C | (retained ? SQUEEZED_RETAINED : 0));
        for (unsigned head = 0; head < heads; ++head) {
            const uint64_t kv_offset = static_cast<uint64_t>(head) * keys * 128 * 2;
            auto key = persistent(*plan, "key_" + std::to_string(head), 1, kv_offset, 128, keys, inputs[1]);
            auto value = persistent(*plan, "value_" + std::to_string(head), 2, kv_offset, 128, keys, inputs[2]);
            for (unsigned start = 0; start < queries; start += tile, ++operation) {
                const unsigned rows = std::min(tile, queries - start), slot = operation % slots;
                const uint64_t row = static_cast<uint64_t>(head) * queries + start;
                const std::string suffix = std::to_string(operation);
                auto query = persistent(*plan, "query_" + suffix, 0, row * 128 * 2, 128, rows, inputs[0]);
                auto output = persistent(*plan, "output_" + suffix, 3, row * 128 * 2, 128, rows, inputs[3]);
                const bool custom = fast >= 2;
                auto score = tensor(*plan, "scores_" + suffix, keys, rows,
                                    custom ? syn_type_fp16 : syn_type_bf16, scratch[slot], 0);
                auto probability = tensor(*plan, "probability_" + suffix, keys, rows, syn_type_bf16,
                                          scratch[slot], alias ? 0 : score_bytes);
                synTensor sum = nullptr;
                synTensor numerator = output;
                if (custom) {
                    sum = tensor(*plan, "sum_" + suffix, 1, rows, syn_type_float, nullptr, 0);
                    numerator = tensor(*plan, "numerator_" + suffix, 128, rows, syn_type_float, nullptr, 0);
                }
                synTensor qk_inputs[] = {query, key};
                synNodeId qk_node = 0, pv_node = 0;
                check(synNodeCreateWithId(plan->graph, qk_inputs, &score, 2, 1, &qk_params, sizeof(qk_params),
                      "gemm", ("qk_" + suffix).c_str(), &qk_node, nullptr, nullptr), "qk");
                std::vector<synTensor> softmax_outputs{probability};
                if (custom) {
                    softmax_outputs.push_back(sum);
                } else if (fast && retained) {
                    softmax_outputs.push_back(persistent(*plan, "maximum_" + suffix, 4, row * 2,
                                                        rows, 0, inputs[4]));
                    softmax_outputs.push_back(persistent(*plan, "inverse_" + suffix, 5, row * 2,
                                                        rows, 0, inputs[5]));
                }
                const void* softmax_params = custom ? nullptr : fast ? static_cast<const void*>(&accelerated) : static_cast<const void*>(&ordinary);
                const unsigned softmax_size = custom ? 0 : fast ? sizeof(accelerated) : sizeof(ordinary);
                const char* custom_guids[] = {"qwen_d2_softmax_f16", "qwen_d2_exp_f16", "qwen_d2_softmax_u4_f16", "qwen_d2_exp_u4_f16",
                                              "qwen_d2_core_f16", "qwen_d2_core_u4_f16"};
                const char* softmax_guid = custom ? custom_guids[fast - 2] : fast ? "softmax_fwd_hf8" : "softmax_fwd_bf16";
                check(synNodeCreate(plan->graph, &score, softmax_outputs.data(), 1, softmax_outputs.size(),
                      softmax_params, softmax_size, softmax_guid,
                      ("softmax_" + suffix).c_str(), nullptr, nullptr), "softmax");
                synTensor pv_inputs[] = {probability, value};
                check(synNodeCreateWithId(plan->graph, pv_inputs, &numerator, 2, 1, &pv_params, sizeof(pv_params),
                      "gemm", ("pv_" + suffix).c_str(), &pv_node, nullptr, nullptr), "pv");
                if (custom) {
                    auto quotient = tensor(*plan, "quotient_" + suffix, 128, rows, syn_type_float, nullptr, 0);
                    const synTensor divide_inputs[] = {numerator, sum};
                    check(synNodeCreate(plan->graph, divide_inputs, &quotient, 2, 1, nullptr, 0,
                          "div_fwd_f32", ("normalize_" + suffix).c_str(), nullptr, nullptr), "normalize");
                    check(synNodeCreate(plan->graph, &quotient, &output, 1, 1, nullptr, 0,
                          "cast_f32_to_bf16", ("output_cast_" + suffix).c_str(), nullptr, nullptr), "cast");
                }
                if (previous[slot]) {
                    check(synNodeDependencySet(plan->graph, &previous[slot], &qk_node, 1, 1), "dependency");
                }
                if (pipeline && last_qk) {
                    check(synNodeDependencySet(plan->graph, &last_qk, &qk_node, 1, 1), "query_order");
                    check(synNodeDependencySet(plan->graph, &qk_node, &last_pv, 1, 1), "query_ahead");
                    check(synNodeDependencySet(plan->graph, &last_pv, &pv_node, 1, 1), "value_order");
                }
                last_qk = qk_node;
                last_pv = pv_node;
                previous[slot] = pv_node;
            }
        }
        check(synGraphCompile(&plan->recipe, plan->graph, recipe_name, nullptr), "compile");
        check(synWorkspaceGetSize(&plan->workspace, plan->recipe), "workspace");
        std::vector<const char*> names;
        std::vector<uint64_t> ids(plan->bindings.size());
        for (const auto& binding : plan->bindings) names.push_back(binding.name.c_str());
        check(synTensorRetrieveIds(plan->recipe, names.data(), ids.data(), ids.size()), "tensor_ids");
        for (unsigned index = 0; index < ids.size(); ++index) plan->bindings[index].id = ids[index];
        std::printf("RMW_RECIPE nodes=%u bindings=%zu workspace=%lu\n", operation * 3, plan->bindings.size(), plan->workspace);
        std::fflush(stdout);
        return plan.release();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "RMW_BUILD_ERROR %s\n", error.what());
        return nullptr;
    }
}

extern "C" uint64_t workspace_bytes(void* handle) {
    return static_cast<Plan*>(handle)->workspace;
}

extern "C" int launch_attention(void* handle, void* stream, const uint64_t* addresses,
                                 uint64_t workspace, int reduced) {
    auto& plan = *static_cast<Plan*>(handle);
    std::vector<synLaunchTensorInfo> tensors;
    bool seen[6] = {};
    for (const auto& binding : plan.bindings) {
        if (reduced && seen[binding.buffer]) continue;
        seen[binding.buffer] = true;
        synLaunchTensorInfo info{};
        info.tensorName = binding.name.c_str();
        info.tensorId = binding.id;
        info.tensorType = DATA_TENSOR;
        info.pTensorAddress = addresses[binding.buffer] + binding.offset;
        for (unsigned index = 0; index < binding.dims; ++index) info.tensorSize[index] = binding.sizes[index];
        tensors.push_back(info);
    }
    return synLaunch(static_cast<synStreamHandle>(stream), tensors.data(), tensors.size(), workspace, plan.recipe, 0);
}

extern "C" void destroy_attention(void* handle) {
    delete static_cast<Plan*>(handle);
}
