// Plan A RMW SRAM Attention runtime:
// - Single RMW section across all slots (prevents multi-bundle serialization)
// - Software-pipelined QK / Softmax / PV node creation & dependencies for slots >= 2
// - Logical 3D reshape around softmax_fwd_hf8 / softmax_fwd_bf16 to eliminate internal SRAM memcpy_dma
// - 4 persistent tensors (query, key, value, output) via logical split/concat
// - Supports tail-handling modes:
//     mask_mode = 0: standard (no extra softmax input; can output retained max/inv if retained=1)
//     mask_mode = 1: FUSED_ADD with [keys, 1, 1] bf16 mask tensor (buffer 4)
//     mask_mode = 2: validCountAxis = 0 with [1] int32 valid_count tensor (buffer 4)
// - C++ multi-head stream launcher (launch all 32 heads in one C call on default Synapse stream)

#include <synapse_api.h>
#include <synapse_common_types.hpp>
#include <perf_lib_layer_params.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Binding {
    std::string name;
    unsigned buffer; // 0=Q, 1=K, 2=V, 3=O, 4=Max/Mask/ValidCount, 5=Inv
    uint64_t offset;
    uint64_t id = 0;
    unsigned dims;
    uint64_t sizes[3];
};

struct PlanA {
    synGraphHandle graph = nullptr;
    synRecipeHandle recipe = nullptr;
    std::vector<synSectionHandle> sections;
    std::vector<Binding> bindings;
    std::vector<synLaunchTensorInfo> launch_info;
    uint64_t workspace = 0;
    unsigned queries = 0;
    unsigned keys = 0;
    unsigned group_heads = 1;
    uint64_t head_strides[6] = {};

    ~PlanA() {
        if (recipe) synRecipeDestroy(recipe);
        if (graph) synGraphDestroy(graph);
        for (auto s : sections) synSectionDestroy(s);
    }
};

static void check(synStatus status, const char* op) {
    if (status != synSuccess) {
        throw std::runtime_error(std::string(op) + " returned " + std::to_string(status));
    }
}

static synSectionHandle make_section(PlanA& plan, bool persistent, bool rmw) {
    synSectionHandle s = nullptr;
    check(synSectionCreate(&s, 0, plan.graph), "section");
    plan.sections.push_back(s);
    check(synSectionSetPersistent(s, persistent), "persistent");
    if (rmw) check(synSectionSetRMW(s, true), "rmw");
    return s;
}

static synTensor make_tensor(PlanA& plan, const std::string& name, unsigned dims,
                             uint64_t s0, uint64_t s1, uint64_t s2,
                             synDataType dtype, synSectionHandle sec, uint64_t offset) {
    synTensor t = nullptr;
    check(synTensorHandleCreate(&t, plan.graph, DATA_TENSOR, name.c_str()), "tensor");
    synTensorGeometry g{};
    g.dims = dims;
    g.sizes[0] = s0;
    g.sizes[1] = s1;
    g.sizes[2] = s2;
    check(synTensorSetGeometry(t, &g, synGeometryMaxSizes), "geometry");
    check(synTensorSetDeviceDataType(t, dtype), "dtype");
    if (sec) check(synTensorAssignToSection(t, sec, offset), "assign");
    return t;
}

static synTensor make_persistent(PlanA& plan, const std::string& name, unsigned buffer, uint64_t offset,
                                 unsigned dims, uint64_t s0, uint64_t s1, uint64_t s2,
                                 synDataType dtype, synSectionHandle sec) {
    auto t = make_tensor(plan, name, dims, s0, s1, s2, dtype, sec, offset);
    Binding b{name, buffer, offset, 0, dims, {s0, s1, s2}};
    plan.bindings.push_back(b);
    return t;
}

extern "C" void* build_plan_a(unsigned queries, unsigned keys, unsigned group_heads,
                              unsigned tile, unsigned slots, int fast, int retained,
                              int no_dma, int mask_mode, int use_split,
                              const char* recipe_name) {
    try {
        if (!queries || !keys || !group_heads || !tile || !slots) {
            throw std::runtime_error("invalid geometry");
        }
        const uint64_t mask_sram_bytes = (mask_mode == 1) ? (((uint64_t)keys * 2 + 255ULL) & ~255ULL) : 0ULL;
        const uint64_t score_bytes = ((uint64_t)keys * tile * 2 + 255ULL) & ~255ULL;
        if (mask_sram_bytes + slots * score_bytes > 16ULL * 1024ULL * 1024ULL) {
            throw std::runtime_error("RMW capacity exceeded");
        }

        auto plan = std::make_unique<PlanA>();
        plan->queries = queries;
        plan->keys = keys;
        plan->group_heads = group_heads;
        plan->head_strides[0] = (uint64_t)queries * 128 * 2;
        plan->head_strides[1] = (uint64_t)keys * 128 * 2;
        plan->head_strides[2] = (uint64_t)keys * 128 * 2;
        plan->head_strides[3] = (uint64_t)queries * 128 * 2;
        // For mask_mode 1 or 2, buffer 4 is shared across all heads (head_stride = 0).
        // For retained stats (1 element/row = 2 bytes), pad per-head stride to 128-byte alignment!
        const uint64_t stat_head_stride = ((uint64_t)queries * 2 + 127ULL) & ~127ULL;
        plan->head_strides[4] = (mask_mode > 0) ? 0ULL : stat_head_stride;
        plan->head_strides[5] = stat_head_stride;

        check(synGraphCreate(&plan->graph, synDeviceGaudi2), "graph");
        std::vector<synSectionHandle> io_secs;
        for (unsigned i = 0; i < 6; ++i) io_secs.push_back(make_section(*plan, true, false));
        synSectionHandle rmw_sec = make_section(*plan, false, true);

        unsigned tiles_per_head = (queries + tile - 1) / tile;
        unsigned num_ops = group_heads * tiles_per_head;
        unsigned split_axis = 1;
        unsigned split_axis0 = 0;

        std::vector<synTensor> k_h(group_heads), v_h(group_heads);
        std::vector<synTensor> q_t(num_ops), o_t(num_ops), m_t(num_ops), inv_t(num_ops);
        std::vector<unsigned> op_head(num_ops), op_rows(num_ops);

        synTensor mask_tensor = nullptr;
        if (mask_mode == 1) {
            auto mask_dram = make_persistent(*plan, "mask", 4, 0, 3, keys, 1, 1, syn_type_bf16, io_secs[4]);
            mask_tensor = make_tensor(*plan, "mask_sram", 3, keys, 1, 1, syn_type_bf16, rmw_sec, 0);
            check(synNodeCreate(plan->graph, &mask_dram, &mask_tensor, 1, 1, nullptr, 0, "memcpy", "copy_mask", nullptr, nullptr), "copy_mask");
        } else if (mask_mode == 2) {
            mask_tensor = make_persistent(*plan, "valid_count", 4, 0, 1, 1, 1, 1, syn_type_int32, io_secs[4]);
        }

        if (use_split) {
            auto q_full = make_persistent(*plan, "query", 0, 0, 2, 128, queries * group_heads, 1, syn_type_bf16, io_secs[0]);
            auto k_full = make_persistent(*plan, "key", 1, 0, 2, 128, keys * group_heads, 1, syn_type_bf16, io_secs[1]);
            auto v_full = make_persistent(*plan, "val", 2, 0, 2, 128, keys * group_heads, 1, syn_type_bf16, io_secs[2]);
            auto o_full = make_persistent(*plan, "output", 3, 0, 2, 128, queries * group_heads, 1, syn_type_bf16, io_secs[3]);
            std::vector<synTensor> m_h(group_heads, nullptr), inv_h(group_heads, nullptr);
            if (fast && retained && mask_mode == 0) {
                for (unsigned h = 0; h < group_heads; ++h) {
                    m_h[h] = make_persistent(*plan, "maximum_" + std::to_string(h), 4, h * stat_head_stride,
                                             no_dma ? 2 : 1, queries, 1, 1, syn_type_bf16, io_secs[4]);
                    inv_h[h] = make_persistent(*plan, "inverse_" + std::to_string(h), 5, h * stat_head_stride,
                                               no_dma ? 2 : 1, queries, 1, 1, syn_type_bf16, io_secs[5]);
                }
            }
            if (group_heads == 1) {
                k_h[0] = k_full;
                v_h[0] = v_full;
            } else {
                for (unsigned h = 0; h < group_heads; ++h) {
                    k_h[h] = make_tensor(*plan, "k_h" + std::to_string(h), 2, 128, keys, 1, syn_type_bf16, nullptr, 0);
                    v_h[h] = make_tensor(*plan, "v_h" + std::to_string(h), 2, 128, keys, 1, syn_type_bf16, nullptr, 0);
                }
                check(synNodeCreate(plan->graph, &k_full, k_h.data(), 1, group_heads, &split_axis, sizeof(split_axis), "split", "split_k", nullptr, nullptr), "split_k");
                check(synNodeCreate(plan->graph, &v_full, v_h.data(), 1, group_heads, &split_axis, sizeof(split_axis), "split", "split_v", nullptr, nullptr), "split_v");
            }
            unsigned op = 0;
            for (unsigned h = 0; h < group_heads; ++h) {
                for (unsigned start = 0; start < queries; start += tile, ++op) {
                    unsigned rows = std::min(tile, queries - start);
                    std::string suf = std::to_string(op);
                    q_t[op] = make_tensor(*plan, "q_" + suf, 2, 128, rows, 1, syn_type_bf16, nullptr, 0);
                    o_t[op] = make_tensor(*plan, "o_" + suf, 2, 128, rows, 1, syn_type_bf16, nullptr, 0);
                    if (fast && retained && mask_mode == 0) {
                        m_t[op] = make_tensor(*plan, "m_" + suf, no_dma ? 2 : 1, rows, 1, 1, syn_type_bf16, nullptr, 0);
                        inv_t[op] = make_tensor(*plan, "inv_" + suf, no_dma ? 2 : 1, rows, 1, 1, syn_type_bf16, nullptr, 0);
                    }
                    op_head[op] = h;
                    op_rows[op] = rows;
                }
            }
            if (num_ops == 1) {
                q_t[0] = q_full;
                o_t[0] = o_full;
                if (fast && retained && mask_mode == 0) {
                    m_t[0] = m_h[0];
                    inv_t[0] = inv_h[0];
                }
            } else {
                check(synNodeCreate(plan->graph, &q_full, q_t.data(), 1, num_ops, &split_axis, sizeof(split_axis), "split", "split_q", nullptr, nullptr), "split_q");
                check(synNodeCreate(plan->graph, o_t.data(), &o_full, num_ops, 1, &split_axis, sizeof(split_axis), "concat", "concat_o", nullptr, nullptr), "concat_o");
                if (fast && retained && mask_mode == 0) {
                    for (unsigned h = 0; h < group_heads; ++h) {
                        if (tiles_per_head == 1) {
                            m_t[h] = m_h[h];
                            inv_t[h] = inv_h[h];
                        } else {
                            check(synNodeCreate(plan->graph, &m_t[h * tiles_per_head], &m_h[h], tiles_per_head, 1, &split_axis0, sizeof(split_axis0), "concat", ("concat_m_" + std::to_string(h)).c_str(), nullptr, nullptr), "concat_m");
                            check(synNodeCreate(plan->graph, &inv_t[h * tiles_per_head], &inv_h[h], tiles_per_head, 1, &split_axis0, sizeof(split_axis0), "concat", ("concat_inv_" + std::to_string(h)).c_str(), nullptr, nullptr), "concat_inv");
                        }
                    }
                }
            }
        } else {
            unsigned op = 0;
            for (unsigned h = 0; h < group_heads; ++h) {
                const uint64_t kv_off = (uint64_t)h * keys * 128 * 2;
                k_h[h] = make_persistent(*plan, "k_h" + std::to_string(h), 1, kv_off, 2, 128, keys, 1, syn_type_bf16, io_secs[1]);
                v_h[h] = make_persistent(*plan, "v_h" + std::to_string(h), 2, kv_off, 2, 128, keys, 1, syn_type_bf16, io_secs[2]);
                for (unsigned start = 0; start < queries; start += tile, ++op) {
                    unsigned rows = std::min(tile, queries - start);
                    uint64_t row = (uint64_t)h * queries + start;
                    std::string suf = std::to_string(op);
                    q_t[op] = make_persistent(*plan, "q_" + suf, 0, row * 128 * 2, 2, 128, rows, 1, syn_type_bf16, io_secs[0]);
                    o_t[op] = make_persistent(*plan, "o_" + suf, 3, row * 128 * 2, 2, 128, rows, 1, syn_type_bf16, io_secs[3]);
                    if (fast && retained && mask_mode == 0) {
                        m_t[op] = make_persistent(*plan, "m_" + suf, 4, row * 2, 1, rows, 1, 1, syn_type_bf16, io_secs[4]);
                        inv_t[op] = make_persistent(*plan, "inv_" + suf, 5, row * 2, 1, rows, 1, 1, syn_type_bf16, io_secs[5]);
                    }
                    op_head[op] = h;
                    op_rows[op] = rows;
                }
            }
        }

        std::vector<synTensor> s_t(num_ops), sv_t(num_ops), pv_t(num_ops), p_t(num_ops);
        for (unsigned i = 0; i < num_ops; ++i) {
            unsigned rows = op_rows[i];
            unsigned slot = i % slots;
            uint64_t base_off = mask_sram_bytes + slot * score_bytes;
            std::string suf = std::to_string(i);
            s_t[i] = make_tensor(*plan, "s_" + suf, 2, keys, rows, 1, syn_type_bf16, rmw_sec, base_off);
            p_t[i] = make_tensor(*plan, "p_" + suf, 2, keys, rows, 1, syn_type_bf16, rmw_sec, base_off);
            if (no_dma) {
                sv_t[i] = make_tensor(*plan, "sv_" + suf, 3, keys, rows, 1, syn_type_bf16, nullptr, 0);
                pv_t[i] = make_tensor(*plan, "pv_" + suf, 3, keys, rows, 1, syn_type_bf16, nullptr, 0);
            }
        }

        synGEMMParams qk_p(false, true), pv_p(false, false);
        ns_Softmax::Params ordinary{0};
        ns_Softmax::ParamsV8 accelerated{};
        accelerated.dim = 0;
        accelerated.dataTypeSize = DATA_32BIT;
        accelerated.validCountAxis = (mask_mode == 2) ? 0 : -1;
        SoftmaxMode_t mode_flags = SOFTMAX_HF8_1C;
        if (retained && mask_mode == 0) mode_flags = static_cast<SoftmaxMode_t>(mode_flags | SQUEEZED_RETAINED);
        if (mask_mode == 1) mode_flags = static_cast<SoftmaxMode_t>(mode_flags | FUSED_ADD);
        accelerated.mode = mode_flags;

        const void* sm_params = fast ? static_cast<const void*>(&accelerated) : static_cast<const void*>(&ordinary);
        unsigned sm_size = fast ? sizeof(accelerated) : sizeof(ordinary);
        const char* sm_guid = fast ? "softmax_fwd_hf8" : "softmax_fwd_bf16";

        std::vector<synNodeId> qk_id(num_ops), sm_id(num_ops), pv_id(num_ops);
        auto emit_qk = [&](unsigned i) {
            std::string suf = std::to_string(i);
            synTensor qk_in[] = {q_t[i], k_h[op_head[i]]};
            check(synNodeCreateWithId(plan->graph, qk_in, &s_t[i], 2, 1, &qk_p, sizeof(qk_p), "gemm", ("qk_" + suf).c_str(), &qk_id[i], nullptr, nullptr), "qk");
            if (no_dma) {
                check(synNodeCreate(plan->graph, &s_t[i], &sv_t[i], 1, 1, nullptr, 0, "reshape", ("r1_" + suf).c_str(), nullptr, nullptr), "r1");
            }
        };
        auto emit_sm_pv = [&](unsigned i) {
            std::string suf = std::to_string(i);
            std::vector<synTensor> sm_in{no_dma ? sv_t[i] : s_t[i]};
            if (mask_mode > 0) sm_in.push_back(mask_tensor);
            std::vector<synTensor> sm_out{no_dma ? pv_t[i] : p_t[i]};
            if (fast && retained && mask_mode == 0) {
                sm_out.push_back(m_t[i]);
                sm_out.push_back(inv_t[i]);
            }
            check(synNodeCreateWithId(plan->graph, sm_in.data(), sm_out.data(), sm_in.size(), sm_out.size(), sm_params, sm_size, sm_guid, ("sm_" + suf).c_str(), &sm_id[i], nullptr, nullptr), "sm");
            if (no_dma) {
                check(synNodeCreate(plan->graph, &pv_t[i], &p_t[i], 1, 1, nullptr, 0, "reshape", ("r2_" + suf).c_str(), nullptr, nullptr), "r2");
            }
            synTensor pv_in[] = {p_t[i], v_h[op_head[i]]};
            check(synNodeCreateWithId(plan->graph, pv_in, &o_t[i], 2, 1, &pv_p, sizeof(pv_p), "gemm", ("pv_" + suf).c_str(), &pv_id[i], nullptr, nullptr), "pv");
        };

        if (slots >= 2) {
            emit_qk(0);
            for (unsigned i = 1; i < num_ops; ++i) {
                emit_qk(i);
                emit_sm_pv(i - 1);
            }
            emit_sm_pv(num_ops - 1);
        } else {
            for (unsigned i = 0; i < num_ops; ++i) {
                emit_qk(i);
                emit_sm_pv(i);
            }
        }

        for (unsigned i = 0; i < num_ops; ++i) {
            if (i >= slots) {
                check(synNodeDependencySet(plan->graph, &pv_id[i - slots], &qk_id[i], 1, 1), "slot_reuse");
            }
            if (slots >= 2 && i >= 1) {
                check(synNodeDependencySet(plan->graph, &qk_id[i - 1], &qk_id[i], 1, 1), "qk_order");
                check(synNodeDependencySet(plan->graph, &qk_id[i], &pv_id[i - 1], 1, 1), "qk_before_prev_pv");
                check(synNodeDependencySet(plan->graph, &pv_id[i - 1], &pv_id[i], 1, 1), "pv_order");
            }
        }

        check(synGraphCompile(&plan->recipe, plan->graph, recipe_name, nullptr), "compile");
        check(synWorkspaceGetSize(&plan->workspace, plan->recipe), "workspace");

        std::vector<const char*> names;
        std::vector<uint64_t> ids(plan->bindings.size());
        for (const auto& b : plan->bindings) names.push_back(b.name.c_str());
        check(synTensorRetrieveIds(plan->recipe, names.data(), ids.data(), ids.size()), "tensor_ids");
        plan->launch_info.resize(plan->bindings.size());
        for (size_t i = 0; i < plan->bindings.size(); ++i) {
            plan->bindings[i].id = ids[i];
            synLaunchTensorInfo info{};
            info.tensorName = plan->bindings[i].name.c_str();
            info.tensorId = ids[i];
            info.tensorType = DATA_TENSOR;
            for (unsigned d = 0; d < plan->bindings[i].dims; ++d) {
                info.tensorSize[d] = plan->bindings[i].sizes[d];
            }
            plan->launch_info[i] = info;
        }
        return plan.release();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "PLAN_A_BUILD_ERROR %s\n", e.what());
        return nullptr;
    }
}

extern "C" uint64_t plan_a_workspace_bytes(void* handle) {
    return static_cast<PlanA*>(handle)->workspace;
}

extern "C" int launch_plan_a(void* handle, void* stream, const uint64_t* base_addresses,
                             uint64_t workspace, unsigned total_heads) {
    auto& plan = *static_cast<PlanA*>(handle);
    const size_t n = plan.bindings.size();
    synLaunchTensorInfo* info = plan.launch_info.data();
    for (unsigned h = 0; h < total_heads; h += plan.group_heads) {
        for (size_t i = 0; i < n; ++i) {
            const auto& b = plan.bindings[i];
            info[i].pTensorAddress = base_addresses[b.buffer] + b.offset + (uint64_t)h * plan.head_strides[b.buffer];
        }
        synStatus st = synLaunch(static_cast<synStreamHandle>(stream), info, n, workspace, plan.recipe, 0);
        if (st != synSuccess) return static_cast<int>(st);
    }
    return 0;
}

extern "C" void destroy_plan_a(void* handle) {
    delete static_cast<PlanA*>(handle);
}
