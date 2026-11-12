// Broadcast a tensor scalar through the vector pipeline.
float64 first_group(float64 x) {
    const int writes = SW_WR_LOWER_GROUP0 | SW_WR_UPPER_GROUP0 | SW_WR_LOWER_GROUP1 | SW_WR_UPPER_GROUP1 |
                       SW_WR_LOWER_GROUP2 | SW_WR_UPPER_GROUP2 | SW_WR_LOWER_GROUP3 | SW_WR_UPPER_GROUP3;
    return v_f32_mov_dual_group_all_b(x, 0xffffffff, 0, 0, 0, 0, writes, 0);
}
#define BROADCAST(T, P) v_f32_shuffle_b(first_group(v_f32_ld_tnsr_b(P, T)), (unsigned char)128, 0, 0)
__local__ float64 pk_cache[8][2], k_cache[8][2], q_cache[8][2];

// One index-space member owns four value columns of one FP32 recurrent state.
void main(tensor state, tensor previous_key, tensor previous_value,
          tensor previous_decay, tensor previous_beta, tensor query, tensor key,
          tensor value, tensor decay, tensor beta, tensor program,
          tensor committed, tensor output, int values, int heads, int batches,
          int previous_rows, int rows) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    int instructions[8];
    for (int order = 0; order < rows; ++order) {
        int5 position = {order, 0, 0, 0, 0};
        instructions[order] = s_i32_ld_g((__global__ int*)gen_addr(position, program));
    }
    int5 count_position = {rows, 0, 0, 0, 0};
    previous_rows = s_i32_ld_g((__global__ int*)gen_addr(count_position, program));
    for (int batch = begin[2]; batch < end[2] && batch < batches; ++batch) {
        for (int head = begin[1]; head < end[1] && head < heads; ++head) {
            int5 gate_position = {0, head, batch, 0, 0};
            float64 previous_decays = first_group(v_f32_ld_tnsr_b(gate_position, previous_decay));
            float64 previous_betas = first_group(v_f32_ld_tnsr_b(gate_position, previous_beta));
            float64 decays = first_group(v_f32_ld_tnsr_b(gate_position, decay));
            float64 betas = first_group(v_f32_ld_tnsr_b(gate_position, beta));
            for (int row = 0; row < previous_rows; ++row) {
                for (int part = 0; part < 2; ++part) {
                    int5 key_position = {part * 64, row, head, batch, 0};
                    pk_cache[row][part] = v_f32_ld_tnsr_b(key_position, previous_key);
                }
            }
            for (int row = 0; row < rows; ++row) {
                for (int part = 0; part < 2; ++part) {
                    int5 key_position = {part * 64, row, head, batch, 0};
                    k_cache[row][part] = v_f32_ld_tnsr_b(key_position, key);
                    q_cache[row][part] = v_f32_ld_tnsr_b(key_position, query);
                }
            }
            for (int block = begin[0]; block < end[0]; ++block) {
                float64 first[4], second[4];
                #pragma unroll
                for (int lane = 0; lane < 4; ++lane) {
                    int5 position = {0, block * 4 + lane, head, batch, 0};
                    first[lane] = v_f32_ld_tnsr_b(position, state);
                    position[0] = 64;
                    second[lane] = v_f32_ld_tnsr_b(position, state);
                }
                for (int row = 0; row < previous_rows; ++row) {
                    float64 d = v_f32_shuffle_b(previous_decays, (unsigned char)(128 + row), 0, 0);
                    float64 b = v_f32_shuffle_b(previous_betas, (unsigned char)(128 + row), 0, 0);
                    int5 value_position = {block * 4, row, head, batch, 0};
                    float64 value_group = first_group(v_f32_ld_tnsr_b(value_position, previous_value));
                    float64 k0 = pk_cache[row][0];
                    float64 k1 = pk_cache[row][1];
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) {
                        float64 v = v_f32_shuffle_b(value_group, (unsigned char)(128 + lane), 0, 0);
                        first[lane] *= d;
                        second[lane] *= d;
                        float64 prediction = v_f32_reduce_add(first[lane] * k0 + second[lane] * k1);
                        float64 update = (v - prediction) * b;
                        first[lane] += k0 * update;
                        second[lane] += k1 * update;
                    }
                }
                #pragma unroll
                for (int lane = 0; lane < 4; ++lane) {
                    int5 position = {0, block * 4 + lane, head, batch, 0};
                    v_f32_st_tnsr(position, committed, first[lane]);
                    position[0] = 64;
                    v_f32_st_tnsr(position, committed, second[lane]);
                }
                for (int order = 0; order < rows; ++order) {
                    int instruction = instructions[order];
                    int row = instruction & 15;
                    float64 d = v_f32_shuffle_b(decays, (unsigned char)(128 + row), 0, 0);
                    float64 b = v_f32_shuffle_b(betas, (unsigned char)(128 + row), 0, 0);
                    int5 value_position = {block * 4, row, head, batch, 0};
                    float64 value_group = first_group(v_f32_ld_tnsr_b(value_position, value));
                    float64 k0 = k_cache[row][0];
                    float64 q0 = q_cache[row][0];
                    float64 k1 = k_cache[row][1];
                    float64 q1 = q_cache[row][1];
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) {
                        float64 v = v_f32_shuffle_b(value_group, (unsigned char)(128 + lane), 0, 0);
                        float64 next_first = first[lane] * d;
                        float64 next_second = second[lane] * d;
                        float64 prediction = v_f32_reduce_add(next_first * k0 + next_second * k1);
                        float64 update = (v - prediction) * b;
                        next_first += k0 * update;
                        next_second += k1 * update;
                        float64 result = v_f32_reduce_add(next_first * q0 + next_second * q1);
                        int5 output_position = {block * 4 + lane, head, row, batch, 0};
                        v_f32_st_tnsr_partial(output_position, output, result, 0, 0);
                        if (instruction & 32) {
                            first[lane] = next_first;
                            second[lane] = next_second;
                        }
                    }
                }
            }
        }
    }
}
