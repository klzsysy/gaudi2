float64 first_group(float64 x) {
    const int writes = SW_WR_LOWER_GROUP0 | SW_WR_UPPER_GROUP0 | SW_WR_LOWER_GROUP1 | SW_WR_UPPER_GROUP1 |
                       SW_WR_LOWER_GROUP2 | SW_WR_UPPER_GROUP2 | SW_WR_LOWER_GROUP3 | SW_WR_UPPER_GROUP3;
    return v_f32_mov_dual_group_all_b(x, 0xffffffff, 0, 0, 0, 0, writes, 0);
}

// Preserve the parallel FP32 algebra used by _commit and _tree_output.
// Each member owns four value columns and the corresponding 128-element states.
void main(tensor state, tensor previous_key, tensor previous_value,
          tensor previous_decay, tensor previous_beta, tensor query, tensor key,
          tensor value, tensor decay, tensor beta, tensor program,
          tensor committed, tensor output, int values, int heads, int batches,
          int previous_rows, int rows) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    int5 count_position = {1, 0, 0, 0, 0};
    const int pending = s_i32_ld_g((__global__ int*)gen_addr(count_position, program));
    for (int batch = begin[2]; batch < end[2] && batch < batches; ++batch) {
        for (int head = begin[1]; head < end[1] && head < heads; ++head) {
            int5 gate_position = {0, head, batch, 0, 0};
            float64 pd = v_f32_shuffle_b(first_group(v_f32_ld_tnsr_b(gate_position, previous_decay)), 128, 0, 0);
            float64 pb = v_f32_shuffle_b(first_group(v_f32_ld_tnsr_b(gate_position, previous_beta)), 128, 0, 0);
            float64 d = v_f32_shuffle_b(first_group(v_f32_ld_tnsr_b(gate_position, decay)), 128, 0, 0);
            float64 b = v_f32_shuffle_b(first_group(v_f32_ld_tnsr_b(gate_position, beta)), 128, 0, 0);
            int5 position = {0, 0, head, batch, 0};
            float64 pk0 = v_f32_ld_tnsr_b(position, previous_key);
            float64 q0 = v_f32_ld_tnsr_b(position, query);
            float64 k0 = v_f32_ld_tnsr_b(position, key);
            position[0] = 64;
            float64 pk1 = v_f32_ld_tnsr_b(position, previous_key);
            float64 q1 = v_f32_ld_tnsr_b(position, query);
            float64 k1 = v_f32_ld_tnsr_b(position, key);
            float64 qk = v_f32_reduce_add(q0 * k0 + q1 * k1);
            for (int block = begin[0]; block < end[0]; ++block) {
                int5 value_position = {block * 4, 0, head, batch, 0};
                float64 previous_values = first_group(v_f32_ld_tnsr_b(value_position, previous_value));
                float64 current_values = first_group(v_f32_ld_tnsr_b(value_position, value));
                #pragma unroll
                for (int lane = 0; lane < 4; ++lane) {
                    int5 state_position = {0, block * 4 + lane, head, batch, 0};
                    float64 s0 = v_f32_ld_tnsr_b(state_position, state);
                    state_position[0] = 64;
                    float64 s1 = v_f32_ld_tnsr_b(state_position, state);
                    if (pending) {
                        float64 pv = v_f32_shuffle_b(previous_values, (unsigned char)(128 + lane), 0, 0);
                        float64 prediction = v_f32_reduce_add(pk0 * s0 + pk1 * s1);
                        float64 update = pb * (pv - pd * prediction);
                        s0 = s0 * pd + pk0 * update;
                        s1 = s1 * pd + pk1 * update;
                    }
                    state_position[0] = 0;
                    v_f32_st_tnsr(state_position, committed, s0);
                    state_position[0] = 64;
                    v_f32_st_tnsr(state_position, committed, s1);
                    float64 v = v_f32_shuffle_b(current_values, (unsigned char)(128 + lane), 0, 0);
                    float64 prediction = v_f32_reduce_add(k0 * s0 + k1 * s1);
                    float64 update = b * (v - d * prediction);
                    float64 result = d * v_f32_reduce_add(q0 * s0 + q1 * s1) + qk * update;
                    int5 output_position = {block * 4 + lane, head, 0, batch, 0};
                    v_f32_st_tnsr_partial(output_position, output, result, 0, 0);
                }
            }
        }
    }
}
