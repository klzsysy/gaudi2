#ifndef VALUE_TILE
#define VALUE_TILE 4
#endif
float64 broadcast_group(float64 x) {
    const int writes = SW_WR_LOWER_GROUP0 | SW_WR_UPPER_GROUP0 | SW_WR_LOWER_GROUP1 | SW_WR_UPPER_GROUP1 |
                       SW_WR_LOWER_GROUP2 | SW_WR_UPPER_GROUP2 | SW_WR_LOWER_GROUP3 | SW_WR_UPPER_GROUP3;
    return v_f32_mov_dual_group_all_b(x, 0xffffffff, 0, 0, 0, 0, writes, 0);
}

void main(tensor state, tensor query, tensor key, tensor value, tensor decay, tensor beta,
          tensor updated, tensor output, int values, int heads, int batches) {
    const int5 begin = get_index_space_offset();
    const int5 end = begin + get_index_space_size();
    for (int batch = begin[2]; batch < end[2] && batch < batches; ++batch) {
        for (int head = begin[1]; head < end[1] && head < heads; ++head) {
            int5 gate_position = {0, head, batch, 0, 0};
            float64 d = v_f32_shuffle_b(broadcast_group(v_f32_ld_tnsr_b(gate_position, decay)), 128, 0, 0);
            float64 b = v_f32_shuffle_b(broadcast_group(v_f32_ld_tnsr_b(gate_position, beta)), 128, 0, 0);
            int5 position = {0, head, batch, 0, 0};
            float64 q0 = v_f32_ld_tnsr_b(position, query);
            float64 k0 = v_f32_ld_tnsr_b(position, key);
            position[0] = 64;
            float64 q1 = v_f32_ld_tnsr_b(position, query);
            float64 k1 = v_f32_ld_tnsr_b(position, key);
            float64 qk = v_f32_reduce_add(q0 * k0 + q1 * k1);
            for (int block = begin[0]; block < end[0]; ++block) {
                int5 value_position = {block * VALUE_TILE, head, batch, 0, 0};
                float64 current_values = broadcast_group(v_f32_ld_tnsr_b(value_position, value));
                #pragma unroll
                for (int lane = 0; lane < VALUE_TILE; ++lane) {
                    int5 state_position = {0, block * VALUE_TILE + lane, head, batch, 0};
                    float64 s0 = v_f32_ld_tnsr_b(state_position, state);
                    state_position[0] = 64;
                    float64 s1 = v_f32_ld_tnsr_b(state_position, state);
                    float64 v = v_f32_shuffle_b(current_values, (unsigned char)(128 + lane), 0, 0);
                    float64 prediction = v_f32_reduce_add(k0 * s0 + k1 * s1);
                    float64 delta = b * (v - d * prediction);
                    float64 result = d * v_f32_reduce_add(q0 * s0 + q1 * s1) + qk * delta;
                    state_position[0] = 0;
                    v_f32_st_tnsr(state_position, updated, s0 * d + k0 * delta);
                    state_position[0] = 64;
                    v_f32_st_tnsr(state_position, updated, s1 * d + k1 * delta);
                    int5 output_position = {block * VALUE_TILE + lane, head, batch, 0, 0};
                    v_f32_st_tnsr_partial(output_position, output, result, 0, 0);
                }
            }
        }
    }
}
