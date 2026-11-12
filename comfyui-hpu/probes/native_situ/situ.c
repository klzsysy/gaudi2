static float64 activate(float64 hidden, float64 gate) {
    float64 capped_gate = 4.0f * v_tanh_fast_f32(gate * 0.25f);
    float64 gated = capped_gate * v_sigmoid_f32(gate);
    return gated * (25.0f * v_tanh_fast_f32(hidden * 0.04f));
}

#ifdef PADE
static bfloat128 reciprocal(bfloat128 value) {
    float128 expanded = v_convert_bf16_to_f32_all_b(value, 0);
    expanded.v1 = v_reciprocal_fast_f32(expanded.v1);
    expanded.v2 = v_reciprocal_fast_f32(expanded.v2);
    return v_convert_f32_to_bf16_all_b(expanded, 0);
}

static bfloat128 tanh_pade(bfloat128 value) {
    value = v_bf16_max_b(v_bf16_min_b(value, 5.0f), -5.0f);
    bfloat128 square = value * value;
    bfloat128 numerator = value * (1.0f + square * (0.1282051282f + square * (0.0027972028f + square * 0.0000074f)));
    bfloat128 denominator = 1.0f + square * (0.4615384615f + square * (0.0233100233f + square * 0.0002072002f));
    bfloat128 result = numerator * reciprocal(denominator);
    return v_bf16_max_b(v_bf16_min_b(result, 1.0f), -1.0f);
}
#endif

static bfloat128 load_activation(tensor input, int col, int row, int columns) {
    int5 position = {col, row, 0, 0, 0};
#ifdef PADE
    bfloat128 hidden = v_bf16_ld_tnsr_b(position, input);
    position[0] += columns;
    bfloat128 gate = v_bf16_ld_tnsr_b(position, input);
    bfloat128 tangent = tanh_pade(gate * 0.25f);
    bfloat128 sum = 1.0f + tangent;
    bfloat128 gated = (2.0f * tangent) * sum * sum * reciprocal(1.0f + tangent * tangent);
    return gated * (25.0f * tanh_pade(hidden * 0.04f));
#else
    float128 hidden = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(position, input), 0);
    position[0] += columns;
    float128 gate = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(position, input), 0);
    float128 result;
    result.v1 = activate(hidden.v1, gate.v1);
    result.v2 = activate(hidden.v2, gate.v2);
    return v_convert_f32_to_bf16_all_b(result, 0);
#endif
}

#ifdef QUANTIZE
// Reserve 1 KiB of the 16 KiB special-function VLM for compiler spills.
__local__ bfloat128 values[60];
void main(tensor input, tensor output, tensor scale_output, int columns, int rows) {
    int5 begin = get_index_space_offset();
    int5 end = begin + get_index_space_size();
    for (int row = begin[0]; row < end[0] && row < rows; ++row) {
        float64 peak = 0;
        for (int col = 0; col < columns; col += 128) {
            bfloat128 rounded = load_activation(input, col, row, columns);
            if (col < columns - 512) values[col / 128] = rounded;
            float128 result = v_convert_bf16_to_f32_all_b(rounded, 0);
            peak = v_f32_max_b(peak, v_f32_sel_less_f32_b(result.v1, 0.0f, -result.v1, result.v1));
            peak = v_f32_max_b(peak, v_f32_sel_less_f32_b(result.v2, 0.0f, -result.v2, result.v2));
        }
        float64 scale = v_f32_max_b(v_f32_reduce_max(peak), 1e-12f) * (1.0f / 240.0f);
        float64 inverse = v_reciprocal_f32(scale);
        int5 position = {0, row, 0, 0, 0};
        v_f32_st_tnsr_partial(position, scale_output, scale, 0, 0);
        for (int col = 0; col < columns; col += 256) {
            bfloat128 first_value, second_value;
            if (col < columns - 512) {
                first_value = values[col / 128];
                second_value = values[col / 128 + 1];
            } else {
                first_value = load_activation(input, col, row, columns);
                second_value = load_activation(input, col + 128, row, columns);
            }
            float128 first = v_convert_bf16_to_f32_all_b(first_value, 0);
            float128 second = v_convert_bf16_to_f32_all_b(second_value, 0);
            first.v1 *= inverse;
            first.v2 *= inverse;
            second.v1 *= inverse;
            second.v2 *= inverse;
            bfloat256 normalized;
            normalized.v1 = v_convert_f32_to_bf16_all_b(first, 0);
            normalized.v2 = v_convert_f32_to_bf16_all_b(second, 0);
            position[0] = col;
            v_f8_st_tnsr(position, output, v_convert_bf16_to_f8_all_b(normalized, SW_LINEAR));
        }
    }
}
#else
void main(tensor input, tensor output, int columns, int rows) {
    int5 begin = get_index_space_offset();
    int5 end = begin + get_index_space_size();
    for (int row = begin[0]; row < end[0] && row < rows; ++row) {
        for (int tile = 0; tile < columns / 128; ++tile) {
            int5 position = {tile * 128, row, 0, 0, 0};
            bfloat128 result = load_activation(input, position[0], row, columns);
            v_bf16_st_tnsr(position, output, result);
        }
    }
}
#endif
