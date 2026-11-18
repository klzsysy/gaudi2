// SPDX-License-Identifier: MIT
// Stable FP16 scores to unnormalized BF16 probabilities and FP32 row sums.
// The output may share the score allocation after the row maximum is known.
#include "reduce_max_no_lut.h"
#include "reduce_no_lut.h"

#ifdef QWEN_EXP_CORE
// Finite-input builtin exp sequence from the installed compiler, also described
// in jchen-38/gaudi2-single-card-vllm at 72fafcf2, exp_lut_half.c (MIT).
// The exact row maximum makes the argument nonpositive. Clamping the lower
// bound keeps the exponent addition inside the normal BF16 range.
static bfloat128 exp_finite(bfloat128 value) {
    value = v_bf16_max_b(value, (bfloat128)-87.0f);
    bfloat128 scaled = v_bf16_mac_b(value, (bfloat128)1.4453125f, (bfloat128)0.5f);
    bfloat128 integer = v_bf16_nearbyint_b(scaled, 198400);
    short128 exponent = v_convert_bf16_to_i16_b(integer, 198464);
    bfloat128 residual = v_bf16_mac_b(integer, (bfloat128)-0.6875f, value);
    residual = v_bf16_mac_b(integer, (bfloat128)-0.005645751953125f, residual);
    bfloat128 biased = residual + (bfloat128)1.5f;
    ushort128 index = v_u16_sub_b(*(ushort128*)&biased, (ushort128)16199, 1);
    bfloat128 mantissa = v_bf16_lookup_1c(index, 138, 1, (bfloat128)0, 1, 0);
    short128 bits = (exponent << 7) + *(short128*)&mantissa;
    return *(bfloat128*)&bits;
}
#endif

void main(tensor scores, tensor probabilities, tensor sums) {
    int5 start = get_index_space_offset(), end = start + get_index_space_size();
    int columns = get_dim_size(scores, 0);
    for (int row = start[0]; row < end[0]; ++row) {
#ifdef QWEN_EXP_CORE
        half128 narrow_maximum = -65504.0f;
#else
        float64 maximum = -3.402823466e38f;
#endif
#ifdef QWEN_UNROLL
#pragma clang loop unroll_count(4)
#endif
        for (int column = 0; column < columns; column += 128) {
            half128 value = v_f16_ld_tnsr_b((int5){column, row, 0, 0, 0}, scores);
            value = v_f16_sel_less_u16_b(read_lane_id_2b_b() + column, (unsigned short)columns,
                                         value, (half128)-65504.0f);
#ifdef QWEN_EXP_CORE
            narrow_maximum = v_f16_max_b(narrow_maximum, value);
#else
            float128 wide = v_convert_f16_to_f32_all_b(value);
            maximum = v_f32_max_b(maximum, v_f32_max_b(wide.v1, wide.v2));
#endif
        }
#ifdef QWEN_EXP_CORE
        float128 wide = v_convert_f16_to_f32_all_b(narrow_maximum);
        float64 maximum = v_f32_max_b(wide.v1, wide.v2);
#endif
        maximum = max64_no_lut(maximum);
        float128 pair;
        pair.v1 = maximum;
        pair.v2 = maximum;
        half128 max_half = v_convert_f32_to_f16_all_b(pair);
        float128 accumulator;
        accumulator.v1 = 0.0f;
        accumulator.v2 = 0.0f;
#ifdef QWEN_UNROLL
#pragma clang loop unroll_count(4)
#endif
        for (int column = 0; column < columns; column += 128) {
            int5 position = {column, row, 0, 0, 0};
            half128 value = v_f16_ld_tnsr_b(position, scores);
#if defined(QWEN_EXP_BF16) || defined(QWEN_EXP_CORE)
            float128 difference = v_convert_f16_to_f32_all_b(v_f16_sub_b(value, max_half));
#ifdef QWEN_EXP_CORE
            bfloat128 probability = exp_finite(v_convert_f32_to_bf16_all_b(difference));
#else
            bfloat128 probability = exp_bf16(v_convert_f32_to_bf16_all_b(difference));
#endif
#else
            half128 exponent = v_f16_mul_b(v_f16_sub_b(value, max_half), (half128)1.4426950408889634f);
            exponent = v_f16_max_b(exponent, (half128)-126.0f);
            short128 integer = v_convert_f16_to_i16_b(exponent, SW_RD);
            half128 fraction = v_f16_sub_b(exponent, v_convert_i16_to_f16_b(integer));
            half128 polynomial = v_f16_mac_b(fraction, (half128)0.0096181291f, (half128)0.0555041087f);
            polynomial = v_f16_mac_b(polynomial, fraction, (half128)0.2402265070f);
            polynomial = v_f16_mac_b(polynomial, fraction, (half128)0.6931471806f);
            polynomial = v_f16_mac_b(polynomial, fraction, (half128)1.0f);
            bfloat128 mantissa = v_convert_f32_to_bf16_all_b(v_convert_f16_to_f32_all_b(polynomial));
            short128 bits = (integer + (short128)127) << 7;
            bfloat128 probability = v_bf16_mul_b(mantissa, *((bfloat128*)&bits));
#endif
            probability = v_bf16_sel_less_u16_b(read_lane_id_2b_b() + column, (unsigned short)columns,
                                               probability, (bfloat128)0.0f);
            v_bf16_st_tnsr(position, probabilities, probability);
            accumulator = v_bf16_mac_acc32_b(probability, (bfloat128)1.0f, accumulator, 0, 1, 0);
        }
        float64 sum = sum64_no_lut(accumulator.v1 + accumulator.v2);
        v_f32_st_tnsr((int5){0, row, 0, 0, 0}, sums, sum);
    }
}
