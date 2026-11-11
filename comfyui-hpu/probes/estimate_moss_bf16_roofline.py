"""CPU-only arithmetic for the 2026-11-10 MOSS roofline investigation.

These are optimistic work/peak estimates, not a device benchmark. See
MOSS_HARDWARE_LIMIT_AND_5090_2026-11-10.md for evidence and exclusions.
Run from any directory with Python 3; no third-party dependencies.
"""
import json


def estimate():
    # Product b1163be0; checkpoint shapes and tokenizer checked on the server.
    dim, ffn, layers, steps = 3072, 8192, 42, 25
    image_tokens, positive_tokens, negative_tokens = 4096, 307, 23
    np, nn = image_tokens + positive_tokens, image_tokens + negative_tokens
    peak_flops, hbm_bytes_s = 432e12, 2.45e12
    measured_dit_s = 18.384110546  # Fixed caption, five warm requests, median.
    measured_full_s = 29.672968674  # Full workflow, five warm requests, median.
    cached_smoke_s = 18.764  # One daily-service request, not a five-run median.

    # FMA = 2 FLOPs. Two CFG branches per step, each at its own sequence length.
    linear = steps * layers * 2 * (np + nn) * (4 * dim**2 + 3 * dim * ffn)
    attention = steps * layers * 4 * dim * (np**2 + nn**2)
    # Image input/output projections, time/modulation projections, and two
    # once-per-conditioning TextEmbedder blocks. Bias/nonlinear FLOPs excluded.
    image_io = steps * 2 * 4 * image_tokens * 128 * dim
    time_modulation = steps * 2 * 2 * (256 * dim + dim**2 + 6 * dim**2 + 2 * dim**2)
    text_preparation = 2 * (positive_tokens + negative_tokens) * 4096 * dim
    text_preparation += 2 * sum(
        2 * n * (4 * dim**2 + 3 * dim * ffn) + 4 * n**2 * dim
        for n in (positive_tokens, negative_tokens)
    )
    matrix_flops = linear + attention + image_io + time_modulation + text_preparation

    # Same-caption prior synchronized diagnostic: 129 main verify calls,
    # 258 MTP head calls, 388 lm_head calls, one 1970-token prefill.
    # The latest run has identical output IDs but was not instrumented for these
    # internal counts. This is a fixed-trace reference, not a universal minimum.
    backbone_bytes = 13_839_131_136
    lm_head_bytes = 2_034_237_440
    mtp_bytes = 486_581_248
    decode_weight_bytes = 129 * backbone_bytes + 388 * lm_head_bytes + 258 * mtp_bytes
    # BF16 weight bytes numerically equal 2*parameter count; omit tiny norm
    # parameter distinctions, attention, and GDN non-weight work here.
    prefill_encode_linear_flops = backbone_bytes * (1970 + positive_tokens + negative_tokens)
    partial_pipeline = matrix_flops / peak_flops + decode_weight_bytes / hbm_bytes_s
    partial_pipeline += prefill_encode_linear_flops / peak_flops

    return {
        "inputs": {
            "product": "b1163be0", "resolution": [1024, 1024], "steps": steps,
            "cfg": 2.5, "image_tokens": image_tokens,
            "conditioning_tokens": [positive_tokens, negative_tokens],
            "sequence_lengths": [np, nn], "layers": layers, "dim": dim, "ffn_dim": ffn,
            "gaudi2_bf16_peak_tflops": peak_flops / 1e12,
            "gaudi2_hbm_tb_s": hbm_bytes_s / 1e12,
        },
        "dit": {
            "linear_pflop": linear / 1e15, "attention_pflop": attention / 1e15,
            "other_matrix_tflop": (image_io + time_modulation + text_preparation) / 1e12,
            "total_matrix_pflop": matrix_flops / 1e15,
            "optimistic_matrix_floor_s": matrix_flops / peak_flops,
            "matrix_s_at_80_percent_peak": matrix_flops / (0.8 * peak_flops),
            "matrix_s_at_60_percent_peak": matrix_flops / (0.6 * peak_flops),
            "matrix_s_at_old_4096cube_probe_275_76_tflops": matrix_flops / 275.76e12,
            "measured_warm_median_s": measured_dit_s,
            "effective_matrix_tflops": matrix_flops / measured_dit_s / 1e12,
            "effective_matrix_peak_percent": 100 * matrix_flops / measured_dit_s / peak_flops,
            "current_over_matrix_floor": measured_dit_s / (matrix_flops / peak_flops),
            "hypothetical_fp8_linears_bf16_attention_matrix_floor_s":
                linear / 865e12 + (matrix_flops - linear) / peak_flops,
        },
        "text_reference_trace": {
            "backbone_bytes": backbone_bytes, "lm_head_bytes": lm_head_bytes,
            "mtp_bytes": mtp_bytes, "verify_calls": 129,
            "mtp_calls": 258, "lm_head_calls": 388,
            "weight_scan_tb": decode_weight_bytes / 1e12,
            "ideal_weight_scan_s": decode_weight_bytes / hbm_bytes_s,
            "prefill_and_encoding_linear_floor_s": prefill_encode_linear_flops / peak_flops,
        },
        "pipeline": {
            "incomplete_optimistic_work_floor_s": partial_pipeline,
            "measured_full_warm_median_s": measured_full_s,
            "current_over_partial_work_floor": measured_full_s / partial_pipeline,
            "exclusions": ["GDN state/recurrence", "KV traffic", "nonlinear operations",
                           "unhidden activation traffic", "CPU/runtime/synchronization",
                           "VAE", "PNG/output", "loading/compilation"],
        },
        "external_target_ratios_not_matched_benchmarks": {
            "our_cached_one_run_s": cached_smoke_s,
            "5090_bf16_cached_report_s": 14.7, "5090_int8_cached_report_s": 5.8,
            "over_5090_bf16": cached_smoke_s / 14.7,
            "reduction_needed_for_14_7s_percent": 100 * (1 - 14.7 / cached_smoke_s),
            "over_5090_int8": cached_smoke_s / 5.8,
            "reduction_needed_for_5_8s_percent": 100 * (1 - 5.8 / cached_smoke_s),
        },
    }


if __name__ == "__main__":
    print(json.dumps(estimate(), indent=2))
