#!/usr/bin/env python3
"""Convert a Comfy/Kitchen INT8 ConvRot Qwen3.5 checkpoint to plain BF16.

This is an investigation probe. It removes Comfy quantization metadata and
materializes only the quantized 2-D linear weights in their original BF16
basis. It does not modify a source checkpoint and does not touch product code.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from comfy_kitchen.backends.eager.quantization import (
    DTYPE_TO_CODE,
    dequantize_int8_convrot_weight_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_quant_configs(handle) -> dict[str, dict]:
    configs: dict[str, dict] = {}
    for key in handle.keys():
        if not key.endswith(".comfy_quant"):
            continue
        prefix = key[: -len(".comfy_quant")]
        raw = handle.get_tensor(key).numpy().tobytes().decode("utf-8")
        configs[prefix] = json.loads(raw)
    return configs


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if not source.is_file():
        raise SystemExit(f"source does not exist: {source}")
    if destination.exists() and not args.overwrite:
        raise SystemExit(f"destination exists; pass --overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    output: dict[str, torch.Tensor] = {}
    converted: list[str] = []
    removed: list[str] = []
    copied_int8: list[str] = []

    with safe_open(str(source), framework="pt", device="cpu") as handle:
        configs = load_quant_configs(handle)
        keys = set(handle.keys())
        drop = set()
        for prefix in configs:
            drop.add(f"{prefix}.comfy_quant")
            for suffix in (
                ".weight_scale",
                ".weight_scale_2",
                ".weight_s_rel",
                ".weight_s_channel",
                ".weight_codebook",
                ".weight_correction",
            ):
                candidate = f"{prefix}{suffix}"
                if candidate in keys:
                    drop.add(candidate)

        for key in sorted(keys):
            if key in drop:
                removed.append(key)
                continue
            tensor = handle.get_tensor(key)
            prefix = key[: -len(".weight")] if key.endswith(".weight") else None
            config = configs.get(prefix) if prefix is not None else None
            if config is not None:
                if config.get("format") != "int8_tensorwise" or not config.get("convrot", False):
                    raise RuntimeError(f"unsupported quantization for {key}: {config}")
                scale_key = f"{prefix}.weight_scale"
                if scale_key not in keys:
                    raise RuntimeError(f"missing scale for {key}")
                scale = handle.get_tensor(scale_key)
                groupsize = int(config.get("convrot_groupsize", 256))
                tensor = dequantize_int8_convrot_weight_dtype(
                    tensor,
                    scale,
                    groupsize,
                    DTYPE_TO_CODE[torch.bfloat16],
                )
                if tensor.dtype != torch.bfloat16:
                    tensor = tensor.to(torch.bfloat16)
                converted.append(key)
            elif tensor.dtype == torch.int8:
                copied_int8.append(key)
            output[key] = tensor.contiguous()

    temp_name = destination.with_name(f".{destination.name}.part-{os.getpid()}")
    try:
        save_file(
            output,
            str(temp_name),
            metadata={"format": "pt", "source": source.name, "conversion": "int8_convrot_to_bf16"},
        )
        os.replace(temp_name, destination)
    finally:
        if temp_name.exists():
            temp_name.unlink()

    with safe_open(str(destination), framework="pt", device="cpu") as handle:
        remaining_quant = [key for key in handle.keys() if key.endswith(".comfy_quant")]
        bad_converted = [key for key in converted if handle.get_tensor(key).dtype != torch.bfloat16]
        if remaining_quant or bad_converted:
            raise RuntimeError(
                f"verification failed: remaining_quant={remaining_quant[:3]}, "
                f"bad_converted={bad_converted[:3]}"
            )

    report = {
        "source": str(source),
        "destination": str(destination),
        "converted_weights": len(converted),
        "copied_unquantized_int8": copied_int8,
        "removed_metadata_or_scales": len(removed),
        "destination_bytes": destination.stat().st_size,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
