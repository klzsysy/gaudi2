"""Replay captured Qwen blocks through the product prefix entry and ordinary block."""
import argparse
import json
from pathlib import Path

from qwen_d2_plan_b_screen import baseline_forward, diff_stat, comfy, torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(32)
    model = comfy.sd.load_diffusion_model(
        "/home/sonny/ComfyUI-Models/models/diffusion_models/qwen_image_2.1_turbo_fp8.safetensors")
    comfy.model_management.load_models_gpu([model])
    results = {}
    for path in sorted(args.inputs.glob('forward-*.pt')):
        d = torch.load(path, weights_only=True)
        block = model.model.diffusion_model.transformer_blocks[d['layer']]
        x, pe, pk, pv = [d[k].to('hpu') for k in ('x', 'pe', 'prefix_k', 'prefix_v')]
        pk, pv = torch.stack((pk, pv), 1).unbind(1)
        mod = tuple(tuple(t.to('hpu') for t in v) if isinstance(v, tuple) else v.to('hpu') for v in d['mod'])
        expected = baseline_forward(block, x.clone(), mod, pe, pk, pv, d['layer'])
        actual = block._forward_with_prefix(x.clone(), mod, pe, pk, pv,
                                            dict(hpu_fast_attention=True, block_index=d['layer']), [])
        results[path.name] = diff_stat(expected, actual)
        print(path.name, results[path.name], flush=True)
        args.output.write_text(json.dumps(results, indent=2))
    assert len(results) == 6 and all(row['bitwise'] for row in results.values())


if __name__ == '__main__':
    main()
