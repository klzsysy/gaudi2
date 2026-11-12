"""Compare mixed-attention graph expressions on captured production blocks."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.use_pytorch_cross_attention = True
import comfy.sd
import comfy.model_management
from comfy.ldm.moss_image import model as moss
from habana_frameworks.torch.utils.debug import _get_fallback_op_count
from moss_attention_dataflow_candidate import install


parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('--modes', default='baseline,rank3,swap-qk,swap-av,swap-both,combined-scale,baseline')
parser.add_argument('--captures', type=Path)
parser.add_argument('--pattern', default='input-*-12-0.pt')
options = parser.parse_args()
options.root.mkdir(parents=True, exist_ok=True)
captures = options.captures or options.root.parent / 'performance-2026-11-12-fp8-deep/baseline/captures'
torch.set_grad_enabled(False)
torch.set_num_threads(32)
patcher = comfy.sd.load_diffusion_model('/home/sonny/ComfyUI-Models/models/diffusion_models/Moody-Moss-Mix-V4.1_00001__clean-fp8.safetensors')
comfy.model_management.load_models_gpu([patcher])
records = []
for source in sorted(captures.glob(options.pattern)):
    data = torch.load(source, weights_only=True)
    block = patcher.model.diffusion_model.layers[data['layer']]
    values = (data['x'].to('hpu'), tuple(t.to('hpu') for t in data['mod']), data['pe'].to('hpu'))
    settings = dict(fp8_attention=True, _moss_hpu_blockwise_rope=data['blockwise'])
    reference = None
    for mode in options.modes.split(','):
        install(mode)
        torch._dynamo.reset()
        moss._compiled_block_forward.cache_clear()
        for _ in range(3):
            output = block(values[0].clone(), *values[1:], settings)
        torch.hpu.synchronize()
        actual = output.cpu()
        if reference is None:
            assert mode == 'baseline'
            reference = actual
        delta = actual.float() - reference.float()
        samples = []
        for _ in range(7):
            start = time.perf_counter_ns()
            for _ in range(10):
                output = block(values[0].clone(), *values[1:], settings)
            torch.hpu.synchronize()
            samples.append((time.perf_counter_ns() - start) / 1e9 / 10)
        record = dict(source=source.name, mode=mode, shape=list(values[0].shape),
                      equal=torch.equal(actual, reference), differing=int((actual != reference).sum()),
                      rrmse=float(delta.norm() / reference.float().norm()), maximum=float(delta.abs().max()),
                      finite=bool(actual.isfinite().all()), repeat_equal=torch.equal(actual, output.cpu()),
                      seconds=statistics.median(samples), samples=samples, fallback=_get_fallback_op_count())
        records.append(record)
        (options.root / 'attention-dataflow-screen.json').write_text(json.dumps(records, indent=2))
        print(json.dumps(record), flush=True)
        assert record['finite'] and not record['fallback'], record
