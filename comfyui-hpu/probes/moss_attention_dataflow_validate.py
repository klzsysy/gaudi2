"""Replay an attention candidate across layers, timesteps and resolutions."""
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
parser.add_argument('--captures', type=Path, action='append', required=True)
parser.add_argument('--mode', default='batch-query-softmax2d-av2d-1536')
parser.add_argument('--reuse-reference', action='store_true')
parser.add_argument('--reference-dir', type=Path)
options = parser.parse_args()
options.root.mkdir(parents=True, exist_ok=True)
reference_directory = options.reference_dir or options.root / 'reference'
reference_directory.mkdir(parents=True, exist_ok=True)
sources = [source for directory in options.captures for source in sorted(directory.glob('input-*.pt'))]
assert sources, 'No captured inputs found'
torch.set_grad_enabled(False)
torch.set_num_threads(32)
patcher = comfy.sd.load_diffusion_model('/home/sonny/ComfyUI-Models/models/diffusion_models/Moody-Moss-Mix-V4.1_00001__clean-fp8.safetensors')
comfy.model_management.load_models_gpu([patcher])
records = []
for mode in ((options.mode,) if options.reuse_reference else ('baseline', options.mode)):
    install(mode)
    torch._dynamo.reset()
    moss._compiled_block_forward.cache_clear()
    for index, source in enumerate(sources):
        data = torch.load(source, weights_only=True)
        block = patcher.model.diffusion_model.layers[data['layer']]
        x, pe = data['x'].to('hpu'), data['pe'].to('hpu')
        mod = tuple(t.to('hpu') for t in data['mod'])
        settings = dict(fp8_attention=True, _moss_hpu_blockwise_rope=data['blockwise'])
        for _ in range(3):
            output = block(x.clone(), mod, pe, settings)
        actual = output.cpu()
        path = reference_directory / f'{index:03}.pt'
        if mode == 'baseline':
            torch.save(dict(source=str(source), output=actual), path)
        saved = torch.load(path, weights_only=True)
        assert saved['source'] == str(source)
        expected = saved['output']
        delta = actual.float() - expected.float()
        samples = []
        for _ in range(5):
            start = time.perf_counter_ns()
            for _ in range(5):
                output = block(x.clone(), mod, pe, settings)
            torch.hpu.synchronize()
            samples.append((time.perf_counter_ns() - start) / 1e9 / 5)
        record = dict(mode=mode, index=index, source=str(source), layer=data['layer'], shape=list(x.shape),
                      equal=torch.equal(actual, expected), differing=int((actual != expected).sum()),
                      rrmse=float(delta.norm() / expected.float().norm()), maximum=float(delta.abs().max()),
                      repeat_equal=torch.equal(actual, output.cpu()), finite=bool(actual.isfinite().all()),
                      seconds=statistics.median(samples), samples=samples, fallback=_get_fallback_op_count())
        records.append(record)
        filename = 'validation-retry.json' if options.reuse_reference else 'validation.json'
        (options.root / filename).write_text(json.dumps(records, indent=2))
        print(json.dumps(record), flush=True)
        assert record['finite'] and not record['fallback'], record
