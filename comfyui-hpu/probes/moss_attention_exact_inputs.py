"""Check a compiler SRAM budget against all 90 recorded production block inputs."""
import argparse
import ctypes
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
import habana_frameworks.torch.utils.debug as debug


parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('--sram-mib', type=int, default=0)
options = parser.parse_args()
root = options.root
reference_directory = root / 'exact-inputs-baseline'
reference_directory.mkdir(exist_ok=True)
label = f'exact-inputs-{options.sram_mib or "baseline"}'
torch.set_grad_enabled(False)
torch.set_num_threads(32)
patcher = comfy.sd.load_diffusion_model('/home/sonny/ComfyUI-Models/models/diffusion_models/Moody-Moss-Mix-V4.1_00001__clean-fp8.safetensors')
comfy.model_management.load_models_gpu([patcher])
api = ctypes.CDLL('/usr/lib/habanalabs/libSynapse.so')
api.synConfigurationSet.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
api.synConfigurationGet.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint64]
if options.sram_mib:
    for name, value in [('ENABLE_EXPERIMENTAL_FLAGS', '1'),
                        ('SRAM_SLICER_MAX_CAPACITY_BYTES', str(options.sram_mib * 1024**2))]:
        assert api.synConfigurationSet(name.encode(), value.encode()) == 0
        buffer = ctypes.create_string_buffer(1024)
        assert api.synConfigurationGet(name.encode(), buffer, 1024) == 0
        assert buffer.value.decode() == value

sources = sorted((root.parent / 'performance-2026-11-12-fp8-deep/baseline/captures').glob('input-*.pt'))
sources += sorted((root.parent / 'performance-2026-11-12-evidence/quality/captures').glob('input-*.pt'))
assert len(sources) == 90, len(sources)
records = []
for index, source in enumerate(sources):
    data = torch.load(source, weights_only=True)
    block = patcher.model.diffusion_model.layers[data['layer']]
    x, pe = data['x'].to('hpu'), data['pe'].to('hpu')
    mod = tuple(value.to('hpu') for value in data['mod'])
    settings = dict(fp8_attention=True, _moss_hpu_blockwise_rope=data['blockwise'])
    for _ in range(3):
        output = block(x.clone(), mod, pe, settings)
    actual = output.cpu()
    path = reference_directory / f'{index:03}.pt'
    if not options.sram_mib:
        torch.save(dict(source=str(source), output=actual), path)
    reference = torch.load(path, weights_only=True)
    assert reference['source'] == str(source)
    expected = reference['output']
    delta = actual.float() - expected.float()
    samples = []
    for _ in range(3):
        start = time.perf_counter_ns()
        for _ in range(5):
            output = block(x.clone(), mod, pe, settings)
        torch.hpu.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e9 / 5)
    record = dict(index=index, source=str(source), layer=data['layer'], shape=list(x.shape),
                  bitwise_equal=torch.equal(actual, expected), differing_elements=int((actual != expected).sum()),
                  max_absolute=float(delta.abs().max()), repeat_equal=torch.equal(actual, output.cpu()),
                  seconds=statistics.median(samples), samples=samples, fallback=debug._get_fallback_op_count())
    records.append(record)
    (root / f'{label}.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(record), flush=True)
