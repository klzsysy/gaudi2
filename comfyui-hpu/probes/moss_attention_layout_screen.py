"""Screen lossless attention layout and scheduling changes on complete blocks."""
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
from comfy.ldm.moss_image import model as moss
from comfy.ldm.modules import attention
from comfy.ldm.modules.attention import AttentionTensorContainer
import habana_frameworks.torch.utils.debug as debug


parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('--cases', default='baseline')
parser.add_argument('--label', default='baseline')
parser.add_argument('--configuration', default='{}')
options = parser.parse_args()
root = options.root
root.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
torch.set_num_threads(32)
model = comfy.sd.load_diffusion_model('/home/sonny/ComfyUI-Models/models/diffusion_models/Moody-Moss-Mix-V4.1_00001__clean-fp8.safetensors')
comfy.model_management.load_models_gpu([model])
original_av = attention._hpu_fp8_attention_av
original_heads = moss._attention_heads


def layout_attention(mode):
    def forward(q, k, v):
        query, key, value = q[0, 0], k[0, 0], v[0, 0]
        if mode == 'qkv-contiguous':
            query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        value_scale = value.abs().amax().float().clamp_min(1e-12) / 240.0
        quantized_value = torch.ops.hpu.cast_to_fp8_v2(value, value_scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
        scale = q.shape[-1] ** -0.25
        query = query * scale
        key = (key * scale).transpose(-1, -2)
        if mode in ('key-contiguous', 'qk-contiguous'):
            key = key.contiguous()
        if mode == 'qk-contiguous':
            query = query.contiguous()
        scores = query @ key
        probabilities = torch.ops.hpu.softmax_fp8.Scalar_scales(scores, -1, 1.0, 128.0)
        probability_scale = torch.full((), 1 / 128.0, device=q.device, dtype=torch.float32)
        transpose_value = mode == 'av-transpose'
        if transpose_value:
            quantized_value = quantized_value.transpose(-1, -2).contiguous()
        output = torch.ops.hpu.fp8_gemm_v2(probabilities, False, quantized_value, transpose_value, None, q.dtype,
                                         probability_scale, value_scale, None, False)
        return output.unsqueeze(0).unsqueeze(0)
    return forward


def scheduled_heads(mode):
    def forward(q, k, v, heads, comfy_attention, transformer_options):
        order = list(range(heads))
        if mode == 'heads-reverse':
            order.reverse()
        elif mode == 'heads-even-odd':
            order = order[::2] + order[1::2]
        elif mode == 'heads-interleave':
            order = [index for pair in zip(order[:heads // 2], order[heads // 2:]) for index in pair]
        if mode == 'heads-contiguous':
            tensors = [tensor.unflatten(-1, (heads, -1)).transpose(1, 2).contiguous().unbind(1)
                       for tensor in (q, k, v)]
        else:
            tensors = [tensor.chunk(heads, dim=-1) for tensor in (q, k, v)]
        outputs = [None] * heads
        for index in order:
            inputs = (AttentionTensorContainer(tensor[index]) for tensor in tensors)
            outputs[index] = moss.optimized_attention(*inputs, 1, transformer_options=transformer_options,
                                                      preferred_attention=comfy_attention)
        if mode == 'heads-stack':
            return torch.stack(outputs, dim=-2).flatten(-2)
        return torch.cat(outputs, dim=-1)
    return forward


api = ctypes.CDLL('/usr/lib/habanalabs/libSynapse.so')
api.synConfigurationGet.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint64]
api.synConfigurationSet.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
requested = json.loads(options.configuration)
set_status = {}
if requested:
    for name, value in dict(ENABLE_EXPERIMENTAL_FLAGS='1', **requested).items():
        set_status[name] = api.synConfigurationSet(name.encode(), value.encode())
    (root / f'{options.label}-set-status.json').write_text(json.dumps(set_status, indent=2))
    assert not any(set_status.values()), set_status
configuration = {}
for name in ['ENABLE_EXPERIMENTAL_FLAGS', 'ENABLE_LAYERED_PIPELINE_BRAIN', 'ENABLE_LAYERED_BRAIN_ATTENTION_SEEDS',
             'ENABLE_LAYERED_BRAIN_MULTI_MME_ATTENTION_SEEDS', 'ENABLE_LAYERED_BRAIN_MULTI_MME_SEEDS',
             'ENABLE_FLASH_ATTENTION_SLICING', 'SRAM_SLICER_MAX_CAPACITY_BYTES', 'DEFAULT_PIPELINE_DEPTH',
             'ENABLE_SHARED_MULTIBUF_PER_SLICED_CHAIN']:
    buffer = ctypes.create_string_buffer(1024)
    status = api.synConfigurationGet(name.encode(), buffer, 1024)
    configuration[name] = dict(status=status, value=buffer.value.decode())
(root / f'{options.label}-configuration.json').write_text(json.dumps(configuration, indent=2))
assert all(configuration[name]['value'] == value for name, value in requested.items()), configuration
captures = root.parent / 'performance-2026-11-12-fp8-deep/baseline/captures'
records = []
for condition in [307, 23]:
    data = torch.load(captures / f'input-{condition}-12-0.pt', weights_only=True)
    block = model.model.diffusion_model.layers[0]
    x, pe = data['x'].to('hpu'), data['pe'].to('hpu')
    mod = tuple(value.to('hpu') for value in data['mod'])
    settings = dict(fp8_attention=True, _moss_hpu_blockwise_rope=data['blockwise'])
    reference_path = root / f'baseline-{condition}.pt'
    for mode in options.cases.split(','):
        attention._hpu_fp8_attention_av = original_av
        moss._attention_heads = original_heads
        if mode.startswith('heads-'):
            moss._attention_heads = scheduled_heads(mode)
        elif mode != 'baseline':
            attention._hpu_fp8_attention_av = layout_attention(mode)
        torch._dynamo.reset()
        moss._compiled_block_forward.cache_clear()
        for _ in range(3):
            output = block(x.clone(), mod, pe, settings)
        torch.hpu.synchronize()
        actual = output.cpu()
        if options.label == 'baseline' and mode == 'baseline':
            torch.save(actual, reference_path)
        reference = torch.load(reference_path, weights_only=True)
        delta = actual.float() - reference.float()
        equal = torch.equal(actual, reference)
        samples = []
        for _ in range(5):
            start = time.perf_counter_ns()
            for _ in range(10):
                output = block(x.clone(), mod, pe, settings)
            torch.hpu.synchronize()
            samples.append((time.perf_counter_ns() - start) / 1e9 / 10)
        record = dict(condition=condition, mode=mode, label=options.label, bitwise_equal=equal,
                      differing_elements=int((actual != reference).sum()),
                      max_absolute=float(delta.abs().max()),
                      rrmse=float((delta.square().sum(dtype=torch.float64) /
                                   reference.float().square().sum(dtype=torch.float64)).sqrt()),
                      seconds=statistics.median(samples), samples=samples,
                      repeat_equal=torch.equal(actual, output.cpu()), fallback=debug._get_fallback_op_count())
        records.append(record)
        (root / f'{options.label}.json').write_text(json.dumps(records, indent=2))
        print(json.dumps(record), flush=True)
