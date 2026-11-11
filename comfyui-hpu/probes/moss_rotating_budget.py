"""Real 42-layer weight rotation, split-region grouping, and matrix ceilings."""
import json
import os
from pathlib import Path
import statistics
import time

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True
import comfy.ops
from comfy.ldm.moss_image import model as moss
import habana_frameworks.torch.utils.debug as debug

torch.set_grad_enabled(False)
torch.set_num_threads(16)
root = Path(os.environ['MOSS_COMPLETION_ROOT'])
blocks = []
for layer in range(42):
    block = moss.Block(3072, 8192, 24, fused_qkv=True, dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
    block.load_state_dict(torch.load(root/f'captures/weights-{layer}.pt', weights_only=True))
    blocks.append(block.to('hpu'))
records = []


def measure(label, fn, condition, iterations=1):
    start = time.perf_counter()
    out = fn()
    torch.hpu.synchronize()
    cold = time.perf_counter() - start
    value = out.cpu().float()
    for _ in range(2):
        fn()
    torch.hpu.synchronize()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.hpu.synchronize()
        samples.append((time.perf_counter()-start)/iterations)
    row = dict(label=label, condition=condition, cold_s=cold, median_s=statistics.median(samples), samples_s=samples,
               fallback=debug._get_fallback_op_count(), memory=torch.hpu.memory_stats())
    records.append(row)
    (root/'rotating-budget.json').write_text(json.dumps(records, indent=2))
    print(json.dumps({k:v for k,v in row.items() if k!='memory'}), flush=True)
    return value


linear = torch.compile(torch.nn.functional.linear, backend='hpu_backend', dynamic=None, options={'force_static_compile': True})
options = {'_moss_hpu_blockwise_rope': True}
for condition in (307, 23):
    data = torch.load(root/f'captures/input-{condition}-0-0.pt', weights_only=True)
    x = data['args'][0].to('hpu')
    mod = tuple(t.to('hpu') for t in data['args'][1])
    pe = data['args'][2].to('hpu')
    assert moss.hpu.blockwise_coefficients(pe)
    def chain(selected):
        current = x.clone()
        for block in selected:
            current = block(current, mod, pe, options)
        return current
    baseline = measure('rotating-42', lambda: chain(blocks), condition)
    torch.save(baseline, root/f'rotating-output-{condition}.pt')
    measure('repeat-one-42', lambda: chain([blocks[0]]*42), condition)
    one = measure('single-block', lambda: chain([blocks[0]]), condition, iterations=10)
    torch.save(one, root/f'eager-output-{condition}.pt')
    b = blocks[0]
    y1 = moss.modulate(x, b.norm1, mod, 0, None)
    original_attention, attention_output = moss._attention, []
    def capture_attention(*values, **kwargs):
        value = original_attention(*values, **kwargs)
        attention_output.append(value)
        return value
    moss._attention = capture_attention
    att = b._forward_attention(y1, pe, options)
    moss._attention = original_attention
    y2 = moss.modulate(moss.residual(x.clone(), att, mod, 2, None), b.norm2, mod, 3, None)
    hidden = comfy.ops._eager_input_act(b.proj['ffn_up_gate'](y2), 'situ')
    for name, activation in [('qkv',y1),('o',attention_output[0]),('ffn_up_gate',y2),('ffn_down',hidden)]:
        def matrices():
            for block in blocks:
                value = linear(activation, block.proj[name].weight)
            return value
        measure('linear-42/'+name, matrices, condition, iterations=2)
        row = records[-1]
        row['tflops'] = 42 * 2 * activation.numel() * b.proj[name].weight.shape[0] / row['median_s'] / 1e12
    # Preserve the two known rounding boundaries while widening the Python region.
    original_mod, original_res = moss.modulate, moss.residual
    moss.modulate = torch.compiler.disable(original_mod)
    moss.residual = torch.compiler.disable(original_res)
    def pair(a, b, x, mod, pe):
        x = a(x, mod, pe, options)
        return b(x, mod, pe, options)
    grouped = torch.compile(pair, backend='hpu_backend', dynamic=None, options={'force_static_compile': True})
    def paired_chain():
        current = x.clone()
        for i in range(0,42,2):
            current = grouped(blocks[i], blocks[i+1], current, mod, pe)
        return current
    actual = measure('pair-split-42', paired_chain, condition)
    records[-1]['rrmse_vs_baseline'] = float((actual-baseline).square().mean().sqrt()/baseline.square().mean().sqrt())
    moss.modulate, moss.residual = original_mod, original_res
    torch._dynamo.reset()
    moss._compiled_block_ops.cache_clear()
    moss._compiled_attention.cache_clear()
    (root/'rotating-budget.json').write_text(json.dumps(records, indent=2))
