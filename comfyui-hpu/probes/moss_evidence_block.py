"""Profile one captured production block and record its FX graph metadata."""
import argparse
import functools
import json
from pathlib import Path
import time

import torch
from comfy.cli_args import args as comfy_args
comfy_args.disable_dynamic_vram = True
comfy_args.use_pytorch_cross_attention = True
import comfy.sd
import comfy.model_management
from comfy.ldm.moss_image import model as moss
from torch._dynamo.backends.registry import lookup_backend


parser = argparse.ArgumentParser()
parser.add_argument('output', type=Path)
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--iterations', type=int, default=3)
parser.add_argument('--native', action='store_true')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
torch.set_num_threads(32)
backend = lookup_backend('hpu_backend')
graphs = []


def capture_backend(graph, inputs, **kwargs):
    nodes = []
    for node in graph.graph.nodes:
        value = node.meta.get('example_value', node.meta.get('val'))
        record = dict(name=node.name, op=node.op, target=str(node.target), inputs=[n.name for n in node.all_input_nodes])
        if isinstance(value, torch.Tensor):
            record.update(shape=[str(n) for n in value.shape], stride=[str(n) for n in value.stride()], dtype=str(value.dtype))
        nodes.append(record)
    graphs.append(dict(code=graph.code,nodes=nodes))
    (args.output/'fx-graphs.json').write_text(json.dumps(graphs,indent=2))
    return backend(graph, inputs, **kwargs)


@functools.lru_cache(None)
def compiled_block():
    return torch.compile(moss.Block._forward_fused, backend=capture_backend, dynamic=None,
                         options={'force_static_compile':True})


moss._compiled_block_forward = compiled_block
patcher = comfy.sd.load_diffusion_model(str(args.model))
comfy.model_management.load_models_gpu([patcher])
data = torch.load(args.input, weights_only=True)
block = patcher.model.diffusion_model.layers[data['layer']]
x, pe = data['x'].to('hpu'), data['pe'].to('hpu')
mod = tuple(value.to('hpu') for value in data['mod'])
options = dict(fp8_attention=True, _moss_hpu_blockwise_rope=data['blockwise'])
for _ in range(5):
    output=block(x.clone(),mod,pe,options)
torch.hpu.synchronize()
start=time.perf_counter()
for _ in range(20):
    output=block(x.clone(),mod,pe,options)
torch.hpu.synchronize()
seconds=(time.perf_counter()-start)/20
if args.native:
    from moss_native_profile import NativeProfile
    context=NativeProfile(args.output)
else:
    context=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                              torch.profiler.ProfilerActivity.HPU])
with context as profile:
    for _ in range(args.iterations):
        output=block(x.clone(),mod,pe,options)
    torch.hpu.synchronize()
if not args.native:
    profile.profiler.export_chrome_trace(str(args.output/'block.trace.json'))
(args.output/'run.json').write_text(json.dumps(dict(input=str(args.input),seconds=seconds,iterations=args.iterations,
                                                  finite=bool(output.cpu().float().isfinite().all())),indent=2))
