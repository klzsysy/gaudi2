"""Verify dynamic query padding across more lengths than Dynamo's default cache limit."""
import json
from pathlib import Path
import sys

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.use_pytorch_cross_attention = True
from habana_frameworks.torch.utils.debug import _get_fallback_op_count
from torch._dynamo.backends.registry import lookup_backend
from moss_attention_dataflow_candidate import candidate


root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
torch.set_num_threads(32)
graphs = []
backend = lookup_backend('hpu_backend')
function = candidate('batch-query-linear-softmax2d-av2d-1536')


def observed_backend(graph, inputs, **kwargs):
    graphs.append(graph.code)
    (root / 'graphs.json').write_text(json.dumps(graphs, indent=2))
    return backend(graph, inputs, **kwargs)


def forward(q, k, v):
    assert torch.compiler.is_compiling(), 'Dynamic lengths fell back to eager execution'
    return function(q, k, v)


def reference(q, k, v):
    query, key, value = q[0, 0], k[0, 0], v[0, 0]
    value_scale = value.abs().amax().float().clamp_min(1e-12) / 240.0
    quantized = torch.ops.hpu.cast_to_fp8_v2(value, value_scale.reciprocal(), False, False, torch.float8_e4m3fn)[0]
    scale = q.shape[-1] ** -0.25
    scores = (query * scale) @ (key * scale).transpose(-1, -2)
    probabilities = torch.ops.hpu.softmax_fp8.Scalar_scales(scores, -1, 1.0, 128.0)
    probability_scale = torch.full((), 1 / 128.0, device=q.device, dtype=torch.float32)
    output = torch.ops.hpu.fp8_gemm_v2(probabilities, False, quantized, False, None, q.dtype,
                                    probability_scale, value_scale, None, False)
    return output.unsqueeze(0).unsqueeze(0)


compiled = torch.compile(forward, backend=observed_backend, dynamic=None, options={'force_static_compile': True})
baseline = torch.compile(reference, backend='hpu_backend', dynamic=None, options={'force_static_compile': True})
records = []
for length in (8192, 8193, 8194, 8195, 8196, 8197, 8198, 8199, 8200, 8583, 8629, 9216, 16385, 8192):
    torch.manual_seed(173)
    q, k, v = torch.randn(1, 1, length, 384, dtype=torch.bfloat16).to('hpu').chunk(3, dim=-1)
    expected = baseline(q, k, v).cpu()
    actual = compiled(q, k, v).cpu()
    record = dict(length=length, equal=torch.equal(actual, expected), graph_count=len(graphs),
                  maximum=float((actual.float() - expected.float()).abs().max()),
                  fallback=_get_fallback_op_count())
    records.append(record)
    (root / 'dynamic-lengths.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(record), flush=True)
    assert record['equal'] and not record['fallback'], record
