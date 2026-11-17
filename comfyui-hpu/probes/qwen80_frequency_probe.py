"""Compare table and scalar-step rotary phases at the actual decode positions."""
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from comfy.text_encoders import qwen35, llama

torch.set_grad_enabled(False)
torch.set_num_threads(32)
root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
model = SimpleNamespace(config=qwen35.Qwen35Config())
compute = qwen35.Qwen35Transformer.compute_freqs_cis
positions = torch.arange(6899, device='hpu').unsqueeze(0)
table = llama.rope_matrix(compute(model, positions, 'hpu'))
source = inspect.getsource(llama.precompute_freqs_cis).replace(
    'inv_freq_expanded.float() @ position_ids_expanded.float()',
    'inv_freq_expanded.float() * position_ids_expanded.float()')
namespace = dict(vars(llama))
exec(compile(source, '<elementwise-frequency>', 'exec'), namespace)
config = model.config
precise = llama.rope_matrix(namespace['precompute_freqs_cis'](
    int(config.head_dim * config.partial_rotary_factor), positions, config.rope_theta,
    rope_dims=config.mrope_section, interleaved_mrope=True, device='hpu'))
rows = []
for position in (0, 17, 2803, 3072, 4096, 4711, 6898):
    reference = llama.rope_matrix(compute(model, positions[:, position:position + 1].clone(), 'hpu')).cpu()
    for mode, values in (('matmul', table), ('elementwise', precise)):
        actual = values[:, :, position:position + 1].cpu()
        rows.append(dict(position=position, mode=mode, exact=torch.equal(actual, reference),
                         max_abs=(actual-reference).abs().max().item()))
(root / 'results.json').write_text(json.dumps(rows, indent=2))
print(rows, flush=True)
