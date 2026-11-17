"""Check request-local sampling before building a complete decoder graph."""
import json
from pathlib import Path
import sys
import torch
import habana_frameworks.torch.core as htcore

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
mode = sys.argv[2]
probs = torch.full((1, 20), .05, dtype=torch.bfloat16, device='hpu')
rows = []
if mode == 'eager':
    from habana_frameworks.torch.dynamo.compile_backend import shared_layer
    shared_layer.hpu_supported_op_list.add('habana_multinomial')

    def sample(seed, probabilities):
        result = torch.ops.hpu.habana_multinomial(seed, probabilities, 1, False)
        seed.add_(1)
        return result

    compiled = torch.compile(sample, backend='hpu_backend', fullgraph=True, dynamic=False,
                             options={'force_static_compile': True})
    for repeat in range(2):
        seed = torch.tensor([42], dtype=torch.int32, device='hpu')
        rows.append([compiled(seed, probs).cpu().item() for _ in range(12)])
else:
    for repeat in range(2):
        generator = torch.Generator(device='hpu').manual_seed(42)
        htcore.mark_step()
        torch.hpu.synchronize()
        graph = torch.hpu.HPUGraph()
        graph.capture_begin()
        result = torch.multinomial(probs, 1, generator=generator)
        graph.capture_end()
        htcore.mark_step()
        torch.hpu.synchronize()
        sequence = [result.cpu().item()]
        for _ in range(11):
            graph.replay(asynchronous=True)
            sequence.append(result.cpu().item())
        rows.append(sequence)
record = dict(mode=mode, draws=rows, reproducible=rows[0] == rows[1], varied=len(set(rows[0])) > 1)
(root / 'results.json').write_text(json.dumps(record, indent=2))
print(record, flush=True)
