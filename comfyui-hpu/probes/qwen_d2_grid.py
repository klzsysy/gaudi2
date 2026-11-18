"""Scan joint query/head tiles after the D2 block-compilation result."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time
import types

import torch
from qwen_d2_attention import tiled_attention
from comfy.ldm.modules.attention import _hpu_fast_attention


def grouped(q, k, v, tile, group):
    outputs = []
    for start in range(0, q.shape[1], group):
        outputs.append(tiled_attention(q[:, start:start+group], k[:, start:start+group],
                                       v[:, start:start+group], tile, 'loop', False))
    return torch.cat(outputs, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', type=Path)
    parser.add_argument('output', type=Path)
    options = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(32)
    configurations = [(tile, group) for tile in (1536, 2048, 3072, 4096) for group in (4, 8, 16, 32)]
    configurations += [(1024, 32), (8192, 4), (8192, 8), (2048, 1)]
    compiled = {'baseline': torch.compile(_hpu_fast_attention, backend='hpu_backend', dynamic=None,
                                          options={'force_static_compile': True})}
    for tile, group in configurations:
        name = f'q{tile}-g{group}'
        function = types.FunctionType(grouped.__code__.replace(co_name='d2_' + name), globals(), argdefs=(tile, group))
        compiled[name] = torch.compile(function, backend='hpu_backend', dynamic=None,
                                       options={'force_static_compile': True})
    results = []
    for forward in (7, 8):
        source = options.inputs / f'forward-{forward:02d}-layer-15-part-1.pt'
        saved = torch.load(source, weights_only=True)
        q, k, v = [saved[name].reshape(1, -1, 32, 128).transpose(1, 2).to('hpu') for name in ('q', 'k', 'v')]
        reference = compiled['baseline'](q, k, v).cpu().float()
        for name, function in compiled.items():
            gc.collect()
            started = time.perf_counter()
            actual = function(q, k, v).cpu().float()
            cold = time.perf_counter()-started
            assert actual.isfinite().all()
            for _ in range(3):
                function(q, k, v)
            torch.hpu.synchronize()
            times = []
            for _ in range(5):
                started = time.perf_counter()
                for _ in range(3):
                    output = function(q, k, v)
                torch.hpu.synchronize()
                times.append((time.perf_counter()-started)/3)
            record = dict(input=source.name, name=name, cold=cold, seconds=statistics.median(times),
                          times=times, rrmse=float((actual-reference).norm()/reference.norm()),
                          equal=torch.equal(actual, reference))
            results.append(record)
            (options.output / 'grid.json').write_text(json.dumps(results, indent=2))
            print(json.dumps({key: value for key, value in record.items() if key != 'times'}), flush=True)


if __name__ == '__main__':
    main()
