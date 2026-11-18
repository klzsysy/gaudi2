"""Capture this SDK's MME descriptors and SCAL configuration in isolation."""
import argparse
import json
import os
from pathlib import Path

import torch
import habana_frameworks.torch.core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    os.chdir(options.output)
    torch.set_grad_enabled(False)
    torch.set_num_threads(32)
    torch.manual_seed(17)
    a = torch.randint(-2, 3, (options.m, options.k)).bfloat16()
    b = torch.randint(-2, 3, (options.k, options.n)).bfloat16()
    function = torch.compile(lambda a, b: a @ b, backend='hpu_backend', fullgraph=True)
    output = function(a.to('hpu'), b.to('hpu')).cpu()
    reference = (a.float() @ b.float()).bfloat16()
    result = dict(shape=[options.m, options.n, options.k], equal=torch.equal(output, reference),
                  torch_version=torch.__version__)
    (options.output / 'descriptor-result.json').write_text(json.dumps(result, indent=2))
    assert result['equal'], result
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
