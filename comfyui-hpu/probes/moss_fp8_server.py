"""Reference execution from the same FP8 checkpoint, with resident decoded BF16 weights."""
import os
from pathlib import Path
import runpy
import comfy.options
comfy.options.enable_args_parsing()
import comfy.ops as ops
import torch
if os.environ.get('MOSS_FP8_MODE') == 'reference':
    pick = ops.pick_operations

    def pick_reference(*args, **kwargs):
        kwargs['load_device'] = torch.device('cpu')
        return pick(*args, **kwargs)
    ops.pick_operations = pick_reference
    load = ops._load_quantized_module

    def load_reference(module, *args, **kwargs):
        load(module, *args, **kwargs)
        if isinstance(module.weight, ops.QuantizedTensor) and module.quant_format == 'float8_e4m3fn':
            module.weight = torch.nn.Parameter(module.weight.dequantize(), requires_grad=False)
            module.layout_type = None
            module.quant_format = None
    ops._load_quantized_module = load_reference
runpy.run_path(str(Path(__file__).with_name('moss_observed_server.py')), run_name='__main__')
