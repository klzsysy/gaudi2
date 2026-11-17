"""Replace the current immediate recurrence while retaining its surrounding casts."""
import inspect
import os
from pathlib import Path
import textwrap
import torch
from comfy.text_encoders import qwen35

torch.ops.load_library(str(Path(os.environ['QWEN_IMMEDIATE_ROOT']) / 'bridge/qwen_gdn_immediate_bridge.so'))


@torch.library.register_fake('custom_op::qwen_gdn_immediate')
def metadata(state, query, key, value, decay, beta):
    return torch.empty_like(state), torch.empty_like(value)


def recurrence(state, query, key, value, decay, beta):
    updated, output = torch.ops.custom_op.qwen_gdn_immediate(
        state.transpose(-1, -2).contiguous(), query, key, value, decay, beta)
    return updated.transpose(-1, -2), output


source = textwrap.dedent(inspect.getsource(qwen35.GatedDeltaNet._decode_single))
start = source.index('    prediction = ')
end = source.index('    output = output.to', start)
source = source[:start] + '''    updated, output = recurrence(state, query, key, value, decay, beta)
    state.copy_(updated)
''' + source[end:]
namespace = dict(vars(qwen35), recurrence=recurrence)
qwen35.recurrence = recurrence
exec(compile(source, '<immediate-tpc>', 'exec'), namespace)
qwen35.GatedDeltaNet._decode_single = namespace['_decode_single']
