"""Count actual Qwen3-VL text rows and FP8 language projection work on CPU."""
import argparse
import json
import math
from pathlib import Path
import struct

from comfy.cli_args import args
args.cpu = True
from comfy.text_encoders.qwen3vl import Qwen3VLTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
a = parser.parse_args()
workflow = json.loads((a.root / 'native-clip/workflow.json').read_text())
tokenizer = Qwen3VLTokenizer()
lengths = {}
for label in ('prompt', 'negative_prompt'):
    tokens = tokenizer.tokenize_with_weights(workflow['459:452']['inputs'][label], prevent_empty_text=True)
    lengths[label] = [len(batch) for batch in tokens['qwen3vl_8b']]
model = Path('/home/sonny/ComfyUI-Models/models/text_encoders') / workflow['459:453']['inputs']['clip_name']
with model.open('rb') as stream:
    header = json.loads(stream.read(struct.unpack('<Q', stream.read(8))[0]))
weights = {k:v['shape'] for k,v in header.items() if k.startswith('model.layers.') and k.endswith('.weight') and v['dtype']=='F8_E4M3'}
result = dict(token_lengths=lengths, converted_linears=len(weights),
              fp8_weight_elements=sum(math.prod(v) for v in weights.values()))
result['fp8_linear_flops'] = 2 * sum(sum(v) for v in lengths.values()) * result['fp8_weight_elements']
(a.root / 'clip-work.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
