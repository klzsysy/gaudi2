"""Compare natural token sequences and paired PNGs from fixed-tree workflows."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


root = Path(sys.argv[1])
directories = [root / 'quality-baseline', root / 'quality-product']
events = [[json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
          for directory in directories]
records = []
for precision in ('bf16', 'fp8'):
    for prompt in ('teapot', 'poster', 'person'):
        label = f'{precision}-{prompt}'
        generated = []
        for directory, rows in zip(directories, events):
            history = json.loads((directory / f'{label}-history.json').read_text())
            prompt_id = history['prompt'][1]
            generated.append(next(row for row in rows
                                  if row.get('prompt_id') == prompt_id and row['kind'] == 'generate_end'))
        images = [np.asarray(Image.open(directory / f'{label}.png').convert('RGB')).astype(np.int16)
                  for directory in directories]
        difference = np.abs(images[0] - images[1])
        record = dict(label=label, baseline_tokens=generated[0]['output_tokens'],
                      product_tokens=generated[1]['output_tokens'],
                      tokens_equal=generated[0]['token_ids'] == generated[1]['token_ids'],
                      text_equal=generated[0]['text'] == generated[1]['text'],
                      pixel_mae=float(difference.mean()), pixel_max=int(difference.max()))
        records.append(record)
(root / 'natural-quality.json').write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2))
