"""Image differences and contact sheet for paired prefill quality requests."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

root = Path(sys.argv[1])
records, pairs = [], []
for path in sorted(root.glob('*-baseline.png')):
    candidate = path.with_name(path.name.replace('-baseline', '-candidate'))
    if not candidate.exists():
        continue
    before, after = Image.open(path).convert('RGB'), Image.open(candidate).convert('RGB')
    a, b = np.asarray(before, dtype=np.float32), np.asarray(after, dtype=np.float32)
    delta = a - b
    records.append(dict(name=path.stem.removesuffix('-baseline'), mae=float(np.abs(delta).mean()),
                        rmse=float(np.sqrt((delta * delta).mean())), maximum=float(np.abs(delta).max())))
    repeat = path.with_name(path.stem + '-repeat.png')
    if repeat.exists():
        noise = a - np.asarray(Image.open(repeat).convert('RGB'), dtype=np.float32)
        records[-1]['baseline_repeat_mae'] = float(np.abs(noise).mean())
    pairs.append((before, after, records[-1]['name']))
sheet = Image.new('RGB', (1024, 280 * len(pairs)), 'white')
draw = ImageDraw.Draw(sheet)
for row, (before, after, name) in enumerate(pairs):
    for col, (image, label) in enumerate(((before, 'baseline'), (after, 'candidate'))):
        image.thumbnail((500, 256))
        sheet.paste(image, (col * 512, row * 280 + 24))
        draw.text((col * 512 + 4, row * 280 + 4), name + ' / ' + label, fill='black')
sheet.save(root / 'contact-sheet.jpg')
(root / 'image-differences.json').write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2))
