"""Fixed-caption timings, phase AIP and image deltas; no inferred hardware MFU."""
import csv
import datetime
import json
from pathlib import Path
import statistics
import sys

import numpy as np
from PIL import Image

root = Path(sys.argv[1])
summary = {}
reference_path = Path(sys.argv[2]) if len(sys.argv) > 2 else root / 'original/image-1.png'
reference = np.asarray(Image.open(reference_path).convert('RGB'), dtype=np.float64)
for directory in root.iterdir():
    if not (directory / 'runs.json').exists():
        continue
    runs = json.loads((directory / 'runs.json').read_text())
    if not runs:
        continue
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    samples = []
    for row in list(csv.reader((directory / 'hpu.csv').open()))[1:]:
        if len(row) == 5:
            stamp = datetime.datetime.strptime(row[0].replace(' CST', ''), '%a %b %d %H:%M:%S %Y').replace(tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
            samples.append((stamp, *map(float, row[1:])))
    warm = runs[1:]
    record = {'count': len(warm), 'cold': runs[0]['seconds'], 'request_seconds': [r['seconds'] for r in warm],
              'dit_seconds': [r['stages']['18:7']['seconds'] for r in warm], 'fallback': [r['fallback'] for r in runs]}
    if warm:
        record['request_median'] = statistics.median(record['request_seconds'])
        record['dit_median'] = statistics.median(record['dit_seconds'])
        selected = []
        for r in warm:
            es = [e for e in events if e.get('prompt_id') == r['prompt_id'] and e.get('node') == '18:7']
            lo = next(e['wall_ns'] / 1e9 for e in es if e['kind'] == 'node_start')
            hi = next(e['wall_ns'] / 1e9 for e in es if e['kind'] == 'node_end')
            selected.extend(s for s in samples if lo <= s[0] and s[0] + 1 <= hi)
        record['dit_telemetry'] = {'samples': len(selected), 'aip_mean': statistics.mean(s[1] for s in selected),
                                   'aip_max': max(s[1] for s in selected), 'power_54V_mean': statistics.mean(s[4] for s in selected)}
        record['peak_allocator_GiB'] = max(r['memory']['MaxInUse'] for r in runs) / 2**30
        pixels = np.asarray(Image.open(directory / 'image-1.png').convert('RGB'), dtype=np.float64)
        if pixels.shape == reference.shape:
            delta = pixels - reference
            mse = np.mean(delta ** 2)
            record['image'] = {'MAE_255': float(np.mean(abs(delta))), 'RMSE_255': float(mse ** .5),
                               'PSNR_dB': float(10 * np.log10(255**2 / mse)) if mse else None,
                               'exact': bool(np.array_equal(pixels, reference))}
        record['warm_images_equal'] = all(np.array_equal(np.asarray(Image.open(directory / f'image-{r["index"]}.png').convert('RGB')), pixels) for r in warm)
        record['within_run_MAE_255'] = [float(np.mean(abs(np.asarray(Image.open(directory / f'image-{r["index"]}.png').convert('RGB'), dtype=np.float64) - pixels))) for r in warm]
    summary[directory.name] = record
(root / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
