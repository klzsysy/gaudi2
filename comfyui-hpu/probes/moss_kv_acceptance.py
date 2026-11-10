"""Run changed prompt/MTP shapes, model reload, and the original INT8 workflow."""
import copy
import json
import os
from pathlib import Path
import runpy
import sys
import time
import urllib.request


directory = Path(sys.argv[1])
source = json.loads((directory / 'b0-source.json').read_text())
probe = runpy.run_path(str(Path(__file__).with_name('moss_baseline.py')))
run = probe['run']

for name, megapixels, mtp, prompt in [
    ('512-auto', 0.25, 'auto', 'A red ceramic teapot beside a sunlit window, product photography.'),
    ('768-off', 0.5625, 'off', 'A small orange cat asleep on a blue velvet armchair, realistic photograph.'),
]:
    if os.environ.get('MOSS_ACCEPTANCE_SKIP_SHORT') == '1':
        continue
    graph = copy.deepcopy(source)
    graph['19']['inputs']['megapixels'] = megapixels
    graph['18:17']['inputs']['value'] = prompt
    graph['18:11']['inputs'].update(max_length=64, mtp=mtp)
    run(name, graph, diagnostic=True)

probe['api']('/free', {'unload_models': True, 'free_memory': True})
for _ in range(150):
    stats = probe['api']('/system_stats')
    if stats['devices'][0]['vram_free'] > 80 * 2**30:
        (directory / 'unloaded-system-stats.json').write_text(json.dumps(stats, indent=2))
        break
    time.sleep(.2)
else:
    raise TimeoutError('Model unload did not release HPU memory')
run('bf16-reloaded', source, diagnostic=True)
quantized = copy.deepcopy(source)
quantized['18:2']['inputs']['clip_name'] = 'qwen3.5_9b_int8_convrot.safetensors'
run('int8-user-1', quantized, diagnostic=True)
run('int8-user-2', quantized, diagnostic=True)

events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
runs = [json.loads(line) for line in (directory / 'runs.jsonl').read_text().splitlines()]
for record in runs:
    if not record['diagnostic']:
        continue
    checks = [e for e in events if e.get('prompt_id') == record['prompt_id'] and e['kind'] == 'tensor_check']
    assert len(checks) >= 2 and all(e['finite'] for e in checks), checks
    assert not record['fallback_counts_cumulative'], record
print(json.dumps({'diagnostic_requests': sum(r['diagnostic'] for r in runs), 'finite': True, 'fallback': {}}), flush=True)
