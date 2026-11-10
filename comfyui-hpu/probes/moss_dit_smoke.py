"""Same-server shape changes, unload/reload and original INT8 user workflows."""
import copy
import json
import os
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

directory = Path(sys.argv[1]) / 'acceptance'
directory.mkdir(exist_ok=True)
fixed = json.loads((directory.parent / 'workflow.json').read_text())
user = json.loads(Path(__file__).with_name('moss-user-textgen-bf16.json').read_text())
user['18:2']['inputs']['clip_name'] = 'qwen3.5_9b_int8_convrot.safetensors'


def api(path, data=None):
    request = urllib.request.Request('http://127.0.0.1:8190' + path, data=None if data is None else json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()
        return json.loads(payload) if payload else None


cases = []
for name, megapixels, prompt in [
    ('512-short', 0.25, 'A red ceramic teapot on a wooden table beside a sunlit window, product photography.'),
    ('768-weighted', 0.5625, 'A clean poster featuring (a blue ceramic vase:1.2), three yellow flowers, a white background, and large readable text "MOSS" at the top.'),
    ('1024-reloaded', 1.0, 'A small orange cat asleep on a blue velvet armchair beside a window, soft afternoon sunlight, realistic photograph.'),
]:
    graph = copy.deepcopy(fixed)
    graph['19']['inputs']['megapixels'] = megapixels
    graph['18:17']['inputs']['value'] = prompt
    cases.append((name, graph))
cases.extend([('int8-user-1', user), ('int8-user-2', user)])
if os.environ.get('MOSS_SMOKE_DIRECT_ONLY') == '1':
    cases = cases[:3]
records = []
for name, graph in cases:
    if name == '1024-reloaded':
        api('/free', {'unload_models': True, 'free_memory': True})
        for _ in range(150):
            stats = api('/system_stats')
            if stats['devices'][0]['vram_free'] > 80 * 2**30:
                (directory / 'unloaded-system-stats.json').write_text(json.dumps(stats, indent=2))
                break
            time.sleep(.2)
        else:
            raise TimeoutError('models did not unload')
    (directory / (name + '-workflow.json')).write_text(json.dumps(graph, indent=2))
    prompt_id = api('/prompt', {'prompt': graph, 'extra_data': {'moss_diagnostic': True}})['prompt_id']
    start = time.perf_counter()
    while prompt_id not in (history := api('/history/' + prompt_id)):
        if time.perf_counter() - start > 1800:
            raise TimeoutError(prompt_id)
        time.sleep(.2)
    history = history[prompt_id]
    (directory / (name + '-history.json')).write_text(json.dumps(history))
    assert history['status']['status_str'] == 'success', history['status']
    events = [json.loads(line) for line in (directory.parent / 'events.jsonl').read_text().splitlines()]
    events = [e for e in events if e.get('prompt_id') == prompt_id]
    checks = [e for e in events if e['kind'] == 'tensor_check']
    assert len(checks) >= 2 and all(e['finite'] for e in checks), checks
    end = next(e for e in events if e['kind'] == 'request_end')
    assert not end['fallback_counts'], end['fallback_counts']
    for output in history['outputs'].values():
        for image in output.get('images', []):
            with urllib.request.urlopen('http://127.0.0.1:8190/view?' + urllib.parse.urlencode(image)) as response:
                (directory / (name + '.png')).write_bytes(response.read())
    record = {'name': name, 'prompt_id': prompt_id, 'diagnostic_seconds': time.perf_counter() - start,
              'checks': checks, 'fallback': end['fallback_counts'],
              'tokens': [e['output_tokens'] for e in events if e['kind'] == 'generate_end']}
    records.append(record)
    (directory / 'results.json').write_text(json.dumps(records, indent=2))
    print('acceptance', name, record['diagnostic_seconds'], record['tokens'], 'PASS', flush=True)
