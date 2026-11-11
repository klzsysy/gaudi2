"""Paired prefill quality, changing shapes, reload, and original INT8 smoke."""
import copy
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

root = Path(sys.argv[1])
if (root / 'skip-quality').exists():
    print('Quality moved to the subsequent candidate covering FP32 encoding.', flush=True)
    sys.exit(0)
destination = root / 'quality'
destination.mkdir(exist_ok=True)
source = json.loads((root / 'source.json').read_text())


def api(path, payload=None):
    request = urllib.request.Request('http://127.0.0.1:8190' + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        value = response.read()
        return json.loads(value) if value else None


prompts = [
    ('short', .25, 'A red ceramic teapot on a wooden table beside a sunlit window, product photography.'),
    ('poster', .5625, 'A clean poster featuring (a blue ceramic vase:1.2), three yellow flowers, a white background, and large readable text "MOSS" at the top.'),
    ('person', 1., 'Portrait photograph of an elderly woman wearing a red scarf, detailed face and hands, holding a small white teacup, natural window light.'),
    ('composition', 1., 'A wide photograph of a busy outdoor market: a green bicycle in the left foreground, a vendor arranging oranges at the center, two shoppers under a blue umbrella on the right, brick storefronts in the background.'),
    ('long', .5625, 'An atmospheric architectural photograph of a quiet seaside bookshop at dusk. Through the large front window, wooden shelves hold colorful books, a reading lamp illuminates a round table, and an orange cat sleeps on a dark green armchair. Outside, rainwater on the pavement reflects the warm interior and a soft purple sky. The view is carefully composed with the doorway on the left and a parked bicycle on the right, realistic materials, distinct foreground and background, soft cinematic light.'),
    ('street', 1., source['18:17']['inputs']['value']),
    ('int8', 1., source['18:17']['inputs']['value']),
]
records = []
for name, size, prompt in prompts:
    if name == 'person':
        api('/free', {'unload_models': True, 'free_memory': True})
        for _ in range(150):
            stats = api('/system_stats')
            if stats['devices'][0]['vram_free'] > 80 * 2**30:
                (destination / 'unloaded.json').write_text(json.dumps(stats))
                break
            time.sleep(.2)
        else:
            raise TimeoutError('unload')
    variants = (False,) if name == 'int8' else (True, False, True) if name in ('poster', 'street') else (True, False)
    for index, baseline in enumerate(variants):
        label = f'{name}-' + ('baseline' if baseline else 'candidate')
        if index == 2:
            label += '-repeat'
        graph = copy.deepcopy(source)
        graph['18:16']['inputs']['switch'] = False
        graph['18:17']['inputs']['value'] = prompt
        graph['18:7']['inputs']['seed'] = 42
        graph['19']['inputs']['megapixels'] = size
        if name == 'int8':
            graph['18:16']['inputs']['switch'] = True
            graph['18:2']['inputs']['clip_name'] = 'qwen3.5_9b_int8_convrot.safetensors'
        start = time.perf_counter()
        prompt_id = api('/prompt', {'prompt': graph, 'extra_data': {'moss_recompute': True, 'moss_diagnostic': True,
            'moss_prefill_baseline': baseline, 'moss_rope_baseline': baseline,
            'moss_norm_baseline': baseline}})['prompt_id']
        while prompt_id not in (history := api('/history/' + prompt_id)):
            if time.perf_counter() - start > 1200:
                raise TimeoutError(prompt_id)
            time.sleep(.2)
        history = history[prompt_id]
        assert history['status']['status_str'] == 'success', history['status']
        (destination / (label + '-history.json')).write_text(json.dumps(history))
        events = [json.loads(s) for s in (root / 'events.jsonl').read_text().splitlines()]
        events = [e for e in events if e.get('prompt_id') == prompt_id]
        checks = [e for e in events if e['kind'] == 'tensor_check']
        assert len(checks) >= 2 and all(e['finite'] for e in checks)
        end = next(e for e in events if e['kind'] == 'request_end')
        assert not end['fallback_counts'], end['fallback_counts']
        for output in history['outputs'].values():
            for image in output.get('images', []):
                with urllib.request.urlopen('http://127.0.0.1:8190/view?' + urllib.parse.urlencode(image)) as response:
                    (destination / (label + '.png')).write_bytes(response.read())
        records.append(dict(label=label, prompt_id=prompt_id, diagnostic_seconds=time.perf_counter()-start, checks=checks))
        (destination / 'results.json').write_text(json.dumps(records, indent=2))
        print(label, 'PASS', flush=True)
