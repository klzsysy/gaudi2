"""Verify ordinary API cache invalidation and same-shape prompt changes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
from PIL import Image


URL = 'http://192.168.10.21:8189'
PRODUCT = Path('/home/sonny/ComfyUI-Gaudi')
BASE = Path('/home/sonny/gaudi2/comfyui-hpu')


def request(path, value=None):
    data = json.dumps(value).encode() if value is not None else None
    req = urllib.request.Request(URL + path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    args.root.mkdir()
    queue = request('/queue')
    assert not queue['queue_running'] and not queue['queue_pending'], queue
    source = json.loads((BASE / 'performance-qwen-startup-fix-2026-11-17/validation/api/cold-product.json').read_text())
    original = source['prompt'][2]
    prompt = original['459:471']['inputs']['prompt']
    blue = prompt.replace('灰褐色', '深蓝色')
    red = prompt.replace('灰褐色', '深红色')
    cases = [('blue-first', blue, 'auto'), ('red-first', red, 'auto'),
             ('blue-return', blue, 'auto'), ('red-recompute', red, 'off'),
             ('blue-recompute', blue, 'off'), ('system-change', blue, 'auto'),
             ('original', prompt, 'auto')]
    records = []
    references = {}
    logfile = PRODUCT / 'logs/hpu.log'
    log_offset = logfile.stat().st_size
    for label, text, mtp in cases:
        graph = copy.deepcopy(original)
        graph['459:471']['inputs'].update(prompt=text, mtp=mtp)
        graph['461']['inputs']['filename_prefix'] = 'gaudi/qwen_cache_audit/' + label
        if label == 'system-change':
            graph['459:475']['inputs']['value'] += '\nPreserve the deep blue color of the dress explicitly in the final description.'
        pid = request('/prompt', {'prompt': graph})['prompt_id']
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            history = request('/history/' + pid)
            if pid in history:
                break
            time.sleep(.5)
        else:
            raise TimeoutError(pid)
        result = history[pid]
        (args.root / (label + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        assert result['status']['status_str'] == 'success', result['status']
        cached = next(m[1]['nodes'] for m in result['status']['messages'] if m[0] == 'execution_cached')
        if label in ('blue-first', 'red-first', 'red-recompute', 'blue-recompute', 'system-change'):
            assert '459:471' not in cached, (label, cached)
        caption = result['outputs']['459:472']['text']
        item = result['outputs']['461']['images'][0]
        image = PRODUCT / 'output' / item['subfolder'] / item['filename']
        pixels = np.asarray(Image.open(image).convert('RGBA'))
        if label.endswith('first'):
            references[text] = caption, pixels
        equivalent = None
        if label in ('blue-return', 'red-recompute', 'blue-recompute'):
            previous_caption, previous_pixels = references[text]
            equivalent = caption == previous_caption and np.array_equal(pixels, previous_pixels)
            assert equivalent, label
        stamps = [m[1]['timestamp'] for m in result['status']['messages'] if m[0] in ('execution_start', 'execution_success')]
        row = dict(label=label, prompt_id=pid, seconds=(stamps[-1] - stamps[0]) / 1000,
                   prompt_sha256=hashlib.sha256(text.encode()).hexdigest(), cached_nodes=cached,
                   caption_sha256=hashlib.sha256(json.dumps(caption, ensure_ascii=False).encode()).hexdigest(),
                   rgba_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                   repeat_or_recompute_equal=equivalent, image=str(image))
        records.append(row)
        (args.root / 'requests.json').write_text(json.dumps(records, indent=2))
        print(row, flush=True)
    assert references[blue][0] != references[red][0]
    assert not np.array_equal(references[blue][1], references[red][1])
    original_image = source['outputs']['461']['images'][0]
    original_pixels = np.asarray(Image.open(PRODUCT / 'output' / original_image['subfolder'] / original_image['filename']).convert('RGBA'))
    assert caption == source['outputs']['459:472']['text']
    assert np.array_equal(pixels, original_pixels)
    with logfile.open('rb') as stream:
        stream.seek(log_offset)
        logs = stream.read()
    (args.root / 'ordinary.log').write_bytes(logs)
    manifest = json.loads((PRODUCT / '.run/deployed-staged.json').read_text())
    hashes = {name: hashlib.sha256((PRODUCT / name).read_bytes()).hexdigest() for name in manifest['file_sha256']}
    assert hashes == manifest['file_sha256']
    summary = dict(deployment=manifest['identity'], requests=records,
                   blue_red_caption_different=True, blue_red_rgba_different=True,
                   original_restored_caption_and_rgba_equal=True,
                   source_sha256=hashes, queue=request('/queue'),
                   batch_as_strided_warnings=logs.count(b'UserWarning: hpu::batch_as_strided'))
    (args.root / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
