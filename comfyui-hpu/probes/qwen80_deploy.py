"""Deploy the validated index snapshot and verify ordinary API generation."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

import numpy as np
from PIL import Image

BASE = Path('/home/sonny/gaudi2/comfyui-hpu')
ROOT = BASE / 'performance-qwen80-2026-11-16/deployment'
PRODUCT = Path('/home/sonny/ComfyUI-Gaudi')
URL = 'http://192.168.10.21:8189'


def request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(URL + path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        content = response.read()
    return json.loads(content) if content else None


def generate(label, graph, original, reference):
    graph = copy.deepcopy(graph)
    graph['461']['inputs']['filename_prefix'] = 'gaudi/qwen80_deployed/' + label
    pid = request('/prompt', dict(prompt=graph))['prompt_id']
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        history = request('/history/' + pid)
        if pid in history:
            break
        time.sleep(1)
    else:
        raise TimeoutError(pid)
    result = history[pid]
    (ROOT / (label + '-history.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    assert result['status']['status_str'] == 'success', result['status']
    assert result['outputs']['459:472']['text'] == original['outputs']['459:472']['text']
    item = result['outputs']['461']['images'][0]
    path = PRODUCT / 'output' / item['subfolder'] / item['filename']
    assert np.array_equal(reference, np.asarray(Image.open(path).convert('RGBA')))
    stamps = [m[1]['timestamp'] for m in result['status']['messages'] if m[0] in ('execution_start', 'execution_success')]
    cached = next(m[1]['nodes'] for m in result['status']['messages'] if m[0] == 'execution_cached')
    assert not {'459:471', '459:452', '459:486', '459:457'} & set(cached), cached
    row = dict(label=label, prompt_id=pid, seconds=(stamps[-1] - stamps[0]) / 1000,
               cached_nodes=cached, caption_equal=True, rgba_equal=True, image=str(path))
    print(json.dumps(row), flush=True)
    return row


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    queue = request('/queue')
    assert not queue['queue_running'] and not queue['queue_pending'], queue
    backup = ROOT / 'baseline'
    backup.mkdir(exist_ok=True)
    paths = list(manifest['file_sha256']) + ['.run/deployed-version', '.run/deployed-staged.json']
    existing = []
    for name in paths:
        path = PRODUCT / name
        if path.exists():
            destination = backup / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            existing.append(name)
    for name, expected in manifest['file_sha256'].items():
        assert hashlib.sha256((ROOT / 'candidate' / name).read_bytes()).hexdigest() == expected
    subprocess.run(['./stop-hpu.sh'], cwd=PRODUCT, check=True)
    success = False
    try:
        for name in manifest['file_sha256']:
            target = PRODUCT / name
            temporary = target.with_suffix(target.suffix + '.deploy')
            shutil.copy2(ROOT / 'candidate' / name, temporary)
            os.replace(temporary, target)
        (PRODUCT / '.run/deployed-version').write_text(manifest['identity'] + '\n')
        (PRODUCT / '.run/deployed-staged.json').write_text(json.dumps(manifest, indent=2))
        subprocess.run(['./start-hpu.sh'], cwd=PRODUCT, check=True, timeout=180)
        previous = BASE / 'performance-qwen21-latest-2026-11-16'
        original = json.loads((previous / 'original.json').read_text())
        reference = np.asarray(Image.open(previous / 'original.png').convert('RGBA'))
        graph = json.loads((BASE / 'workflows/qwen21-latest-2026-11-16-api.json').read_text())
        runs = [generate('cold', graph, original, reference)]
        graph['459:471']['inputs']['mtp'] = 'off'
        runs.append(generate('warm-off', graph, original, reference))
        graph['459:471']['inputs']['mtp'] = 'auto'
        runs.append(generate('warm-auto', graph, original, reference))
        request('/free', dict(unload_models=True, free_memory=True))
        samples = []
        for _ in range(6):
            time.sleep(10)
            samples.append(request('/system_stats'))
        (ROOT / 'unload.json').write_text(json.dumps(samples, indent=2))
        runs.append(generate('reload', graph, original, reference))
        actual = {name: hashlib.sha256((PRODUCT / name).read_bytes()).hexdigest() for name in manifest['file_sha256']}
        assert actual == manifest['file_sha256']
        manifest.update(status='deployed_awaiting_human_acceptance', runs=runs,
                        deployed_sha256=actual, queue=request('/queue'))
        assert not manifest['queue']['queue_running'] and not manifest['queue']['queue_pending']
        (ROOT / 'deployment.json').write_text(json.dumps(manifest, indent=2))
        (PRODUCT / '.run/deployed-staged.json').write_text(json.dumps(manifest, indent=2))
        success = True
    finally:
        if not success:
            subprocess.run(['./stop-hpu.sh'], cwd=PRODUCT, check=True)
            for name in paths:
                if name in existing:
                    shutil.copy2(backup / name, PRODUCT / name)
                else:
                    (PRODUCT / name).unlink(missing_ok=True)
            subprocess.run(['./start-hpu.sh'], cwd=PRODUCT, check=True, timeout=180)


if __name__ == '__main__':
    main()
