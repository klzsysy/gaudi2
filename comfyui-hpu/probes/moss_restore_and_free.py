"""Validate complete model release, reload, and normal cached operation."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


root = Path(os.environ['MOSS_DIT_NEXT'])
url = 'http://192.168.10.21:8189'
source = json.loads((root / 'probes/moss-user-textgen-bf16.json').read_text())


def api(route, data=None):
    request = urllib.request.Request(url + route, data=None if data is None else json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = response.read()
        return json.loads(payload) if payload else None


def run(label, graph):
    start = time.perf_counter()
    pid = api('/prompt', {'prompt': graph})['prompt_id']
    while pid not in (history := api('/history/' + pid)):
        if time.perf_counter() - start > 1200:
            raise TimeoutError(pid)
        time.sleep(.2)
    h = history[pid]
    assert h['status']['status_str'] == 'success', h['status']
    messages = dict(h['status']['messages'])
    record = {'label': label, 'prompt_id': pid, 'seconds': (messages['execution_success']['timestamp'] - messages['execution_start']['timestamp']) / 1000,
              'cached': messages.get('execution_cached', {}).get('nodes', [])}
    print(json.dumps(record), flush=True)
    return record


assert api('/queue') == {'queue_running': [], 'queue_pending': []}
records = [run('warmup', source)]
if os.environ.get('MOSS_RESTORE_ONLY') != '1':
    release = root / 'release-product'
    release.mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name('moss_memory_release.py')), str(release), url], check=True)
    actual = json.loads((release / 'unload-series.json').read_text())[-1]['used']
    baseline = json.loads((root / 'full-unload-control/unload-series.json').read_text())[-1]['used']
    assert actual <= baseline + 2**20, (actual, baseline)
    (root / 'complete-unload.json').write_text(json.dumps({'baseline': baseline, 'product': actual}, indent=2))
    records.append(run('reloaded', source))
cached = copy.deepcopy(source)
cached['18:7']['inputs']['seed'] += 1
records.append(run('cached-text', cached))
assert all(nid in records[-1]['cached'] for nid in ('18:11', '18:4', '18:5'))
(root / 'restore-results.json').write_text(json.dumps(records, indent=2))
