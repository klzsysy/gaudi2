"""Observe full-model unload until allocation settles, retrying asynchronous flags."""
import json
from pathlib import Path
import sys
import time
import urllib.request


root = Path(sys.argv[1])
url = sys.argv[2] if len(sys.argv) > 2 else 'http://127.0.0.1:8190'
def api(route, data=None):
    req = urllib.request.Request(url + route, data=None if data is None else json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=20) as response:
        body = response.read()
        return json.loads(body) if body else None

samples = []
start = time.perf_counter()
for index in range(120):
    if index % 10 == 0:
        api('/free', {'unload_models': True, 'free_memory': True})
    stat = api('/system_stats')['devices'][0]
    samples.append({'seconds': time.perf_counter() - start, 'used': stat['vram_total'] - stat['vram_free']})
    (root / 'unload-series.json').write_text(json.dumps(samples, indent=2))
    if index >= 30 and len({x['used'] for x in samples[-12:]}) == 1 and samples[-1]['used'] < samples[0]['used']:
        print(json.dumps({'settled': samples[-1], 'initial': samples[0]}), flush=True)
        break
    time.sleep(.5)
else:
    raise TimeoutError('Unload did not settle')
