"""Check cold, cached and 768px requests through the ordinary deployed service."""
import copy
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

root = Path(sys.argv[1])
url = sys.argv[2]
source = json.loads((root / 'baseline/source.json').read_text())
records = []


def api(route, data=None):
    request = urllib.request.Request(url + route, data=None if data is None else json.dumps(data).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def run(label, graph):
    (root / f'deployment-{label}-source.json').write_text(json.dumps(graph, indent=2))
    prompt_id = api('/prompt', {'prompt': graph})['prompt_id']
    start = time.monotonic()
    while prompt_id not in (history := api('/history/' + prompt_id)):
        if time.monotonic() - start > 1200:
            raise TimeoutError(prompt_id)
        time.sleep(.5)
    history = history[prompt_id]
    (root / f'deployment-{label}-history.json').write_text(json.dumps(history, indent=2))
    assert history['status']['status_str'] == 'success', history['status']
    messages = dict(history['status']['messages'])
    elapsed = (messages['execution_success']['timestamp'] - messages['execution_start']['timestamp']) / 1000
    cached = messages.get('execution_cached', {}).get('nodes', [])
    if label == 'cached':
        assert '18:11' in cached, cached
    for output in history['outputs'].values():
        for item in output.get('images', []):
            with urllib.request.urlopen(url + '/view?' + urllib.parse.urlencode(item)) as response:
                (root / f'deployment-{label}.png').write_bytes(response.read())
    record = dict(label=label, prompt_id=prompt_id, seconds=elapsed, cached_nodes=cached)
    records.append(record)
    print(json.dumps(record), flush=True)


run('cold', source)
cached = copy.deepcopy(source)
cached['18:7']['inputs']['seed'] = (cached['18:7']['inputs']['seed'] + 1) % (2**64)
run('cached', cached)
small = copy.deepcopy(cached)
small['19']['inputs']['megapixels'] = .5625
small['18:11']['inputs']['mtp'] = 'off'
small['18:11']['inputs']['max_length'] = 32
run('768', small)
queue = {key: len(value) for key, value in api('/queue').items()}
stats = api('/system_stats')
result = dict(runs=records, queue=queue, torch_version=stats['system']['pytorch_version'],
              device=stats['devices'][0]['name'])
(root / 'deployment.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
