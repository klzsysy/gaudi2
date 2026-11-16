"""Verify the staged FP8 workflow through the ordinary service API."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('workflow', type=Path)
parser.add_argument('ui_workflow', type=Path)
a = parser.parse_args()
a.root.mkdir(parents=True, exist_ok=True)
url = 'http://192.168.10.21:8189'


def request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url + path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


queue = request('/queue')
assert not queue['queue_running'] and not queue['queue_pending'], queue
graph = json.loads(a.workflow.read_text())
graph['461']['inputs']['filename_prefix'] = 'gaudi/qwen21_fp8/normal'
payload = dict(prompt=graph, extra_data={'extra_pnginfo': {'workflow': json.loads(a.ui_workflow.read_text())}})
pid = request('/prompt', payload)['prompt_id']
print(pid, flush=True)
for _ in range(600):
    history = request('/history/' + pid)
    if pid in history:
        break
    time.sleep(2)
else:
    raise TimeoutError(pid)
result = history[pid]
(a.root / 'normal.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
assert result['status']['status_str'] == 'success', result['status']
stamps = [m[1]['timestamp'] for m in result['status']['messages'] if m[0] in ('execution_start', 'execution_success')]
report = dict(prompt_id=pid, seconds=(stamps[-1]-stamps[0])/1000, queue=request('/queue'), system_stats=request('/system_stats'))
(a.root / 'verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
