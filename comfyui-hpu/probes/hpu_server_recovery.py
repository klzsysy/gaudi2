"""Validate broken HTTP clients and replay saved pending prompts on the actual server."""
import json
import os
from pathlib import Path
import signal
import socket
import struct
import time
import urllib.parse
import urllib.request


root = Path(os.environ['HPU_RECOVERY_DIR'])
checkout = Path.cwd()
url = os.environ['HPU_RECOVERY_URL']
host = urllib.parse.urlsplit(url)
pid = int((checkout / '.run/hpu.pid').read_text())


def check_signal():
    fields = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
    bit = 1 << (signal.SIGPIPE - 1)
    assert int(fields['SigIgn'].strip(), 16) & bit, 'SIGPIPE must be ignored before injection'
    assert not int(fields['SigCgt'].strip(), 16) & bit


def api(route, data=None):
    request = urllib.request.Request(url + route, data=None if data is None else json.dumps(data).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


check_signal()
static = checkout / '.venv/lib/python3.10/site-packages/comfyui_frontend_package/static'
asset = max(static.rglob('*.js'), key=lambda p: p.stat().st_size)
route = '/' + asset.relative_to(static).as_posix()
disconnects = []
for _ in range(8):
    with socket.create_connection((host.hostname, host.port), timeout=5) as client:
        client.sendall(f'GET {route} HTTP/1.1\r\nHost: {host.netloc}\r\nConnection: close\r\n\r\n'.encode())
        status = client.recv(512).split(b'\r\n', 1)[0].decode()
        assert '200' in status, status
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
        disconnects.append(status)
os.kill(pid, signal.SIGPIPE)
check_signal()
assert api('/queue') == {'queue_running': [], 'queue_pending': []}
(root / 'disconnect-check.json').write_text(json.dumps({'pid': pid, 'asset': route, 'aborted_downloads': len(disconnects), 'explicit_sigpipe': True}, indent=2))
records = []
for saved in json.loads((root / 'pending.json').read_text()):
    payload = {'prompt': saved['prompt'], 'extra_data': saved['extra_data']}
    if saved['extra_data'].get('client_id'):
        payload['client_id'] = saved['extra_data']['client_id']
    start = time.perf_counter()
    response = api('/prompt', payload)
    prompt_id = response['prompt_id']
    print('submitted', saved['original_prompt_id'], '->', prompt_id, flush=True)
    while prompt_id not in (history := api('/history/' + prompt_id)):
        if time.perf_counter() - start > 1200:
            raise TimeoutError(prompt_id)
        time.sleep(.5)
    history = history[prompt_id]
    (root / (prompt_id + '-history.json')).write_text(json.dumps(history, indent=2))
    assert history['status']['status_str'] == 'success', history['status']
    images = [image for output in history['outputs'].values() for image in output.get('images', [])]
    assert images, 'Expected a generated image'
    for index, image in enumerate(images):
        with urllib.request.urlopen(url + '/view?' + urllib.parse.urlencode(image), timeout=30) as response:
            (root / f'{prompt_id}-{index}.png').write_bytes(response.read())
    check_signal()
    record = {'original_prompt_id': saved['original_prompt_id'], 'replayed_prompt_id': prompt_id,
              'seconds': time.perf_counter() - start, 'status': history['status']['status_str'], 'images': images}
    records.append(record)
    (root / 'replay-results.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(record), flush=True)
