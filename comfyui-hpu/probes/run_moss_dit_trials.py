"""Serial fixed-caption workflows; real recomputation, loader caches retained."""
import json
import hashlib
import os
from pathlib import Path
import statistics
import subprocess
import time
import urllib.parse
import urllib.request

root = Path.cwd()
directory = Path(os.environ['MOSS_DIT_ROOT'])
directory.mkdir(parents=True, exist_ok=True)
probes = Path(__file__).parent
python = str(root / '.venv/bin/python')
base_env = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so', PYTHONPATH=str(root))


def api(path, data=None):
    req = urllib.request.Request('http://127.0.0.1:8190' + path, data=None if data is None else json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


for mode in os.environ.get('MOSS_DIT_CASES', 'original').split(','):
    target = directory / mode
    target.mkdir(exist_ok=True)
    source_files = ('comfy/ldm/moss_image/model.py', 'comfy/ldm/moss_image/hpu.py', 'comfy/model_base.py', 'comfy/ops.py',
                    'comfy/text_encoders/qwen35.py', 'comfy/text_encoders/hpu_gated_delta.py')
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_files if (root / name).exists()}
    (target / 'source-hashes.json').write_text(json.dumps(source_hashes, indent=2))
    workflow = json.loads((probes / 'moss-user-textgen-bf16.json').read_text())
    workflow['18:16']['inputs']['switch'] = False
    workflow['18:17']['inputs']['value'] = Path(os.environ['MOSS_CAPTION']).read_text()
    if 'MOSS_DIT_STEPS' in os.environ:
        workflow['18:7']['inputs']['steps'] = int(os.environ['MOSS_DIT_STEPS'])
    if 'MOSS_DIT_MEGAPIXELS' in os.environ:
        workflow['19']['inputs']['megapixels'] = float(os.environ['MOSS_DIT_MEGAPIXELS'])
    (target / 'workflow.json').write_text(json.dumps(workflow, indent=2))
    env = dict(base_env, MOSS_EVENTS=str(target / 'events.jsonl'), MOSS_DIT_DIRECTORY=str(target), MOSS_DIT_MODE=mode)
    command = [python, str(probes / 'moss_dit_server.py'), '--disable-dynamic-vram', '--disable-pinned-memory', '--disable-async-offload', '--disable-xformers', '--use-pytorch-cross-attention', '--listen', '127.0.0.1', '--port', '8190']
    records = []
    with (target / 'server.log').open('w') as log, (target / 'hpu.csv').open('w') as telemetry_log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        telemetry = None
        try:
            for _ in range(120):
                if server.poll() is not None:
                    raise RuntimeError('server startup failed: ' + str(target))
                try:
                    api('/system_stats')
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError('startup')
            telemetry = subprocess.Popen(['hl-smi', '--query-aip=timestamp,utilization.aip,utilization.memory,memory.used,power.draw', '--format=csv,nounits', '-l', '100ms'], stdout=telemetry_log)
            for i in range(int(os.environ.get('MOSS_DIT_RUNS', '6'))):
                start = time.perf_counter()
                extra = {'moss_diagnostic': os.environ.get('MOSS_DIT_DIAGNOSTIC') == '1'}
                cached_text = os.environ.get('MOSS_DIT_CACHE_TEXT') == '1'
                if cached_text:
                    extra['moss_recompute_nodes'] = ['KSampler', 'VAEDecode']
                prompt_id = api('/prompt', {'prompt': workflow, 'extra_data': extra})['prompt_id']
                while prompt_id not in (history := api('/history/' + prompt_id)):
                    if time.perf_counter() - start > 1800:
                        raise TimeoutError(prompt_id)
                    time.sleep(0.2)
                history = history[prompt_id]
                (target / f'history-{i}.json').write_text(json.dumps(history))
                assert history['status']['status_str'] == 'success', history['status']
                events = [json.loads(line) for line in (target / 'events.jsonl').read_text().splitlines()]
                events = [e for e in events if e.get('prompt_id') == prompt_id]
                stages = {}
                for end in (e for e in events if e['kind'] == 'node_end'):
                    begin = next(e for e in events if e['kind'] == 'node_start' and e['node'] == end['node'])
                    stages[end['node']] = {'class': end['class_type'], 'seconds': (end['mono_ns'] - begin['mono_ns']) / 1e9}
                assert any(x['class'] == 'KSampler' for x in stages.values())
                assert any(x['class'] == 'CLIPTextEncode' for x in stages.values()) == (not cached_text or i == 0)
                assert not any(x['class'] == 'TextGenerate' for x in stages.values())
                request_start = next(e for e in events if e['kind'] == 'request_start')
                request_end = next(e for e in events if e['kind'] == 'request_end')
                record = {'index': i, 'prompt_id': prompt_id, 'seconds': (request_end['mono_ns'] - request_start['mono_ns']) / 1e9,
                          'stages': stages, 'fallback': request_end['fallback_counts'], 'memory': request_end['memory_stats']}
                records.append(record)
                (target / 'runs.json').write_text(json.dumps(records, indent=2))
                for output in history['outputs'].values():
                    for image in output.get('images', []):
                        with urllib.request.urlopen('http://127.0.0.1:8190/view?' + urllib.parse.urlencode(image)) as response:
                            (target / f'image-{i}.png').write_bytes(response.read())
                print(mode, i, record['seconds'], stages.get('18:7'), record['fallback'], flush=True)
            if os.environ.get('MOSS_DIT_SMOKE') == '1':
                subprocess.run([python, str(probes / 'moss_dit_smoke.py'), str(target)], env=env, check=True)
        finally:
            if telemetry is not None:
                telemetry.terminate()
                telemetry.wait(timeout=10)
            server.terminate()
            server.wait(timeout=60)
    if len(records) > 1:
        summary = {'request_median': statistics.median(r['seconds'] for r in records[1:]),
                   'dit_median': statistics.median(r['stages']['18:7']['seconds'] for r in records[1:])}
        (target / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(mode, summary, flush=True)
