"""Run cold + five warm full workflows per mode, preserving loader caches."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import time
import urllib.request

root = Path.cwd()
directory = Path(os.environ['MOSS_EXPERIMENT_DIR'])
probes = directory / 'probes'
base_env = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so', PYTHONPATH=str(root))
python = str(root / '.venv/bin/python')
for mode in os.environ.get('MOSS_CASES', 'compiled,original').split(','):
    target = directory / ('full-' + mode)
    target.mkdir(exist_ok=True)
    (target / 'experiment-options.json').write_text(json.dumps({name: os.environ.get(name) for name in ('MOSS_KV_MODE', 'MOSS_DECODE_REGIONS', 'MOSS_DISABLE_ATTENTION_REGION', 'MOSS_CLIP_FILE')}, indent=2))
    (target / 'source-sha256.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [*Path('comfy/text_encoders').glob('hpu_*.py'), Path('comfy/text_encoders/llama.py'), Path('comfy/text_encoders/qwen35.py'), Path('comfy/ldm/moss_image/model.py'), Path('comfy/ops.py')]}, indent=2))
    workflow = json.loads((probes / 'moss-user-textgen-bf16.json').read_text())
    if os.environ.get('MOSS_CLIP_FILE'):
        workflow['18:2']['inputs']['clip_name'] = os.environ['MOSS_CLIP_FILE']
    (target / 'b0-source.json').write_text(json.dumps(workflow, indent=2))
    cmd = [python, str(probes / 'moss_observed_server.py'), '--disable-dynamic-vram', '--disable-pinned-memory', '--disable-async-offload', '--disable-xformers', '--use-pytorch-cross-attention', '--listen', '127.0.0.1', '--port', '8190']
    if mode == 'original':
        cmd.append('--disable-comfy-compiler')
    env = dict(base_env, MOSS_EVENTS=str(target / 'events.jsonl'))
    with (target / 'server.log').open('w') as server_log, (target / 'hpu.csv').open('w') as telemetry_log, (target / 'client.log').open('w') as client_log:
        server = subprocess.Popen(cmd, env=env, stdout=server_log, stderr=subprocess.STDOUT)
        telemetry = None
        try:
            for _ in range(120):
                if server.poll() is not None:
                    raise RuntimeError('server exited during startup')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8190/system_stats', timeout=2) as response:
                        json.load(response)
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError('server startup')
            telemetry = subprocess.Popen(['hl-smi', '--query-aip=timestamp,utilization.aip,utilization.memory,memory.used,power.draw', '--format=csv,nounits', '-l', '100ms'], stdout=telemetry_log)
            subprocess.run([python, str(probes / 'moss_baseline.py'), str(target)], env=dict(env, MOSS_B0_RUNS=os.environ.get('MOSS_B0_RUNS', '6'), MOSS_B1_RUNS='0'), stdout=client_log, stderr=subprocess.STDOUT, check=True)
            if os.environ.get('MOSS_KV_ACCEPTANCE') == '1':
                subprocess.run([python, str(probes / 'moss_kv_acceptance.py'), str(target)], env=env, stdout=client_log, stderr=subprocess.STDOUT, check=True)
            print(mode, (target / 'baseline-summary.json').read_text(), flush=True)
        finally:
            if telemetry is not None:
                telemetry.terminate()
                telemetry.wait(timeout=10)
            server.terminate()
            server.wait(timeout=60)
    time.sleep(3)
