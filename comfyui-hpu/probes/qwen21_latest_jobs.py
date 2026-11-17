"""Reproduce the frozen November 16 request and restore ordinary serving."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request

import psutil

BASE = Path('/home/sonny/gaudi2/comfyui-hpu')
ROOT = BASE / 'performance-qwen21-latest-2026-11-16'
PRODUCT = Path('/home/sonny/ComfyUI-Gaudi')
WORKFLOW = BASE / 'workflows/qwen21-latest-2026-11-16-api.json'
PROBES = BASE / 'probes'
URL = 'http://192.168.10.21:8189'
stop_monitor = threading.Event()
job = 'initial'


def request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(URL + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def source_hashes():
    files = json.loads((ROOT / 'source-manifest.json').read_text())
    return {name: hashlib.sha256((PRODUCT / name).read_bytes()).hexdigest() for name in files}


def monitor():
    fields = ['power.draw', 'utilization.aip', 'memory.used', 'temperature.aip',
              'clocks.current.soc', 'stats.violation.power', 'stats.violation.thermal']
    with (ROOT / 'telemetry.jsonl').open('w') as stream:
        while not stop_monitor.is_set():
            started = time.monotonic()
            record = {'wall_ns': time.time_ns(), 'job': job}
            try:
                values = subprocess.check_output(
                    ['hl-smi', '--query-aip=' + ','.join(fields), '--format=csv,noheader,nounits'],
                    text=True, timeout=5).strip().split(',')
                record['hpu'] = dict(zip(fields, [v.strip() for v in values]))
                processes = []
                for process in psutil.process_iter(['pid', 'cmdline', 'cpu_times', 'memory_info']):
                    command = ' '.join(process.info['cmdline'] or [])
                    if 'python' in command and 'qwen21_deep_jobs.py' not in command and any(name in command for name in
                            ('main.py', 'qwen21_fp8_server', 'qwen21_text_profile.py')):
                        cpu = process.info['cpu_times']
                        processes.append({'pid': process.pid, 'cpu_seconds': cpu.user + cpu.system,
                                          'rss': process.info['memory_info'].rss,
                                          'threads': [{'id': t.id, 'cpu_seconds': t.user_time + t.system_time}
                                                      for t in process.threads()]})
                record['processes'] = processes
            except (OSError, subprocess.SubprocessError, psutil.Error) as error:
                record['error'] = str(error)
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            stop_monitor.wait(max(0, .25 - (time.monotonic() - started)))


def run(label, command, environment):
    global job
    job = label
    print('start', label, flush=True)
    with (ROOT / (label + '-job.log')).open('w') as log:
        subprocess.run(command, cwd=PRODUCT, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=1800)
    print('finished', label, flush=True)


def native_environment(label, environment):
    directory = ROOT / label
    directory.mkdir(exist_ok=True)
    template = BASE / 'performance-qwen21-fp8-opt-2026-11-15/native-text/profiler.json'
    config = json.loads(template.read_text())
    config['GeneralSettings']['values']['outdir']['value'] = str(directory / 'hw')
    config['GeneralSettings']['values']['session']['value'] = label
    path = directory / 'profiler.json'
    path.write_text(json.dumps(config, indent=2))
    return dict(environment, HABANA_PROFILE='1', HABANA_PROF_CONFIG=str(path))


def verify_restoration():
    graph = json.loads(WORKFLOW.read_text())
    graph['461']['inputs']['filename_prefix'] = 'gaudi/qwen21_latest/normal'
    pid = request('/prompt', {'prompt': graph})['prompt_id']
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        history = request('/history/' + pid)
        if pid in history:
            break
        time.sleep(1)
    else:
        raise TimeoutError(pid)
    result = history[pid]
    (ROOT / 'restored-normal.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    assert result['status']['status_str'] == 'success', result['status']
    (ROOT / 'restored-queue.json').write_text(json.dumps(request('/queue')))


def main():
    global job
    ROOT.mkdir(exist_ok=True)
    queue = request('/queue')
    assert not queue['queue_running'] and not queue['queue_pending'], queue
    before = source_hashes()
    (ROOT / 'source-hashes-before.json').write_text(json.dumps(before, indent=2))
    environment = dict(os.environ, PT_HPU_LAZY_MODE='0',
                       GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
                       PYTHONPATH=os.pathsep.join((str(PRODUCT), str(PROBES))))
    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    try:
        subprocess.run(['./stop-hpu.sh'], cwd=PRODUCT, check=True)
        command = [sys.executable, str(PROBES / 'qwen21_deep_jobs.py')]
        common = ['--workflow', str(WORKFLOW), '--output-root', str(ROOT),
                  '--server-module', 'qwen21_fp8_server']
        run('workflow', command + ['workflow', '--runs', '4', '--cached', '2',
                                  '--fp8-diagnostics'] + common, environment)
        run('native-dit', command + ['native-dit', '--runs', '1', '--cached', '0',
                                    '--native-dit'] + common,
            native_environment('native-dit', environment))
        run('native-text', [sys.executable, str(PROBES / 'qwen21_text_profile.py'),
                            str(ROOT / 'native-text'), '--workflow', str(WORKFLOW),
                            '--native', '--profile-start-token', '1536', '--window-tokens', '1568'],
            native_environment('native-text', environment))
    finally:
        job = 'restoration'
        try:
            subprocess.run(['./start-hpu.sh'], cwd=PRODUCT, check=True, timeout=180)
            verify_restoration()
            after = source_hashes()
            (ROOT / 'source-verification.json').write_text(json.dumps({
                'all_source_hashes_match': before == after,
                'files': len(before), 'deployed_version': (PRODUCT / '.run/deployed-version').read_text().strip()
            }, indent=2))
            assert before == after
        finally:
            stop_monitor.set()
            worker.join(timeout=10)


if __name__ == '__main__':
    main()
