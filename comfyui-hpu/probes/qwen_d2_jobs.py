"""Serialize D2 experiments and restore the accepted ordinary service."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

import qwen21_latest_jobs as lifecycle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('jobs', type=Path)
    options = parser.parse_args()
    options.root.mkdir(parents=True, exist_ok=True)
    lifecycle.ROOT = options.root
    lock = (lifecycle.BASE / '.qwen80-pe.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    queue = lifecycle.request('/queue')
    assert not queue['queue_running'] and not queue['queue_pending'], queue
    environment = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
                       PYTHONPATH=os.pathsep.join((str(lifecycle.PRODUCT), str(lifecycle.PROBES))))
    try:
        subprocess.run(['./stop-hpu.sh'], cwd=lifecycle.PRODUCT, check=True)
        time.sleep(2)
        for job in json.loads(options.jobs.read_text()):
            directory = options.root / job['name']
            directory.mkdir(parents=True, exist_ok=False)
            source = Path(job['arguments'][0])
            snapshot = directory / 'source'
            snapshot.mkdir()
            files = list(lifecycle.PROBES.glob('qwen_d2_*.py')) + list(lifecycle.PROBES.glob('qwen_d2_*.cpp'))
            files += [path for path in (lifecycle.PROBES / 'qwen_d2_softmax').rglob('*')
                      if path.suffix in ('.py', '.c', '.cpp')]
            if source not in files:
                files.append(source)
            hashes = {}
            for path in files:
                relative = path.relative_to(lifecycle.PROBES) if path.is_relative_to(lifecycle.PROBES) else Path(path.name)
                destination = snapshot / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            (directory / 'job.json').write_text(json.dumps(dict(job=job, source_sha256=hashes), indent=2))
            started = time.monotonic()
            print('start', job['name'], flush=True)
            with (directory / 'job.log').open('w') as log:
                process = subprocess.Popen([sys.executable, *job['arguments']], cwd=lifecycle.PRODUCT,
                    env=dict(environment, **job.get('environment', {})), stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait(timeout=job.get('timeout', 900))
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    code = 'timeout'
            record = dict(name=job['name'], returncode=code, seconds=time.monotonic()-started)
            with (options.root / 'jobs.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
            if code != 0 and job.get('stop_on_error', False):
                raise RuntimeError(f"D2 job {job['name']} failed with {code}")
    finally:
        subprocess.run(['./start-hpu.sh'], cwd=lifecycle.PRODUCT, check=True, timeout=180)
        lifecycle.verify_restoration()
        (options.root / 'deployed-version.txt').write_text(
            (lifecycle.PRODUCT / '.run/deployed-version').read_text())
        print('ordinary service restored and generation passed', flush=True)


if __name__ == '__main__':
    main()
