"""Check KV boundary reuse and prefix alternatives before final D1 deployment."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import qwen21_latest_jobs as lifecycle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    lifecycle.ROOT = root
    lock = (lifecycle.BASE / '.qwen80-pe.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    queue = lifecycle.request('/queue')
    assert not queue['queue_running'] and not queue['queue_pending'], queue
    files = json.loads((root / 'files.json').read_text())
    backup = root / 'baseline-source'
    backup.mkdir()
    existing = []
    for name in files:
        source = lifecycle.PRODUCT / name
        if source.exists():
            destination = backup / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            existing.append(name)
    environment = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
                       PYTHONPATH=os.pathsep.join((str(lifecycle.PRODUCT), str(lifecycle.PROBES))))
    try:
        subprocess.run(['./stop-hpu.sh'], cwd=lifecycle.PRODUCT, check=True)
        for name in files:
            shutil.copy2(root / 'candidate-source' / name, lifecycle.PRODUCT / name)
        lifecycle.run('attention-tests', [sys.executable, '-m', 'pytest', '-q',
            'tests-unit/comfy_test/hpu_fast_attention_test.py'], environment)
        lifecycle.run('prefix', [sys.executable, str(lifecycle.PROBES / 'qwen_d1_prefix.py'),
            str(root.parent / 'workflow/capture'), str(root / 'prefix.json')], environment)
        (root / 'validated-hashes.json').write_text(json.dumps({
            name: hashlib.sha256((lifecycle.PRODUCT / name).read_bytes()).hexdigest() for name in files}, indent=2))
    finally:
        for name in files:
            if name in existing:
                shutil.copy2(backup / name, lifecycle.PRODUCT / name)
            else:
                (lifecycle.PRODUCT / name).unlink(missing_ok=True)
        subprocess.run(['./start-hpu.sh'], cwd=lifecycle.PRODUCT, check=True, timeout=180)
        (root / 'restored-queue.json').write_text(json.dumps(lifecycle.request('/queue')))


if __name__ == '__main__':
    main()
