"""Run isolated startup probes and restore the ordinary service."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import qwen21_latest_jobs as lifecycle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('modes', nargs='+')
    args = parser.parse_args()
    args.root.mkdir(exist_ok=True)
    lock = (lifecycle.BASE / '.qwen80-pe.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    queue = lifecycle.request('/queue')
    assert not queue['queue_running'] and not queue['queue_pending'], queue
    environment = dict(os.environ, PT_HPU_LAZY_MODE='0',
                       GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
                       PYTHONPATH=os.pathsep.join((str(lifecycle.PRODUCT), str(lifecycle.PROBES))),
                       TORCH_LOGS='recompiles')
    history = lifecycle.BASE / 'performance-qwen-startup-2026-11-16/history-before.json'
    try:
        subprocess.run(['./stop-hpu.sh'], cwd=lifecycle.PRODUCT, check=True)
        for mode in args.modes:
            with (args.root / (mode + '.log')).open('w') as log:
                result = subprocess.run([sys.executable, str(lifecycle.PROBES / 'qwen_startup_probe.py'),
                    str(args.root / mode), '--mode', mode, '--history', str(history)],
                    cwd=lifecycle.PRODUCT, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
                print(mode, result.returncode, flush=True)
    finally:
        subprocess.run(['./start-hpu.sh'], cwd=lifecycle.PRODUCT, check=True, timeout=180)
        (args.root / 'restored-queue.json').write_text(json.dumps(lifecycle.request('/queue')))


if __name__ == '__main__':
    main()
