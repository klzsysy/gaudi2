"""Screen sampled-token readback batching against the ordinary FP8 loop."""
import os
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import qwen21_latest_jobs as lifecycle

base = lifecycle.BASE / 'performance-qwen80-2026-11-16'
lifecycle.ROOT = base / 'readback'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--after', type=Path, default=base / 'restored-normal.json')
    args = parser.parse_args()
    root = lifecycle.ROOT
    root.mkdir(exist_ok=True)
    # The first profiler job owns the card through its ordinary-service check.
    deadline = time.monotonic() + 1800
    while not args.after.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Initial profiler restoration did not finish')
        time.sleep(2)
    queue = lifecycle.request('/queue')
    assert not queue['queue_running'] and not queue['queue_pending'], queue
    environment = dict(os.environ, PT_HPU_LAZY_MODE='0',
                       GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
                       PYTHONPATH=os.pathsep.join((str(lifecycle.PRODUCT), str(lifecycle.PROBES))))
    command = [sys.executable, str(lifecycle.PROBES / 'qwen21_text_profile.py')]
    common = ['--workflow', str(lifecycle.WORKFLOW), '--no-region-labels', '--plain-only',
              '--plain-runs', '3', '--window-tokens', '256', '--window-steps', '64',
              '--profile-start-token', '64']
    try:
        subprocess.run(['./stop-hpu.sh'], cwd=lifecycle.PRODUCT, check=True)
        for name in ('baseline', 'defer4', 'defer8'):
            timings = root / name / 'timings.jsonl'
            if timings.exists() and sum(json.loads(line)['mode'] == 'plain' for line in timings.read_text().splitlines()) >= 3:
                continue
            trial = ['--trial', 'hardware80:' + name] if name != 'baseline' else []
            lifecycle.run(name, command + [str(root / name)] + common + trial, environment)
    finally:
        subprocess.run(['./start-hpu.sh'], cwd=lifecycle.PRODUCT, check=True, timeout=180)
        lifecycle.verify_restoration()


if __name__ == '__main__':
    main()
