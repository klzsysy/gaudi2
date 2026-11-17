"""Test the opaque short-RoPE boundary found in the low-overhead trace."""
import json
import argparse
import os
import subprocess
import sys
import time

import psutil
import qwen21_latest_jobs as lifecycle

base = lifecycle.BASE / 'performance-qwen80-2026-11-16'
lifecycle.ROOT = base / 'rope'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int, default=164519)
    parser.add_argument('--trials', nargs='+', default=['baseline', 'rope-inline-none', 'rope-native'])
    args = parser.parse_args()
    root = lifecycle.ROOT
    root.mkdir(exist_ok=True)
    deadline = time.monotonic() + 2400
    while psutil.pid_exists(args.wait_pid):
        if time.monotonic() > deadline:
            raise TimeoutError('Product validation still owns the card')
        time.sleep(2)
    queue = lifecycle.request('/queue')
    assert not queue['queue_running'] and not queue['queue_pending'], queue
    teacher = root / 'teacher.json'
    teacher.write_text((base / 'readback/baseline/plain-tokens.json').read_text())
    environment = dict(os.environ, PT_HPU_LAZY_MODE='0',
                       GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
                       PYTHONPATH=os.pathsep.join((str(lifecycle.PRODUCT), str(lifecycle.PROBES))))
    command = [sys.executable, str(lifecycle.PROBES / 'qwen21_text_profile.py')]
    common = ['--workflow', str(lifecycle.WORKFLOW), '--no-region-labels', '--logits',
              '--teacher', str(teacher), '--window-tokens', '256', '--window-steps', '64',
              '--profile-start-token', '64']
    try:
        subprocess.run(['./stop-hpu.sh'], cwd=lifecycle.PRODUCT, check=True)
        outcomes = {}
        for name in args.trials:
            trial = ['--trial', 'hardware80:' + name] if name != 'baseline' else []
            try:
                lifecycle.run(name, command + [str(root / name)] + common + trial, environment)
                outcomes[name] = 0
            except subprocess.CalledProcessError as error:
                outcomes[name] = error.returncode
                if error.returncode < 0:
                    raise
            finally:
                (root / 'trial-exits.json').write_text(json.dumps(outcomes, indent=2))
    finally:
        subprocess.run(['./start-hpu.sh'], cwd=lifecycle.PRODUCT, check=True, timeout=180)
        lifecycle.verify_restoration()


if __name__ == '__main__':
    main()
