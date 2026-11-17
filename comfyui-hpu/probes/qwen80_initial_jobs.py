"""Low-interference PE profiling with ordinary service restoration."""
import shutil
import os
import subprocess
import sys

import qwen21_latest_jobs as lifecycle

lifecycle.ROOT = lifecycle.BASE / 'performance-qwen80-2026-11-16'


def main():
    root = lifecycle.ROOT
    root.mkdir(exist_ok=True)
    previous = lifecycle.BASE / 'performance-qwen21-latest-2026-11-16'
    shutil.copyfile(previous / 'source-manifest.json', root / 'source-manifest.json')
    queue = lifecycle.request('/queue')
    assert not queue['queue_running'] and not queue['queue_pending'], queue
    environment = dict(os.environ, PT_HPU_LAZY_MODE='0',
                       GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
                       PYTHONPATH=os.pathsep.join((str(lifecycle.PRODUCT), str(lifecycle.PROBES))))
    command = [sys.executable, str(lifecycle.PROBES / 'qwen21_text_profile.py')]
    common = ['--workflow', str(lifecycle.WORKFLOW), '--no-region-labels',
              '--window-steps', '64']
    try:
        subprocess.run(['./stop-hpu.sh'], cwd=lifecycle.PRODUCT, check=True)
        lifecycle.run('trace-early', command + [str(root / 'trace-early'), '--window-tokens', '160',
                                              '--profile-start-token', '64'] + common, environment)
        lifecycle.run('trace-late', command + [str(root / 'trace-late'), '--window-tokens', '1632',
                                             '--profile-start-token', '1536'] + common, environment)
    finally:
        subprocess.run(['./start-hpu.sh'], cwd=lifecycle.PRODUCT, check=True, timeout=180)
        lifecycle.verify_restoration()


if __name__ == '__main__':
    main()
