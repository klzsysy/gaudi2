"""Validate the D1 product snapshot before staged deployment."""
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
from qwen_d1_validation import workflows
from qwen_d1_hardware import measure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--skip-all-layers', action='store_true')
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
            target = backup / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            existing.append(name)
    environment = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
                       PYTHONPATH=os.pathsep.join((str(lifecycle.PRODUCT), str(lifecycle.PROBES))))
    try:
        subprocess.run(['./stop-hpu.sh'], cwd=lifecycle.PRODUCT, check=True)
        if not args.skip_all_layers:
            lifecycle.run('all-layers', [sys.executable, str(lifecycle.PROBES / 'qwen_d1_screen.py'),
                str(root.parent / 'workflow'), '--all-inputs', '--names', 'baseline', 'center-split-fast',
                '--output', 'all-layers'], environment)
        for name in files:
            shutil.copy2(root / 'candidate-source' / name, lifecycle.PRODUCT / name)
        lifecycle.run('attention-tests', [sys.executable, '-m', 'pytest', '-q',
            'tests-unit/comfy_test/hpu_fast_attention_test.py'], environment)
        tests = ['generation_readback_test.py', 'hpu_qwen_generation_test.py', 'qwen35_decode_test.py',
                 'hpu_qwen_norm_test.py', 'hpu_qwen21_block_test.py', 'hpu_mlp_test.py', 'hpu_rope_test.py',
                 'hpu_kv_attention_test.py', 'hpu_moss_attention_test.py']
        lifecycle.run('suite', [sys.executable, '-m', 'pytest', '-q',
            *['tests-unit/comfy_test/' + name for name in tests],
            'tests-unit/execution_test/test_text_generation_cache.py'], environment)
        lifecycle.run('cache-admission', [sys.executable, '-m', 'pytest', '-q',
            'tests-unit/comfy_test/qwen21_cache_admission_test.py'], environment)
        api = root / 'api'
        api.mkdir()
        workflows(api, dict(environment, QWEN_D1_PRODUCT='1'), ['product'])
        history = json.loads((api / 'warm-0-product.json').read_text())
        measure(root / 'hardware', environment, history)
        (root / 'validated-hashes.json').write_text(json.dumps({
            name: hashlib.sha256((lifecycle.PRODUCT / name).read_bytes()).hexdigest() for name in files}, indent=2))
    finally:
        for name in files:
            target = lifecycle.PRODUCT / name
            if name in existing:
                shutil.copy2(backup / name, target)
            else:
                target.unlink(missing_ok=True)
        subprocess.run(['./start-hpu.sh'], cwd=lifecycle.PRODUCT, check=True, timeout=180)
        lifecycle.verify_restoration()


if __name__ == '__main__':
    main()
