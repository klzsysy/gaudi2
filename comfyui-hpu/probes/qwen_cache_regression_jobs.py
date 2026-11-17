"""Validate cache regression tests with the accepted product and restore serving."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

import qwen21_latest_jobs as lifecycle
import qwen80_deploy as deployment


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
    existing = []
    backup = root / 'baseline'
    backup.mkdir()
    for name in files:
        source = lifecycle.PRODUCT / name
        if source.exists():
            target = backup / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            existing.append(name)
    environment = dict(os.environ, PT_HPU_LAZY_MODE='0',
        GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
        PYTHONPATH=os.pathsep.join((str(lifecycle.PRODUCT), str(lifecycle.PROBES))))
    passed = False
    try:
        subprocess.run(['./stop-hpu.sh'], cwd=lifecycle.PRODUCT, check=True)
        for name in files:
            shutil.copy2(root / 'candidate' / name, lifecycle.PRODUCT / name)
        lifecycle.run('cache-regression', [sys.executable, '-m', 'pytest', '-q', *files,
            '-k', 'isolates_same_length or text_generation_cache'], environment)
        tests = ['generation_readback_test.py', 'hpu_qwen_generation_test.py', 'qwen35_decode_test.py',
                 'hpu_qwen_norm_test.py', 'hpu_qwen21_block_test.py', 'hpu_mlp_test.py',
                 'hpu_rope_test.py', 'hpu_kv_attention_test.py']
        lifecycle.run('suite', [sys.executable, '-m', 'pytest', '-q',
            *['tests-unit/comfy_test/' + name for name in tests], files[1]], environment)
        (root / 'validated-hashes.json').write_text(json.dumps({
            name: hashlib.sha256((lifecycle.PRODUCT / name).read_bytes()).hexdigest() for name in files}, indent=2))
        passed = True
    finally:
        for name in files:
            path = lifecycle.PRODUCT / name
            if name in existing:
                shutil.copy2(backup / name, path)
            else:
                path.unlink(missing_ok=True)
        subprocess.run(['./start-hpu.sh'], cwd=lifecycle.PRODUCT, check=True, timeout=180)
        if passed:
            reference_path = lifecycle.BASE / 'performance-qwen-startup-fix-2026-11-17/validation/api/cold-product.json'
            reference = json.loads(reference_path.read_text())
            item = reference['outputs']['461']['images'][0]
            pixels = np.asarray(Image.open(lifecycle.PRODUCT / 'output' / item['subfolder'] / item['filename']).convert('RGBA'))
            deployment.ROOT = root
            row = deployment.generate('cache-regression-restored', reference['prompt'][2], reference, pixels)
            (root / 'normal-verification.json').write_text(json.dumps(row, indent=2))
        (root / 'restored-queue.json').write_text(json.dumps(lifecycle.request('/queue')))


if __name__ == '__main__':
    main()
