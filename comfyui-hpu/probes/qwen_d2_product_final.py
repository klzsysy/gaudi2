"""Validate shared-storage regression, paired workflows, and D2 hardware counters."""
import argparse
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
    options = parser.parse_args()
    root = options.root
    manifest = json.loads((root / 'manifest.json').read_text())
    backup = root / 'baseline-source'
    backup.mkdir()
    for name, digest in manifest['baseline_sha256'].items():
        path = lifecycle.PRODUCT / name
        if digest is None:
            assert not path.exists(), name
        else:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
            target = backup / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    try:
        for name, digest in manifest['file_sha256'].items():
            source = root / 'candidate' / name
            assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
            target = lifecycle.PRODUCT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        tests = ['hpu_fast_attention_test.py', 'hpu_qwen21_block_test.py',
                 'hpu_qwen_generation_test.py', 'qwen35_decode_test.py', 'hpu_qwen_norm_test.py',
                 'hpu_mlp_test.py', 'hpu_rope_test.py', 'hpu_kv_attention_test.py', 'hpu_moss_attention_test.py',
                 'generation_readback_test.py', 'qwen21_cache_admission_test.py']
        with (root / 'tests.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'pytest', '-q',
                *['tests-unit/comfy_test/' + name for name in tests],
                'tests-unit/execution_test/test_text_generation_cache.py'], cwd=lifecycle.PRODUCT,
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
        print('shared-storage and cache admission regressions passed', flush=True)
        environment = dict(os.environ, QWEN_D1_PRODUCT='1', QWEN_VALIDATION_SERVER='qwen_d2_product_server')
        api = root / 'api'
        api.mkdir()
        workflows(api, environment, ['product'])
        history = json.loads((api / 'warm-0-product.json').read_text())
        measure(root / 'hardware', environment, history)
        actual = {name: hashlib.sha256((lifecycle.PRODUCT / name).read_bytes()).hexdigest()
                  for name in manifest['file_sha256']}
        assert actual == manifest['file_sha256']
        (root / 'validated-hashes.json').write_text(json.dumps(actual, indent=2))
    finally:
        for name, digest in manifest['baseline_sha256'].items():
            path = lifecycle.PRODUCT / name
            if digest is None:
                path.unlink(missing_ok=True)
            else:
                shutil.copy2(backup / name, path)


if __name__ == '__main__':
    main()
