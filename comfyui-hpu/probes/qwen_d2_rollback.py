"""Withdraw the D2 candidate and verify the accepted D1 ordinary service."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image

import qwen21_latest_jobs as lifecycle
import qwen80_deploy as deployment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root, product = args.root, lifecycle.PRODUCT
    result_dir = root / 'rollback'
    result_dir.mkdir(exist_ok=True)
    deployment.ROOT = result_dir
    lock = (lifecycle.BASE / '.qwen80-pe.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    manifest = json.loads((root / 'deployment/deployment.json').read_text())
    backup = root / 'deployment/baseline'
    for name, digest in manifest['baseline_sha256'].items():
        path = product / name
        current = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        assert current in (digest, manifest['file_sha256'][name]), name
        if digest is not None:
            assert hashlib.sha256((backup / name).read_bytes()).hexdigest() == digest, name
    assert (backup / '.run/deployed-version').read_text().strip() == manifest['baseline']
    queue = deployment.request('/queue')
    assert not queue['queue_running'] and not queue['queue_pending'], queue
    logfile = product / 'logs/hpu.log'
    offset = logfile.stat().st_size
    subprocess.run(['./stop-hpu.sh'], cwd=product, check=True)
    try:
        for name, digest in manifest['baseline_sha256'].items():
            target = product / name
            if digest is None:
                target.unlink(missing_ok=True)
            else:
                temporary = target.with_suffix(target.suffix + '.rollback')
                shutil.copy2(backup / name, temporary)
                os.replace(temporary, target)
        for name in ('.run/deployed-version', '.run/deployed-staged.json'):
            shutil.copy2(backup / name, product / name)
    finally:
        subprocess.run(['./start-hpu.sh'], cwd=product, check=True, timeout=180)
    expected = json.loads((root / 'rollback-source-hashes.json').read_text())
    mismatches = [name for name, digest in expected.items()
                  if hashlib.sha256((product / name).read_bytes()).hexdigest() != digest]
    assert not mismatches, mismatches
    reference_path = lifecycle.BASE / 'performance-qwen-d1-2026-11-17/product3/api/full-0-product.json'
    reference = json.loads(reference_path.read_text())
    item = reference['outputs']['461']['images'][0]
    pixels = np.asarray(Image.open(product / 'output' / item['subfolder'] / item['filename']).convert('RGBA'))
    graph = copy.deepcopy(reference['prompt'][2])
    runs = []
    for label, mode in (('d2-rollback-cold', 'auto'), ('d2-rollback-warm', 'off')):
        graph['459:471']['inputs']['mtp'] = mode
        runs.append(deployment.generate(label, graph, reference, pixels))
    queue = deployment.request('/queue')
    assert not queue['queue_running'] and not queue['queue_pending'], queue
    with logfile.open('rb') as stream:
        stream.seek(offset)
        logs = stream.read()
    (result_dir / 'ordinary.log').write_bytes(logs)
    record = dict(status='withdrawn_restored_d1', candidate=manifest['identity'],
                  restored_commit=manifest['baseline'], user_decision='Discard D2 product changes and retain investigation records',
                  source_files_checked=len(expected), source_mismatches=mismatches,
                  removed_files=[name for name, digest in manifest['baseline_sha256'].items() if digest is None],
                  restored_sha256={name: digest for name, digest in manifest['baseline_sha256'].items() if digest is not None},
                  runs=runs, queue=queue, batch_as_strided_warnings=logs.count(b'UserWarning: hpu::batch_as_strided'),
                  recompile_limits=logs.count(b'hit config.recompile_limit'))
    (result_dir / 'rollback.json').write_text(json.dumps(record, indent=2))
    print('D2 withdrawn; accepted D1 hashes and ordinary generations verified', flush=True)


if __name__ == '__main__':
    main()
