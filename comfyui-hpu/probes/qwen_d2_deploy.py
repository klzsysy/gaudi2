"""Deploy the validated unstaged D2 snapshot and verify ordinary generations."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

import qwen21_latest_jobs as lifecycle
import qwen80_deploy as deployment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--validation', type=Path, required=True)
    args = parser.parse_args()
    root, product = args.root, deployment.PRODUCT
    deployment.ROOT = root
    lock = (lifecycle.BASE / '.qwen80-pe.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['staged'] is False
    assert json.loads((args.validation / 'validated-hashes.json').read_text()) == manifest['file_sha256']
    queue = deployment.request('/queue')
    assert not queue['queue_running'] and not queue['queue_pending'], queue
    for name, digest in manifest['baseline_sha256'].items():
        path = product / name
        assert (hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None) == digest, name
    for name, digest in manifest['file_sha256'].items():
        assert hashlib.sha256((root / 'candidate' / name).read_bytes()).hexdigest() == digest, name
    paths = list(manifest['file_sha256']) + ['.run/deployed-version', '.run/deployed-staged.json']
    backup = root / 'baseline'
    backup.mkdir()
    existing = []
    for name in paths:
        source = product / name
        if source.exists():
            target = backup / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            existing.append(name)
    logfile = product / 'logs/hpu.log'
    log_offset = logfile.stat().st_size
    subprocess.run(['./stop-hpu.sh'], cwd=product, check=True)
    time.sleep(2)
    success = False
    try:
        for name in manifest['file_sha256']:
            target = product / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.deploy')
            shutil.copy2(root / 'candidate' / name, temporary)
            os.replace(temporary, target)
        (product / '.run/deployed-version').write_text(manifest['identity'] + '\n')
        (product / '.run/deployed-staged.json').write_text(json.dumps(manifest, indent=2))
        subprocess.run(['./start-hpu.sh'], cwd=product, check=True, timeout=180)
        cases = [('cold', 'full-0-product', 'auto'), ('warm', 'full-0-product', 'off'),
                 ('blue', 'blue-product', 'auto'), ('red', 'red-product', 'auto'),
                 ('blue-return', 'blue-product', 'off'), ('original', 'full-0-product', 'auto')]
        runs = []
        for label, reference_label, mtp in cases:
            reference = json.loads((args.validation / 'api' / (reference_label + '.json')).read_text())
            graph = copy.deepcopy(reference['prompt'][2])
            graph['459:471']['inputs']['mtp'] = mtp
            item = reference['outputs']['461']['images'][0]
            pixels = np.asarray(Image.open(product / 'output' / item['subfolder'] / item['filename']).convert('RGBA'))
            runs.append(deployment.generate('d2-' + label, graph, reference, pixels))
        deployment.request('/free', dict(unload_models=True, free_memory=True))
        unloaded = []
        for _ in range(6):
            time.sleep(10)
            unloaded.append(deployment.request('/system_stats'))
        (root / 'unload.json').write_text(json.dumps(unloaded, indent=2))
        runs.append(deployment.generate('d2-reload', graph, reference, pixels))
        actual = {name: hashlib.sha256((product / name).read_bytes()).hexdigest() for name in manifest['file_sha256']}
        assert actual == manifest['file_sha256']
        queue = deployment.request('/queue')
        assert not queue['queue_running'] and not queue['queue_pending'], queue
        with logfile.open('rb') as stream:
            stream.seek(log_offset)
            logs = stream.read()
        (root / 'ordinary.log').write_bytes(logs)
        alias_warnings = logs.count(b'UserWarning: hpu::batch_as_strided')
        recompile_limits = logs.count(b'hit config.recompile_limit')
        assert alias_warnings == recompile_limits == 0, (alias_warnings, recompile_limits)
        manifest.update(staged=True, status='deployed_awaiting_human_acceptance', human_acceptance=False,
                        runs=runs, deployed_sha256=actual, queue=queue,
                        batch_as_strided_warnings=alias_warnings, recompile_limits=recompile_limits)
        (root / 'deployment.json').write_text(json.dumps(manifest, indent=2))
        (product / '.run/deployed-staged.json').write_text(json.dumps(manifest, indent=2))
        success = True
        print('D2 unstaged snapshot deployed and ordinary validation passed', flush=True)
    finally:
        if not success:
            subprocess.run(['./stop-hpu.sh'], cwd=product, check=True)
            time.sleep(2)
            for name in paths:
                if name in existing:
                    shutil.copy2(backup / name, product / name)
                else:
                    (product / name).unlink(missing_ok=True)
            subprocess.run(['./start-hpu.sh'], cwd=product, check=True, timeout=180)
            lifecycle.ROOT = root
            lifecycle.verify_restoration()


if __name__ == '__main__':
    main()
