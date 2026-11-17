"""Deploy a staged snapshot and replay the edited prompts on the ordinary API."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image

import qwen80_deploy as deployment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--validation', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    product = deployment.PRODUCT
    deployment.ROOT = root
    manifest = json.loads((root / 'manifest.json').read_text())
    validated = json.loads((args.validation / 'validated-hashes.json').read_text())
    for name, digest in validated.items():
        assert digest == manifest['file_sha256'][name]
    queue = deployment.request('/queue')
    assert not queue['queue_running'] and not queue['queue_pending'], queue
    backup = root / 'baseline'
    backup.mkdir()
    paths = list(manifest['file_sha256']) + ['.run/deployed-version', '.run/deployed-staged.json']
    for name in paths:
        target = backup / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(product / name, target)
    for name, digest in manifest['file_sha256'].items():
        assert hashlib.sha256((root / 'candidate' / name).read_bytes()).hexdigest() == digest
    logfile = product / 'logs/hpu.log'
    log_offset = logfile.stat().st_size
    subprocess.run(['./stop-hpu.sh'], cwd=product, check=True)
    success = False
    try:
        for name in manifest['file_sha256']:
            target = product / name
            temporary = target.with_suffix(target.suffix + '.deploy')
            shutil.copy2(root / 'candidate' / name, temporary)
            os.replace(temporary, target)
        (product / '.run/deployed-version').write_text(manifest['identity'] + '\n')
        (product / '.run/deployed-staged.json').write_text(json.dumps(manifest, indent=2))
        subprocess.run(['./start-hpu.sh'], cwd=product, check=True, timeout=180)
        cases = [('cold', 'cold-product'), ('edit-0', 'edit-0-product'), ('edit-1', 'edit-1-product'),
                 ('edit-2', 'edit-2-product'), ('edit-3', 'edit-3-product'), ('warm', 'cold-product')]
        rows = []
        for label, reference_label in cases:
            reference = json.loads((args.validation / 'api' / (reference_label + '.json')).read_text())
            graph = copy.deepcopy(reference['prompt'][2])
            if label == 'warm':
                graph['459:471']['inputs']['mtp'] = 'off'
            item = reference['outputs']['461']['images'][0]
            pixels = np.asarray(Image.open(product / 'output' / item['subfolder'] / item['filename']).convert('RGBA'))
            rows.append(deployment.generate('startup-' + label, graph, reference, pixels))
        actual = {name: hashlib.sha256((product / name).read_bytes()).hexdigest() for name in manifest['file_sha256']}
        assert actual == manifest['file_sha256']
        queue = deployment.request('/queue')
        assert not queue['queue_running'] and not queue['queue_pending'], queue
        with logfile.open('rb') as stream:
            stream.seek(log_offset)
            logs = stream.read()
        (root / 'ordinary.log').write_bytes(logs)
        manifest.update(status='deployed_awaiting_human_acceptance', human_acceptance=False,
                        runs=rows, deployed_sha256=actual, queue=queue,
                        batch_as_strided_warnings=logs.count(b'UserWarning: hpu::batch_as_strided'))
        (root / 'deployment.json').write_text(json.dumps(manifest, indent=2))
        (product / '.run/deployed-staged.json').write_text(json.dumps(manifest, indent=2))
        success = True
    finally:
        if not success:
            subprocess.run(['./stop-hpu.sh'], cwd=product, check=True)
            for name in paths:
                shutil.copy2(backup / name, product / name)
            subprocess.run(['./start-hpu.sh'], cwd=product, check=True, timeout=180)


if __name__ == '__main__':
    main()
