"""Deploy the accepted-for-review PE snapshot using its validated natural image."""
import argparse
import json
from pathlib import Path
import threading
import numpy as np
from PIL import Image
import qwen80_deploy as deployment
import qwen21_latest_jobs as lifecycle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    deployment.ROOT = args.root
    reference = json.loads(args.reference.read_text())
    item = reference['outputs']['461']['images'][0]
    path = deployment.PRODUCT / 'output' / item['subfolder'] / item['filename']
    pixels = np.asarray(Image.open(path).convert('RGBA'))
    original_generate = deployment.generate

    def generate(label, graph, unused_history, unused_pixels):
        return original_generate(label, graph, reference, pixels)

    deployment.generate = generate
    lifecycle.ROOT = args.root
    lifecycle.job = 'ordinary-deployment'
    lifecycle.stop_monitor.clear()
    monitor = threading.Thread(target=lifecycle.monitor, daemon=True)
    monitor.start()
    try:
        deployment.main()
    finally:
        lifecycle.stop_monitor.set()
        monitor.join(timeout=10)


if __name__ == '__main__':
    main()
