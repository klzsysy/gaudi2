"""Repeat the original product, then validate the final text optimization serially."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--wait-pid', type=int)
a = p.parse_args()
if a.wait_pid:
    while Path(f'/proc/{a.wait_pid}').exists():
        time.sleep(5)

base = Path('/home/sonny/gaudi2/comfyui-hpu')
root = base/'performance-qwen21-2026-11-15'
product = Path('/home/sonny/ComfyUI-Gaudi')
env = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so')
files = ['qwen35.py', 'hpu_mlp.py']
for name in files:
    shutil.copy2(root/'product-baseline'/name, product/'comfy/text_encoders'/name)
command = [sys.executable, str(base/'probes/qwen21_deep_jobs.py'),
           '--workflow', str(base/'workflows/qwen21-latest-2026-11-15-api.json'),
           '--output-root', str(root)]
try:
    subprocess.run(command+['latest-reference-repeat', '--runs', '4', '--cached', '0'], env=env, check=True)
finally:
    for name in files:
        shutil.copy2(root/'product-final-source'/name, product/'comfy/text_encoders'/name)
with (root/'product-tests-final.log').open('w') as log:
    subprocess.run([sys.executable, '-m', 'pytest', 'tests-unit/comfy_test/hpu_mlp_test.py',
                    'tests-unit/comfy_test/hpu_gdn_region_test.py', '-q', '-x'],
                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
subprocess.run(command+['latest-final', '--runs', '4', '--cached', '3', '--free'], env=env, check=True)
