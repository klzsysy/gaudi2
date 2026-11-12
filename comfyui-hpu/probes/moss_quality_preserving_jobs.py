"""Compare complete workloads, recheck restored quality, and restart the ordinary service."""
import os
from pathlib import Path
import subprocess
import sys


root = Path(os.environ['MOSS_QUALITY_ROOT'])
probes = Path(__file__).parent
env = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
           PYTHONPATH=str(Path.cwd()))
jobs = [
    ('pipeline-sram32', 'moss_fp8_deep_jobs.py',
     dict(MOSS_DEEP_ROOT=str(root), MOSS_DEEP_CASE='pipeline-sram32', MOSS_DEEP_SERVER='moss_sram_server.py')),
    ('pipeline-baseline', 'moss_fp8_deep_jobs.py',
     dict(MOSS_DEEP_ROOT=str(root), MOSS_DEEP_CASE='pipeline-baseline')),
    ('quality-rollback', 'moss_fp8_text_quality.py',
     dict(MOSS_SYSTEM_ROOT=str(root), MOSS_QUALITY_MODE='fp8', MOSS_QUALITY_DIRECTORY='quality-rollback')),
]
try:
    for name, script, settings in jobs:
        with (root / f'{name}.log').open('w') as log:
            subprocess.run([sys.executable, str(probes / script)], env=dict(env, **settings),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print(name, 'complete', flush=True)
finally:
    with (root / 'restore-service.log').open('w') as log:
        subprocess.run(['./start-hpu.sh'], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
