"""Validate product entry points, then paired natural quality, on one HPU."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ['MOSS_SYSTEM_ROOT'])
probes = Path(__file__).parent
jobs = [
    ('product', 'moss_fp8_system_jobs.py', {'MOSS_SYSTEM_CASES': 'native-product', 'MOSS_FP8_TEXT': '0', 'MOSS_SYSTEM_PROFILE': '0'}),
    ('quality', 'moss_fp8_text_quality.py', {}),
]
for name, script, settings in jobs:
    with (root / (name + '-final.log')).open('w') as log:
        subprocess.run([sys.executable, str(probes / script)], env=dict(os.environ, **settings), stdout=log, stderr=subprocess.STDOUT, check=True)
    print(name, 'complete', flush=True)
