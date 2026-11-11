"""Serial native/reference timing, quality and lifecycle checks for the imported DiT."""
import json
import os
from pathlib import Path
import subprocess
import sys
root = Path(os.environ['MOSS_FP8_ROOT'])
source = json.loads((root / 'fixed-source.json').read_text())
for mode in ('reference', 'native'):
    destination = root / ('fixed-' + mode)
    destination.mkdir(exist_ok=True)
    (destination / 'source.json').write_text(json.dumps(source))
    env = dict(os.environ, MOSS_FP8_ROOT=str(destination), MOSS_FP8_MODE=mode, MOSS_FP8_RUNS='6', MOSS_FP8_ACCEPTANCE='1')
    with (destination / 'workflows.log').open('w') as log:
        subprocess.run([sys.executable, str(Path(__file__).with_name('moss_fp8_workflows.py'))], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(mode, 'completed', flush=True)
