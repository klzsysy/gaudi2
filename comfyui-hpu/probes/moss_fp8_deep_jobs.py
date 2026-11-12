"""Run a fixed-work FP8 pipeline case using the existing measurement contract."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ['MOSS_DEEP_ROOT'])
name = os.environ.get('MOSS_DEEP_CASE', 'baseline')
target = root / name
target.mkdir(parents=True, exist_ok=True)
source = root.parent / 'performance-2026-11-11-fp8-system/native-release/source.json'
(target / 'source.json').write_text(source.read_text())
env = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH=os.environ.get('GC_KERNEL_PATH', '/usr/lib/habanalabs/libtpc_kernels.so'),
           PYTHONPATH=str(Path.cwd()), MOSS_PIPELINE_DIR=str(target), MOSS_MTP_FIXED='1',
           MOSS_MTP_PROGRAM=str(root.parent / 'performance-2026-11-11-completion/mtp-record-298/program.pt'),
           MOSS_RESULT_DIR=str(target / 'mtp'), MOSS_MTP_PROFILE='0', MOSS_MTP_LOGITS=os.environ.get('MOSS_DEEP_LOGITS', '0'),
           MOSS_MTP_LOGITS_GENERATION='7',
           MOSS_SERVER_SCRIPT=os.environ.get('MOSS_DEEP_SERVER', 'moss_fp8_deep_server.py'), MOSS_PIPELINE_CACHE_RUNS='0', MOSS_RELEASE='1')
with (target / 'job.log').open('w') as log:
    subprocess.run([sys.executable, str(Path(__file__).with_name('moss_pipeline_measure.py'))], env=env,
                   stdout=log, stderr=subprocess.STDOUT, check=True)
print(name, 'complete', flush=True)
