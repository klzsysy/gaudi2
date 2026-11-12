"""Run isolated compiler-layout experiments and record requested/read-back settings."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


root = Path(os.environ['MOSS_QUALITY_ROOT'])
base = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
            PYTHONPATH=str(Path.cwd()))
cases = [
    ('layered', dict(ENABLE_LAYERED_PIPELINE_BRAIN='1')),
    ('layered-attention', dict(ENABLE_LAYERED_PIPELINE_BRAIN='1', ENABLE_LAYERED_BRAIN_ATTENTION_SEEDS='1',
                              ENABLE_LAYERED_BRAIN_MULTI_MME_ATTENTION_SEEDS='1', ENABLE_LAYERED_BRAIN_MULTI_MME_SEEDS='1')),
    ('flash-slicing', dict(ENABLE_FLASH_ATTENTION_SLICING='1')),
    ('sram-32', dict(SRAM_SLICER_MAX_CAPACITY_BYTES=str(32 * 1024**2))),
    ('sram-24', dict(SRAM_SLICER_MAX_CAPACITY_BYTES=str(24 * 1024**2))),
    ('depth-2', dict(DEFAULT_PIPELINE_DEPTH='2')),
    ('depth-5', dict(DEFAULT_PIPELINE_DEPTH='5')),
    ('shared-multibuffer', dict(ENABLE_SHARED_MULTIBUF_PER_SLICED_CHAIN='1')),
]
for label, settings in cases:
    use_api = os.environ.get('MOSS_COMPILER_USE_API') == '1'
    if use_api:
        label = 'api-' + label
    start = time.monotonic()
    command = [sys.executable, str(Path(__file__).with_name('moss_attention_layout_screen.py')), str(root), '--label', label]
    if use_api:
        command.extend(['--configuration', json.dumps(settings)])
    with (root / f'{label}.log').open('w') as stream:
        process = subprocess.Popen(command, env=base if use_api else dict(base, **settings), stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=360)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            code = 'timeout'
    record = dict(label=label, requested=settings, returncode=code, seconds=time.monotonic() - start)
    with (root / 'compiler-jobs.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)
