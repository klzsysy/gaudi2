"""Check the built product GDN extension through regression and workflow entry points."""
import os
from pathlib import Path
import subprocess
import sys


root = Path(os.environ['MOSS_OPT_ROOT'])
probes = Path(__file__).parent
env = dict(os.environ, PT_HPU_LAZY_MODE='0', PYTHONPATH=str(Path.cwd()),
           GC_KERNEL_PATH=f'/usr/lib/habanalabs/libtpc_kernels.so:{Path.cwd()}/build/hpu/libcomfy_gdn_tpc.so')
tests = sorted(str(path) for path in Path('tests-unit/comfy_test').glob('hpu_*test.py'))
tests += ['tests-unit/comfy_test/qwen35_fp8_conversion_test.py',
          'tests-unit/comfy_quant/test_mixed_precision.py', 'tests-unit/comfy_test/model_detection_test.py']
test_program = ('from comfy.text_encoders import hpu_gdn; assert hpu_gdn.available; '
                'print("product_native_gdn_enabled", flush=True); import pytest; '
                f'raise SystemExit(pytest.main({tests + ["-q"]!r}))')
jobs = [
    ('product-tests', [sys.executable, '-c', test_program], {}),
    ('pipeline-product', [sys.executable, str(probes / 'moss_fp8_deep_jobs.py')],
     dict(MOSS_DEEP_ROOT=str(root), MOSS_DEEP_CASE='pipeline-product')),
    ('quality-product', [sys.executable, str(probes / 'moss_fp8_text_quality.py')],
     dict(MOSS_SYSTEM_ROOT=str(root), MOSS_QUALITY_MODE='fp8', MOSS_QUALITY_DIRECTORY='quality-product')),
]
for name, command, settings in jobs:
    with (root / f'{name}.log').open('w') as stream:
        subprocess.run(command, env=dict(env, **settings), stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(name, 'complete', flush=True)
