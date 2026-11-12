"""Serialize final regression, fixed-work timing, natural images and reload."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ['MOSS_FUSION_ROOT'])
label = os.environ.get('MOSS_FUSION_RELEASE_CASE', 'release')
probes = Path(__file__).parent
env = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
           PYTHONPATH=str(Path.cwd()))
tests = sorted(str(path) for path in Path('tests-unit/comfy_test').glob('hpu_*test.py'))
tests += ['tests-unit/comfy_test/qwen35_fp8_conversion_test.py',
          'tests-unit/comfy_quant/test_mixed_precision.py', 'tests-unit/comfy_test/model_detection_test.py']
jobs = [
    (f'tests-{label}', [sys.executable, '-m', 'pytest', *tests, '-q'], {}),
    (f'pipeline-{label}', [sys.executable, str(probes / 'moss_fp8_deep_jobs.py')],
     {'MOSS_DEEP_ROOT': str(root), 'MOSS_DEEP_CASE': label, 'MOSS_DEEP_PROFILE': '1'}),
    (f'quality-{label}', [sys.executable, str(probes / 'moss_fp8_text_quality.py')],
     {'MOSS_SYSTEM_ROOT': str(root), 'MOSS_QUALITY_MODE': 'fp8', 'MOSS_QUALITY_DIRECTORY': f'quality-{label}'}),
]
for name, command, settings in jobs:
    with (root / (name + '.log')).open('w') as stream:
        subprocess.run(command, env=dict(env, **settings), stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(name, 'complete', flush=True)
