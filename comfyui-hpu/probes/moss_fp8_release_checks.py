"""Serial regression, asynchronous reload acceptance, and final fixed workload."""
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ['MOSS_SYSTEM_ROOT'])
probes = Path(__file__).parent
base = dict(os.environ, PT_HPU_LAZY_MODE='0', PT_HPU_EAGER_PIPELINE_ENABLE='1',
            GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so', PYTHONPATH=str(Path.cwd()))
tests = sorted(str(path) for path in Path('tests-unit/comfy_test').glob('hpu_*test.py'))
tests += ['tests-unit/comfy_test/qwen35_fp8_conversion_test.py', 'tests-unit/comfy_quant/test_mixed_precision.py', 'tests-unit/comfy_test/model_detection_test.py']
jobs = [
    ('tests-release', [sys.executable, '-m', 'pytest', *tests, '-q'], {}),
    ('reload-release', [sys.executable, str(probes / 'moss_fp8_text_quality.py')], {'MOSS_QUALITY_MODE': 'fp8', 'MOSS_QUALITY_DIRECTORY': 'quality-verified'}),
    ('pipeline-release', [sys.executable, str(probes / 'moss_fp8_system_jobs.py')], {'MOSS_SYSTEM_CASES': 'native-release', 'MOSS_FP8_TEXT': '0', 'MOSS_SYSTEM_PROFILE': '0'}),
]
for name, command, settings in jobs:
    with (root / (name + '.log')).open('w') as log:
        subprocess.run(command, env=dict(base, **settings), stdout=log, stderr=subprocess.STDOUT, check=True)
    print(name, 'complete', flush=True)
