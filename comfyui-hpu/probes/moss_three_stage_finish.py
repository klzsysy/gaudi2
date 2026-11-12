"""Serialize final BF16, paired natural-quality and regression checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


root = Path(sys.argv[1])
probes = Path(__file__).parent
python = sys.executable
env = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
           PYTHONPATH=str(Path.cwd()))
tests = sorted(str(path) for path in Path('tests-unit/comfy_test').glob('hpu_*test.py'))
tests += ['tests-unit/comfy_test/qwen35_decode_test.py', 'tests-unit/comfy_test/qwen35_fp8_conversion_test.py',
          'tests-unit/comfy_quant/test_mixed_precision.py', 'tests-unit/comfy_test/model_detection_test.py']
quality = dict(MOSS_SYSTEM_ROOT=str(root), MOSS_QUALITY_SERVER='moss_three_stage_server.py',
               MOSS_QUALITY_MTP='tree3s1')
jobs = [
    ('product-bf16', [python, str(probes / 'moss_three_stage_pipeline.py'), str(root), 'product-functional-bf16',
                      '--dit', 'bf16', '--text', 'bf16', '--logits'], {}),
    ('product-bf16-logits', [python, str(probes / 'moss_three_stage_compare.py'),
                            str(root / 'bf16-baseline/mtp/mtp-logits-7.pt'),
                            str(root / 'product-functional-bf16/mtp/mtp-logits-7.pt'),
                            str(root / 'product-functional-bf16-logits.json')], {}),
    ('quality-baseline', [python, str(probes / 'moss_fp8_text_quality.py')],
     dict(quality, MOSS_QUALITY_DIRECTORY='quality-baseline', MOSS_TEXT_ROPE_BASELINE='1')),
    ('quality-product', [python, str(probes / 'moss_fp8_text_quality.py')],
     dict(quality, MOSS_QUALITY_DIRECTORY='quality-product', MOSS_TEXT_ROPE_BASELINE='0')),
    ('regressions', [python, '-m', 'pytest', *tests, '-q'], {}),
]
for name, command, settings in jobs:
    start = time.monotonic()
    with (root / f'finish-{name}.log').open('w') as log:
        result = subprocess.run(command, env=dict(env, **settings), stdout=log, stderr=subprocess.STDOUT)
    record = dict(name=name, returncode=result.returncode, seconds=time.monotonic() - start)
    with (root / 'finish-jobs.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
