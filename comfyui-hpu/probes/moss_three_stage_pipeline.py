"""Run a selected existing DiT/text pair through the fixed-work API benchmark."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('label')
parser.add_argument('--dit', choices=['fp8', 'bf16'], default='fp8')
parser.add_argument('--text', choices=['fp8', 'bf16'], default='fp8')
parser.add_argument('--rope', action='store_true')
parser.add_argument('--rope-mode', choices=['functional', 'foreach', 'opaque-none', 'opaque-functional-none'], default='functional')
parser.add_argument('--logits', action='store_true')
options = parser.parse_args()
target = options.root / options.label
target.mkdir(parents=True, exist_ok=True)
source = json.loads((options.root.parent / 'performance-2026-11-11-fp8-system/native-release/source.json').read_text())
source['18:1']['inputs']['unet_name'] = ('Moody-Moss-Mix-V4.1_00001__clean-fp8.safetensors'
                                      if options.dit == 'fp8' else 'moss_image_sft_bf16.safetensors')
source['18:2']['inputs']['clip_name'] = f'qwen3.5_9b_{options.text}_from_int8.safetensors'
(target / 'source.json').write_text(json.dumps(source, indent=2))
env = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
           PYTHONPATH=str(Path.cwd()), MOSS_PIPELINE_DIR=str(target), MOSS_MTP_FIXED='1',
           MOSS_MTP_PROGRAM=str(options.root.parent / 'performance-2026-11-11-completion/mtp-record-298/program.pt'),
           MOSS_RESULT_DIR=str(target / 'mtp'), MOSS_MTP_PROFILE='0',
           MOSS_MTP_LOGITS='1' if options.logits else '0', MOSS_MTP_LOGITS_GENERATION='7',
           MOSS_SERVER_SCRIPT='moss_three_stage_server.py', MOSS_PIPELINE_CACHE_RUNS='0', MOSS_RELEASE='1')
if options.rope:
    env['MOSS_TEXT_ROPE'] = options.rope_mode
with (target / 'job.log').open('w') as log:
    subprocess.run([sys.executable, str(Path(__file__).with_name('moss_pipeline_measure.py'))],
                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
print(options.label, 'complete', flush=True)
