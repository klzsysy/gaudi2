"""Serial fixed-MTP workflow comparison with cold, warm and separate profiles."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ['MOSS_SYSTEM_ROOT'])
source = json.loads((root.parent / 'performance-2026-11-11-bf16-next/source.json').read_text())
source['18:1']['inputs']['unet_name'] = 'Moody-Moss-Mix-V4.1_00001__clean-fp8.safetensors'
for name in os.environ.get('MOSS_SYSTEM_CASES', 'baseline,native-text').split(','):
    target = root / name
    target.mkdir(parents=True, exist_ok=True)
    graph = json.loads(json.dumps(source))
    native = name != 'baseline'
    if native:
        graph['18:2']['inputs']['clip_name'] = 'qwen3.5_9b_fp8_from_int8.safetensors'
    (target / 'source.json').write_text(json.dumps(graph, indent=2))
    env = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
               PYTHONPATH=str(Path.cwd()), MOSS_PIPELINE_DIR=str(target), MOSS_MTP_FIXED='1',
               MOSS_MTP_PROGRAM=str(root.parent / 'performance-2026-11-11-completion/mtp-record-298/program.pt'),
               MOSS_RESULT_DIR=str(target / 'mtp'), MOSS_MTP_PROFILE=os.environ.get('MOSS_SYSTEM_PROFILE', '1'), MOSS_MTP_PROFILE_GENERATION='7',
               MOSS_SERVER_SCRIPT='moss_fp8_system_server.py', MOSS_PIPELINE_CACHE_RUNS='0', MOSS_RELEASE='1',
               MOSS_FP8_TEXT=os.environ.get('MOSS_FP8_TEXT', '1') if native else '0', MOSS_MTP_LOGITS=os.environ.get('MOSS_SYSTEM_PROFILE', '1'), MOSS_MTP_LOGITS_GENERATION='7')
    with (target / 'job.log').open('w') as log:
        subprocess.run([sys.executable, str(Path(__file__).with_name('moss_pipeline_measure.py'))], env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print(name, 'complete', flush=True)
