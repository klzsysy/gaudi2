"""Serial FP8 stage captures using the calibrated AXI response-beat setup."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('workflow', type=Path)
parser.add_argument('--stages', nargs='+', default=['text', 'dit', 'clip'])
parser.add_argument('--caption-history', type=Path)
a = parser.parse_args()
probes = Path(__file__).parent
base = json.loads((probes.parent / 'performance-qwen-hbm-2026-11-15/text0-c16383-b255-confirm/profiler.json').read_text())
for stage in a.stages:
    directory = a.root / ('native-' + stage)
    directory.mkdir(parents=True, exist_ok=True)
    config = json.loads(json.dumps(base))
    config['GeneralSettings']['values']['outdir']['value'] = str(directory / 'hw')
    config['GeneralSettings']['values']['session']['value'] = 'fp8-' + stage
    configuration = directory / 'profiler.json'
    configuration.write_text(json.dumps(config, indent=2))
    environment = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
                       PYTHONPATH=str(Path.cwd()), HABANA_PROFILE='1', HABANA_PROF_CONFIG=str(configuration))
    if stage == 'text':
        command = [sys.executable, str(probes / 'qwen21_text_profile.py'), str(directory),
                   '--workflow', str(a.workflow), '--native', '--window-tokens', '64', '--profile-start-token', '32']
    else:
        workflow = json.loads(a.workflow.read_text())
        history = json.loads((a.caption_history or a.root / 'fp8-off/full-1.json').read_text())
        workflow['459:452']['inputs']['prompt'] = history['outputs']['459:472']['text'][0]
        for node in ('459:471', '459:472', '459:473', '459:474', '459:475'):
            workflow.pop(node)
        frozen = directory / 'workflow.json'
        frozen.write_text(json.dumps(workflow, ensure_ascii=False, indent=2))
        command = [sys.executable, str(probes / 'qwen21_deep_jobs.py'), directory.name,
                   '--output-root', str(a.root), '--workflow', str(frozen), '--runs', '2', '--cached', '0',
                   '--server-module', 'qwen21_fp8_server', '--native-' + stage]
    with (a.root / ('native-' + stage + '-job.log')).open('w') as log:
        result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    print(stage, result.returncode, flush=True)
    if result.returncode:
        raise RuntimeError('Inspect failed capture log before continuing')
