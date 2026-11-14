"""Serial candidate screening after the full-workflow baseline releases the HPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--wait-pid', type=int)
p.add_argument('--phase2', action='store_true')
p.add_argument('--phase3', action='store_true')
p.add_argument('--phase4', action='store_true')
p.add_argument('--phase5', action='store_true')
a = p.parse_args()
if a.wait_pid:
    while Path(f'/proc/{a.wait_pid}').exists():
        time.sleep(5)

base = Path('/home/sonny/gaudi2/comfyui-hpu')
root = base / 'performance-qwen21-2026-11-15'
env = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
           PYTHONPATH=str(Path.cwd()))
teacher = base / 'performance-qwen21-bf16-2026-11-13/teacher.json'
jobs = []
for name, trial in [('text-reference', ''), ('text-materialized', 'next:materialized'),
                    ('text-singlecommit', 'next:singlecommit')]:
    jobs.append((name, ['qwen21_text_profile.py', str(root/name), '--teacher', str(teacher), '--logits', '--trial', trial]))
for name, trial in [('block-reference', ''), ('block-modulation', 'next:modulation'),
                    ('block-heads16', 'next:heads16'), ('block-heads8', 'next:heads8'),
                    ('block-heads4', 'next:heads4'), ('block-heads1', 'next:heads1')]:
    command = ['qwen21_block_replay.py', str(root/'latest-baseline/block.pt'), str(root/name), '--trial', trial]
    if name != 'block-reference':
        command += ['--reference', str(root/'block-reference/reference.pt')]
    else:
        command += ['--profile']
    jobs.append((name, command))
if a.phase2:
    env['QWEN_AUDIT'] = str(root/'materialized-audit.jsonl')
    workflow = str(base/'workflows/qwen21-latest-2026-11-15-api.json')
    jobs = [
        ('text-singlecommit', ['qwen21_text_profile.py', str(root/'text-singlecommit'), '--teacher', str(teacher), '--logits', '--trial', 'next:singlecommit']),
        ('text-audit', ['qwen21_text_profile.py', str(root/'text-audit'), '--window-tokens', '64', '--plain-only', '--trial', 'next:audit_materialized']),
        ('block-native', ['qwen21_block_replay.py', str(root/'latest-baseline/block.pt'), str(root/'block-native'), '--reference', str(root/'block-reference/reference.pt'), '--trial', 'next:native_sdpa']),
        ('text-latest-trace', ['qwen21_text_profile.py', str(root/'text-latest-trace'), '--workflow', workflow, '--window-tokens', '64']),
        ('text-latest-cpu', ['qwen21_text_profile.py', str(root/'text-latest-cpu'), '--workflow', workflow, '--window-tokens', '64', '--cpu-profile']),
    ]
if a.phase3:
    env['QWEN_AUDIT'] = str(root/'materialized-audit.jsonl')
    jobs = [(name, ['qwen21_text_profile.py', str(root/name), '--teacher', str(teacher), '--logits', '--trial', trial])
            for name, trial in [('text-tail', 'next:tail'), ('text-singlecommit-retry', 'next:singlecommit')]]
if a.phase4:
    env['QWEN_AUDIT'] = str(root/'tail-audit.jsonl')
    jobs = [(name, ['qwen21_text_profile.py', str(root/name), '--teacher', str(teacher), '--logits', '--trial', trial])
            for name, trial in [('text-tail2', 'next:tail2'), ('text-mlpresidual', 'next:mlpresidual')]]
    jobs += [('text-tail-audit', ['qwen21_text_profile.py', str(root/'text-tail-audit'), '--window-tokens', '64', '--plain-only', '--trial', 'next:audit_tail'])]
if a.phase5:
    jobs = [(name, ['qwen21_text_profile.py', str(root/name), '--teacher', str(teacher), '--logits', '--trial', trial])
            for name, trial in [('text-core', 'next:core'), ('text-rope-cache', 'next:rope_cache')]]
results = []
for name, command in jobs:
    print('starting', name, flush=True)
    with (root/f'{name}.log').open('w') as log:
        start = time.monotonic()
        result = subprocess.run([sys.executable, str(base/'probes'/command[0]), *command[1:]],
                                env=env, stdout=log, stderr=subprocess.STDOUT)
    results.append(dict(name=name, returncode=result.returncode, seconds=time.monotonic()-start))
    phase = next((i for i in range(2, 6) if getattr(a, f'phase{i}')), None)
    report_name = f'screen-jobs-phase{phase}.json' if phase else 'screen-jobs.json'
    (root/report_name).write_text(json.dumps(results, indent=2))
    print(results[-1], flush=True)
