"""Serial isolated PE experiments, with ordinary-service restoration."""
import argparse
import hashlib
import fcntl
import json
import os
import subprocess
import shutil
import sys
import time
from pathlib import Path
import psutil

import qwen21_latest_jobs as lifecycle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--trials', nargs='+', required=True)
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--quality', action='store_true')
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--restore-health-only', action='store_true')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    lifecycle.ROOT = args.root
    while args.wait_pid and psutil.pid_exists(args.wait_pid):
        time.sleep(2)
    lock = (lifecycle.BASE / '.qwen80-pe.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    queue = lifecycle.request('/queue')
    assert not queue['queue_running'] and not queue['queue_pending'], queue
    teacher = lifecycle.BASE / 'performance-qwen21-latest-2026-11-16/native-text/plain-tokens.json'
    environment = dict(os.environ, PT_HPU_LAZY_MODE='0',
                       GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
                       PYTHONPATH=os.pathsep.join((str(lifecycle.PRODUCT), str(lifecycle.PROBES))))
    outcomes = {}
    try:
        subprocess.run(['./stop-hpu.sh'], cwd=lifecycle.PRODUCT, check=True)
        for trial in args.trials:
            snapshot = args.root / (trial + '-source')
            snapshot.mkdir()
            hashes = {}
            paths = list(lifecycle.PROBES.glob('qwen80_*.py')) + list((lifecycle.PROBES / 'native_gdn_immediate').glob('*.*'))
            paths += [lifecycle.PROBES / 'qwen21_text_profile.py']
            for path in paths:
                relative = path.relative_to(lifecycle.PROBES)
                destination = snapshot / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
            (snapshot / 'hashes.json').write_text(json.dumps(hashes, indent=2))
            product_files = ['comfy/text_encoders/llama.py', 'comfy/text_encoders/qwen35.py',
                             'comfy/text_encoders/hpu_kv_attention.py', 'comfy/text_encoders/hpu_mlp.py',
                             'comfy/ops.py', 'comfy/quant_ops.py']
            product_hashes = {name: hashlib.sha256((lifecycle.PRODUCT / name).read_bytes()).hexdigest()
                              for name in product_files}
            (snapshot / 'product-hashes.json').write_text(json.dumps(product_hashes, indent=2))
            current_environment = environment.copy()
            for part in trial.split('+'):
                if part.startswith('tpc'):
                    build = lifecycle.BASE / 'performance-qwen80-pe-2026-11-16/tpc-build'
                    current_environment['QWEN_IMMEDIATE_ROOT'] = str(build)
                    current_environment['GC_KERNEL_PATH'] += ':' + str(build / part[3:] / 'libqwen_gdn_immediate_tpc.so')
            command = [sys.executable, str(lifecycle.PROBES / 'qwen21_text_profile.py'),
                       str(args.root / trial), '--workflow', str(lifecycle.WORKFLOW),
                       '--window-steps', '64', '--profile-start-token', '64', '--no-region-labels']
            if args.tokens:
                command += ['--teacher', str(args.teacher or teacher), '--window-tokens', str(args.tokens)]
            elif trial != 'baseline' or args.teacher is not None:
                command += ['--teacher', str(args.teacher or args.root / 'baseline/plain-tokens.json')]
            command += ['--logits'] if args.quality else ['--plain-only', '--plain-runs', str(args.runs)]
            if trial != 'baseline':
                command += ['--trial', 'hardware80:pe:' + trial]
            if trial.startswith('token'):
                block = trial[5:].split('-')[0]
                command = [sys.executable, str(lifecycle.PROBES / 'qwen80_token_probe.py'),
                           str(args.root / trial), '--block', block, '--tokens', str(args.tokens),
                           '--runs', str(args.runs)]
                if 'vertical' in trial:
                    command += ['--vertical', '--align']
                if 'native' in trial:
                    command += ['--native-rope']
                if args.quality:
                    command += ['--logits']
            if trial.startswith('graph'):
                current_environment['PT_HPU_LAZY_MODE'] = '1'
                current_environment['PT_HPU_WEIGHT_SHARING'] = '0'
                command = [sys.executable, str(lifecycle.PROBES / 'qwen80_token_probe.py'),
                           str(args.root / trial), '--block', trial[5:].split('-')[0], '--tokens', str(args.tokens),
                           '--runs', str(args.runs), '--graph']
                if args.quality:
                    command += ['--logits']
                if 'functional' in trial:
                    command += ['--functional-graph']
            if trial == 'attention-micro':
                command = [sys.executable, str(lifecycle.PROBES / 'qwen80_attention_probe.py'), str(args.root / trial)]
            if trial == 'matrix-micro':
                command = [sys.executable, str(lifecycle.PROBES / 'qwen80_matrix_probe.py'), str(args.root / trial)]
            if trial == 'conv-micro':
                command = [sys.executable, str(lifecycle.PROBES / 'qwen80_conv_probe.py'), str(args.root / trial)]
            if trial == 'frequency-micro':
                command = [sys.executable, str(lifecycle.PROBES / 'qwen80_frequency_probe.py'), str(args.root / trial)]
            if trial == 'padding-micro':
                command = [sys.executable, str(lifecycle.PROBES / 'qwen80_padding_probe.py'), str(args.root / trial)]
            if trial.startswith('rng-'):
                rng_mode = trial.removeprefix('rng-')
                current_environment['PT_HPU_LAZY_MODE'] = '1' if rng_mode == 'graph' else '0'
                current_environment['PT_HPU_WEIGHT_SHARING'] = '0'
                command = [sys.executable, str(lifecycle.PROBES / 'qwen80_rng_probe.py'), str(args.root / trial), rng_mode]
            try:
                lifecycle.run(trial, command, current_environment)
                outcomes[trial] = 0
            except subprocess.CalledProcessError as error:
                outcomes[trial] = error.returncode
                if error.returncode < 0:
                    raise
            finally:
                (args.root / 'exits.json').write_text(json.dumps(outcomes, indent=2))
    finally:
        subprocess.run(['./start-hpu.sh'], cwd=lifecycle.PRODUCT, check=True, timeout=180)
        if args.restore_health_only:
            (args.root / 'restored-queue.json').write_text(json.dumps(lifecycle.request('/queue')))
        else:
            lifecycle.verify_restoration()


if __name__ == '__main__':
    main()
