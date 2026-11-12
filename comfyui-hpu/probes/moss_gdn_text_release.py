"""Verify final-product FP8 logits and paired BF16 text timing and logits."""
import json
import os
from pathlib import Path
import subprocess
import sys


root = Path(os.environ['MOSS_OPT_ROOT'])
vendor = '/usr/lib/habanalabs/libtpc_kernels.so'
native = f'{vendor}:{Path.cwd()}/build/hpu/libcomfy_gdn_tpc.so'
env = dict(os.environ, PT_HPU_LAZY_MODE='0', PYTHONPATH=str(Path.cwd()),
           MOSS_WORKFLOW=str(root / 'pipeline-baseline/source.json'),
           MOSS_MTP_PROGRAM=str(root.parent / 'performance-2026-11-11-completion/mtp-record-298/program.pt'),
           MOSS_TEXT_PROFILE_REPEATS='0', MOSS_MTP_LOGITS='1', MOSS_MTP_LOGITS_GENERATION='2')
for label, dtype, library, repeats in [('text-product', 'fp8', native, '1'),
                                       ('text-bf16-baseline', 'bf16', vendor, '6'),
                                       ('text-bf16-product', 'bf16', native, '6')]:
    directory = root / label
    directory.mkdir(exist_ok=True)
    settings = dict(env, GC_KERNEL_PATH=library, MOSS_RESULT_DIR=str(directory), MOSS_TEXT_PLAIN_REPEATS=repeats,
                    MOSS_CLIP_FILE=f'/home/sonny/ComfyUI-Models/models/text_encoders/qwen3.5_9b_{dtype}_from_int8.safetensors')
    with (root / f'{label}.log').open('w') as stream:
        subprocess.run([sys.executable, str(Path(__file__).with_name('moss_evidence_text.py'))],
                       env=settings, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(label, 'complete', flush=True)

import torch
torch.set_num_threads(4)
comparisons = {}
for dtype, first, second in [('fp8', 'text-baseline', 'text-product'),
                             ('bf16', 'text-bf16-baseline', 'text-bf16-product')]:
    expected = torch.load(root / first / 'mtp-logits-2.pt', weights_only=True, map_location='cpu')
    actual = torch.load(root / second / 'mtp-logits-2.pt', weights_only=True, map_location='cpu')
    assert expected.shape == actual.shape
    divergences, equal, squared, reference_squared, maximum = [], 0, 0.0, 0.0, 0.0
    for a, b in zip(expected.split(16), actual.split(16)):
        a, b = a.float(), b.float()
        assert b.isfinite().all()
        p, q = a.log_softmax(-1), b.log_softmax(-1)
        divergences.extend((p.exp() * (p - q)).sum(-1).tolist())
        equal += int((a.argmax(-1) == b.argmax(-1)).sum())
        squared += float((a - b).square().sum())
        reference_squared += float(a.square().sum())
        maximum = max(maximum, float((a - b).abs().max()))
    comparisons[dtype] = dict(shape=list(actual.shape), top1_equal=equal, kl_mean=sum(divergences) / len(divergences),
                              kl_max=max(divergences), rrmse=(squared / reference_squared) ** .5, max_absolute=maximum)
(root / 'product-logits.json').write_text(json.dumps(comparisons, indent=2))
print(json.dumps(comparisons), flush=True)
