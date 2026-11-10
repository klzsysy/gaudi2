"""Check cache tree compaction independently of attention compilation."""
import json
import os
import torch
from comfy.cli_args import args
from comfy.text_encoders.llama import FixedKVCache
from comfy.text_encoders.qwen35 import MTP_MODES, VerifyTree


torch.set_grad_enabled(False)
method = os.environ.get('MOSS_COMMIT_METHOD', 'original')
if method == 'clone':
    def commit(cache, base, path):
        for row, source in enumerate(path[1:], 1):
            if source != row:
                dst, src = (base + row) % cache.key.shape[2], (base + source) % cache.key.shape[2]
                cache.key[:, :, dst] = cache.key[:, :, src].clone()
                cache.value[:, :, dst] = cache.value[:, :, src].clone()
    FixedKVCache.commit_path = commit
elif method == 'index':
    def commit(cache, base, path):
        rows = [(row, source) for row, source in enumerate(path[1:], 1) if source != row]
        if rows:
            dst = torch.tensor([(base + row) % cache.key.shape[2] for row, _ in rows], device=cache.key.device)
            src = torch.tensor([(base + source) % cache.key.shape[2] for _, source in rows], device=cache.key.device)
            cache.key.index_copy_(2, dst, cache.key.index_select(2, src))
            cache.value.index_copy_(2, dst, cache.value.index_select(2, src))
    FixedKVCache.commit_path = commit
for disabled in (True, False):
    args.disable_comfy_compiler = disabled
    torch.manual_seed(73)
    cache = FixedKVCache.zeros(1, 1, 1025, 64, 'hpu', torch.bfloat16, FixedKVCache.shared(1, 'hpu'))
    topology = VerifyTree(MTP_MODES['tree2s2'][1], 8, 'hpu')
    cache.tracker['tree'] = topology
    cache.index = 254
    cache.key.copy_(torch.randn(cache.key.shape, dtype=torch.bfloat16))
    cache.value.copy_(torch.randn(cache.value.shape, dtype=torch.bfloat16))
    for turn in range(3):
        cache.prepare(8)
        q = torch.randn(1, 4, 8, 64, dtype=torch.bfloat16).to('hpu')
        k = torch.randn(1, 1, 8, 64, dtype=torch.bfloat16).to('hpu')
        v = torch.randn_like(k)
        cache.write(k, v)
        base, source = cache.index, topology.children[0][-1]
        before = cache.key.cpu().clone()
        expected = k[:, :, source].cpu().clone()
        cache.attend(q, k, v).cpu()
        cache.advance(8)
        cache.rollback(6)
        cache.commit_path(base, [0, source])
        actual = cache.key[:, :, base + 1].cpu()
        print(json.dumps({'method': method, 'disabled': disabled, 'turn': turn, 'base': base, 'source': source,
                          'write_error': (before[:, :, base + source] - expected).abs().max().item(),
                          'commit_error': (actual - expected).abs().max().item(),
                          'unchanged_destination': bool(torch.equal(before[:, :, base + 1], actual)),
                          'input_error': (k[:, :, source].cpu() - expected).abs().max().item()}), flush=True)
        topology.update(MTP_MODES['tree4s1m8' if turn == 0 else 'tree2s2'][1])
