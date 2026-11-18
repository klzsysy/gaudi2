"""Execution-local D2 SRAM prototype with compiled input packing and numeric gates."""
import math
import os
from pathlib import Path
import sys
import time

import torch
from habana_frameworks.torch.utils.experimental import _data_ptr
from comfy.ldm.modules.attention import _hpu_fast_attention


SCORE_FACTOR = 261.25 / 184.6649652337873


def prepare(q, k, v, query_buffer, key_buffer, value_buffer):
    centered = (k.float() - k.float().mean(dim=-2, keepdim=True)).to(k.dtype)
    query_buffer[:, :q.shape[-2]].copy_(q[0] * .0625)
    query_buffer[:, q.shape[-2]:].zero_()
    value_buffer[:, :v.shape[-2]].copy_(v[0])
    value_buffer[:, v.shape[-2]:].zero_()
    key_buffer[:, :, :k.shape[-2]].copy_(centered[0].transpose(-1, -2))
    key_buffer[:, :, k.shape[-2]:].zero_()
    query_square = q[0].float().square().sum(-1)
    key_square = centered[0].float().square().sum(-1)
    upper = (query_square.amax(-1) * key_square.amax(-1)).sqrt() * (.0625 * SCORE_FACTOR)
    maximum_value = v[0].abs().amax(dim=(-1, -2)).float()
    finite = query_square.sum(-1) + key_square.sum(-1) + v[0].float().sum((-1, -2))
    return torch.stack([upper, maximum_value, finite])


class NativeAttention:
    def __init__(self, root, query_length=8160, capacity=9472, heads=32, valid=9445):
        source = Path(root) / 'native-parts1/mme-direct-attention/attention'
        os.environ['H3DIRECT_ROOT'] = str(source)
        os.environ['H3DIRECT_POW2Q'] = '1'
        sys.path.insert(0, str(source))
        import h3_direct_attention as direct
        direct.GROUP = 64
        blocks = (query_length + 8191) // 8192 * 64
        direct.geometry = lambda keys: (keys, blocks, 1)
        self.direct = direct
        self.query_length = query_length
        self.capacity = capacity
        self.heads = heads
        self.plan = direct._Plan(False, heads, capacity, valid)
        self.prepare = torch.compile(prepare, backend='hpu_backend', dynamic=None,
                                     options={'force_static_compile': True})
        self.fallback = torch.compile(_hpu_fast_attention, backend='hpu_backend', dynamic=None,
                                      options={'force_static_compile': True})
        self.records = []

    def __call__(self, q, k, v):
        assert q.shape == (1, self.heads, self.query_length, 128)
        assert k.shape == v.shape and k.shape[:2] == q.shape[:2] and k.shape[-1] == 128
        assert k.shape[-2] <= self.capacity
        started = time.perf_counter()
        plan = self.plan
        assert plan.set_valid(k.shape[-2]), 'Native length update failed'
        gates = self.prepare(q, k, v, plan.qh, plan.kth.view(self.heads, 128, self.capacity), plan.vh)
        output = torch.empty_like(plan.qh)
        upper, vmax, finite = gates.cpu().tolist()
        torch.hpu.synchronize()
        allowed = self.direct._gate_decisions(upper, vmax, [math.isfinite(value) for value in finite], k.shape[-2])
        packed = time.perf_counter()
        good = [head for head, valid in enumerate(allowed) if valid]
        plan.run_heads_stream(good, _data_ptr(output))
        computed = time.perf_counter()
        bad = [head for head, valid in enumerate(allowed) if not valid]
        result = output[:, :self.query_length]
        if bad:
            indices = torch.tensor(bad, device=q.device)
            result[indices] = self.fallback(q[:, indices], k[:, indices], v[:, indices])[0]
        torch.hpu.synchronize()
        self.records.append(dict(keys=k.shape[-2], direct_heads=good, fallback_heads=bad,
                                 packing_seconds=packed-started, native_seconds=computed-packed,
                                 seconds=time.perf_counter()-started))
        return result.unsqueeze(0)

    def close(self):
        torch.hpu.synchronize()
        self.plan.release()
        self.plan = None
