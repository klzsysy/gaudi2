"""Independent-head SDPA grouping using the existing operation and real block data."""
import json
import os
from pathlib import Path
import statistics
import time
import re

import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True
import comfy.ops
from comfy.ldm.moss_image import model as moss
from comfy.ldm.modules.attention import attention_basic
import habana_frameworks.torch.utils.debug as debug

torch.set_grad_enabled(False)
torch.set_num_threads(16)
root = Path(os.environ['MOSS_NEXT_ROOT'])
captures = Path(os.environ['MOSS_CAPTURE_ROOT'])
original = comfy.ops.scaled_dot_product_attention
block = moss.Block(3072, 8192, 24, fused_qkv=True, dtype=torch.bfloat16, operations=comfy.ops.manual_cast)
block.load_state_dict(torch.load(captures/'weights-0.pt', weights_only=True))
block.to('hpu')
records = []


def measure(function, count=10):
    start = time.perf_counter()
    output = function()
    torch.hpu.synchronize()
    cold = time.perf_counter()-start
    reference = output.cpu().float()
    del output
    for _ in range(3):
        function()
    torch.hpu.synchronize()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(count):
            function()
        torch.hpu.synchronize()
        samples.append((time.perf_counter()-start)/count)
    return reference, dict(cold_s=cold, median_ms=statistics.median(samples)*1000, samples_ms=[x*1000 for x in samples])


def grouped(width,layout):
    def attention(q, k, v, *values, **kwargs):
        assert not values and kwargs.get('attn_mask') is None
        assert q.shape[1] == k.shape[1] == v.shape[1]
        if layout == 'basic':
            return attention_basic(q,k,v,q.shape[1],skip_reshape=True,skip_output_reshape=True,scale=kwargs.get('scale',q.shape[-1]**-.5))
        if layout == 'batch':
            shape = q.shape
            result = original(q.flatten(0,1).unsqueeze(1),k.flatten(0,1).unsqueeze(1),v.flatten(0,1).unsqueeze(1),**kwargs)
            return result.reshape(shape[0],shape[1],shape[2],v.shape[-1])
        if layout == 'flat':
            shape = q.shape
            result = original(q.flatten(0,1),k.flatten(0,1),v.flatten(0,1),**kwargs)
            return result.reshape(shape[0],shape[1],shape[2],v.shape[-1])
        if width >= q.shape[1]:
            return original(q,k,v,**kwargs)
        outputs = [original(q[:, start:start+width], k[:, start:start+width], v[:, start:start+width], **kwargs)
                   for start in range(0,q.shape[1],width)]
        if layout == 'sequence':
            combined = torch.cat([part.transpose(1,2).flatten(2) for part in outputs],dim=-1)
            return combined.reshape(q.shape[0],q.shape[2],q.shape[1],v.shape[-1]).transpose(1,2)
        return torch.cat(outputs,dim=1)
    return attention


for condition in (307,23):
    data = torch.load(captures/f'input-{condition}-0-0.pt', weights_only=True)
    x = data['args'][0].to('hpu')
    mod = tuple(t.to('hpu') for t in data['args'][1])
    pe = data['args'][2].to('hpu')
    options = {'_moss_hpu_blockwise_rope': True}
    assert moss.hpu.blockwise_coefficients(pe)
    captured = []
    def capture(q,k,v,*values,**kwargs):
        captured.append((q,k,v))
        return original(q,k,v,*values,**kwargs)
    comfy.ops.scaled_dot_product_attention = capture
    block._forward_attention(moss.modulate(x,block.norm1,mod,0,None),pe,options)
    q,k,v = captured[0]
    cases = os.environ.get('MOSS_HEAD_CASES','head24,head12,head8,head6,head4,head3,head2,head1').split(',')
    for case in cases:
        layout,width = re.fullmatch(r'([a-z]+)(\d+)',case).groups()
        width = int(width)
        torch._dynamo.reset()
        moss._compiled_block_ops.cache_clear()
        moss._compiled_attention.cache_clear()
        attention = grouped(width,layout)
        comfy.ops.scaled_dot_product_attention = attention
        compiled = torch.compile(attention,backend='hpu_backend',dynamic=False,options={'force_static_compile':True})
        av,at = measure(lambda:compiled(q,k,v))
        bv,bt = measure(lambda:block(x.clone(),mod,pe,options))
        if case == 'head24':
            baseline_attention, baseline_block = av,bv
        record = dict(condition=condition,case=case,heads_per_group=width,attention=at,block=bt,
                      attention_rrmse=float((av-baseline_attention).square().mean().sqrt()/baseline_attention.square().mean().sqrt()),
                      block_rrmse=float((bv-baseline_block).square().mean().sqrt()/baseline_block.square().mean().sqrt()),
                      fallback=debug._get_fallback_op_count())
        records.append(record)
        (root/os.environ.get('MOSS_HEAD_RESULT','head-groups.json')).write_text(json.dumps(records,indent=2))
        print(json.dumps(record),flush=True)
    comfy.ops.scaled_dot_product_attention = original
