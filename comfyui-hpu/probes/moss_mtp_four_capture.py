"""Capture the real four-row attention input before a compiler failure."""
import os
from pathlib import Path
import runpy
import torch
from comfy.cli_args import args
args.disable_dynamic_vram=True
args.disable_pinned_memory=True
args.disable_async_offload=True
args.use_pytorch_cross_attention=True
from comfy.text_encoders import qwen35
if os.environ.get('MOSS_SHORT_STATIC'):
    import moss_short_static
    moss_short_static.install(os.environ['MOSS_SHORT_STATIC'])

root=Path(os.environ['MOSS_RESULT_DIR'])
root.mkdir(parents=True,exist_ok=True)
if os.environ.get('MOSS_FOUR_UNPADDED')=='1':
    from comfy.text_encoders import hpu_gated_delta
    from moss_tree_delta import parallel_tree
    hpu_gated_delta._tree_output=parallel_tree
if os.environ.get('MOSS_STAGE_FENCES')=='1':
    import moss_four_fences
    moss_four_fences.install()
original=qwen35.GatedAttention.forward
seen=False


def cpu(tensor):
    return tensor.detach().cpu().clone()


def forward(module,x,attention_mask=None,freqs_cis=None,optimized_attention=None,past_key_value=None):
    global seen
    if x.shape[1]==4 and not seen:
        seen=True
        cache=past_key_value
        config=dict(hidden_size=module.hidden_size,num_attention_heads=module.num_heads,num_key_value_heads=module.num_kv_heads,
                    head_dim=module.head_dim,partial_rotary_factor=module.rotary_dim/module.head_dim,
                    rms_norm_eps=module.q_norm.eps,rms_norm_add=module.q_norm.add)
        record=dict(x=cpu(x),weights={k:cpu(v) for k,v in module.state_dict().items()},config=config,
                    freqs=cpu(freqs_cis) if isinstance(freqs_cis,torch.Tensor) else tuple(cpu(v) for v in freqs_cis),
                    key=cpu(cache.key),value=cpu(cache.value),index=cache.index,seqlen=cpu(cache.seqlen),position=cpu(cache.position),
                    mask=cpu(cache.tracker['tree'].mask))
        torch.save(record,root/'attention-four.pt')
    return original(module,x,attention_mask,freqs_cis,optimized_attention,past_key_value)


qwen35.GatedAttention.forward=forward
runpy.run_path(str(Path(__file__).with_name('moss_gdn_experiment.py')),run_name='__main__')
