import os
from pathlib import Path
import torch
import habana_frameworks.torch

torch.ops.load_library(str(Path(os.environ['MOSS_OPT_ROOT'])/'gdn-build/moss_gdn_bridge.so'))


@torch.library.register_fake('custom_op::moss_gdn_core')
def metadata(state,pk,pv,pd,pb,q,k,v,d,b,program):
    return torch.empty_like(state),state.new_empty((state.shape[0],q.shape[2],state.shape[1],state.shape[2]))


def core(state,pk,pv,pd,pb,q,k,v,d,b,program,pending=None):
    if pending is None:
        pending=program.new_full((1,),pk.shape[-1])
    control=torch.cat((program.to(torch.int32),pending.reshape(1).to(torch.int32)))
    committed,output=torch.ops.custom_op.moss_gdn_core(
        state.transpose(-1,-2).contiguous(),pk.transpose(-1,-2).contiguous(),pv.transpose(-1,-2).contiguous(),
        pd.transpose(1,2).contiguous(),pb.transpose(1,2).contiguous(),q.transpose(-1,-2).contiguous(),
        k.transpose(-1,-2).contiguous(),v.transpose(-1,-2).contiguous(),d.transpose(1,2).contiguous(),
        b.transpose(1,2).contiguous(),control)
    return committed.transpose(-1,-2).contiguous(),output
