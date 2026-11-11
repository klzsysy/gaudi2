"""Parallel, division-free DeltaNet recurrence for at most eight tree rows."""
import torch


def path_weights(decay, parents):
    rows = decay.shape[-1]
    identity = torch.eye(rows,dtype=decay.dtype,device=decay.device)
    positions = torch.arange(rows,device=parents.device)
    transition = decay.unsqueeze(-1) * (parents.unsqueeze(-1)==positions).to(decay.dtype)
    paths = identity
    power = transition
    for _ in range((rows-1).bit_length()):
        paths = paths + power @ paths
        power = power @ power
    base = (paths @ (decay*(parents<0)).unsqueeze(-1)).squeeze(-1)
    return paths,base,identity


def corrections(state,k,v,decay,beta,parents):
    keys = k.transpose(-1,-2)
    values = v.transpose(-1,-2)
    decay = decay.transpose(1,2)
    beta = beta.transpose(1,2)
    paths,base,identity = path_weights(decay,parents)
    lower = (keys @ k) * paths * beta.unsqueeze(-1) * (1-identity)
    updates = beta.unsqueeze(-1) * (values-base.unsqueeze(-1)*(keys @ state))
    power = -lower
    for _ in range((keys.shape[-2]-1).bit_length()):
        updates = updates + power @ updates
        power = power @ power
    return updates,paths,base


def parallel_commit(state,k,v,decay,beta):
    parents = torch.arange(k.shape[-1],device=k.device)-1
    updates,paths,base = corrections(state,k,v,decay,beta,parents)
    return state*base[..., -1,None,None] + (k*paths[..., -1,None,:]) @ updates


def parallel_tree(state,q,k,v,decay,beta,program):
    rows = program & 15
    committed = (program & 32) != 0
    positions = torch.arange(program.shape[0],device=program.device)
    previous = torch.where((positions.unsqueeze(0)<positions.unsqueeze(1)) & committed.unsqueeze(0),positions,-1).amax(-1)
    parent_order = torch.where(previous>=0,rows.index_select(0,previous.clamp(min=0)),-1)
    parents = torch.empty_like(rows).scatter(0,rows,parent_order)
    updates,paths,base = corrections(state,k,v,decay,beta,parents)
    queries = q.transpose(-1,-2)
    output = base.unsqueeze(-1)*(queries @ state) + ((queries @ k)*paths) @ updates
    return output.transpose(1,2)
