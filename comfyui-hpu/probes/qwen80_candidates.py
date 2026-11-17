"""Process-local experiments for the current Qwen FP8 generation loop."""
import inspect
import textwrap

import torch
from comfy.text_encoders import hpu_kv_attention, llama, qwen35


def install(mode):
    if mode.startswith('pe:'):
        from qwen80_pe_candidates import install as install_pe
        return install_pe(mode.removeprefix('pe:'))
    if mode.startswith('rope-'):
        original = qwen35.apply_rope

        def rope(query, key, frequencies):
            if (query.device.type == 'hpu' and query.dtype == key.dtype == torch.bfloat16
                    and query.ndim == key.ndim == 4 and query.shape[-2:] == key.shape[-2:] == (1, 64)
                    and torch.compiler.is_compiling()):
                matrix = llama.rope_matrix(frequencies)
                if matrix.ndim == 5:
                    matrix = matrix.unsqueeze(0)
                if mode == 'rope-native':
                    cosine = torch.cat((matrix[..., 0, 0], matrix[..., 1, 1]), dim=-1)
                    sine = torch.cat((matrix[..., 1, 0], -matrix[..., 0, 1]), dim=-1)
                    query = torch.ops.hpu.rotary_pos_embedding(query.float(), sine, cosine, None, 0, 0).to(query.dtype)
                    key = torch.ops.hpu.rotary_pos_embedding(key.float(), sine, cosine, None, 0, 0).to(key.dtype)
                    return query, key
                return hpu_kv_attention._rope_forward(query, key, matrix)
            return original(query, key, frequencies)

        qwen35.apply_rope = rope
        if mode == 'rope-inline-none':
            compiled = torch.compile(qwen35._hpu_decode, backend='hpu_backend', fullgraph=True, dynamic=False,
                                     options={'force_static_compile': True, 'decomposition_mode': 'none'})
            qwen35._compiled_hpu_decode = lambda: compiled
        return
    if not mode.startswith('defer'):
        raise ValueError(mode)
    chunk = int(mode.removeprefix('defer'))
    assert chunk in (2, 4, 8, 16)
    source = textwrap.dedent(inspect.getsource(llama.BaseGenerate.generate))
    source = source.replace('generated_token_ids = []', 'generated_token_ids = []\n    pending_tokens = []')
    original = '''            token_id = decode_tokens[0].item()
            generated_token_ids.append(token_id)'''
    assert original in source
    source = source.replace(original, '            pending_tokens.append(decode_tokens[0].clone())')
    original = '''            pbar.update(1)

            if token_id in stop_tokens:
                break'''
    assert original in source
    replacement = f'''            if len(pending_tokens) == {chunk} or step + 1 == max_length:
                received = torch.stack(pending_tokens).reshape(-1).tolist()
                pending_tokens.clear()
                completed = False
                consumed = 0
                for token_id in received:
                    generated_token_ids.append(token_id)
                    consumed += 1
                    if token_id in stop_tokens:
                        completed = True
                        break
                pbar.update(consumed)
                if completed:
                    break'''
    source = source.replace(original, replacement)
    namespace = dict(vars(llama))
    exec(compile(source, '<qwen80-buffered-generation>', 'exec'), namespace)
    llama.BaseGenerate.generate = namespace['generate']
