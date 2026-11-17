"""Complete token-graph experiment with request-local device RNG seeds."""
import argparse
import copy
import gc
import faulthandler
import inspect
import json
from pathlib import Path
import time
import textwrap

import torch
import habana_frameworks.torch.core as htcore
from comfy.cli_args import args as comfy_args
comfy_args.disable_dynamic_vram = True
comfy_args.disable_pinned_memory = True
comfy_args.disable_async_offload = True
comfy_args.use_pytorch_cross_attention = True
import comfy.sd
from comfy.text_encoders import llama, qwen35


def token_block(owner, token, rotary_table, caches, penalty, position, seed, teacher, offset, count, generator=None,
                functional=False):
    if functional:
        original_caches = caches
        caches = [copy.copy(cache) for cache in caches]
        original_token, original_penalty, original_position, original_offset = token, penalty, position, offset
    results = []
    scores = []
    for _ in range(count):
        attention_cache = next(cache for cache in caches if type(cache) is qwen35.HPUSingleTokenKV)
        if functional:
            for cache in caches:
                if type(cache) is qwen35.HPUSingleTokenKV:
                    cache.position = position.expand_as(attention_cache.position)
                    cache.seqlen = (position + 1).int()
        else:
            attention_cache.position.copy_(position.expand_as(attention_cache.position))
            attention_cache.seqlen.copy_((position + 1).int())
        rotary = rotary_table.reshape(rotary_table.shape[2], -1).index_select(0, position.int())
        rotary = rotary.reshape(1, 1, 1, rotary_table.shape[-3], 2, 2)
        hidden = owner.model.embed_tokens(token).to(torch.bfloat16).clone()
        hidden = qwen35._hpu_decode(owner.model, hidden, rotary, caches)
        logits = owner.model.lm_head(hidden)[:, -1]
        scores.append(logits)
        probs, indices = owner._processed_probs(logits, 1., 20, .95, 0., 1., [], 1.5, penalty)
        if generator is None:
            choice = torch.ops.hpu.habana_multinomial(seed, probs, 1, False)
        else:
            choice = torch.multinomial(probs, 1, generator=generator)
        sampled = indices.int().gather(1, choice).long()
        actual_sample = sampled
        if teacher is not None:
            sampled = teacher.index_select(0, offset.int()).reshape(1, 1).long()
        if functional:
            token = sampled
            penalty = penalty.index_fill(0, token[0], 1)
            position = position + 1
            offset = offset + 1
        else:
            token.copy_(sampled)
            penalty.index_fill_(0, token[0], 1)
            position.add_(1)
            offset.add_(1)
        if generator is None:
            seed.add_(1)
        # Expose the actual sample so teacher forcing cannot prune the head/sampler.
        results.append(torch.stack((token.clone(), actual_sample), dim=-1))
    if functional:
        for original, current in zip(original_caches, caches):
            original.key.copy_(current.key)
            original.value.copy_(current.value)
        original_token.copy_(token)
        original_penalty.copy_(penalty)
        original_position.copy_(position)
        original_offset.copy_(offset)
    return torch.cat(results, dim=1), torch.cat(scores, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--block', type=int, default=1)
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--natural', action='store_true')
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--vertical', action='store_true')
    parser.add_argument('--native-rope', action='store_true')
    parser.add_argument('--align', action='store_true')
    parser.add_argument('--logits', action='store_true')
    parser.add_argument('--functional-graph', action='store_true')
    options = parser.parse_args()
    requested_tokens = options.tokens
    if options.block >= 8 or options.align:
        options.tokens = 1 + ((options.tokens - 1 + options.block - 1) // options.block) * options.block
    options.output.mkdir(parents=True, exist_ok=True)
    if options.native_rope:
        from qwen80_pe_candidates import install
        install('native-rope')
    base = Path('/home/sonny/gaudi2/comfyui-hpu')
    graph = json.loads((base / 'workflows/qwen21-latest-2026-11-16-api.json').read_text())
    settings = graph['459:471']['inputs']
    teacher_ids = json.loads((base / 'performance-qwen80-pe-2026-11-16/full-quality/baseline/plain-tokens.json').read_text())
    torch.set_grad_enabled(False)
    torch.set_num_threads(32)
    faulthandler.dump_traceback_later(180, repeat=True)
    if not options.graph:
        from qwen80_partition import install
        install()
    vertical = options.vertical or options.block >= 8
    if vertical and not options.graph:
        from habana_frameworks.torch.dynamo.compile_backend.partitioner import HabanaPartitioner
        initialize = HabanaPartitioner.__init__

        def initialize_vertical(self, *values, **kwargs):
            initialize(self, *values, **kwargs)
            self.skip_horizontal_fusion = True

        HabanaPartitioner.__init__ = initialize_vertical
    if options.graph:
        comfy_args.disable_comfy_compiler = True
        original_rope = qwen35.apply_rope

        def graph_rope(query, key, frequencies):
            if query.shape[-2] != 1:
                return original_rope(query, key, frequencies)
            matrix = llama.rope_matrix(frequencies).reshape(1, -1, 4)
            cosine = torch.cat((matrix[..., 0], matrix[..., 3]), dim=-1).reshape(1, 1, 1, -1)
            sine = torch.cat((-matrix[..., 1], matrix[..., 2]), dim=-1).reshape(1, 1, 1, -1)
            query_out = torch.ops.hpu.rotary_pos_embedding(query.float(), sine, cosine, None, 0, 0)
            key_out = torch.ops.hpu.rotary_pos_embedding(key.float(), sine, cosine, None, 0, 0)
            return query_out.to(query.dtype), key_out.to(key.dtype)

        qwen35.apply_rope = graph_rope
        if options.functional_graph:
            source = textwrap.dedent(inspect.getsource(qwen35.GatedDeltaNet._decode_single))
            source = source.replace('cache.conv_state.copy_(window[..., 1:])', 'cache.key = window[..., 1:].clone()')
            source = source.replace('state.copy_(state * decay.unsqueeze(-1) + key.unsqueeze(-1) * update.unsqueeze(-2))',
                                    'cache.value = state * decay.unsqueeze(-1) + key.unsqueeze(-1) * update.unsqueeze(-2)')
            namespace = dict(vars(qwen35))
            exec(compile(source, '<functional-gdn>', 'exec'), namespace)
            qwen35.GatedDeltaNet._decode_single = namespace['_decode_single']

            def write(cache, key, value):
                indices = cache.position[:key.shape[2]]
                cache.key = cache.key.index_copy(2, indices, key)
                cache.value = cache.value.index_copy(2, indices, value)

            qwen35.HPUSingleTokenKV.write = write
    else:
        from habana_frameworks.torch.dynamo.compile_backend import shared_layer
        shared_layer.hpu_supported_op_list.add('habana_multinomial')
        sampler = torch.compile(lambda seed, probs: torch.ops.hpu.habana_multinomial(seed, probs, 1, False),
                                backend='hpu_backend', fullgraph=True, dynamic=False,
                                options={'force_static_compile': True})
        smoke = sampler(torch.tensor([42], dtype=torch.int32, device='hpu'),
                        torch.full((1, 20), .05, dtype=torch.bfloat16, device='hpu')).cpu()
        assert smoke.shape == (1, 1) and 0 <= smoke.item() < 20
        print('request-local sampler smoke passed', flush=True)
    clip = comfy.sd.load_clip(['/home/sonny/ComfyUI-Models/models/text_encoders/' + graph['459:473']['inputs']['clip_name']],
                              clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
    prompts = clip.tokenize(settings['prompt'], system_prompt=graph['459:475']['inputs']['value'], min_length=1, thinking=settings['thinking'])
    clip.load_model(prompts)
    compiled = token_block if options.graph else torch.compile(
        token_block, backend='hpu_backend', fullgraph=True, dynamic=False,
        options={'force_static_compile': True})
    saved_logits = []
    stage_times = {}

    def generate(owner, embeds=None, *generation_args, **kwargs):
        generation_start = time.perf_counter()
        embeds = embeds.to(torch.bfloat16)
        if embeds.ndim == 2:
            embeds = embeds.unsqueeze(0)
        capacity = embeds.shape[1] + settings['max_length']
        caches = owner.init_kv_cache(1, capacity, embeds.device, embeds.dtype)
        hidden, _, _ = owner.model(None, embeds=embeds, attention_mask=None, past_key_values=caches,
                                   position_ids=kwargs.get('position_ids'))
        prefill_logits = owner.logits(hidden)[:, -1]
        if options.logits and run == options.runs:
            saved_logits.append(prefill_logits.detach().cpu())
        generator = torch.Generator(device=embeds.device).manual_seed(settings['sampling_mode.seed'])
        penalty = torch.zeros(owner.model.config.vocab_size, device=embeds.device, dtype=torch.bool)
        first = owner.sample_token(prefill_logits, 1., 20, .95, 0., 1., [], generator,
                                   presence_penalty=1.5, penalty_mask=penalty)
        teacher = None if options.natural else torch.tensor(teacher_ids, device=embeds.device, dtype=torch.int32)
        token = first if teacher is None else teacher[:1].reshape(1, 1).long().clone()
        penalty.index_fill_(0, token[0], 1)
        proxies = []
        for cache in caches:
            if type(cache) is qwen35.LinearKV:
                proxy = qwen35.ImmediateLinearKV(cache.key, cache.value, 1, None, None,
                                                cache.g_decay, cache.dt_bias, cache.norm_weight)
            else:
                proxy = qwen35.HPUSingleTokenKV(cache.key, cache.value, 1, cache.position, cache.seqlen, {'tree': None})
            proxies.append(proxy)
        positions = torch.arange(capacity, device=embeds.device).unsqueeze(0)
        rotary = llama.rope_matrix(owner.model.compute_freqs_cis(positions, embeds.device))
        position = torch.tensor([embeds.shape[1]], device=embeds.device, dtype=torch.long)
        seed = torch.tensor([settings['sampling_mode.seed'] % (2**31 - 1)], device=embeds.device, dtype=torch.int32)
        offset = torch.ones(1, device=embeds.device, dtype=torch.long)
        ids = token.reshape(-1).tolist()
        decode_start = time.perf_counter()
        stage_times['prefill_and_setup_seconds'] = decode_start - generation_start
        graphs = {}
        for start in range(1, options.tokens, options.block):
            comfy.model_management.throw_exception_if_processing_interrupted()
            count = min(options.block, options.tokens - start)
            if options.graph:
                if count not in graphs:
                    htcore.mark_step()
                    torch.hpu.synchronize()
                    graph = torch.hpu.HPUGraph()
                    graph.capture_begin()
                    result, logits = compiled(owner, token, rotary, proxies, penalty, position, seed, teacher, offset, count,
                                              generator, options.functional_graph)
                    graph.capture_end()
                    htcore.mark_step()
                    torch.hpu.synchronize()
                    graphs[count] = graph, result, logits
                else:
                    graph, result, logits = graphs[count]
                    graph.replay(asynchronous=True)
                received = result.reshape(-1, 2)[:, 0].tolist()
            else:
                result, logits = compiled(owner, token, rotary, proxies, penalty, position, seed, teacher, offset, count)
                received = result.reshape(-1, 2)[:, 0].tolist()
            if options.logits and run == options.runs:
                saved_logits.append(logits.detach().cpu())
            for value in received:
                ids.append(value)
                if options.natural and value in owner.model.config.stop_tokens:
                    return ids
        return ids

    qwen35.Qwen35.generate = generate
    for run in range(options.runs + 1):
        gc.collect()
        torch.hpu.synchronize()
        start = time.perf_counter()
        ids = clip.generate(prompts, max_length=settings['max_length'], seed=settings['sampling_mode.seed'])
        torch.hpu.synchronize()
        record = dict(run=run, seconds=time.perf_counter() - start, tokens=len(ids), block=options.block,
                      mode='warm' if run == 0 else ('logits' if options.logits and run == options.runs else 'plain'),
                      vertical_partitioning=vertical,
                      bitset_partitioning=not options.graph,
                      requested_tokens=requested_tokens,
                      teacher_match=None if options.natural else ids == teacher_ids[:len(ids)])
        record.update(stage_times)
        print(record, flush=True)
        if run == 0:
            faulthandler.cancel_dump_traceback_later()
        with (options.output / 'timings.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        (options.output / f'tokens-{run}.json').write_text(json.dumps(ids))
    if saved_logits:
        torch.save(torch.cat(saved_logits), options.output / 'logits.pt')


if __name__ == '__main__':
    main()
