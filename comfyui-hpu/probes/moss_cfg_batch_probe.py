"""Investigation-only left padding to measure CFG batching; no production policy."""
import os
import torch
import torch.nn.functional as F
import comfy.conds
import comfy.model_base


class PaddedContext(comfy.conds.CONDRegular):
    def can_concat(self, other):
        return self.cond.shape[0] == other.cond.shape[0] and self.cond.shape[2:] == other.cond.shape[2:] and self.cond.device == other.cond.device

    def concat(self, others):
        values = [self.cond] + [x.cond for x in others]
        length = max(x.shape[1] for x in values)
        return torch.cat([F.pad(x, (0, 0, length - x.shape[1], 0)) for x in values])


class PaddedBias(comfy.conds.CONDRegular):
    def can_concat(self, other):
        return self.cond.shape[0] == other.cond.shape[0] and self.cond.device == other.cond.device

    def concat(self, others):
        values = [self.cond] + [x.cond for x in others]
        length = max(x.shape[1] for x in values)
        return torch.cat([F.pad(x, (length - x.shape[1], 0), value=float('-inf')) for x in values])


class PreparedRoPE(comfy.conds.CONDConstant):
    def process_cond(self, batch_size, **kwargs):
        value = dict(self.cond)
        value['pe'] = value['pe'].expand(batch_size, *value['pe'].shape[1:])
        return self._copy_with(value)

    def can_concat(self, other):
        return self.cond['shape'] == other.cond['shape'] and self.cond['rope_options'] == other.cond['rope_options']

    def concat(self, others):
        if not others:
            return self.cond
        values = [self.cond] + [x.cond for x in others]
        length = max(x['text_len'] for x in values)
        spatial = self.cond['shape'][0] * self.cond['shape'][1]
        positions = []
        for item in values:
            pe = item['pe']
            padding = pe.new_zeros((pe.shape[0], length - item['text_len'], *pe.shape[2:]))
            positions.append(torch.cat((pe[:, :spatial], padding, pe[:, spatial:]), dim=1))
        return dict(self.cond, pe=torch.cat(positions), text_len=length)


def install():
    original = comfy.model_base.MOSSImage.extra_conds
    def extra_conds(model, **kwargs):
        # This prototype is only for the plain text-to-image fixture.
        assert not kwargs.get('reference_latents') and kwargs.get('hooks') is None
        prepared = os.environ.get('MOSS_CFG_PREPARED') == '1'
        out = original(model, **kwargs) if prepared else original(model, **dict(kwargs, hooks=object()))
        context = kwargs['cross_attn']
        out['c_crossattn'] = PaddedContext(context)
        bias = kwargs.get('context_attention_bias')
        if bias is None:
            bias = torch.zeros(context.shape[:2], device=context.device, dtype=context.dtype)
        out['context_attention_bias'] = PaddedBias(bias)
        if prepared:
            assert 'prepared_context' in out
            out['prepared_context'] = PaddedContext(out['prepared_context'].cond)
            out['prepared_rope'] = PreparedRoPE(out['prepared_rope'].cond)
        return out
    comfy.model_base.MOSSImage.extra_conds = extra_conds
