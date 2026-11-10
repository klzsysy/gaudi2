"""Record preparation counts without module hooks or device synchronization."""
import functools

import comfy.model_base
from comfy.ldm.moss_image import model as moss


def install(event):
    for owner, method, label in ((moss.MOSSImage, 'rope', 'rope'), (moss.TextEmbedder, 'forward', 'text_embedder')):
        original = getattr(owner, method)
        def wrap(function, label):
            @functools.wraps(function)
            def measured(*values, **kwargs):
                event('preparation_call', region=label)
                return function(*values, **kwargs)
            return measured
        setattr(owner, method, wrap(original, label))
    original = comfy.model_base.MOSSImage.extra_conds
    @functools.wraps(original)
    def extra_conds(model, **kwargs):
        output = original(model, **kwargs)
        options = model.current_patcher.model_options if model.current_patcher is not None else {}
        event('preparation_result', prepared='prepared_context' in output,
              model_option_keys=list(options), transformer_option_keys=list(options.get('transformer_options', {})),
              has_hooks=kwargs.get('hooks') is not None, area=kwargs.get('area'), latent_shapes=kwargs.get('latent_shapes'))
        return output
    comfy.model_base.MOSSImage.extra_conds = extra_conds
