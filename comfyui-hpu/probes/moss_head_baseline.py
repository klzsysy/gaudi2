"""Select the previous all-head operation through the same public dispatcher."""
from comfy.ldm.moss_image import model as moss
product_head = moss._attention_heads


def together(q,k,v,heads,comfy_attention,transformer_options):
    tensors = (moss.AttentionTensorContainer(t) for t in (q,k,v))
    return moss.optimized_attention(*tensors,heads,transformer_options=transformer_options,preferred_attention=comfy_attention)


def install():
    moss._attention_heads = together
