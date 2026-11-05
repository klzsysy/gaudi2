"""Diagnostic only: run from the independent ComfyUI checkout, with no HPU server running."""
import sys
import torch
import comfy.sd
import comfy.utils
from PIL import Image
from safetensors.torch import load_file
import numpy as np

torch.set_grad_enabled(False)
torch.set_num_threads(16)
if sys.argv[1] == 'vae':
    samples = load_file('output/gaudi/qwen21-cache-off_00001_.latent')['latent_tensor']
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file('models/vae/qwen_image_2.1_vae_bf16.safetensors'), device=torch.device('cpu'), dtype=torch.float32)
    pixels = vae.decode(samples).detach().cpu().numpy()[0]
    print('pixels', pixels.shape, np.isfinite(pixels).all(), pixels.std(), flush=True)
    Image.fromarray(np.clip(pixels * 255, 0, 255).astype(np.uint8)).save('test-results/qwen-cpu-vae.png')
else:
    patcher = comfy.sd.load_diffusion_model('models/diffusion_models/qwen_image_2.1_turbo_bf16.safetensors', model_options={'load_device':torch.device('cpu'), 'dtype':torch.bfloat16})
    model = patcher.model.diffusion_model
    torch.manual_seed(42)
    x = torch.randn(1,64,4,4,dtype=torch.bfloat16)
    context = torch.randn(1,16,4096,dtype=torch.bfloat16)
    t = torch.tensor([0.9],dtype=torch.bfloat16)
    references = {}
    def hook(name):
        def check(m, inputs, out):
            if not isinstance(out,torch.Tensor): return
            value=out.detach().float().cpu().clone()
            if out.device.type=='cpu':
                references[name]=value
            else:
                ref=references[name]
                print(name, 'rms',value.square().mean().sqrt().item(), 'relative_error',((value-ref).square().mean()/ref.square().mean().clamp_min(1e-10)).sqrt().item(),flush=True)
        return check
    for name,module in model.named_modules():
        if name in ['txt_in','img_in','time_text_embed','modulation','norm_out','proj_out'] or name.startswith('transformer_blocks.0.') or name in ['transformer_blocks.0','transformer_blocks.1','transformer_blocks.31']:
            module.register_forward_hook(hook(name))
    opts={'qwen_image21_cache':{'device':'off'}}
    ref=model(x,t,context,transformer_options=opts)
    print('CPU complete',ref.float().square().mean().sqrt().item(),flush=True)
    model.to('hpu')
    out=model(x.to('hpu'),t.to('hpu'),context.to('hpu'),transformer_options=opts)
    print('HPU complete',out.float().cpu().square().mean().sqrt().item(),flush=True)
