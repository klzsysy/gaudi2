"""Compare actual Qwen VAE layers on CPU and HPU, using the same small latent."""
import sys
import torch, comfy.sd, comfy.utils
from safetensors.torch import load_file

dtype = torch.float32 if "--fp32" in sys.argv else torch.bfloat16
torch.set_grad_enabled(False)
torch.set_num_threads(16)
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file('models/vae/qwen_image_2.1_vae_bf16.safetensors'), device=torch.device('cpu'), dtype=dtype)
model = vae.first_stage_model.to(dtype=dtype)
x = load_file('output/gaudi/qwen21-cache-off_00001_.latent')['latent_tensor'].to(dtype)
if '--full' not in sys.argv:
    x=x[:, :, :2, :2]
refs = {}
counts = {}
def hook(name):
    def check(m,i,o):
        if not isinstance(o,torch.Tensor): return
        key=(name,counts.get(name,0)); counts[name]=key[1]+1
        val=o.detach().flatten()[::max(1,o.numel()//8192)].float().cpu().clone()
        if o.device.type=='cpu': refs[key]=val
        else:
            ref=refs[key]
            err=((val-ref).square().mean()/ref.square().mean().clamp_min(1e-10)).sqrt().item()
            print(name,list(o.shape),'err',round(err,5),'rms',round(val.square().mean().sqrt().item(),5),flush=True)
    return check
for name,module in model.named_modules():
    if '--full' not in sys.argv or name in ['conv2','decoder.conv1','decoder.middle.0','decoder.middle.1','decoder.middle.2','decoder.upsamples.0','decoder.upsamples.1','decoder.upsamples.2','decoder.upsamples.3','decoder.upsamples.4','decoder.head.0','decoder.head.1','decoder.head.2','decoder']:
        if name.startswith('decoder') or name=='conv2': module.register_forward_hook(hook(name))
a=model.decode(x)
print('CPU done',flush=True)
counts.clear()
model.to('hpu')
b=model.decode(x.to('hpu'))
print('DONE',flush=True)
