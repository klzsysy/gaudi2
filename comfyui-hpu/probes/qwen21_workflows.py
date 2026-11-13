"""Build the first Turbo workflow from the two official T2I templates."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = json.loads((root / 'workflows/qwen21-official-t2i.json').read_text())
turbo = json.loads((root / 'workflows/qwen21-official-turbo-t2i.json').read_text())
base_outer = next(n for n in base['nodes'] if n['id'] == 459)
outer = next(n for n in turbo['nodes'] if n['id'] == 459)
outer['widgets_values'][0] = base_outer['widgets_values'][0]
outer['widgets_values'][2] = True
outer['widgets_values'][8] = 43
outer['widgets_values'][9] = 'qwen_image_2.1_turbo_bf16.safetensors'
original_enhancer = outer['widgets_values'][12]
outer['widgets_values'][12] = 'qwen3.5_9b_qwen_image_2.1_pe_t2i.bf16_from_int8.safetensors'
graph = turbo['definitions']['subgraphs'][0]
for node in graph['nodes']:
    if node['id'] == 451:
        node['widgets_values'][0] = outer['widgets_values'][9]
        node['properties']['models'][0]['name'] = outer['widgets_values'][9]
        node['properties']['models'][0]['url'] = 'https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/' + outer['widgets_values'][9]
    if node['id'] == 473:
        node['widgets_values'][0] = outer['widgets_values'][12]
        node['properties'].pop('models', None)
    if node['id'] == 488:
        node['widgets_values'] = [43, 'fixed']
next(n for n in turbo['nodes'] if n['id'] == 483)['widgets_values'][0] += '\n\nHPU preset uses a local BF16 decode of the matching INT8 prompt enhancer. Enable refine_prompt to rewrite the prompt. Thinking mode is enabled, and the original INT8 file remains available in the PE_model selector.'
next(n for n in turbo['nodes'] if n['id'] == 461)['widgets_values'][0] = 'gaudi/qwen21_turbo'
(root / 'workflows/qwen21-turbo-hpu.json').write_text(json.dumps(turbo, ensure_ascii=False, indent=2) + '\n')

prompt = outer['widgets_values'][0]
system = next(n for n in graph['nodes'] if n['id'] == 475)['widgets_values'][0]
api = {}
def node(key, kind, **inputs):
    api[str(key)] = {'class_type': kind, 'inputs': inputs}
node(13, 'ResolutionSelector', aspect_ratio='1:1 (Square)', megapixels=1.0, multiple=8)
node(451, 'UNETLoader', unet_name=outer['widgets_values'][9], weight_dtype='default')
node(453, 'CLIPLoader', clip_name='qwen3vl_8b_int8_convrot.safetensors', type='qwen_image', device='default')
node(454, 'VAELoader', vae_name='qwen_image_2.1_vae_bf16.safetensors')
node(473, 'CLIPLoader', clip_name=outer['widgets_values'][12], type='stable_diffusion', device='default')
node(475, 'PrimitiveStringMultiline', value=system)
node(471, 'TextGenerate', clip=['473',0], prompt=prompt, max_length=4096, sampling_mode='on', **{
    'sampling_mode.temperature':1.0, 'sampling_mode.top_k':20, 'sampling_mode.top_p':0.95,
    'sampling_mode.min_p':0.0, 'sampling_mode.repetition_penalty':1.0, 'sampling_mode.seed':43,
    'sampling_mode.presence_penalty':1.5}, thinking=True, use_default_template=True, mtp='auto', system_prompt=['475',0])
node(474, 'ComfySwitchNode', switch=False, on_false=prompt, on_true=['471',0])
node(472, 'PreviewAny', source=['474',0])
node(452, 'TextEncodeQwenImage21', clip=['453',0], prompt=['472',0], negative_prompt='', resolution=1024)
node(456, 'EmptyLatentImage', width=['13',0], height=['13',1], batch_size=1)
node(490, 'QwenImage21Cache', model=['451',0], device='auto', dtype='default')
node(489, 'CFGGuider', model=['490',0], positive=['452',0], negative=['452',1], cfg=1.0)
node(488, 'RandomNoise', noise_seed=43)
node(487, 'KSamplerSelect', sampler_name='euler')
node(484, 'ManualSigmas', sigmas=next(n for n in graph['nodes'] if n['id']==484)['widgets_values'][0])
node(486, 'SamplerCustomAdvanced', noise=['488',0], guider=['489',0], sampler=['487',0], sigmas=['484',0], latent_image=['456',0])
node(457, 'VAEDecode', samples=['486',0], vae=['454',0])
node(461, 'SaveImageAdvanced', images=['457',0], filename_prefix='gaudi/qwen21_turbo', format='png', **{'format.bit_depth':'8-bit','format.input_color_space':'sRGB'})
for name, enhanced in [('direct',False),('enhanced',True)]:
    api['474']['inputs']['switch'] = enhanced
    (root / f'workflows/qwen21-turbo-{name}-api.json').write_text(json.dumps(api, ensure_ascii=False, indent=2) + '\n')

api['473']['inputs']['clip_name'] = original_enhancer
(root / 'workflows/qwen21-turbo-enhanced-int8-api.json').write_text(json.dumps(api, ensure_ascii=False, indent=2) + '\n')
