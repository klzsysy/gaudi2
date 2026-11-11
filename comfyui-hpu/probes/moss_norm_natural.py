"""Natural ending prompt set; variable token counts are not matched-work timing."""
import json
import os
from pathlib import Path
import runpy
import time
import torch
from comfy.cli_args import args
args.disable_dynamic_vram = True
args.disable_pinned_memory = True
args.disable_async_offload = True
args.use_pytorch_cross_attention = True
from comfy.text_encoders import qwen35
if os.environ.get('MOSS_NORM_EAGER') == '1':
    qwen35.RMSNorm.forward = qwen35.LlamaRMSNorm.forward
state = runpy.run_path(str(Path(__file__).with_name('moss_gdn_experiment.py')),run_name='__main__')
clip = state['clip']
root = state['directory']
prompts = [
 ('short','A red ceramic teapot on a wooden table beside a sunlit window, product photography.'),
 ('poster','A clean poster featuring (a blue ceramic vase:1.2), three yellow flowers, a white background, and large readable text "MOSS" at the top.'),
 ('person','Portrait photograph of an elderly woman wearing a red scarf, detailed face and hands, holding a small white teacup, natural window light.'),
 ('composition','A wide photograph of a busy outdoor market: a green bicycle in the left foreground, a vendor arranging oranges at the center, two shoppers under a blue umbrella on the right, brick storefronts in the background.'),
 ('long','An atmospheric architectural photograph of a quiet seaside bookshop at dusk. Through the large front window, wooden shelves hold colorful books, a reading lamp illuminates a round table, and an orange cat sleeps on a dark green armchair. Outside, rainwater on the pavement reflects the warm interior and a soft purple sky. The view is carefully composed with the doorway on the left and a parked bicycle on the right, realistic materials, distinct foreground and background, soft cinematic light.'),
]
records=[]
for name,prompt in prompts:
    tokens=clip.tokenize(prompt,system_prompt=state['workflow']['18:14']['inputs']['value'],min_length=1,thinking=False)
    for repeat in range(2):
        torch.hpu.synchronize();start=time.perf_counter()
        ids=clip.generate(tokens,max_length=4096,do_sample=True,temperature=.7,top_k=64,top_p=.95,min_p=.05,seed=1,mtp=True)
        torch.hpu.synchronize()
        record=dict(name=name,repeat=repeat,seconds=time.perf_counter()-start,tokens=len(ids),natural_end=len(ids)<4096,ids=ids,text=clip.decode(ids))
        assert record['natural_end']
        records.append(record)
        (root/'natural.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))
        print(name,repeat,record['seconds'],len(ids),flush=True)
