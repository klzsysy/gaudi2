"""Paired quality contact sheets and pixel/repeat measurements."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

root = Path(sys.argv[1])
names = [name for name in ['short','poster','person','composition','long','street']
         if (root/f'{name}-baseline.png').exists() and (root/f'{name}-candidate.png').exists()]
metrics = {}
for page in range((len(names)+2)//3):
    subset = names[page*3:page*3+3]
    canvas = Image.new('RGB',(1024,len(subset)*540),'white')
    draw = ImageDraw.Draw(canvas)
    for row,name in enumerate(subset):
        a = np.asarray(Image.open(root/f'{name}-baseline.png').convert('RGB')).astype(float)
        b = np.asarray(Image.open(root/f'{name}-candidate.png').convert('RGB')).astype(float)
        values = dict(mae=float(np.abs(a-b).mean()),maximum=float(np.abs(a-b).max()))
        repeat = root/f'{name}-baseline-repeat.png'
        if repeat.exists():
            c = np.asarray(Image.open(repeat).convert('RGB')).astype(float)
            values['baseline_repeat_mae'] = float(np.abs(a-c).mean())
        metrics[name] = values
        for col,variant in enumerate(['baseline','candidate']):
            image = Image.open(root/f'{name}-{variant}.png').convert('RGB')
            image.thumbnail((512,512))
            canvas.paste(image,(col*512,row*540+28))
            draw.text((col*512+8,row*540+8),name+' '+variant,fill='black')
    canvas.save(root/f'contact-{page}.jpg')
(root/'pixel-comparison.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
