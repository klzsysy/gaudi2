"""Expand current-product quality inputs and capture three layers/timesteps."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.parse
import urllib.request


root=Path(os.environ['MOSS_EVIDENCE_ROOT'])/'quality'
root.mkdir(parents=True,exist_ok=True)
source=json.loads((root.parent/'dit-six-retry/source.json').read_text())
url='http://127.0.0.1:8190'
prompts=[
    ('short','A single green pear on a plain gray background.'),
    ('count_text','A precise studio photograph of exactly four objects arranged left to right on a white shelf: a red cube, a blue sphere, a yellow cone, and a green cylinder. Above them, large black uppercase lettering reads FOUR SHAPES. All four objects are separated and fully visible. Frontal view, clear outlines, neutral light.'),
    ('texture','Macro photograph of a pale gray knitted wool scarf folded beside translucent white lace and a brushed silver spoon, soft diffuse overcast light, subtle low-contrast texture, fine individual fibers, realistic materials, shallow depth of field, restrained highlights.'),
    ('long','A wide cinematic photograph of a quiet bookshop at dusk. In the foreground on the left, an elderly bookseller wearing a dark green cardigan carefully wraps a small blue book in brown paper. Both hands are visible on the counter. A brass desk lamp casts warm light onto the paper and the wood grain. A sleeping orange cat lies beside a stack of three books. Behind the counter, tall shelves of worn books recede toward a rain-streaked window. Outside the window, bicycles stand beneath a plane tree and reflections shimmer on wet paving stones. A young woman wearing a red scarf sits by the window reading a cream-colored book. Her face is shown in profile. A white ceramic cup and a clear glass of water sit on her round table. Over the entrance, a sign has large legible letters reading NIGHT BOOKS. The interior combines amber lamplight with cool blue twilight from outside. Preserve natural facial features, realistic fingers, fabric folds, subtle dust in the light, detailed book bindings, straight shelves, believable reflections, and coherent perspective. Keep the bookseller, reader, cat, window, and entrance visually distinct. Use a restrained photographic color palette and a natural camera perspective with foreground and background depth. The composition should feel peaceful, lived-in, and documentary, with no decorative border.')]


def api(route,data=None):
    request=urllib.request.Request(url+route,data=None if data is None else json.dumps(data).encode(),
                                   headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=120) as response:
        return json.load(response)


env=dict(os.environ,PT_HPU_LAZY_MODE='0',GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so',
         PYTHONPATH=str(Path.cwd()),MOSS_PIPELINE_DIR=str(root),MOSS_EVENTS=str(root/'events.jsonl'),
         MOSS_DEEP_CAPTURE='1',MOSS_DEEP_CAPTURE_ALL='1',MOSS_DEEP_PROFILE='0')
command=[sys.executable,str(Path(__file__).with_name('moss_fp8_deep_server.py')),'--disable-dynamic-vram',
         '--disable-pinned-memory','--disable-async-offload','--disable-xformers','--use-pytorch-cross-attention',
         '--listen','127.0.0.1','--port','8190']
with (root/'server.log').open('w') as log:
    server=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            if server.poll() is not None:
                raise RuntimeError('Quality server exited')
            try:
                api('/system_stats')
                break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError('Quality server startup')
        records=[]
        for index,(label,prompt) in enumerate(prompts):
            graph=copy.deepcopy(source)
            graph['18:4']['inputs']['text']=prompt
            graph['18:7']['inputs']['seed']+=index+101
            (root/f'{label}-source.json').write_text(json.dumps(graph,indent=2))
            start=time.monotonic()
            prompt_id=api('/prompt',dict(prompt=graph,extra_data=dict(moss_diagnostic=True,
                          moss_recompute_nodes=['CLIPTextEncode','KSampler','VAEDecode'])))['prompt_id']
            while prompt_id not in (history:=api('/history/'+prompt_id)):
                if server.poll() is not None:
                    raise RuntimeError('Quality server exited during request')
                if time.monotonic()-start>1200:
                    raise TimeoutError(prompt_id)
                time.sleep(.3)
            history=history[prompt_id]
            (root/f'{label}-history.json').write_text(json.dumps(history,indent=2))
            assert history['status']['status_str']=='success',history['status']
            for output in history['outputs'].values():
                for item in output.get('images',[]):
                    with urllib.request.urlopen(url+'/view?'+urllib.parse.urlencode(item)) as response:
                        (root/f'{label}.png').write_bytes(response.read())
            events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
            events=[e for e in events if e.get('prompt_id')==prompt_id]
            checks=[e for e in events if e['kind']=='tensor_check']
            assert len(checks)>=2 and all(e['finite'] for e in checks)
            assert not next(e for e in events if e['kind']=='request_end')['fallback_counts']
            records.append(dict(label=label,prompt_id=prompt_id,seconds=time.monotonic()-start,checks=checks))
            (root/'results.json').write_text(json.dumps(records,indent=2))
            print(label,'complete',flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
