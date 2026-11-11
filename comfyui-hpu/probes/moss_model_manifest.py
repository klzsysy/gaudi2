"""Read-only checkpoint and deployed-source hashes for this MOSS benchmark."""
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ['MOSS_BF16_ROOT'])
checkout = Path.cwd()
workflow = json.loads((root / 'baseline/source.json').read_text())
models = {}
for node, field, folders in [('18:1', 'unet_name', ('diffusion_models', 'unet')),
                             ('18:2', 'clip_name', ('text_encoders', 'clip')),
                             ('18:3', 'vae_name', ('vae',))]:
    name = workflow[node]['inputs'][field]
    path = next(checkout / 'models' / folder / name for folder in folders if (checkout / 'models' / folder / name).exists()).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    stat = path.stat()
    models[field] = dict(name=name, path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest.hexdigest())
    print(name, digest.hexdigest(), flush=True)
files = ['comfy/ldm/moss_image/model.py', 'comfy/ldm/moss_image/hpu.py', 'comfy/text_encoders/qwen35.py',
         'comfy/text_encoders/hpu_gated_delta.py', 'comfy/model_base.py', 'comfy/ops.py']
manifest = dict(models=models, product=(checkout / '.run/deployed-version').read_text().strip(),
                sources={name: hashlib.sha256((checkout/name).read_bytes()).hexdigest() for name in files},
                workflow_sha256=hashlib.sha256((root/'baseline/source.json').read_bytes()).hexdigest(),
                torch=json.loads((root/'teacher-baseline/metadata.json').read_text())['torch'])
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
