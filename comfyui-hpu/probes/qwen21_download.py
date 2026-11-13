"""Download the requested prompt enhancer using the existing server HF settings."""
import hashlib
import json
import os
from pathlib import Path
import time

from huggingface_hub import hf_hub_download

started = time.monotonic()
print(json.dumps({'endpoint_configured': bool(os.environ.get('HF_ENDPOINT')), 'token_configured': bool(os.environ.get('HF_TOKEN'))}), flush=True)
path = hf_hub_download(
    repo_id='Comfy-Org/Qwen-Image-2.1',
    filename='text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors',
    local_dir='/home/sonny/ComfyUI-Models/models',
    token=os.environ.get('HF_TOKEN'),
)
hash_value = hashlib.file_digest(open(path, 'rb'), 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
if hash_value is None:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    hash_value = digest.hexdigest()
assert hash_value == '9182abae56fe05459840a86d22abd21f972061c92fce032630af680c8c5178d3'
print(json.dumps({'file': path, 'bytes': Path(path).stat().st_size, 'sha256': hash_value, 'seconds': time.monotonic() - started}), flush=True)
