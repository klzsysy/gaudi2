"""Replay the paired text experiment under ComfyUI's inference-mode context."""
from pathlib import Path
import runpy

import torch


with torch.inference_mode():
    runpy.run_path(str(Path(__file__).with_name('moss_text_rope_paired.py')), run_name='__main__')
