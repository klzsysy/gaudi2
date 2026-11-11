"""Fixed teacher trajectory, switching only GDN prefill compilation."""
import os
from pathlib import Path
import runpy

from comfy.text_encoders import hpu_gated_delta

if os.environ.get('MOSS_PREFILL_BASELINE') == '1':
    hpu_gated_delta.compiled_prefill = lambda function: function
runpy.run_path(str(Path(__file__).with_name('moss_gdn_experiment.py')), run_name='__main__')
