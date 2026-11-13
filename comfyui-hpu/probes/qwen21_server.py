"""Qwen investigation launcher using the existing workflow measurement hooks."""
import os
from pathlib import Path
import runpy
import comfy.options
comfy.options.enable_args_parsing()
if os.environ.get('QWEN21_CANDIDATE') == '1':
    from qwen21_candidate import install
    install()
if os.environ.get('QWEN21_COMPARE'):
    from qwen21_quality import install
    install()
runpy.run_path(str(Path(__file__).with_name('moss_observed_server.py')),run_name='__main__')
