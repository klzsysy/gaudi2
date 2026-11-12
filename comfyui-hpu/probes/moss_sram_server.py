from pathlib import Path
import runpy

import comfy.options
comfy.options.enable_args_parsing()
import moss_sram_candidate

runpy.run_path(str(Path(__file__).with_name('moss_fp8_deep_server.py')), run_name='__main__')
