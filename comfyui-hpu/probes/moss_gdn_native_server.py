from pathlib import Path
import os
import runpy

import comfy.options
comfy.options.enable_args_parsing()
import native_gdn.install

runpy.run_path(str(Path(__file__).with_name(os.environ.get('MOSS_GDN_SERVER_BASE', 'moss_fp8_deep_server.py'))), run_name='__main__')
