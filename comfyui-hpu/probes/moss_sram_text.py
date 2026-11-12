from pathlib import Path
import runpy

import moss_sram_candidate

runpy.run_path(str(Path(__file__).with_name('moss_evidence_text.py')), run_name='__main__')
