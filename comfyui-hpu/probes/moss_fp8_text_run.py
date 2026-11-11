"""Standalone text benchmark with the native FP8 candidate."""
from pathlib import Path
import runpy
import os
import moss_fp8_text_candidate
moss_fp8_text_candidate.install()
runpy.run_path(str(Path(__file__).with_name("moss_mtp_program.py" if os.environ.get("MOSS_MTP_FIXED") == "1" else "moss_gdn_experiment.py")),run_name="__main__")
