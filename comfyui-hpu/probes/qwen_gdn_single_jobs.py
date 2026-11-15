"""End-to-end fixed-token comparisons for the single-token GDN kernel."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path("/home/sonny/gaudi2/comfyui-hpu/performance-qwen-gdn-single-2026-11-15")
probe = Path(__file__).with_name("qwen21_text_profile.py")
teacher = Path("/home/sonny/gaudi2/comfyui-hpu/performance-qwen21-2026-11-15/text-reference/plain-tokens.json")
environment = dict(os.environ, PT_HPU_LAZY_MODE="0", PYTHONPATH="/home/sonny/ComfyUI-Gaudi",
                   GC_KERNEL_PATH="/usr/lib/habanalabs/libtpc_kernels.so")
try:
    for name in ("reference", "native"):
        command = [sys.executable, str(probe), str(root / name), "--teacher", str(teacher), "--logits"]
        current = environment.copy()
        if name == "native":
            command += ["--trial", "next:gdn_single"]
            current["QWEN_SINGLE_ROOT"] = str(root)
            current["GC_KERNEL_PATH"] += ":" + str(root / "gdn-build/libqwen_gdn_single_tpc.so")
        start = time.perf_counter()
        with (root / f"{name}.log").open("w") as log:
            result = subprocess.run(command, env=current, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        print(dict(name=name, returncode=result.returncode, seconds=time.perf_counter()-start), flush=True)
        if result.returncode:
            raise RuntimeError(f"Failed in {name}")
finally:
    subprocess.run(["./start-hpu.sh"], check=True, timeout=180)
