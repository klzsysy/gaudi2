"""Evaluate whole-decoder execution in isolated eager and lazy processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--modes", nargs="+", default=["eager", "compile", "lazy", "graph"])
args = parser.parse_args()
args.root.mkdir(parents=True, exist_ok=True)
environment = dict(os.environ, PYTHONPATH="/home/sonny/ComfyUI-Gaudi",
                   GC_KERNEL_PATH="/usr/lib/habanalabs/libtpc_kernels.so")
try:
    for mode in args.modes:
        directory = args.root / mode
        start = time.perf_counter()
        command = [sys.executable, str(Path(__file__).with_name("qwen_single_decode.py")),
                   str(directory), "--mode", "eager" if mode == "gdn" else mode, "--input", str(args.input)]
        current_environment = dict(environment, PT_HPU_LAZY_MODE="1" if mode in ("lazy", "graph") else "0")
        if mode == "gdn":
            native_root = "/home/sonny/gaudi2/comfyui-hpu/performance-qwen-gdn-single-2026-11-15"
            command += ["--gdn-single"]
            current_environment["QWEN_SINGLE_ROOT"] = native_root
            current_environment["GC_KERNEL_PATH"] += ":" + native_root + "/gdn-build/libqwen_gdn_single_tpc.so"
        with (args.root / f"{mode}.log").open("w") as log:
            result = subprocess.run(command, env=current_environment,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        record = dict(mode=mode, returncode=result.returncode, seconds=time.perf_counter() - start)
        with (args.root / "jobs.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(record, flush=True)
        if result.returncode < 0:
            raise RuntimeError("Decoder probe terminated abnormally, inspect its log before continuing")
finally:
    subprocess.run(["./start-hpu.sh"], check=True, timeout=180)
