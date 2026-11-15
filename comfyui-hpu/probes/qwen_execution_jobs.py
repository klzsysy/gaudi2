"""Serial candidate validation with queue checks and ordinary-service restoration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--modes", nargs="+", required=True)
parser.add_argument("--stage", choices=("text", "workflow"), default="text")
parser.add_argument("--teacher", type=Path)
parser.add_argument("--workflow", type=Path)
parser.add_argument("--logits", action="store_true")
parser.add_argument("--runs", type=int, default=3)
parser.add_argument("--case-graphs", type=Path)
parser.add_argument("--native", action="store_true")
parser.add_argument("--reload", action="store_true")
parser.add_argument("--window-tokens", type=int)
args = parser.parse_args()
product = Path("/home/sonny/ComfyUI-Gaudi")
probes = Path(__file__).parent
args.root.mkdir(parents=True, exist_ok=True)
native = probes.parent / "performance-qwen-gdn-single-2026-11-15"
source_files = [product / "comfy/text_encoders" / name for name in
                ("qwen35.py", "hpu_mlp.py", "hpu_gated_delta.py", "hpu_kv_attention.py", "llama.py")]


def hashes():
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}


def queue():
    with urllib.request.urlopen("http://192.168.10.21:8189/queue", timeout=5) as response:
        return json.load(response)


pending = queue()
assert not pending["queue_running"] and not pending["queue_pending"], pending
before = hashes()
environment = dict(os.environ, PT_HPU_LAZY_MODE="0", PYTHONPATH=str(product),
                   GC_KERNEL_PATH="/usr/lib/habanalabs/libtpc_kernels.so")
try:
    subprocess.run(["./stop-hpu.sh"], cwd=product, check=True, timeout=75)
    for mode in args.modes:
        label = mode.replace("+", "-")
        directory = args.root / label
        assert not directory.exists(), f"Preserve existing run {directory}"
        trial = "" if mode == "reference" else "execution:" + mode
        if args.stage == "text":
            command = [sys.executable, str(probes / "qwen21_text_profile.py"), str(directory)]
            if args.teacher:
                command += ["--teacher", str(args.teacher)]
            if args.native:
                command += ["--native", "--window-tokens", "560"]
            else:
                command += ["--logits"] if args.logits else ["--plain-only", "--plain-runs", str(args.runs)]
                if args.window_tokens:
                    command += ["--window-tokens", str(args.window_tokens)]
        else:
            command = [sys.executable, str(probes / "qwen21_deep_jobs.py"), label,
                       "--output-root", str(args.root), "--runs", str(args.runs), "--cached", "0"]
            if args.case_graphs:
                command += ["--case-graphs", str(args.case_graphs)]
            if args.teacher:
                command += ["--teacher", str(args.teacher)]
            if args.reload:
                command += ["--reload"]
        if trial:
            command += ["--trial", trial]
        if args.workflow:
            command += ["--workflow", str(args.workflow)]
        current_environment = environment.copy()
        if args.native:
            source_config = probes.parent / "performance-qwen-hbm-2026-11-15/text0-c16383-b255-confirm/profiler.json"
            config = json.loads(source_config.read_text())
            config["GeneralSettings"]["values"]["outdir"]["value"] = str(directory / "hw")
            config["GeneralSettings"]["values"]["session"]["value"] = "execution-hbm"
            directory.mkdir(parents=True)
            configuration = directory / "profiler.json"
            configuration.write_text(json.dumps(config, indent=2))
            current_environment.update(HABANA_PROFILE="1", HABANA_PROF_CONFIG=str(configuration))
        if "gdn_native" in mode:
            current_environment["QWEN_SINGLE_ROOT"] = str(native)
            current_environment["GC_KERNEL_PATH"] += ":" + str(native / "gdn-build/libqwen_gdn_single_tpc.so")
        start = time.perf_counter()
        with (args.root / f"{label}.log").open("w") as log:
            result = subprocess.run(command, cwd=product, env=current_environment,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        record = dict(mode=mode, stage=args.stage, returncode=result.returncode,
                      elapsed_seconds=time.perf_counter() - start, command=command)
        with (args.root / "jobs.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if result.returncode:
            raise RuntimeError(f"Candidate {mode} failed, inspect its log before continuing")
finally:
    subprocess.run(["./start-hpu.sh"], cwd=product, check=True, timeout=180)
    restored = queue()
    after = hashes()
    report = dict(deployed=(product / ".run/deployed-version").read_text().strip(),
                  source_hashes_before=before, source_hashes_after=after,
                  source_unchanged=before == after, queue=restored)
    (args.root / "restoration.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    assert before == after
