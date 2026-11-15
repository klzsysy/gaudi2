"""Validate product integration with automatic source rollback on failure."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request


root = Path("/home/sonny/gaudi2/comfyui-hpu/deploy-qwen-execution-2026-11-15")
product = Path("/home/sonny/ComfyUI-Gaudi")
probes = root.parent / "probes"
files = [str(path.relative_to(root / "candidate")) for path in (root / "candidate").rglob("*.py")]
environment = dict(os.environ, PT_HPU_LAZY_MODE="0", GC_KERNEL_PATH="/usr/lib/habanalabs/libtpc_kernels.so",
                   PYTHONPATH=str(product))
with urllib.request.urlopen("http://192.168.10.21:8189/queue", timeout=10) as response:
    queue = json.load(response)
assert not queue["queue_running"] and not queue["queue_pending"], queue
backup = root / "baseline"
if not backup.exists():
    backup.mkdir()
    for name in files:
        if (product / name).exists():
            destination = backup / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(product / name, destination)
    shutil.copy2(product / ".run/deployed-version", backup / "deployed-version")
subprocess.run(["./stop-hpu.sh"], cwd=product, check=True, timeout=75)
success = False


def run(label, command, timeout=900):
    start = time.perf_counter()
    with (root / f"{label}.log").open("w") as log:
        result = subprocess.run(command, cwd=product, env=environment, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout)
    record = dict(label=label, command=command, returncode=result.returncode, seconds=time.perf_counter() - start)
    with (root / "jobs.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    if result.returncode:
        raise RuntimeError(f"Failed in {label}")


try:
    for name in files:
        target = product / name
        temporary = target.with_suffix(target.suffix + ".deploy")
        shutil.copy2(root / "candidate" / name, temporary)
        os.replace(temporary, target)
    tests = ["hpu_qwen_generation_test.py", "hpu_gdn_region_test.py", "hpu_mlp_test.py", "hpu_qwen_norm_test.py",
             "hpu_kv_attention_test.py", "hpu_rope_test.py", "hpu_textgen_test.py"]
    run("tests", [sys.executable, "-m", "pytest", "-q", "-x"] + ["tests-unit/comfy_test/" + name for name in tests])
    teacher = root.parent / "performance-qwen21-2026-11-15/text-reference/plain-tokens.json"
    run("product-text", [sys.executable, str(probes / "qwen21_text_profile.py"), str(root / "product-text"),
                         "--teacher", str(teacher), "--logits"])
    reference = root.parent / "performance-qwen-execution-2026-11-15/combined/gdn_direct-sdpa-whole/logits.pt"
    run("comparison", [sys.executable, str(probes / "qwen_execution_summary.py"), str(root), "--reference", str(reference)])
    source_hashes = {name: hashlib.sha256((product / name).read_bytes()).hexdigest() for name in files}
    (root / "validated-sources.json").write_text(json.dumps(source_hashes, indent=2))
    success = True
    print("Validation complete; product files are ready for version marking and service startup", flush=True)
finally:
    if not success:
        for name in files:
            target = product / name
            if (backup / name).exists():
                shutil.copy2(backup / name, target)
            else:
                target.unlink(missing_ok=True)
        shutil.copy2(backup / "deployed-version", product / ".run/deployed-version")
        subprocess.run(["./start-hpu.sh"], cwd=product, check=True, timeout=180)
