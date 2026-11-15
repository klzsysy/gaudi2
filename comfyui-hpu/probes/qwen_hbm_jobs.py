"""Serial calibration jobs with one configuration and process per capture."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--batch", choices=["controls", "windows", "native", "completion", "beats", "confirm"], default="controls")
args = parser.parse_args()
args.root.mkdir(parents=True, exist_ok=True)
base_path = args.root / "base-profiler.json"
if not base_path.exists():
    subprocess.run(["/opt/habanalabs/bin/synprof_configuration", "--config-filename", str(base_path),
                    "--gaudi2", "--use-template", "profile_api_light", "--hmmu", "on", "--hmmu_bmon", "on",
                    "--host", "off", "--invoc", "json,hltv", "-b", "128",
                    "-o", str(args.root / "hw"), "-s", "hbm"], check=True)
base = json.loads(base_path.read_text())
jobs = [("copy", 128, 16384, 1024), ("copy", 2048, 16384, 1024),
        ("fill", 512, 16384, 1024), ("sum", 512, 16384, 1024),
        ("gemv", 512, 16384, 1024), ("idle", 512, 16384, 1024),
        ("copy", 512, 4096, 4096), ("copy", 512, 16384, 16384)]
if args.batch == "windows":
    jobs = [(op, 512, capture, width) for op in ("copy", "fill", "gemv")
            for capture, width in ((16383, 255), (65535, 255), (4095, 255))]
if args.batch == "native":
    jobs = [("synapse_dma", mib, 4095, 255) for mib in (128, 512, 2048)]
    jobs += [(op, mib, 4095, width) for op, mib, width in
             (("sum", 512, 255), ("copy", 512, 63), ("gemv", 512, 63),
              ("copy", 128, 255), ("copy", 2048, 255), ("text", 0, 4095))]
if args.batch in ("completion", "beats"):
    jobs = [(op, 512, 4095, 255) for op in ("copy", "gemv", "fill")]
    jobs += [("text", 0, 16383, 255)]
if args.batch == "confirm":
    jobs = [(op, mib, 4095, 255) for op, mib in
            (("copy", 128), ("copy", 2048), ("sum", 512), ("gemv", 128), ("gemv", 2048), ("idle", 512))]
    jobs += [("text", 0, 16383, 255)]
environment = dict(os.environ, PT_HPU_LAZY_MODE="0", PYTHONPATH="/home/sonny/ComfyUI-Gaudi",
                   GC_KERNEL_PATH="/usr/lib/habanalabs/libtpc_kernels.so", HABANA_PROFILE="1")
try:
    for operation, mib, capture, width in jobs:
        name = f"{operation}{mib}-c{capture}-b{width}"
        if args.batch in ("completion", "beats"):
            name += "-axi" if args.batch == "completion" else "-beats"
        if args.batch == "confirm":
            name += "-confirm"
        directory = args.root / name
        directory.mkdir(exist_ok=True)
        config = copy.deepcopy(base)
        config["GeneralSettings"]["values"]["outdir"]["value"] = str(directory / "hw")
        config["GeneralSettings"]["values"]["session"]["value"] = name
        for plugin in config["Plugins"]:
            if plugin["name"] == "HwTrace":
                units = plugin["values"]["archProfileUnits"]["gaudi2"]["HMMU"]
                for unit_name, unit in units.items():
                    if unit_name != "enable":
                        for monitor in ("b10", "b11"):
                            unit["bmonParams"][monitor] = {key: {"value": value} for key, value in
                                dict(enable=True, bmon_Bandwidth=True, bmon_Latency=False,
                                     bmon_Compression=False, capture_window=capture, bandwidth_window=width).items()}
                            if args.batch in ("completion", "beats", "confirm"):
                                unit["bmonParams"][monitor]["bmon_AXI_Completion_Bandwidth"] = {"value": True}
        configuration = directory / "profiler.json"
        configuration.write_text(json.dumps(config, indent=2))
        command = [sys.executable, str(Path(__file__).with_name("qwen_hbm_calibrate.py")),
                   str(directory), "--operation", operation, "--mib", str(mib)]
        if operation == "synapse_dma":
            command = [str(args.root / "hbm-dma"), str(directory), str(mib)]
        if operation == "text":
            command = [sys.executable, str(Path(__file__).with_name("qwen_single_decode.py")),
                       str(directory), "--native"]
            if args.batch == "beats":
                command += ["--steps", "4"]
            if args.batch == "confirm":
                command += ["--steps", "16", "--start-token", "512", "--no-snapshot"]
        with (args.root / f"{name}.log").open("w") as log:
            result = subprocess.run(command,
                                    env=dict(environment, HABANA_PROF_CONFIG=str(configuration)),
                                    stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        print(name, result.returncode, flush=True)
        if result.returncode:
            raise RuntimeError(f"Calibration failed in {name}, inspect its log before continuing")
finally:
    subprocess.run(["./start-hpu.sh"], check=True, timeout=180)
