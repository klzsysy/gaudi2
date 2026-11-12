"""Pair the driver utilization formula with sustained controlled workloads."""
import os
from pathlib import Path
import subprocess
import sys


root=Path(os.environ['MOSS_EVIDENCE_ROOT'])/'power-control'
root.mkdir(parents=True,exist_ok=True)
probes=Path(__file__).parent
monitor=subprocess.Popen([sys.executable,str(probes/'moss_evidence_utilization.py'),str(root/'driver.jsonl'),'--seconds','180'])
try:
    subprocess.run([sys.executable,str(probes/'moss_evidence_hardware.py'),str(root)],check=True)
finally:
    monitor.terminate()
    monitor.wait(timeout=10)
