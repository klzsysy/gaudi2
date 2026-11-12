"""Exclusive-card bandwidth, activity telemetry and hardware-profiler controls."""
import argparse
import contextlib
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time

import torch
import habana_frameworks.torch.utils.debug as debug


parser = argparse.ArgumentParser()
parser.add_argument('output', type=Path)
parser.add_argument('--raw', choices=['copy', 'mme', 'tpc'])
parser.add_argument('--iterations', type=int, default=20)
parser.add_argument('--mib', type=int, default=512)
parser.add_argument('--native', action='store_true')
parser.add_argument('--automatic', action='store_true')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
torch.set_num_threads(32)


def telemetry(stream, destination):
    with destination.open('w') as output:
        for line in stream:
            output.write(json.dumps(dict(mono_ns=time.perf_counter_ns(), wall_ns=time.time_ns(),
                                         csv=line.strip())) + '\n')
            output.flush()


def buffers(mib):
    source = torch.ones(mib * 1024 * 1024 // 2, dtype=torch.bfloat16, device='hpu')
    return source, torch.empty_like(source)


def measure(function, iterations):
    for _ in range(5):
        function()
    torch.hpu.synchronize()
    samples = []
    for _ in range(5):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        torch.hpu.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e9 / iterations)
    return dict(samples_seconds=samples, median_seconds=statistics.median(samples))


if args.raw:
    source, destination = buffers(args.mib)
    a = torch.ones((4096, 4096), dtype=torch.bfloat16, device='hpu')
    b, c = torch.ones_like(a), torch.empty_like(a)
    operations = dict(copy=lambda: destination.copy_(source),
                      mme=lambda: torch.mm(a, b, out=c),
                      tpc=lambda: torch.sigmoid(source, out=destination))
    function = operations[args.raw]
    for _ in range(10):
        function()
    torch.hpu.synchronize()
    if args.automatic:
        context=contextlib.nullcontext()
    elif args.native:
        from moss_native_profile import NativeProfile
        context=NativeProfile(args.output)
    else:
        context=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                  torch.profiler.ProfilerActivity.HPU])
    with context as profile:
        start = time.perf_counter_ns()
        for _ in range(args.iterations):
            function()
        torch.hpu.synchronize()
        stop = time.perf_counter_ns()
    if not args.native and not args.automatic:
        profile.profiler.export_chrome_trace(str(args.output/'raw.trace.json'))
    result = dict(operation=args.raw, mib=args.mib, iterations=args.iterations,
                  seconds=(stop-start)/1e9, logical_copy_bytes=args.mib*1024**2*2*args.iterations,
                  fallback=debug._get_fallback_op_count())
    (args.output / 'raw-run.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
else:
    command = ['stdbuf', '-oL', 'hl-smi', '--query-aip=utilization.aip,memory.used,power.draw,temperature.aip,clocks.current.soc,clocks.limit.soc,clocks.limit.tpc,stats.violation.power,stats.violation.thermal',
               '--format=csv,nounits', '-l', '100ms']
    monitor = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    reader = threading.Thread(target=telemetry, args=(monitor.stdout, args.output/'telemetry.jsonl'))
    reader.start()
    result = dict(torch=torch.__version__, cpu_threads=torch.get_num_threads(), bandwidth=[], phases=[])
    try:
        for mib in (128, 512, 2048):
            source, destination = buffers(mib)
            record = measure(lambda: destination.copy_(source), 50)
            record.update(mib_per_buffer=mib, logical_bytes=2*mib*1024**2,
                          effective_GB_s=2*mib*1024**2/record['median_seconds']/1e9)
            result['bandwidth'].append(record)
            del source, destination
        source, destination = buffers(512)
        a = torch.ones((4096,4096), dtype=torch.bfloat16, device='hpu')
        b, c = torch.ones_like(a), torch.empty_like(a)
        for label, function in [('idle', None), ('mme', lambda: torch.mm(a,b,out=c)),
                                ('idle', None), ('tpc_sigmoid', lambda: torch.sigmoid(source,out=destination)),
                                ('idle', None), ('copy', lambda: destination.copy_(source)), ('idle',None)]:
            if function:
                for _ in range(5):
                    function()
                torch.hpu.synchronize()
            start = time.perf_counter_ns()
            iterations = 0
            while (time.perf_counter_ns()-start)/1e9 < 8:
                if function:
                    for _ in range(50):
                        function()
                    torch.hpu.synchronize()
                    iterations += 50
                else:
                    time.sleep(.05)
            result['phases'].append(dict(label=label,start_ns=start,end_ns=time.perf_counter_ns(),iterations=iterations))
            print(json.dumps(result['phases'][-1]),flush=True)
        result['fallback'] = debug._get_fallback_op_count()
        (args.output/'calibration.json').write_text(json.dumps(result,indent=2))
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        reader.join(timeout=10)
