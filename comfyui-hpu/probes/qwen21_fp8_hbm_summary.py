"""Stream calibrated HBM response beats and report profiled/plain denominators."""
import argparse
import collections
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
parser.add_argument('--stages', nargs='+', default=['text', 'dit', 'clip'])
parser.add_argument('--reference-root', type=Path)
a = parser.parse_args()
report = {}
reference_root = a.reference_root or a.root
for stage in a.stages:
    root = a.root / ('native-' + stage)
    config = json.loads((root / 'profiler.json').read_text())
    hw = next(p for p in config['Plugins'] if p['name'] == 'HwTrace')
    bmon = hw['values']['archProfileUnits']['gaudi2']['HMMU']['DCORE0_HMMU0_CS']['bmonParams']['b10']
    factor = (bmon['capture_window']['value']+1)/(bmon['bandwidth_window']['value']+1)*128
    ports = collections.defaultdict(lambda: dict(samples=0, bytes=0, rounding_envelope=0))
    for line in (root / 'native-events.jsonl').open():
        event = json.loads(line)
        if event['category'] != 'BMON' or 'counter_values' not in event:
            continue
        port = ports[event['name']]
        port['samples'] += 1
        port['bytes'] += event['counter_values'][2] * factor
        port['rounding_envelope'] += (event['counter_values'][1] > 0) * factor
    read = sum(v['bytes'] for k,v in ports.items() if 'Read' in k)
    write = sum(v['bytes'] for k,v in ports.items() if 'Write' in k)
    seconds = json.loads((root / 'window.json').read_text())['seconds']
    plain = json.loads((root / 'plain-window.json').read_text())['seconds']
    if (root / 'timings.jsonl').exists():
        rows = [json.loads(line) for line in (root / 'timings.jsonl').read_text().splitlines()]
        plain = statistics.median(r['window_seconds'] for r in rows if r['mode']=='plain')
    record = dict(read_bytes=read, write_bytes=write, ports=dict(ports), profiled_seconds=seconds,
                  plain_seconds=plain, profiler_slowdown=seconds/plain-1,
                  profiled_TBps=(read+write)/seconds/1e12,
                  plain_estimated_TBps=(read+write)/plain/1e12,
                  bandwidth_fraction=(read+write)/plain/2.45e12)
    if stage == 'text':
        weights = json.loads((reference_root / 'weights-pe.json').read_text())
        flops = 2 * sum(r['elements'] for r in weights['converted']) * 16
        record['bytes_per_token'] = (read+write)/16
    elif stage == 'dit':
        trace = next((reference_root / 'fp8-off').glob('*dit.trace.fp8-summary.json'))
        flops = json.loads(trace.read_text())['fp8_linear_flops']
    else:
        flops = json.loads((a.root / 'clip-work.json').read_text())['fp8_linear_flops']
    if flops is not None:
        record.update(fp8_linear_flops=flops, useful_fp8_TFps=flops/plain/1e12,
                      fp8_peak_fraction=flops/plain/865e12)
    report[stage] = record
    print(stage, {k:v for k,v in record.items() if k!='ports'}, flush=True)
(a.root / 'hardware-summary.json').write_text(json.dumps(report, indent=2))
