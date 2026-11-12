"""Collect compact evidence and explicitly retain unresolved measurement units."""
import collections
import hashlib
import json
from pathlib import Path
import statistics
import sys


root=Path(sys.argv[1])


def read(path):
    return json.loads((root/path).read_text())


def lines(path):
    return [json.loads(line) for line in (root/path).read_text().splitlines()]


traces=read('dit-gc/evidence-traces.json')
assert len(traces)==6
budget=collections.Counter()
for trace in traces:
    for key,ms in trace['budget_ms'].items():
        budget[key]+=ms*25/3/1000
graph=read('graph-attribution.json')
unresolved=sum(value for key,value in graph['other_exclusive_seconds'].items() if 'unresolved' in key)
paired=lines('power-control/driver.jsonl')
errors=[max(min(x['expected'])-x['utilization'],x['utilization']-max(x['expected']),0) for x in paired]
precision=read('quality/precision.json')
text={}
for case in ['text','text-gc']:
    rows=lines(f'{case}/perturbation.jsonl')
    groups={}
    for mode in sorted(set(x['mode'] for x in rows)):
        selected=[x for x in rows if x['mode']==mode]
        groups[mode]=dict(count=len(selected),median_seconds=statistics.median(x['seconds'] for x in selected),
                          median_window_seconds=statistics.median(x['window_seconds'] for x in selected))
    text[case]=dict(groups=groups,runs=rows)
counters={}
for case in ['counter-units-periodic','hbm-copy-128','hbm-copy-512','block-hardware-positive','block-hardware-negative']:
    totals=collections.Counter()
    event_counts=collections.Counter()
    for line in (root/case/'native-events.jsonl').open():
        event=json.loads(line)
        event_counts[event['category']]+=1
        if event['category']=='SPMU':
            totals[event['name']]+=event['raw_u64']
    counters[case]=dict(status=read(f'{case}/native-profile-status.json'),spmu_totals=dict(totals),
                        event_counts=dict(event_counts),run=read(f'{case}/raw-run.json') if case.startswith(('hbm','counter')) else read(f'{case}/run.json'))
previous=json.loads((root.parent/'evidence/moss-fp8-fusion-2026-11-12.json').read_text())
checkout=Path('/home/sonny/ComfyUI-Gaudi')
hashes={name:hashlib.sha256((checkout/name).read_bytes()).hexdigest() for name in previous['source_hashes']}
assert hashes==previous['source_hashes']
result=dict(date='2026-11-12',product_commit=previous['product_commit'],server_root=str(root),
    source_hashes=hashes,environment=dict(kernel='5.15.0-203-generic',driver='1.24.2-52a555d',
        torch='2.13.0a0+git2469d18',driver_module_sha256='21fa94a804e1a4a3745dd256cdcb4381bf8762113bd664d843ad5cc304a7613f',
        driver_srcversion='0A86AA51441E652BE8CD384',cpu_cores=32,cpu_threads=64,numa_nodes=1),
    timing=read('timing-summary.json'),device_budget=dict(seconds=dict(budget),total_seconds=sum(budget.values()),traces=traces),
    other_kernel=dict(exclusive_seconds=graph['other_exclusive_seconds'],unresolved_seconds=unresolved,
                      unresolved_fraction=unresolved/budget['other_kernel'],kernel_mapping=graph['kernel_mapping']),
    attention_storage=dict(qk_scores='DRAM BF16 NxN',probabilities='SRAM FP8 query slices shared with AV',
                           positive_qk_tile=[4403,1536],positive_probability_tile=[4403,2048],
                           score_logical_read_write_TB=25*42*24*(4403**2+4119**2)*4/1e12),
    utilization=dict(formula='floor(100 * (clamp(power_mW, 60000, 550000) - 60000) / (550000 - 60000))',
                     samples=len(paired),within_bracket=sum(e==0 for e in errors),within_one_point=sum(e<=1 for e in errors),
                     example=next(x for x in paired if x['utilization']>=35 and x['expected']==[x['utilization']]*2)),
    bandwidth=read('hardware/calibration.json'),hardware_counters=counters,text_perturbation=text,
    quality=dict(cases=len(precision),min_rrmse=min(x['rrmse'] for x in precision),median_rrmse=statistics.median(x['rrmse'] for x in precision),
                 max_rrmse=max(x['rrmse'] for x in precision),by_layer={str(layer):dict(count=sum(x['layer']==layer for x in precision),
                 median_rrmse=statistics.median(x['rrmse'] for x in precision if x['layer']==layer),
                 max_rrmse=max(x['rrmse'] for x in precision if x['layer']==layer)) for layer in [0,20,41]},
                 visual_review={'short':'Single pear on gray background','count_text':'Four colored shapes in requested order and FOUR SHAPES text',
                                'texture':'Knit, lace and metal textures retained','long':'Bookshop, bookseller, reader, cat, window and NIGHT BOOKS retained'},
                 reference='FP32 operations on decoded quantized weights',inputs=precision),
    open_items=['BMON values do not yet reproduce known copy bytes across both buffer sizes. No actual HBM byte total or saturation percentage is asserted.',
                'SPMU counters overlap and have distinct hardware meanings. Their totals do not form additive stall percentages.',
                'A small fraction of other-kernel time remains named but not semantically attributed.',
                'The first six-trace launcher segfaulted before profiling in its first warm request. The accepted launcher completed repeated runs. Root cause remains unconfirmed.'],
    jobs=lines('jobs.jsonl'))
result['bmon_cross_check']=dict(source=str(root/'automatic-copy/hw'),
    method='Automatic multi-enqueue capture, vendor-native HLTV and JSON',
    observation='Vendor-native JSON reproduces the counter values and some Avg > Max records. Byte integration remains uncalibrated.')
if (root/'service-check/deployment.json').exists():
    result['service_check']=read('service-check/deployment.json')
if (root/'final-service-check/deployment.json').exists():
    result['final_service_check']=read('final-service-check/deployment.json')
(root/'evidence-summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(total_device=sum(budget.values()),unresolved=unresolved,quality_cases=len(precision),paired_samples=len(paired))))
