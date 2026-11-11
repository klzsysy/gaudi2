"""End-to-end 298-token fixed-MTP workflows with identical generated captions."""
import collections
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

root = Path(os.environ['MOSS_COMPLETION_ROOT'])
probes = Path(__file__).parent
source = (root/'baseline/source.json').read_text()
base = dict(os.environ, PT_HPU_LAZY_MODE='0', GC_KERNEL_PATH='/usr/lib/habanalabs/libtpc_kernels.so', PYTHONPATH=str(Path.cwd()),
            MOSS_MTP_FIXED='1', MOSS_MTP_PROGRAM=str(root/'mtp-record-298/program.pt'), MOSS_PIPELINE_CACHE_RUNS='0')
summary = {}
for name, settings in [('baseline', {'MOSS_NORM_EAGER':'1'}), ('candidate', {})]:
    destination = root/('matched-'+name)
    destination.mkdir(exist_ok=True)
    (destination/'source.json').write_text(source)
    env = dict(base, **settings, MOSS_PIPELINE_DIR=str(destination), MOSS_RESULT_DIR=str(destination/'mtp'))
    with (root/('matched-'+name+'.log')).open('w') as log:
        subprocess.run([sys.executable,str(probes/'moss_pipeline_measure.py')], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    runs = [json.loads(s) for s in (destination/'runs.jsonl').read_text().splitlines()]
    events = [json.loads(s) for s in (destination/'events.jsonl').read_text().splitlines()]
    samples = [r for r in runs if r['label'].startswith('full-warm')]
    assert len(samples) == 5 and all(r['output_tokens'] == 298 for r in samples)
    stages = collections.defaultdict(list)
    ids = []
    for run in samples:
        request = [e for e in events if e.get('prompt_id') == run['prompt_id']]
        starts = {e['node']:e for e in request if e['kind']=='node_start'}
        times = collections.Counter()
        for e in request:
            if e['kind'] == 'node_end':
                times[e['class_type']] += (e['mono_ns']-starts[e['node']]['mono_ns'])/1e9
            if e['kind'] == 'generate_end':
                ids.append(e['token_ids'])
        for key,value in times.items():
            stages[key].append(value)
    summary[name] = dict(median=statistics.median(r['request_seconds'] for r in samples),
                         samples=[r['request_seconds'] for r in samples],
                         stages={k:statistics.median(v) for k,v in stages.items()},
                         cold=runs[0]['request_seconds'], ids_equal=all(v==ids[0] for v in ids), ids=ids[0])
    (root/'matched-pipeline-summary.json').write_text(json.dumps(summary,indent=2))
    print(name, {k:v for k,v in summary[name].items() if k!='ids'}, flush=True)
assert summary['baseline']['ids'] == summary['candidate']['ids']
