"""Compare device interval unions and launch counts for matched MTP steps."""
import collections
import json
from pathlib import Path
import sys

root=Path(sys.argv[1])
paths={
    'baseline':root.parent/'performance-2026-11-11-completion/mtp-norm-profile/mtp-steps.trace.json',
    'parallel-core':root/'product-profile/mtp-steps.trace.json',
    'candidate':root/'region-product-profile/mtp-steps.trace.json',
}
records={}
for label,path in paths.items():
    events=json.loads(path.read_text())['traceEvents']
    kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X' and e.get('dur',0)>0]
    intervals=sorted((e['ts'],e['ts']+e['dur']) for e in kernels)
    begin,end=intervals[0]
    active=0
    for start,stop in intervals[1:]:
        if start>end:
            active+=end-begin
            begin,end=start,stop
        else:
            end=max(end,stop)
    active+=end-begin
    span=max(stop for _,stop in intervals)-intervals[0][0]
    runtime=collections.Counter(e['name'] for e in events if e.get('cat')=='privateuse1_runtime')
    cpu=collections.Counter(e['name'] for e in events if e.get('cat')=='cpu_op')
    records[label]=dict(trace=str(path),span_ms=span/1000,active_union_ms=active/1000,idle_ms=(span-active)/1000,
                        idle_fraction=(span-active)/span,kernels=len(kernels),runtime_counts=dict(runtime),
                        cpu_linear_calls=cpu['aten::linear'])
(root/'submission-budget.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
