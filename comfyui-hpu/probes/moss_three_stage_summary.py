"""Collect reproducible timing, numerical, quality and deployment evidence."""
import collections
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys


root = Path(sys.argv[1])
commit = sys.argv[2]


def read(name):
    return json.loads((root / name).read_text())


def lines(name):
    return [json.loads(line) for line in (root / name).read_text().splitlines()]


def pipeline(name):
    runs = lines(f'{name}/runs.jsonl')
    events = lines(f'{name}/events.jsonl')
    warm = [run for run in runs if run['label'].startswith('full-warm')]
    assert len(warm) == 5
    assert all(not event['fallback_counts'] for event in events if event['kind'] == 'request_end')
    stages = collections.defaultdict(list)
    for run in warm:
        request = [event for event in events if event.get('prompt_id') == run['prompt_id']]
        starts = {event['node']: event['mono_ns'] for event in request if event['kind'] == 'node_start'}
        times = collections.Counter()
        for event in request:
            if event['kind'] == 'node_end':
                times[event['class_type']] += (event['mono_ns'] - starts[event['node']]) / 1e9
        for key, value in times.items():
            stages[key].append(value)
    checks = lines(f'{name}/mtp/program-checks.jsonl')
    assert all(check['tokens'] == 298 for check in checks)
    return dict(cold_seconds=runs[0]['request_seconds'],
                warm_seconds=[run['request_seconds'] for run in warm],
                median_seconds=statistics.median(run['request_seconds'] for run in warm),
                node_medians={key: statistics.median(values) for key, values in stages.items()},
                program_checks=checks)


attention_runs = read('attention-pipeline/runs.json')
attention_quality = read('attention-pipeline/paired-quality.json')
attention_reference = next(row for row in attention_runs if row['label'] == 'baseline-cold')
attention_candidate = next(row for row in attention_runs if row['label'] == 'long-query-cold')
validation = read('linear-validation/validation-retry.json')
dynamic = read('dynamic-padding/dynamic-lengths.json')
assert len(validation) == 18 and all(row['equal'] for row in validation)
assert len(dynamic) == 14 and all(row['equal'] for row in dynamic)
quality = read('natural-quality.json')
assert len(quality) == 6 and all(row['tokens_equal'] for row in quality)
test_logs = {}
for name in ('finish-regressions.log', 'final-targeted-tests.log', 'final-accepted-regressions.log'):
    text = (root / name).read_text()
    summary = re.findall(r'^\d+ passed[^\n]*$', text, re.MULTILINE)
    assert summary, name
    test_logs[name] = summary[-1]
source_files = ['comfy/ldm/modules/attention.py', 'comfy/text_encoders/hpu_kv_attention.py',
                'tests-unit/comfy_test/hpu_moss_attention_test.py', 'tests-unit/comfy_test/hpu_rope_test.py']
result = dict(
    date='2026-11-13', started='2026-11-12', product_commit=commit,
    final_targeted_test_tree='4d9cc16d8fc372dc580511bed073bb0ce606672e',
    baseline_commit='69fe4a54088adab9e20ce4a71dbc38af370785db', server_root=str(root),
    source_hashes={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source_files},
    environment=dict(device='Gaudi2 HL-225', torch='2.13.0a0+git2469d18', synapse='1.24.2',
                     mode='eager', cpu_threads=32),
    attention=dict(paired=attention_quality, real_inputs=validation, dynamic_lengths=dynamic,
                   dynamic_graph_count=max(row['graph_count'] for row in dynamic),
                   storage=read('long-query/attention-storage.json'),
                   cumulative_allocator_peak_bytes=dict(baseline=attention_reference['memory']['MaxInUse'],
                                                        after_candidate=attention_candidate['memory']['MaxInUse']),
                   memory_scope='Same-process cumulative high water including cold and both recipe sets',
                   unloaded_bytes=5393876352),
    bf16_trials=dict(screen=read('bf16-screen/bf16-shared-screen.json'),
                     rounding_preserved=read('bf16-preserve/bf16-shared-screen.json'),
                     decision='Retain existing BF16 DiT regions after quality-preserving candidates showed no net gain'),
    decision=dict(attention='accepted', bf16_dit='existing regions retained',
                  text='accepted after controlled equality checks and a baseline recheck reproduced the Auto text-insertion issue'),
    pipelines={name: pipeline(name) for name in ['baseline-retry', 'product-functional-fp8',
                                                'bf16-baseline', 'product-functional-bf16']},
    logits=dict(fp8=read('product-functional-fp8-logits.json'),
                bf16=read('product-functional-bf16-logits.json')),
    natural_quality=quality,
    default_auto_review=read('auto-quality-review.json'),
    visual_review=dict(fp8_teapot='Single visible handle', bf16_teapot='No visible handle in both baseline and candidate',
                       posters='Three yellow flowers and MOSS text in both pairs; FP8 text has low contrast',
                       people='Wrinkles, red scarf and white cup preserved'),
    quality_workflow=dict(mtp='tree3s1', stochastic_generation=True, forced_tokens=False,
                          unloaded_bytes=2685928832, reload='512px MTP off and original INT8 successful'),
    tests=test_logs,
    test_scope=dict(broad='341 tests before the dynamic padding formula cleanup',
                    targeted='37 attention and RoPE tests on the final source tree',
                    rollback='335 tests during the original-text baseline recheck'),
    jobs=lines('finish-jobs.jsonl'),
    excluded_trials=dict(functional_decomposition=read('rope-fp8-logits.json'),
                         hot_registration='Runtime kernel replacement corrupted comparisons. Fixed-kernel processes are the accepted comparisons.',
                         process_crashes=['baseline/server.log', 'product-fp8/server.log', 'product-fp8-retry/server.log']),
    remaining_items=['QK score tensors still use DRAM. Probability tiles use SRAM.',
                     'The earlier measurement-process segmentation faults have no confirmed vendor-level root cause.',
                     'Image defects already present in the baseline samples are retained in the visual review.',
                     'Default Auto expansion can insert unrequested visible text with both the original and optimized text implementation.'],
)
assert result['logits']['fp8']['equal'] and result['logits']['bf16']['equal']
if (root / 'deployment.json').exists():
    result['deployment'] = read('deployment.json')
    result['final_queue'] = read('deployment-final-queue.json')
    assert not any(result['final_queue'].values())
    history = read('deployment-0-history.json')
    result['final_auto_text'] = history['outputs']['18:13']['text'][0]
    result['final_auto_text_equal_to_prior_baseline'] = result['final_auto_text'] == result['default_auto_review']['baseline_text']
    result['deployment_artifacts'] = []
    for index in range(3):
        history = read(f'deployment-{index}-history.json')
        assert history['status']['completed'] and history['status']['status_str'] == 'success'
        image_path = root / f'deployment-{index}.png'
        result['deployment_artifacts'].append(dict(
            history=f'deployment-{index}-history.json', image=image_path.name,
            image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest()))
    result['final_visual_review'] = 'First final deployment image retains woman, hummingbird, roses and gown details. No extra text label is visible.'
(root / 'summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(dict(product_commit=commit, tests=test_logs,
                      decision=result['decision'])))
