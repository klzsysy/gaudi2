"""Collect compact timing, precision, lifecycle and source evidence."""
import hashlib
import json
import os
from pathlib import Path
import re
import statistics


root = Path(os.environ['MOSS_OPT_ROOT'])
checkout = Path.cwd()


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def pipeline(label):
    directory = root / label
    runs = read_lines(directory / 'runs.jsonl')
    events = read_lines(directory / 'events.jsonl')
    warm = [row for row in runs if row['label'].startswith('full-warm')]
    generated = []
    peaks = []
    for run in warm:
        selected = [event for event in events if event.get('prompt_id') == run['prompt_id']]
        begin = next(event for event in selected if event['kind'] == 'generate_start')
        end = next(event for event in selected if event['kind'] == 'generate_end')
        generated.append((end['mono_ns'] - begin['mono_ns']) / 1e9)
        request_end = next(event for event in selected if event['kind'] == 'request_end')
        assert not request_end['fallback_counts']
        peaks.append(request_end['memory_stats']['MaxInUse'])
    seconds = [row['request_seconds'] for row in warm]
    return dict(runs=runs, warm_seconds=seconds, median=statistics.median(seconds) if seconds else None,
                generation_seconds=generated, generation_median=statistics.median(generated) if generated else None,
                peak_allocated_bytes=max(peaks) if peaks else None,
                release={path.name: read_json(path) for path in (directory / 'release').glob('*.json')})


evidence = dict(server_root=str(root), baseline_product='0995638a6546bb10d4558417d3327e3137871830',
                final_product_state='0995638a plus uncommitted native GDN changes',
                screens={}, pipelines={}, text={}, precision={}, tests={}, source_sha256={})
for name in ['situ-screen', 'attention-tiles', 'native-screen-lanes', 'native-screen-rows', 'native-screen',
             'gdn-core-scalar', 'gdn-core-vector', 'gdn-core-cached', 'gdn-core-pending', 'gdn-core-screen', 'gdn-paths']:
    evidence['screens'][name] = read_json(root / f'{name}.json')
for label in ['pipeline-baseline', 'pipeline-gdn-tiled-retry', 'pipeline-product']:
    evidence['pipelines'][label] = pipeline(label)
for label in ['text-baseline', 'text-gdn-vector', 'text-gdn-tiled', 'text-product', 'text-bf16-baseline', 'text-bf16-product']:
    rows = read_lines(root / label / 'perturbation.jsonl')
    clean = [row['seconds'] for row in rows if row['mode'] == 'plain'
             and (label == 'text-gdn-vector' or row['index'] >= 2)]
    evidence['text'][label] = dict(runs=rows, clean_seconds=clean,
                                 clean_median=statistics.median(clean) if clean else None,
                                 excluded_logits_generation=2 if label != 'text-gdn-vector' else None,
                                 program_checks=read_lines(root / label / 'program-checks.jsonl'))
for name in ['gdn-real-replay', 'gdn-logits', 'product-logits']:
    evidence['precision'][name] = read_json(root / f'{name}.json')
for name in ['gdn-native-tests', 'product-tests']:
    path = root / f'{name}.log'
    evidence['tests'][name] = re.findall(r'\d+ passed[^\n]*', path.read_text()) if path.exists() else []
sources = ['comfy/text_encoders/hpu_gated_delta.py', 'scripts/gaudi/build_gdn.py', 'scripts/gaudi/start.sh']
sources += [str(path.relative_to(checkout)) for path in (checkout / 'comfy/text_encoders/hpu_gdn').iterdir() if path.is_file()]
for name in sources:
    evidence['source_sha256'][name] = hashlib.sha256((checkout / name).read_bytes()).hexdigest()
evidence['built_libraries'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (checkout / 'build/hpu').glob('*.so')}
evidence['quality'] = {}
for path in (root / 'quality-product').glob('*-history.json'):
    history = read_json(path)
    evidence['quality'][path.stem] = dict(status=history['status']['status_str'], outputs=history['outputs'])
evidence['quality_release'] = read_json(root / 'quality-product/unload-series.json')
evidence['visual_review'] = dict(
    inspected=['fp8-teapot', 'fp8-poster', 'fp8-person'],
    findings='Subjects, MOSS lettering and three flowers are present. The teapot sample has both an overhead handle and a side handle. Natural captions and image details differ from the previous baseline.')
evidence['deployment'] = read_json(root / 'deployment/deployment.json')
evidence['runtime_verification'] = read_json(root / 'deployment-runtime.json')
(root / 'summary.json').write_text(json.dumps(evidence, indent=2))
print(json.dumps({label: {key: value[key] for key in ['median', 'generation_median', 'peak_allocated_bytes']}
                  for label, value in evidence['pipelines'].items()}, indent=2))
