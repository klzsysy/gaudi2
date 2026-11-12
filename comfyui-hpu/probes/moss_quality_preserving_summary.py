"""Archive rollback, exactness, timing, graph-storage and quality evidence."""
import hashlib
import json
import os
from pathlib import Path
import statistics

from PIL import Image, ImageChops, ImageStat


root = Path(os.environ['MOSS_QUALITY_ROOT'])
checkout = Path.cwd()


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def pipeline(label):
    directory = root / label
    runs = lines(directory / 'runs.jsonl')
    warm = [row for row in runs if row['label'].startswith('full-warm')]
    values = [row['request_seconds'] for row in warm]
    return dict(runs=runs, warm_seconds=values, median=statistics.median(values), mean=statistics.mean(values),
                program_checks=lines(directory / 'mtp/program-checks.jsonl'),
                unload=read(directory / 'release/unload-series.json'))


jobs = lines(root / 'compiler-jobs.jsonl')
compiler = []
for job in jobs:
    label = job['label']
    configuration = read(root / f'{label}-configuration.json', {})
    compiler.append(dict(job, configuration=configuration, set_status=read(root / f'{label}-set-status.json'),
                         activation_verified=all(configuration.get(key, {}).get('value') == value
                                                 for key, value in job['requested'].items()),
                         results=read(root / f'{label}.json')))
baseline = read(root / 'exact-inputs-baseline.json')
candidate = read(root / 'exact-inputs-32.json')
assert len(baseline) == len(candidate) == 90
for first, second in zip(baseline, candidate):
    assert first['source'] == second['source']
quality = []
historical = root.parent / 'performance-2026-11-12-fp8-fusion/quality-verified'
for label in ['teapot', 'poster', 'person']:
    reference = Image.open(historical / f'fp8-{label}.png').convert('RGB')
    actual = Image.open(root / f'quality-rollback/fp8-{label}.png').convert('RGB')
    difference = ImageChops.difference(reference, actual)
    old = read(historical / f'fp8-{label}-history.json')
    new = read(root / f'quality-rollback/fp8-{label}-history.json')
    quality.append(dict(label=label, source_graph_equal=old['prompt'][2] == new['prompt'][2],
                        caption_equal=old['outputs'].get('18:13') == new['outputs'].get('18:13'),
                        pixels_equal=difference.getbbox() is None, pixel_mae=sum(ImageStat.Stat(difference).mean) / 3,
                        baseline_pixel_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
                        rollback_pixel_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                        status=new['status']['status_str']))
sources = ['comfy/ldm/moss_image/model.py', 'comfy/ldm/moss_image/hpu.py', 'comfy/ldm/modules/attention.py',
           'comfy/ops.py', 'comfy/quant_ops.py', 'comfy/text_encoders/qwen35.py',
           'comfy/text_encoders/llama.py', 'comfy/text_encoders/hpu_gated_delta.py',
           'scripts/gaudi/start.sh', '.gitignore']
result = dict(product='0995638a6546bb10d4558417d3327e3137871830',
              disposition='GDN candidate withdrawn. Existing product retained. SRAM32 timing benefit not established.',
              rollback_logits=read(root / 'rollback-logits.json'),
              layouts=read(root / 'baseline.json'), compiler=compiler,
              compiler_preflight=read(root / 'compiler-set-preflight.json'),
              baseline_inputs=baseline, sram32_inputs=candidate,
              sram32_exact_count=sum(row['bitwise_equal'] for row in candidate),
              sram32_median_block_time_ratio=statistics.median(second['seconds'] / first['seconds']
                                                               for first, second in zip(baseline, candidate)),
              sram32_logits=read(root / 'sram32-logits.json'),
              pipelines={label: pipeline(label) for label in ['pipeline-baseline', 'pipeline-sram32']},
              attention_storage=read(root / 'attention-storage.json'), historical_quality_comparison=quality,
              quality_unload=read(root / 'quality-rollback/unload-series.json'),
              visual_review='Restored teapot has one side handle. Poster retains MOSS and three yellow flowers. Portrait retains red scarf and white cup. Historical cross-run caption/pixel differences remain unlocalized.',
              deployment=read(root / 'deployment/deployment.json'), runtime=read(root / 'runtime-verification.json'),
              source_sha256={name: hashlib.sha256((checkout / name).read_bytes()).hexdigest() for name in sources},
              native_extension_removed=not (checkout / 'comfy/text_encoders/hpu_gdn').exists()
                                       and not (checkout / 'build/hpu').exists()
                                       and not (checkout / 'scripts/gaudi/build_gdn.py').exists())
(root / 'summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps({label: {key: data[key] for key in ['median', 'mean', 'warm_seconds']}
                  for label, data in result['pipelines'].items()}, indent=2))
