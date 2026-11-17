"""Summarize D1 timing, image changes, trajectories, and calibrated HBM bytes."""
import argparse
import collections
import json
from pathlib import Path
import statistics

import numpy as np
from PIL import Image
import torch


def interval_union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def workflow_summary(root):
    rows = json.loads((root / 'results.json').read_text())
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    groups = collections.defaultdict(list)
    for row in rows:
        record = dict(row)
        for kind in ('SamplerCustomAdvanced', 'TextGenerate', 'TextEncodeQwenImage21', 'VAEDecode'):
            stamps = [event['mono_ns'] for event in events if event.get('prompt_id') == row['prompt_id']
                      and event.get('class_type') == kind and event['kind'] in ('node_start', 'node_end')]
            if stamps:
                record[kind] = (stamps[-1] - stamps[0]) / 1e9
        generated = [event for event in events if event.get('prompt_id') == row['prompt_id'] and event['kind'] == 'generate_end']
        if generated:
            record['output_tokens'] = generated[-1]['output_tokens']
        groups[row['candidate']].append(record)
    timings = {}
    for name, group in groups.items():
        timings[name] = {}
        for prefix in ('warm-', 'full-'):
            selected = [row for row in group if row['label'].startswith(prefix)]
            if not selected:
                continue
            fields = {}
            for key in ('seconds', 'SamplerCustomAdvanced', 'TextGenerate', 'TextEncodeQwenImage21', 'VAEDecode'):
                values = [row[key] for row in selected if key in row]
                if values:
                    fields[key] = dict(median=statistics.median(values), minimum=min(values), maximum=max(values), samples=values)
            timings[name][prefix.rstrip('-')] = fields
    image_pairs = []
    histories = {row['label']: json.loads((root / (row['label'] + '.json')).read_text()) for row in rows}

    def pixels(history):
        item = history['outputs']['461']['images'][0]
        path = Path('/home/sonny/ComfyUI-Gaudi/output') / item['subfolder'] / item['filename']
        return np.asarray(Image.open(path).convert('RGB')).astype(np.float64)

    for label, baseline in histories.items():
        if not label.endswith('-baseline'):
            continue
        case = label.removesuffix('-baseline')
        reference = pixels(baseline)
        for variant in groups:
            if variant == 'baseline' or case + '-' + variant not in histories:
                continue
            history = histories[case + '-' + variant]
            difference = pixels(history) - reference
            mse = np.mean(difference ** 2)
            image_pairs.append(dict(case=case, candidate=variant, rgb_mae=float(abs(difference).mean()),
                                    max_abs=float(abs(difference).max()), psnr=float(10 * np.log10(255 ** 2 / mse)) if mse else None,
                                    pixels_equal=not bool(mse), caption_equal=history['outputs'].get('459:472') == baseline['outputs'].get('459:472')))
    trajectories = {}
    trajectory_rows = [row for row in rows if row['label'].startswith('trajectory-')]
    baseline_row = next(row for row in trajectory_rows if row['candidate'] == 'baseline')
    reference = torch.load(root / (baseline_row['prompt_id'] + '-trajectory.pt'), weights_only=True)
    for row in trajectory_rows:
        values = torch.load(root / (row['prompt_id'] + '-trajectory.pt'), weights_only=True)
        trajectories[row['candidate']] = [dict(index=i + 1, rrmse=float((value.float() - ref.float()).norm() / ref.float().norm()),
                                              finite=bool(value.isfinite().all())) for i, (value, ref) in enumerate(zip(values, reference))]
    cache_checks = {}
    repeated_images = {}
    for name in groups:
        if 'full-0-' + name in histories:
            reference = pixels(histories['warm-0-' + name])
            labels = [label for label in histories if label.endswith('-' + name)
                      and label.startswith(('cold-', 'warm-', 'full-'))]
            repeated_images[name] = {label: bool(np.array_equal(reference, pixels(histories[label]))) for label in labels}
            assert all(repeated_images[name].values()), repeated_images[name]
        if 'blue-return-' + name not in histories:
            continue
        blue, red, returned = (histories[label + '-' + name] for label in ('blue', 'red', 'blue-return'))
        check = dict(returned_caption_equal=blue['outputs']['459:472'] == returned['outputs']['459:472'],
                     returned_pixels_equal=bool(np.array_equal(pixels(blue), pixels(returned))),
                     blue_red_caption_differ=blue['outputs']['459:472'] != red['outputs']['459:472'],
                     blue_red_pixels_differ=not bool(np.array_equal(pixels(blue), pixels(red))))
        assert all(check.values()), check
        cache_checks[name] = check
    return dict(timings=timings, requests=dict(groups), images=image_pairs, trajectories=trajectories,
                cache_checks=cache_checks, repeated_images=repeated_images)


def hardware_summary(root):
    result = {}
    for directory in sorted(root.glob('native-*')):
        configuration = json.loads((directory / 'profiler.json').read_text())
        hw = next(plugin for plugin in configuration['Plugins'] if plugin['name'] == 'HwTrace')
        bmon = hw['values']['archProfileUnits']['gaudi2']['HMMU']['DCORE0_HMMU0_CS']['bmonParams']['b10']
        factor = (bmon['capture_window']['value'] + 1) / (bmon['bandwidth_window']['value'] + 1) * 128
        ports = collections.defaultdict(float)
        engines = {}
        starts = collections.defaultdict(list)
        intervals = collections.defaultdict(list)
        operations = collections.defaultdict(list)
        unmatched_ends = 0
        hardware_markers = 0
        for line in (directory / 'native-events.jsonl').open():
            event = json.loads(line)
            if event['category'] == 'BMON' and 'counter_values' in event:
                ports[event['name']] += event['counter_values'][2] * factor
            if event['category'] == '__metadata' and event['name'] == 'process_name':
                for kind in ('MME', 'TPC'):
                    if event['metadata_name'].startswith('*' + kind):
                        engines[event['pid']] = kind
            if event['pid'] not in engines or event['ph'] not in ('B', 'E'):
                continue
            if event['name'] == 'START_EVENT' or event['name'].startswith(('MMEH_WB1_MON_TS_', 'TPC_SPU_START')):
                hardware_markers += 1
                continue
            key = event['pid'], event['tid'], event['name'], event['id']
            if event['ph'] == 'B':
                starts[key].append(event['ts'])
            elif starts[key]:
                interval = starts[key].pop(), event['ts']
                intervals[engines[event['pid']]].append(interval)
                parts = event['name'].split('/')
                operations[parts[1] if len(parts) >= 3 else event['name']].append(interval)
            else:
                unmatched_ends += 1
        read = sum(value for name, value in ports.items() if 'Read' in name)
        write = sum(value for name, value in ports.items() if 'Write' in name)
        plain = json.loads((directory / 'plain-windows.json').read_text())
        seconds = statistics.median(plain)
        profiled = json.loads((directory / 'window.json').read_text())['seconds']
        result[directory.name] = dict(read_bytes=read, write_bytes=write, total_bytes=read + write,
                                      plain_seconds=seconds, plain_samples=plain, profiled_seconds=profiled,
                                      profiler_slowdown=profiled / seconds - 1, ports=dict(ports),
                                      estimated_TBps=(read + write) / seconds / 1e12,
                                      hbm_peak_fraction=(read + write) / seconds / 2.45e12)
        active = interval_union([interval for group in intervals.values() for interval in group])
        span = active[-1][1] - active[0][0]
        result[directory.name]['activity'] = dict(
            device_span_ms=span / 1000, kernel_coverage=sum(end - start for start, end in active) / span,
            engine_union_ms={name: sum(end - start for start, end in interval_union(group)) / 1000
                             for name, group in intervals.items()},
            operation_union_ms={name: sum(end - start for start, end in interval_union(group)) / 1000
                                for name, group in operations.items()},
            unmatched_ends=unmatched_ends, unmatched_starts=sum(len(group) for group in starts.values()),
            excluded_hardware_markers=hardware_markers)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--product-root', type=Path)
    options = parser.parse_args()
    torch.set_num_threads(8)
    result = dict(baseline='a5308be1c7dd56d8ebb694a8a9a630a072089a77',
                  candidate=workflow_summary(options.root / 'workflow'))
    product_root = options.product_root or options.root / 'product'
    if (product_root / 'validated-hashes.json').exists():
        result['product'] = workflow_summary(product_root / 'api')
        result['hardware'] = hardware_summary(product_root / 'hardware')
    (options.root / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({name: value.get('timings', value) if isinstance(value, dict) else value
                      for name, value in result.items()}, indent=2))


if __name__ == '__main__':
    main()
