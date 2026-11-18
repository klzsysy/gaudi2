"""Summarize compiled tensor placement and calibrated D2 hardware traces."""
import argparse
import collections
import json
import math
from pathlib import Path


def graph_summary(path):
    graphs = json.loads(path.read_text())['graphs']
    results = []
    for graph in graphs:
        if not any(node.get('engine') == 'MME' for node in graph['nodes']):
            continue
        tensors = {tensor['name']: tensor for tensor in graph['tensors']}
        patterns = collections.Counter()
        score_bytes = collections.Counter()
        bundles = collections.defaultdict(collections.Counter)
        for node in graph['nodes']:
            guid = node['guid']
            if node.get('engine') == 'MME':
                kind = 'MME'
            elif 'softmax' in guid:
                kind = 'softmax'
            else:
                continue
            bundles[node.get('bundle_index', 'none')][kind] += 1
            inputs, outputs = [], []
            for names, target in ((node['input_tensors'], inputs), (node['output_tensors'], outputs)):
                for name in names:
                    tensor = tensors.get(name, {})
                    target.append((tuple(tensor.get('max_shape', [])), tensor.get('dtype'),
                                   tensor.get('allocation')))
            patterns[(guid, tuple(inputs), tuple(outputs))] += 1
            if node.get('engine') == 'MME':
                for name in node['output_tensors']:
                    tensor = tensors[name]
                    shape = tensor['max_shape']
                    if len(shape) >= 2 and shape[0] > 128:
                        score_bytes[tensor['allocation']] += math.prod(shape) * tensor['dtype_bit_size'] // 8
        results.append(dict(name=graph['name'], nodes=len(graph['nodes']),
                            workspace_bytes=graph.get('workspace_size'),
                            score_output_bytes=dict(score_bytes),
                            patterns=[dict(guid=key[0], inputs=key[1], outputs=key[2], count=count)
                                      for key, count in patterns.items()],
                            bundles={str(key): dict(value) for key, value in bundles.items()}))
    return results


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def hardware_summary(directory, iterations):
    configuration = json.loads((directory / 'profiler.json').read_text())
    hw = next(plugin for plugin in configuration['Plugins'] if plugin['name'] == 'HwTrace')
    bmon = hw['values']['archProfileUnits']['gaudi2']['HMMU']['DCORE0_HMMU0_CS']['bmonParams']['b10']
    factor = (bmon['capture_window']['value'] + 1) / (bmon['bandwidth_window']['value'] + 1) * 128
    ports = collections.defaultdict(float)
    engines, starts = {}, collections.defaultdict(list)
    intervals, categories = collections.defaultdict(list), collections.defaultdict(list)
    unmatched_ends = 0
    for line in (directory / 'native-events.jsonl').open():
        event = json.loads(line)
        if event['category'] == 'BMON' and 'counter_values' in event:
            ports[event['name']] += event['counter_values'][2] * factor / iterations
        if event['category'] == '__metadata' and event['name'] == 'process_name':
            for kind in ('MME', 'TPC'):
                if event['metadata_name'].startswith('*' + kind):
                    engines[event['pid']] = kind
        if event['pid'] not in engines or event['ph'] not in ('B', 'E'):
            continue
        name = event['name']
        if name == 'START_EVENT' or name.startswith(('MMEH_WB1_MON_TS_', 'TPC_SPU_START')):
            continue
        key = event['pid'], event['tid'], name, event['id']
        if event['ph'] == 'B':
            starts[key].append(event['ts'])
            continue
        if not starts[key]:
            unmatched_ends += 1
            continue
        interval = starts[key].pop(), event['ts']
        intervals[engines[event['pid']]].append(interval)
        if 'softmax' in name:
            category = 'softmax'
        elif 'sdpa_recomp' in name:
            category = 'sdpa-' + engines[event['pid']]
        elif 'gate_up' in name or 'fp8_gemm' in name:
            category = 'fp8-linear'
        elif '/bmm' in name:
            category = 'math-attention-mme'
        else:
            category = engines[event['pid']] + '-other'
        categories[category].append(interval)
    all_intervals = union([item for group in intervals.values() for item in group])
    span = all_intervals[-1][1] - all_intervals[0][0]
    read = sum(value for name, value in ports.items() if 'Read' in name)
    write = sum(value for name, value in ports.items() if 'Write' in name)
    points = []
    for name, group in categories.items():
        for start, end in union(group):
            points.extend(((start, 1, name), (end, -1, name)))
    active = collections.Counter()
    exclusive = collections.defaultdict(float)
    previous = min(point[0] for point in points)
    for timestamp, delta, name in sorted(points):
        names = tuple(sorted(key for key, count in active.items() if count))
        exclusive[' + '.join(names) or 'gap'] += (timestamp - previous) / 1000 / iterations
        active[name] += delta
        previous = timestamp
    return dict(read_bytes=read, write_bytes=write, total_bytes=read+write, ports=dict(ports),
                device_span_ms=span / 1000 / iterations,
                kernel_coverage=sum(end-start for start, end in all_intervals) / span,
                engine_union_ms={name: sum(end-start for start, end in union(group))/1000/iterations
                                 for name, group in intervals.items()},
                category_union_ms={name: sum(end-start for start, end in union(group))/1000/iterations
                                   for name, group in categories.items()},
                exclusive_ms=dict(exclusive), unmatched_ends=unmatched_ends,
                unmatched_starts=sum(len(group) for group in starts.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--iterations', type=int, default=10)
    options = parser.parse_args()
    results = {}
    for path in sorted(options.root.glob('**/result.json')):
        directory = path.parent
        record = dict(measurement=json.loads(path.read_text()))
        if (directory / 'post').exists():
            record['graphs'] = graph_summary(directory / 'post')
        if (directory / 'native-events.jsonl').exists():
            record['hardware'] = hardware_summary(directory, options.iterations)
            record['hardware']['hbm_peak_fraction'] = (record['hardware']['total_bytes'] /
                                                       record['measurement']['seconds'] / 2.45e12)
        results[str(directory.relative_to(options.root))] = record
    (options.root / 'audit.json').write_text(json.dumps(results, indent=2))
    for name, record in results.items():
        score_bytes = collections.Counter()
        for graph in record.get('graphs', []):
            score_bytes.update(graph['score_output_bytes'])
        print(name, round(record['measurement']['seconds'] * 1000, 4), dict(score_bytes), flush=True)


if __name__ == '__main__':
    main()
