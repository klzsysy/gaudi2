"""Run independent device profiles and interleaved workflow comparisons on the single HPU."""
import argparse
import json
import os
from pathlib import Path

import qwen_d1_hardware
import qwen_d1_validation
from qwen_d1_summary import hardware_summary, workflow_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('caption_history', type=Path)
    parser.add_argument('phase', choices=('hardware', 'api'))
    args = parser.parse_args()
    env = dict(os.environ, QWEN_D1_PRODUCT='1', QWEN_VALIDATION_SERVER='qwen_d2_plan_b_server')
    directory = args.root / args.phase
    if args.phase == 'hardware':
        qwen_d1_hardware.measure(directory, env, json.loads(args.caption_history.read_text()))
        summary = hardware_summary(directory)
        for name, record in summary.items():
            events = (directory / name / 'events.jsonl').read_text().splitlines()
            stats = [json.loads(line)['memory_stats'] for line in events if '"memory_stats"' in line]
            record['memory_stats'] = stats[-1]
    else:
        directory.mkdir()
        qwen_d1_validation.workflows(directory, env, ['product'])
        summary = workflow_summary(directory)
    (args.root / (args.phase + '-summary.json')).write_text(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
