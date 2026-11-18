"""Adapt the pinned four-way runtime to Qwen's shorter one/two-way KV tiles."""
import argparse
import difflib
from pathlib import Path
import shutil
import subprocess
import sys


NORMALIZE = r'''// SPDX-License-Identifier: MIT
// Qwen D2, merge unnormalized numerators using the shared fixed exp shift.
#ifndef QWEN_KV_PARTS
#define QWEN_KV_PARTS 4
#endif
#ifndef QUAD_ROWS
#define QUAD_ROWS 1024
#endif
void main(tensor partials, tensor stats, tensor output) {
    int5 start = get_index_space_offset(), end = start + get_index_space_size();
    int queries = get_dim_size(partials, 1) / QWEN_KV_PARTS;
    for (int row = start[1]; row < end[1]; ++row) {
        int stat_row = (row / QUAD_ROWS) * (QWEN_KV_PARTS * QUAD_ROWS) + row % QUAD_ROWS;
        float64 total = 0.0f;
        float128 numerator;
        numerator.v1 = 0.0f;
        numerator.v2 = 0.0f;
        for (int part = 0; part < QWEN_KV_PARTS; ++part) {
            float64 stat = v_f32_ld_tnsr_b((int5){0, stat_row + part * QUAD_ROWS, 0, 0, 0}, stats);
            total += v_broadcast_element_f32(stat, 1);
            bfloat128 value = v_bf16_ld_tnsr_b((int5){0, row + part * queries, 0, 0, 0}, partials);
            float128 term = v_convert_bf16_to_f32_all_b(value);
            numerator.v1 += term.v1;
            numerator.v2 += term.v2;
        }
        total = v_f32_max_b(total, 1e-30f);
        int64 bits = 0x7ef311c3 - *((int64 *)&total);
        float64 inverse = *((float64 *)&bits);
        inverse *= 2.0f - total * inverse;
        inverse *= 2.0f - total * inverse;
        inverse *= 2.0f - total * inverse;
        numerator.v1 *= inverse;
        numerator.v2 *= inverse;
        v_bf16_st_tnsr((int5){0, row, 0, 0, 0}, output, v_convert_f32_to_bf16_all_b(numerator));
    }
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--parts', type=int, choices=(1, 2), required=True)
    options = parser.parse_args()
    original = options.root / 'native-source/mme-direct-attention'
    target = options.root / f'native-parts{options.parts}/mme-direct-attention'
    shutil.copytree(original, target, ignore=shutil.ignore_patterns('build'), dirs_exist_ok=True)
    source = target / 'attention'
    path = source / 'runtime_quad.hpp'
    before = path.read_text()
    after = before.replace('struct QuadPlan {',
                           'static constexpr unsigned quad_parts = QWEN_KV_PARTS;\nstruct QuadPlan {')
    replacements = {
        '(keys+4*quad_key_quantum-1)/(4*quad_key_quantum)':
            '(keys+quad_parts*quad_key_quantum-1)/(quad_parts*quad_key_quantum)',
        'take(4 * n * matrix)': 'take(quad_parts * n * matrix)',
        'p.keys - 3 * first': 'p.keys - (quad_parts - 1) * first',
        'valid <= 3 * first': 'valid <= (quad_parts - 1) * first',
        'b % 4, slot = part == 3': 'b % quad_parts, slot = part == quad_parts - 1',
        'valid - 3 * first': 'valid - (quad_parts - 1) * first',
        '128 * n * factor': '128 * n * factor * 4 / quad_parts',
        'quad_plans[0].n != n/(quad_query_rows/512)':
            'quad_plans[0].n != n/(quad_query_rows/512)*quad_parts/4',
        'valid <= 3 * (quad_first_width(keys))': 'valid <= (quad_parts - 1) * (quad_first_width(keys))',
        'n/=factor;': 'n = n / factor * quad_parts / 4;',
        '(quad_query_rows==1024?24u:16u)': '(quad_query_rows==1024?64u:16u)',
        '(b / 4) * 4 * matrix * factor': '(b / quad_parts) * 4 * matrix * factor',
        'part * n * matrix * factor': 'part * n * matrix * factor * 4 / quad_parts',
        'b % 4 == 3 ? 1 : 0': 'b % quad_parts == quad_parts - 1 ? 1 : 0',
    }
    for old, new in replacements.items():
        assert old in after, old
        after = after.replace(old, new)
    path.write_text(after)
    patches = [''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                            fromfile='a/attention/runtime_quad.hpp', tofile='b/attention/runtime_quad.hpp'))]
    path = source / 'h3_direct_attention.py'
    before = path.read_text()
    after = before.replace('PARTS, QUANTUM = 4, 128', f'PARTS, QUANTUM = {options.parts}, 128')
    after = after.replace('keys - 3 * self.first', 'keys - (PARTS - 1) * self.first')
    path.write_text(after)
    patches.append(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                              fromfile='a/attention/h3_direct_attention.py', tofile='b/attention/h3_direct_attention.py')))
    path = source / 'kernels/normalize_quad_fixed.c'
    before = path.read_text()
    path.write_text(NORMALIZE)
    patches.append(''.join(difflib.unified_diff(before.splitlines(True), NORMALIZE.splitlines(True),
                                              fromfile='a/attention/kernels/normalize_quad_fixed.c',
                                              tofile='b/attention/kernels/normalize_quad_fixed.c')))
    (target / 'qwen-parts.patch').write_text('\n'.join(patches))
    subprocess.run([sys.executable, str(Path(__file__).with_name('qwen_d2_native_build.py')),
                    str(options.root), '--source', str(source), '--parts', str(options.parts)], check=True)


if __name__ == '__main__':
    main()
