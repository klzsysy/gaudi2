"""Build the isolated reference runtime against local SDK headers and MME dumps."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--parts', type=int, default=4)
    options = parser.parse_args()
    source = options.source or options.root / 'native-source/mme-direct-attention/attention'
    build = source / 'build'
    build.mkdir(exist_ok=True)
    sys.path.insert(0, str(source))
    import mme_dump
    descriptors = options.root / 'descriptors-v2'
    records = {}
    directories = sorted(descriptors.glob('g-*')) + sorted((options.root / 'descriptors-wide').glob('g-*'))
    for directory in directories:
        result = json.loads((directory / 'descriptor-result.json').read_text())
        assert result['equal']
        target = source / 'fixtures' / directory.name / 'mme_debug'
        shutil.copytree(directory / 'mme_debug', target, dirs_exist_ok=True)
        converted = mme_dump.load_set(target)
        assert converted
        records[directory.name] = dict(result=result, descriptors=len(converted))
        if result['shape'] == [128, 128, 128]:
            words = mme_dump.load_set(target, keep_slave=False)[0][2]
            header = '/* Local Synapse 1.24.2 GEMM128 dump. RATE_LIMITER uses reference 0x040404. */\n'
            header += 'static const uint32_t desc_words[182] = {\n'
            header += '\n'.join(f'    0x{word:08x},' for word in words) + '\n};\n'
            (source.parent / 'desc128.h').write_text(header)
        elif result['shape'][2] == 128:
            records[directory.name]['coalesced_descriptors'] = len(mme_dump.coalesce_qk(converted))
            mme_dump.repeat_rows_512(converted)
    commands = [[
        'g++', '-std=gnu++17', '-O2', '-Wall', '-Wextra', '-Werror', '-fPIC', '-shared',
        f'-DQWEN_KV_PARTS={options.parts}',
        '-I/usr/include/habanalabs', '-I/usr/include/drm',
        '-I/opt/habanalabs/src/hl-thunk/include/uapi', '-I/opt/habanalabs/src/hl-thunk/include/specs',
        str(source / 'context_bridge.cpp'), '-L/usr/lib/habanalabs', '-Wl,-rpath,/usr/lib/habanalabs',
        '-lSynapse', '-lscal', '-lhl-thunk', '-o', str(build / 'libcontext_bridge.so')]]
    kernels = [
        ('relu', 'relu', []),
        ('normalize_quad_fixed1024', 'normalize_quad_fixed', ['-DQUAD_ROWS=1024', f'-DQWEN_KV_PARTS={options.parts}']),
        ('exp_f16_tile_fixed0_pow2q_g15pad', 'exp_f16_logq', [
            '-DFIXED0_MULTIPLIER=261.25f', '-DMAX_STRIDE=8', '-DF16_AFFINE', '-DF16_PIPE2',
            '-DF16_TAIL256', '-DSINGLE_TILE', '-DF16_SKEW', '-DF16_IDX', '-DF16_FIXED0',
            '-DF16_FIXED0_G8', '-DF16_FIXED0_G15', '-DF16_FIXED0_PAD']),
    ]
    for name, filename, defines in kernels:
        commands.append(['/opt/habanalabs/bin/tpc-clang', '-O2', '-Wall', '-Werror', '-march=gaudi2',
                         *defines, str(source / 'kernels' / (filename + '.c')), '-c', '-o', str(build / (name + '.o'))])
        commands.append([sys.executable, str(source / 'extract_text.py'),
                         str(build / (name + '.o')), str(build / (name + '.bin'))])
    lut = source / 'third_party/synapseai-core/lut'
    commands.append(['g++', '-std=gnu++17', '-O2', '-I' + str(lut), str(source / 'gen_lut.cpp'),
                     str(lut / 'SpecialFuncCoefficientsGen6.cpp'), '-o', str(build / 'gen_lut')])
    commands.append([str(build / 'gen_lut'), str(build / 'lut_gen6_logq2_shift8.bin'), '--logq2-shift8'])
    for command in commands:
        print('build', command[0], command[-1], flush=True)
        subprocess.run(command, check=True)
    records['artifacts'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in build.iterdir() if path.is_file()}
    (source / 'native-build.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
