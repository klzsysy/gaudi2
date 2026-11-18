"""Build the isolated stable softmax kernel against the installed Gaudi2 SDK."""
import argparse
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
options = parser.parse_args()
source = Path(__file__).parent
build = options.root / 'rmw/softmax'
build.mkdir(parents=True, exist_ok=True)
headers = options.root / 'native-source/mme-direct-attention/attention/kernels'
assembly = '.section .rodata\n'
for name, defines in [('softmax', []), ('softmax_exp', ['-DQWEN_EXP_BF16']),
                      ('softmax_u4', ['-DQWEN_UNROLL']), ('softmax_exp_u4', ['-DQWEN_EXP_BF16', '-DQWEN_UNROLL']),
                      ('softmax_core', ['-DQWEN_EXP_CORE']), ('softmax_core_u4', ['-DQWEN_EXP_CORE', '-DQWEN_UNROLL'])]:
    command = ['/opt/habanalabs/bin/tpc-clang', '-march=gaudi2', '-O2', '-Wall', '-Werror',
               '-I' + str(headers), *defines, str(source / 'core.c')]
    subprocess.run(command + ['-c', '-o', str(build / (name + '.o'))], check=True)
    subprocess.run(command + ['-S', '-o', str(build / (name + '.s'))], check=True)
    assembly += f'.global {name}_start\n.global {name}_end\n{name}_start:\n.incbin "{build}/{name}.o"\n{name}_end:\n'
(build / 'blob.S').write_text(assembly + '.section .note.GNU-stack,"",@progbits\n')
subprocess.run(['g++', '-fPIC', '-c', str(build / 'blob.S'), '-o', str(build / 'blob.o')], check=True)
subprocess.run(['g++', '-std=c++17', '-shared', '-fPIC', '-O2', '-I/usr/include/habanalabs',
                str(source / 'glue.cpp'), str(build / 'blob.o'), '-o', str(build / 'libqwen_d2_softmax.next.so')], check=True)
os.replace(build / 'libqwen_d2_softmax.next.so', build / 'libqwen_d2_softmax.so')
