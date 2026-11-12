"""Build isolated TPC/HPU extension artifacts for this investigation."""
import os
from pathlib import Path
import subprocess

import habana_frameworks.torch
from torch.utils.cpp_extension import load


source = Path(__file__).parent
build = Path(os.environ['MOSS_OPT_ROOT']) / 'native-build'
build.mkdir(parents=True, exist_ok=True)
for name, flags in [('situ', []), ('situ_quant', ['-DQUANTIZE'])]:
    if os.environ.get('MOSS_TPC_PADE') == '1':
        flags = flags + ['-DPADE']
    subprocess.run(['/opt/habanalabs/bin/tpc-clang', '-march=gaudi2', '-O2', '-c',
                    *flags, str(source/'situ.c'), '-o', str(build/f'{name}.o')], check=True)
    assembly = build/f'{name}.S'
    assembly.write_text(f'.section .rodata\n.global {name}_start\n.global {name}_end\n{name}_start:\n'
                        f'.incbin "{build/name}.o"\n{name}_end:\n.section .note.GNU-stack,"",@progbits\n')
    subprocess.run(['g++', '-fPIC', '-c', str(assembly), '-o', str(build/f'{name}-blob.o')], check=True)
subprocess.run(['g++', '-shared', '-fPIC', '-O2', '-I/usr/include/habanalabs', str(source/'glue.cpp'),
                str(build/'situ-blob.o'), str(build/'situ_quant-blob.o'), '-o', str(build/'libmoss_situ_tpc.so')], check=True)
package = Path(habana_frameworks.torch.__file__).parent
os.environ['MAX_JOBS'] = '2'
load(name='moss_situ_bridge', sources=[str(source/'bridge.cpp')], build_directory=str(build),
     extra_include_paths=[str(package/'include')],
     extra_ldflags=[f'-L{package/"lib"}', '-lhabana_pytorch2_plugin', f'-Wl,-rpath,{package/"lib"}'],
     is_python_module=False, verbose=True)
print(build, flush=True)
