"""Build tile-specific immediate-state kernels using the installed eager ABI."""
import os
from pathlib import Path
import subprocess
import habana_frameworks.torch
from torch.utils.cpp_extension import load

source = Path(__file__).parent
root = Path(os.environ['QWEN_IMMEDIATE_ROOT'])
package = Path(habana_frameworks.torch.__file__).parent
for tile in (4, 8, 16):
    build = root / str(tile)
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run(['/opt/habanalabs/bin/tpc-clang', '-march=gaudi2', '-O2', f'-DVALUE_TILE={tile}',
                    '-c', str(source / 'core.c'), '-o', str(build / 'gdn.o')], check=True)
    (build / 'gdn.S').write_text(f'.section .rodata\n.global gdn_start\n.global gdn_end\ngdn_start:\n.incbin "{build}/gdn.o"\ngdn_end:\n.section .note.GNU-stack,"",@progbits\n')
    subprocess.run(['g++', '-fPIC', '-c', str(build / 'gdn.S'), '-o', str(build / 'gdn-blob.o')], check=True)
    subprocess.run(['g++', '-shared', '-fPIC', '-O2', f'-DVALUE_TILE={tile}', '-I/usr/include/habanalabs',
                    str(source / 'glue.cpp'), str(build / 'gdn-blob.o'), '-o', str(build / 'libqwen_gdn_immediate_tpc.so')], check=True)
build = root / 'bridge'
build.mkdir(exist_ok=True)
os.environ['MAX_JOBS'] = '2'
load(name='qwen_gdn_immediate_bridge', sources=[str(source / 'bridge.cpp')], build_directory=str(build),
     extra_include_paths=[str(package / 'include')],
     extra_ldflags=[f'-L{package / "lib"}', '-lhabana_pytorch2_plugin', f'-Wl,-rpath,{package / "lib"}'],
     is_python_module=False, verbose=True)
