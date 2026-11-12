"""Build the isolated recurrent-state TPC experiment."""
import os
from pathlib import Path
import subprocess
import habana_frameworks.torch
from torch.utils.cpp_extension import load

source=Path(__file__).parent
build=Path(os.environ['MOSS_OPT_ROOT'])/'gdn-build'
build.mkdir(parents=True,exist_ok=True)
subprocess.run(['/opt/habanalabs/bin/tpc-clang','-march=gaudi2','-O2','-c',str(source/'core.c'),'-o',str(build/'gdn.o')],check=True)
(build/'gdn.S').write_text(f'.section .rodata\n.global gdn_start\n.global gdn_end\ngdn_start:\n.incbin "{build}/gdn.o"\ngdn_end:\n.section .note.GNU-stack,"",@progbits\n')
subprocess.run(['g++','-fPIC','-c',str(build/'gdn.S'),'-o',str(build/'gdn-blob.o')],check=True)
subprocess.run(['g++','-shared','-fPIC','-O2','-I/usr/include/habanalabs',str(source/'glue.cpp'),str(build/'gdn-blob.o'),'-o',str(build/'libmoss_gdn_tpc.so')],check=True)
package=Path(habana_frameworks.torch.__file__).parent
os.environ['MAX_JOBS']='2'
load(name='moss_gdn_bridge',sources=[str(source/'bridge.cpp')],build_directory=str(build),
     extra_include_paths=[str(package/'include')],extra_ldflags=[f'-L{package/"lib"}','-lhabana_pytorch2_plugin',f'-Wl,-rpath,{package/"lib"}'],
     is_python_module=False,verbose=True)
