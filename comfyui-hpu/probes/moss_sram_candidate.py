"""Install one process-local SRAM-budget experiment before model execution."""
import ctypes
import os

import torch


temporary = torch.empty(1, device='hpu')
del temporary
api = ctypes.CDLL('/usr/lib/habanalabs/libSynapse.so')
api.synConfigurationSet.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
api.synConfigurationGet.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_uint64]
for name, value in [('ENABLE_EXPERIMENTAL_FLAGS', '1'),
                    ('SRAM_SLICER_MAX_CAPACITY_BYTES', str(int(os.environ.get('MOSS_SRAM_MIB', '32')) * 1024**2))]:
    assert api.synConfigurationSet(name.encode(), value.encode()) == 0
    buffer = ctypes.create_string_buffer(1024)
    assert api.synConfigurationGet(name.encode(), buffer, 1024) == 0
    assert buffer.value.decode() == value
