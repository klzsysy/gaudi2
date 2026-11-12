"""Read utilization and the same driver's power counter through hl-thunk."""
import argparse
import ctypes
import ctypes.util
import json
from pathlib import Path
import time


parser=argparse.ArgumentParser()
parser.add_argument('output',type=Path)
parser.add_argument('--seconds',type=int,default=12)
args=parser.parse_args()
library=ctypes.CDLL(ctypes.util.find_library('hl-thunk'))
library.hlthunk_open_control_by_bus_id.argtypes=[ctypes.c_char_p]
library.hlthunk_get_device_utilization.argtypes=[ctypes.c_int,ctypes.c_uint32,ctypes.POINTER(ctypes.c_uint32)]
library.hlthunk_get_power_info.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_uint64)]
library.hlthunk_close.argtypes=[ctypes.c_int]
fd=library.hlthunk_open_control_by_bus_id(b'0000:01:00.0')
assert fd>=0,fd
max_power=int(Path('/sys/class/accel/accel0/device/max_power').read_text())
start=time.monotonic()
try:
    with args.output.open('w') as stream:
        while time.monotonic()-start<args.seconds:
            power1,power2=ctypes.c_uint64(),ctypes.c_uint64()
            utilization=ctypes.c_uint32()
            statuses=[library.hlthunk_get_power_info(fd,ctypes.byref(power1)),
                      library.hlthunk_get_device_utilization(fd,1000,ctypes.byref(utilization)),
                      library.hlthunk_get_power_info(fd,ctypes.byref(power2))]
            assert statuses==[0,0,0],statuses
            expected=[int(100*(max(60000,min(p,max_power))-60000)/(max_power-60000)) for p in (power1.value,power2.value)]
            record=dict(mono_ns=time.perf_counter_ns(),power_mw=[power1.value,power2.value],
                        max_power_mw=max_power,dc_power_mw=60000,utilization=utilization.value,expected=expected)
            stream.write(json.dumps(record)+'\n')
            stream.flush()
            time.sleep(.1)
finally:
    library.hlthunk_close(fd)
