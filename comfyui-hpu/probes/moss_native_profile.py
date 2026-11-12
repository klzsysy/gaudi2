"""Invoke the installed Synapse profiler API with an explicit configuration."""
import ctypes
import json
import os
from pathlib import Path
import struct
import math

import torch


class TraceEvent(ctypes.Structure):
    _fields_=[('name',ctypes.c_char_p),('category',ctypes.c_char_p),('type',ctypes.c_char),
              ('timestamp',ctypes.c_longdouble),('duration',ctypes.c_longdouble),
              ('engine_type',ctypes.c_uint32),('engine_index',ctypes.c_uint32),
              ('context_id',ctypes.c_uint64),('arguments',ctypes.c_ubyte*128)]


class NativeProfile:
    def __init__(self, output):
        self.output=Path(output)
        self.api=ctypes.CDLL('/usr/lib/habanalabs/libSynapse.so')
        self.api.synProfilerQueryRequiredMemory.argtypes=[ctypes.c_uint32,ctypes.POINTER(ctypes.c_uint32)]
        self.api.synProfilerSetUserBuffer.argtypes=[ctypes.c_uint32,ctypes.c_void_p]
        self.api.synProfilerStart.argtypes=[ctypes.c_int,ctypes.c_uint32]
        self.api.synProfilerStop.argtypes=[ctypes.c_int,ctypes.c_uint32]
        self.api.synProfilerGetTrace.argtypes=[ctypes.c_int,ctypes.c_uint32,ctypes.c_int,ctypes.c_void_p,
                                              ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
        self.statuses={}
        self.buffer=None

    def check(self,name,status):
        self.statuses[name]=status
        (self.output/'native-profile-status.json').write_text(json.dumps(dict(pid=os.getpid(),statuses=self.statuses),indent=2))
        if status:
            raise RuntimeError(f'{name} returned Synapse status {status}')

    def __enter__(self):
        required=ctypes.c_uint32()
        self.check('query',self.api.synProfilerQueryRequiredMemory(0,ctypes.byref(required)))
        if required.value:
            self.buffer=torch.empty(required.value,dtype=torch.uint8,device='hpu')
            self.check('buffer',self.api.synProfilerSetUserBuffer(0,ctypes.c_void_p(self.buffer.data_ptr())))
        torch.hpu.synchronize()
        self.check('start',self.api.synProfilerStart(1,0))
        return self

    def __exit__(self,kind,value,traceback):
        torch.hpu.synchronize()
        self.check('stop',self.api.synProfilerStop(1,0))
        size,entries=ctypes.c_size_t(),ctypes.c_size_t()
        self.check('trace_query',self.api.synProfilerGetTrace(3,0,1,None,ctypes.byref(size),ctypes.byref(entries)))
        self.statuses['trace_bytes']=size.value
        self.statuses['trace_entries']=entries.value
        buffer=ctypes.create_string_buffer(size.value)
        self.check('trace_read',self.api.synProfilerGetTrace(3,0,1,buffer,ctypes.byref(size),ctypes.byref(entries)))
        events=ctypes.cast(buffer,ctypes.POINTER(TraceEvent))
        with (self.output/'native-events.jsonl').open('w') as stream:
            for index in range(entries.value):
                event=events[index]
                raw=bytes(event.arguments)
                record=dict(name=event.name.decode() if event.name else '',
                            category=event.category.decode() if event.category else '',
                            ph=event.type.decode(),ts=float(event.timestamp),dur=float(event.duration),
                            pid=event.engine_type,tid=event.engine_index,id=event.context_id)
                if event.category==b'SPMU':
                    record['raw_u64']=struct.unpack_from('Q',raw)[0]
                if event.category==b'BMON BW':
                    record['counter_values']=[v if math.isfinite(v) else None for v in struct.unpack_from('ddd',raw)]
                    record['matches']=struct.unpack_from('H',raw,24)[0]
                if event.type==b'M':
                    value=ctypes.c_char_p.from_buffer_copy(raw[:8]).value
                    record['metadata_name']=value.decode() if value else ''
                stream.write(json.dumps(record)+'\n')
        self.check('export',self.api.synProfilerGetTrace(3,0,1,None,None,None))
