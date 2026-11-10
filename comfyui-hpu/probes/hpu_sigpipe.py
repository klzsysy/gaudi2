"""Reproduce a closed sendfile socket poisoning Synapse, then test Python's policy."""
import json
import os
import signal
import socket
import sys
import tempfile

import torch
import habana_frameworks.torch.core


def signal_state():
    with open('/proc/self/status') as stream:
        rows = dict(line.split(':', 1) for line in stream if ':' in line)
    bit = 1 << (signal.SIGPIPE - 1)
    return {'ignored': bool(int(rows['SigIgn'].strip(), 16) & bit),
            'caught': bool(int(rows['SigCgt'].strip(), 16) & bit)}


print('before_init', json.dumps(signal_state()), flush=True)
torch.ones(1, device='hpu').cpu()
print('after_init', json.dumps(signal_state()), flush=True)
if sys.argv[1] == 'restore':
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
print('before_sendfile', json.dumps(signal_state()), flush=True)
writer, reader = socket.socketpair()
reader.close()
with writer, tempfile.TemporaryFile() as payload:
    payload.write(b'x' * 4096)
    payload.flush()
    try:
        os.sendfile(writer.fileno(), payload.fileno(), 0, 4096)
    except BrokenPipeError:
        print('expected_broken_pipe', flush=True)
try:
    output = (torch.ones(1, device='hpu') + 4).cpu()
    torch.testing.assert_close(output, torch.tensor([5.0]))
    print('HPU_AFTER_DISCONNECT_OK', flush=True)
except RuntimeError as exc:
    print('HPU_AFTER_DISCONNECT_FAILED', str(exc), flush=True)
    raise SystemExit(2)
