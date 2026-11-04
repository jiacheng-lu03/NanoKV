"""Shared reader artifact utilities."""
import hashlib
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OP=Path(os.environ.get('NANOKV_CODEC', 'artifacts/mg/codec.pt'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()
