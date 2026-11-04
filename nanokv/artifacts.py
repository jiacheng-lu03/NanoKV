"""Validate portable packed codecs without depending on training-machine paths."""
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import re

@lru_cache(maxsize=16)
def _verified(path, size, mtime_ns, expected):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=expected:
        raise ValueError(f'Packed codec checksum mismatch: {path}')
    return True

def portable_codec_verified(path):
    expected=os.environ.get('NANOKV_PORTABLE_CODEC_SHA256','').lower()
    if not expected:return False
    if not re.fullmatch('[0-9a-f]{64}',expected):
        raise ValueError('NANOKV_PORTABLE_CODEC_SHA256 must be a SHA256 digest')
    p=Path(path).resolve();stat=p.stat()
    return _verified(str(p),stat.st_size,stat.st_mtime_ns,expected)
