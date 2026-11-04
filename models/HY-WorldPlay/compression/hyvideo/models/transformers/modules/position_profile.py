import os
from collections import defaultdict

import torch


_PROFILE = defaultdict(lambda: {"count": 0, "total_ms": 0.0})


def enabled() -> bool:
    return os.environ.get("HY_POSITION_ENCODING_PROFILE", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def reset():
    _PROFILE.clear()


def record(bucket: str, block_idx, elapsed_ms: float):
    key = f"{bucket}:layer_{block_idx}"
    item = _PROFILE[key]
    item["count"] += 1
    item["total_ms"] += float(elapsed_ms)


def cuda_call(bucket: str, block_idx, fn, *args, **kwargs):
    if not enabled():
        return fn(*args, **kwargs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    record(bucket, block_idx, start.elapsed_time(end))
    return result


def snapshot():
    return {key: dict(value) for key, value in _PROFILE.items()}
