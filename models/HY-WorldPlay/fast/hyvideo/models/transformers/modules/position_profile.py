import os
import atexit
import json
from collections import defaultdict
from pathlib import Path

import torch


_PROFILE = defaultdict(lambda: {"count": 0, "total_ms": 0.0})
_DETAIL = []
_STACK = []


def _write_detail():
    path = os.environ.get("HY_POSITION_ENCODING_DETAIL_PATH")
    if path and _DETAIL:
        Path(path).write_text(json.dumps({"diagnostic_only": True, "rows": _DETAIL}, indent=2))


atexit.register(_write_detail)


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
    detail = bool(os.environ.get("HY_POSITION_ENCODING_DETAIL_PATH"))
    parent = _STACK[-1] if _STACK else None
    row_id = len(_DETAIL)
    if detail:
        _DETAIL.append({
            "bucket": bucket, "layer": block_idx,
            "stage": os.environ.get("HY_ROLLOUT_DRIFT_STAGE", "other"),
            "chunk": int(os.environ.get("HY_ROLLOUT_DRIFT_CHUNK", "-1")),
            "step": int(os.environ.get("HY_ROLLOUT_DRIFT_STEP", "-1")),
            "parent": parent,
            "shapes": [list(x.shape) for x in args[:3] if isinstance(x, torch.Tensor)],
        })
        _STACK.append(row_id)
    try:
        result = fn(*args, **kwargs)
    finally:
        if detail:
            _STACK.pop()
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)
    record(bucket, block_idx, elapsed)
    if detail:
        _DETAIL[row_id]["ms"] = elapsed
    return result


def snapshot():
    return {key: dict(value) for key, value in _PROFILE.items()}
