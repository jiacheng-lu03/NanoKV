"""Opt-in shape-only accounting; no device reads or synchronizations."""
import atexit
import json
import os
from collections import Counter
from pathlib import Path

_rows = Counter()

def record(layer, runtime, query, current, text, history, low, anchor, recent, chunk):
    path = os.environ.get("HY_DIRECT_CODEC_MAC_AUDIT_PATH")
    if not path:
        return
    groups = runtime.get("grouped_attention")
    if groups:
        dims = tuple((int(g["head_indices"].numel()), int(g["dim"])) for g in groups)
    else:
        dims = ((16, int(runtime["shape"][1])),)
    _rows[(int(layer), int(chunk), query, current, text, history, low, anchor, recent, dims)] += 1

def finish():
    path = os.environ.get("HY_DIRECT_CODEC_MAC_AUDIT_PATH")
    if not path or not _rows:
        return
    data = []
    for (layer, chunk, q, cur, text, hist, low, anchor, recent, dims), count in sorted(_rows.items()):
        # Per batch, per Main/PRoPE branch. QK and AV share D in this policy.
        baseline = 2*q*(hist+cur)*16*128
        saved = 2*q*low*sum(n*(128-d) for n,d in dims)
        data.append(dict(layer=layer, chunk=chunk, calls=count, query_tokens=q,
                         current_tokens=cur, text_tokens=text, history_tokens=hist,
                         low_tokens=low, anchor_tokens=anchor, recent_tokens=recent,
                         head_dims=dims, visual_baseline_mac=baseline,
                         visual_saved_mac=saved,
                         with_text_baseline_mac=baseline+2*q*text*16*128))
    Path(path).write_text(json.dumps(dict(
        note="Selected-layer denoise calls only; add native fallback layers and first/no-history chunks to the denominator. Excludes projection, softmax, encoding/recompute.",
        rows=data), indent=2)+"\n")

atexit.register(finish)
