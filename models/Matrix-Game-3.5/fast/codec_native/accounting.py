"""Shape-based dense QK+AV MAC accounting, not measured latency."""
import atexit,json,os
from pathlib import Path
_ROWS={}
def record(layer,q,k,low,heads,dim,low_dim_sum,write_cache,batch=1,history=0):
    if not os.environ.get('MG35_CODEC_NATIVE_MAC_JSON'):return
    key=f'{layer}:{"rebuild" if write_cache else "denoise"}'
    row=_ROWS.setdefault(key,dict(layer=int(layer),rebuild=bool(write_cache),calls=0,baseline=0,candidate=0,history_low_tokens=0,history_baseline=0))
    base=2*batch*int(q)*int(k)*heads*dim
    saved=2*batch*int(q)*int(low)*(heads*dim-int(low_dim_sum))
    row['calls']+=1;row['baseline']+=base;row['candidate']+=base-saved;row['history_low_tokens']+=int(low)
    row['history_baseline']+=2*batch*int(q)*int(history)*heads*dim
@atexit.register
def save():
    dest=os.environ.get('MG35_CODEC_NATIVE_MAC_JSON')
    if not dest or not _ROWS:return
    rows=list(_ROWS.values());base=sum(r['baseline'] for r in rows);candidate=sum(r['candidate'] for r in rows)
    history=sum(r['history_baseline'] for r in rows)
    # For history-only policies, (base-candidate)/history is the historical
    # reduction. Native-extension policies must not use this denominator.
    Path(dest).write_text(json.dumps(dict(scope='all self-attention layers, dense QK+AV, includes current/mosaic/rebuild; excludes cross-attention',baseline=base,candidate=candidate,reduction=1-candidate/base,history_baseline=history,history_fraction=history/base,history_only_reduction_if_no_native_projection=(base-candidate)/history if history else 0,rows=rows),indent=2))
