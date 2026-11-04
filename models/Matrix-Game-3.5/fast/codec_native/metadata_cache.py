"""Bounded positional-metadata cache, never a K/V cache sidecar."""
from collections import OrderedDict
from functools import lru_cache
from itertools import groupby
import torch

VIEWS=OrderedDict()
ROUTES=OrderedDict()
MASKS=OrderedDict()


@lru_cache(maxsize=64)
def expanded(frames,tokens):
    return [int(f) for f in frames for _ in range(tokens//len(frames))] if frames else []


def views(camera_info,frames):
    matrices=camera_info[1]
    try:
        identity=tuple((id(m),m._version) for m in matrices)
    except RuntimeError:
        return None  # Do not cache versionless inference-mode tensors.
    key=(identity,tuple(frames))
    if key in VIEWS:
        VIEWS.move_to_end(key)
        return VIEWS[key][1]
    runs=[(f,sum(1 for _ in values)) for f,values in groupby(frames)]
    selected=[f for f,_ in runs] if runs and len({n for _,n in runs})==1 else frames
    index=torch.as_tensor(selected,device=matrices[0].device,dtype=torch.long)
    result=tuple(m.index_select(1,index) for m in matrices)
    # Strong references prevent identity recycling; _version invalidates edits.
    VIEWS[key]=(matrices,result)
    while len(VIEWS)>16:VIEWS.popitem(last=False)
    return result


def tensors():
    for sources,results in VIEWS.values():
        yield from sources
        yield from results
    for value in ROUTES.values():
        yield value['protected_index']
    for sources, visible in MASKS.values():
        if sources is not None: yield sources
        if visible is not None: yield visible


def route(frames, chunk_ids, read_chunk, tokens, recent, stride, sink, native, device):
    """Frame metadata are already CPU values: do not recover them from CUDA masks."""
    key=(tuple(frames),tuple(chunk_ids),read_chunk,tokens,recent,stride,sink,native,str(device))
    if key in ROUTES:
        ROUTES.move_to_end(key)
        return ROUTES[key]
    if tokens and (not frames or tokens % len(frames)):
        raise ValueError('Cache metadata must contain complete frames')
    per=tokens//len(frames) if frames else 0
    active=[True]*len(frames)
    if read_chunk is not None and len(chunk_ids)==len(frames):
        active=[int(c)<0 or int(c)==int(read_chunk) for c in chunk_ids]
    old=max(0,len(frames)-recent)
    selected=set(range(old,len(frames)))|set(range(min(sink,len(frames))))
    if stride and old:selected.update(range(old-1,-1,-stride))
    protected=[a and (native or i in selected) for i,a in enumerate(active)]
    protect_cpu=torch.tensor(protected,dtype=torch.bool).repeat_interleave(per)
    low_cpu=torch.tensor([a and not p for a,p in zip(active,protected)],dtype=torch.bool).repeat_interleave(per)
    index=protect_cpu.nonzero(as_tuple=False).flatten().to(device)
    value=dict(protect_cpu=protect_cpu,low_cpu=low_cpu,protected_index=index,
        protected_frames=[int(f) for f,p in zip(frames,protected) if p for _ in range(per)],
        active_count=sum(active)*per,protected_count=sum(protected)*per)
    ROUTES[key]=value
    while len(ROUTES)>8:ROUTES.popitem(last=False)
    return value


def uniform_visible(hole, prefix, mosaic, current, device):
    """Cache only shared immutable visibility metadata, never feature tensors."""
    if hole is None:return None
    try:identity=(id(hole),hole._version)
    except RuntimeError:identity=None
    key=(identity,prefix,mosaic,current,str(device))
    if identity is not None and key in MASKS:
        MASKS.move_to_end(key)
        return MASKS[key][1]
    if bool(hole.all()):visible=None
    else:
        visible=torch.cat((torch.arange(prefix,device=device),
            prefix+hole.to(device).nonzero(as_tuple=False).flatten(),
            torch.arange(prefix+mosaic,prefix+mosaic+current,device=device)))
    if identity is not None:
        MASKS[key]=(hole,visible)
        while len(MASKS)>8:MASKS.popitem(last=False)
    return visible


def clear():
    VIEWS.clear(); ROUTES.clear(); MASKS.clear(); expanded.cache_clear()


def history_counts(frames,chunk_ids,read_chunk,tokens,recent,stride,sink,native=False):
    if not frames:return 0,0
    assert tokens%len(frames)==0
    active=[True]*len(frames)
    if read_chunk is not None and len(chunk_ids)==len(frames):
        active=[int(c)<0 or int(c)==int(read_chunk) for c in chunk_ids]
    old=max(0,len(frames)-recent)
    protected=set(range(old,len(frames)))|set(range(min(sink,len(frames))))
    if stride and old:protected.update(range(old-1,-1,-stride))
    per=tokens//len(frames)
    a=sum(active)*per
    p=a if native else sum(v and i in protected for i,v in enumerate(active))*per
    return a,p
