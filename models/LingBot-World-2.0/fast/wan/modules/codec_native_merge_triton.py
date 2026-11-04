"""Canonical FP32 logaddexp merge with one BF16 output rounding."""
import torch
import triton
import triton.language as tl
from triton.language.extra import cuda

@triton.jit
def _merge_kernel(A,LA,B,LB,Y,L,D:tl.constexpr,T:tl.constexpr):
    r=tl.program_id(0);j=tl.arange(0,T)
    x=tl.load(LA+r).to(tl.float32);z=tl.load(LB+r).to(tl.float32)
    maximum=tl.maximum(x,z)
    l=maximum+cuda.libdevice.log1p(cuda.libdevice.exp(-tl.abs(x-z)))
    l=tl.where((x==z)&(tl.abs(x)==float('inf')),x,l)
    origin=tl.where(tl.abs(l)!=float('inf'),l,0.)
    wa=cuda.libdevice.exp(x-origin);wb=cuda.libdevice.exp(z-origin)
    av=tl.load(A+r*D+j,j<D,0).to(tl.float32)
    bv=tl.load(B+r*D+j,j<D,0).to(tl.float32)
    tl.store(Y+r*D+j,av*wa+bv*wb,j<D);tl.store(L+r,l)

def merge(a,la,b,lb):
    assert all(x.is_contiguous() and x.is_cuda for x in (a,la,b,lb))
    out=torch.empty_like(a);lse=torch.empty_like(la)
    _merge_kernel[(la.numel(),)](a,la,b,lb,out,lse,a.shape[-1],triton.next_power_of_2(a.shape[-1]),enable_fp_fusion=False)
    return out,lse
