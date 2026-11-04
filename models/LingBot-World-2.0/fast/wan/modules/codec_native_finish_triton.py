"""Fuse centered-value correction, canonical LSE merge and head scatter.

No change to the GEMMs, FA kernels, RoPE, or reduction used for value means.
Keep FP32 operation order and the final single BF16 rounding of the reference.
"""
import triton
import triton.language as tl
from triton.language.extra import cuda

@triton.jit
def _finish(A,LA,B,LB,M,P,HEAD,Y,
            AS0:tl.constexpr,AS1:tl.constexpr,AS2:tl.constexpr,AS3:tl.constexpr,
            LS0:tl.constexpr,LS1:tl.constexpr,LS2:tl.constexpr,
            BS0:tl.constexpr,BS1:tl.constexpr,BS2:tl.constexpr,BS3:tl.constexpr,
            TS0:tl.constexpr,TS1:tl.constexpr,TS2:tl.constexpr,
            MS0:tl.constexpr,MS1:tl.constexpr,MS3:tl.constexpr,
            PS0:tl.constexpr,PS1:tl.constexpr,PS3:tl.constexpr,
            YS0:tl.constexpr,YS1:tl.constexpr,YS2:tl.constexpr,YS3:tl.constexpr,
            H:tl.constexpr,S:tl.constexpr,D:tl.constexpr,CORRECT:tl.constexpr,T:tl.constexpr):
    row=tl.program_id(0); s=row%S; h=(row//S)%H; b=row//(S*H)
    head=tl.load(HEAD+h);j=tl.arange(0,T)
    x=tl.load(LA+b*LS0+head*LS1+s*LS2).to(tl.float32)
    z=tl.load(LB+b*TS0+h*TS1+s*TS2).to(tl.float32)
    l=tl.maximum(x,z)+cuda.libdevice.log1p(cuda.libdevice.exp(-tl.abs(x-z)))
    l=tl.where((x==z)&(tl.abs(x)==float('inf')),x,l)
    origin=tl.where(tl.abs(l)!=float('inf'),l,0.)
    av=tl.load(A+b*AS0+head*AS1+s*AS2+j*AS3,j<D,0).to(tl.float32)
    bv=tl.load(B+b*BS0+h*BS1+s*BS2+j*BS3,j<D,0).to(tl.float32)
    if CORRECT:
        mean=tl.load(M+b*MS0+head*MS1+j*MS3,j<D,0).to(tl.float32)
        projected=tl.load(P+b*PS0+h*PS1+j*PS3,j<D,0).to(tl.float32)
        bv=(bv+mean)-projected
    value=av*cuda.libdevice.exp(x-origin)+bv*cuda.libdevice.exp(z-origin)
    tl.store(Y+b*YS0+head*YS1+s*YS2+j*YS3,value,j<D)

def finish(native,native_lse,lifted,low_lse,heads,result,mean=None,projected=None):
    assert native.is_cuda and lifted.is_cuda and result.dtype==native.dtype
    correct=mean is not None
    assert correct==(projected is not None)
    m=mean if correct else lifted;p=projected if correct else lifted
    b,h,s,d=lifted.shape
    _finish[(b*h*s,)](native,native_lse,lifted,low_lse,m,p,heads,result,
        *native.stride(),*native_lse.stride(),*lifted.stride(),*low_lse.stride(),
        m.stride(0),m.stride(1),m.stride(3),p.stride(0),p.stride(1),p.stride(3),
        *result.stride(),h,s,d,correct,triton.next_power_of_2(d),enable_fp_fusion=False)
    return result
