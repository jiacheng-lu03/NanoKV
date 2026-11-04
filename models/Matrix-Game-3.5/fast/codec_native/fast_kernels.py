"""SM80 elementwise reader kernels; keep original FP64 rotary arithmetic."""
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

@triton.jit
def _rope(X,F,I,Y,N:tl.constexpr,H:tl.constexpr,S:tl.constexpr,D:tl.constexpr,
          XB:tl.constexpr,XH:tl.constexpr,XS:tl.constexpr,XD:tl.constexpr,
          FS:tl.constexpr,FP:tl.constexpr,IH:tl.constexpr,ID:tl.constexpr,
          BLOCK:tl.constexpr):
    t=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);ok=t<N
    p=t%(D//2);s=(t//(D//2))%S;h=(t//(D//2)//S)%H;b=t//(D//2)//S//H
    orig=tl.load(I+h*IH+p*ID,ok,0)
    real=tl.load(F+s*FS+orig*FP,ok,0).to(tl.float64)
    imag=tl.load(F+s*FS+orig*FP+1,ok,0).to(tl.float64)
    a=tl.load(X+b*XB+h*XH+s*XS+2*p*XD,ok,0).to(tl.float64)
    z=tl.load(X+b*XB+h*XH+s*XS+(2*p+1)*XD,ok,0).to(tl.float64)
    # PyTorch's double -> BF16 copy rounds through FP32. Preserve that
    # intermediate conversion explicitly rather than relying on lowering.
    tl.store(Y+2*t,(a*real-z*imag).to(tl.float32),ok);tl.store(Y+2*t+1,(a*imag+z*real).to(tl.float32),ok)

def rope(x,freqs,pairs):
    if freqs.ndim==3:freqs=freqs[:,0]
    f=torch.view_as_real(freqs);out=torch.empty(x.shape,device=x.device,dtype=x.dtype)
    n=x.numel()//2
    _rope[(triton.cdiv(n,256),)](x,f,pairs,out,n,x.shape[1],x.shape[2],x.shape[3],*x.stride(),f.stride(0),f.stride(1),*pairs.stride(),256,enable_fp_fusion=False)
    return out

@triton.jit
def _merge(A,B,LA,LB,O,L,N:tl.constexpr,D:tl.constexpr,
           H:tl.constexpr,S:tl.constexpr,AB:tl.constexpr,AH:tl.constexpr,AS:tl.constexpr,AD:tl.constexpr,
           BB:tl.constexpr,BH:tl.constexpr,BS:tl.constexpr,BD:tl.constexpr,
           LAB:tl.constexpr,LAH:tl.constexpr,LAS:tl.constexpr,LBB:tl.constexpr,LBH:tl.constexpr,LBS:tl.constexpr,
           BLOCK:tl.constexpr):
    t=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);ok=t<N
    d=t%D;s=t//D%S;h=t//D//S%H;b=t//D//S//H
    a=tl.load(LA+b*LAB+h*LAH+s*LAS,ok,float('-inf')).to(tl.float32)
    c=tl.load(LB+b*LBB+h*LBH+s*LBS,ok,float('-inf')).to(tl.float32)
    mx=tl.maximum(a,c);delta=tl.abs(a-c)
    l=mx+libdevice.log1p(libdevice.exp(-delta))
    l=tl.where(a==c,a+0.6931471805599453,l)
    origin=tl.where(l!=float('-inf'),l,0.)
    wa=libdevice.exp(a-origin);wb=libdevice.exp(c-origin)
    x=tl.load(A+b*AB+h*AH+s*AS+d*AD,ok,0).to(tl.float32)
    y=tl.load(B+b*BB+h*BH+s*BS+d*BD,ok,0).to(tl.float32)
    tl.store(O+t,x*wa+y*wb,ok)
    tl.store(L+t//D,l,ok&(d==0))

def merge(a,la,b,lb):
    out=torch.empty(a.shape,device=a.device,dtype=a.dtype);lse=torch.empty(la.shape,device=la.device,dtype=la.dtype)
    n=a.numel()
    _merge[(triton.cdiv(n,256),)](a,b,la,lb,out,lse,n,a.shape[-1],a.shape[1],a.shape[2],*a.stride(),*b.stride(),*la.stride(),*lb.stride(),256,enable_fp_fusion=False)
    return out,lse
