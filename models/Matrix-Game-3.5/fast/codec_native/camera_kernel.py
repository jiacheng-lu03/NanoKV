"""Fused MG full-layout camera transform; retain BF16 matrix/input operands."""
import os
import torch
import triton
import triton.language as tl


@triton.jit
def _camera(X,M,C,Y,N:tl.constexpr,H:tl.constexpr,S:tl.constexpr,D:tl.constexpr,F:tl.constexpr,
            XB:tl.constexpr,XH:tl.constexpr,XS:tl.constexpr,XD:tl.constexpr,
            MB:tl.constexpr,MF:tl.constexpr,MC:tl.constexpr,MI:tl.constexpr,MJ:tl.constexpr,
            CH:tl.constexpr,CG:tl.constexpr,BLOCK:tl.constexpr,
            ACC64:tl.constexpr,ORDER:tl.constexpr,TREE:tl.constexpr):
    t=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);ok=t<N
    d=t%D;s=t//D%S;h=t//D//S%H;b=t//D//S//H
    g=d//8;i=d%8//2;r=d%2
    c=tl.load(C+h*CH+g*CG,ok,0)
    mbase=b*MB+(s//(S//F))*MF+c*MC+i*MI
    xbase=b*XB+h*XH+s*XS+(g*8+r)*XD
    if TREE:
        a0=tl.load(M+mbase,ok,0).to(tl.bfloat16).to(tl.float32)
        a1=tl.load(M+mbase+MJ,ok,0).to(tl.bfloat16).to(tl.float32)
        a2=tl.load(M+mbase+2*MJ,ok,0).to(tl.bfloat16).to(tl.float32)
        a3=tl.load(M+mbase+3*MJ,ok,0).to(tl.bfloat16).to(tl.float32)
        x0=tl.load(X+xbase,ok,0).to(tl.float32)
        x1=tl.load(X+xbase+2*XD,ok,0).to(tl.float32)
        x2=tl.load(X+xbase+4*XD,ok,0).to(tl.float32)
        x3=tl.load(X+xbase+6*XD,ok,0).to(tl.float32)
        acc=(a0*x0+a1*x1)+(a2*x2+a3*x3)
    else:
        acc=tl.full((BLOCK,),0,tl.float64 if ACC64 else tl.float32)
        for ix in tl.static_range(4):
            j=ORDER[ix]
            a=tl.load(M+mbase+j*MJ,ok,0).to(tl.bfloat16).to(tl.float32)
            x=tl.load(X+xbase+j*2*XD,ok,0).to(tl.float32)
            if ACC64:acc=acc+a.to(tl.float64)*x.to(tl.float64)
            else:acc=acc+a*x
    tl.store(Y+t,acc.to(tl.float32),ok)


@triton.jit
def _camera_dot(X,M,C,Y,H:tl.constexpr,S:tl.constexpr,D:tl.constexpr,F:tl.constexpr,
                XB:tl.constexpr,XH:tl.constexpr,XS:tl.constexpr,XD:tl.constexpr,
                MB:tl.constexpr,MF:tl.constexpr,MC:tl.constexpr,MI:tl.constexpr,MJ:tl.constexpr,
                CH:tl.constexpr,CG:tl.constexpr,BM:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    group=tl.program_id(1)
    g=group%(D//8);h=group//(D//8)%H;f=group//(D//8)//H%F;b=group//(D//8)//H//F
    token=f*(S//F)+row//2
    r=row%2
    cam=tl.load(C+h*CH+g*CG)
    j=tl.arange(0,16)
    i=tl.arange(0,16)
    x=tl.load(X+b*XB+h*XH+token[:,None]*XS+(g*8+j[None,:]*2+r[:,None])*XD,
              (row[:,None]<2*(S//F))&(j[None,:]<4),0).to(tl.bfloat16)
    matrix=tl.load(M+b*MB+f*MF+cam*MC+i[None,:]*MI+j[:,None]*MJ,
                   (j[:,None]<4)&(i[None,:]<4),0).to(tl.bfloat16)
    out=tl.dot(x,matrix)
    ptr=Y+((b*H+h)*S+token[:,None])*D+g*8+i[None,:]*2+r[:,None]
    tl.store(ptr,out,(row[:,None]<2*(S//F))&(i[None,:]<4))


def camera(x,matrix,cams,*,mode=None):
    if matrix.shape[0] not in (1,x.shape[0]):raise ValueError('camera batch must match features or broadcast')
    if x.shape[2] % matrix.shape[1]:raise ValueError('frames must divide token sequence')
    if mode is None:mode=os.environ.get('MG35_CODEC_NATIVE_CAMERA_ACC','fp64')
    if mode not in ('fp64','fp32','fp32_0132','tree','dot'):raise ValueError(mode)
    ms=list(matrix.stride())
    if matrix.shape[0]==1:ms[0]=0
    out=torch.empty(x.shape,device=x.device,dtype=x.dtype)
    if mode=='dot':
        b,h,s,d=x.shape;f=matrix.shape[1]
        _camera_dot[(triton.cdiv((s//f)*2,32),b*h*f*(d//8))](
            x,matrix,cams,out,h,s,d,f,*x.stride(),*ms,*cams.stride(),32,num_warps=4)
        return out
    order=(0,1,3,2) if mode=='fp32_0132' else (0,1,2,3)
    _camera[(triton.cdiv(x.numel(),256),)](x,matrix,cams,out,x.numel(),*x.shape[1:],matrix.shape[1],*x.stride(),*ms,*cams.stride(),256,mode=='fp64',order,mode=='tree',enable_fp_fusion=False)
    return out


def main():
    from codec_native.core import camera_selected
    torch.manual_seed(31)
    for d in (32,64,128):
        for frames in (1,2,3):
            ix=(torch.randperm(16,device='cuda')[:d//8].sort().values[:,None]*8+torch.arange(8,device='cuda')).flatten().expand(17,-1).contiguous()
            cams=(ix.reshape(17,-1,8)[...,0]//8%4).contiguous()
            x=torch.randn(2,frames*880,17,d,device='cuda',dtype=torch.bfloat16).permute(0,2,1,3)
            m=torch.randn(2,frames,4,4,4,device='cuda')
            a=camera_selected(x,m,ix);b=camera(x,m,cams)
            rel=float((a.float()-b.float()).norm()/a.float().norm())
            print(dict(dim=d,frames=frames,equal=torch.equal(a,b),rel=rel),flush=True)
            assert torch.equal(a,b)


if __name__=='__main__':main()
