"""Selected native 3D RoPE without materializing complex multipliers.

FP64 arithmetic matches the original complex-double path before BF16 rounding.
No frequency rescaling and no replacement of the retained original pair indices.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _rotate(X,P,F,I,G,Y,N:tl.constexpr,H:tl.constexpr,S:tl.constexpr,D:tl.constexpr,
            FRAME:tl.constexpr,FHALF:tl.constexpr,XS0:tl.constexpr,XS1:tl.constexpr,
            XS2:tl.constexpr,XS3:tl.constexpr,IS0:tl.constexpr,IS1:tl.constexpr,
            FS0:tl.constexpr,FS1:tl.constexpr,PS:tl.constexpr,WIDTH:tl.constexpr,BLOCK:tl.constexpr):
 j=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);ok=j<N
 pair=j%(D//2);t=j//(D//2);s=t%S;t=t//S;h=t%H;b=t//H
 idx=tl.load(I+h*IS0+2*pair*IS1,ok,0)//2
 pos=tl.load(P+s*PS,ok,0)
 if WIDTH > 0:
  w=WIDTH
 else:
  w=tl.load(G+2)
 split=FHALF//3;temporal=FHALF-2*split
 coordinate=tl.where(idx<temporal,pos//FRAME,tl.where(idx<temporal+split,(pos%FRAME)//w,(pos%FRAME)%w))
 fc=coordinate*FS0+idx*FS1
 c=tl.load(F+fc,ok,0).to(tl.float64);sn=tl.load(F+fc+1,ok,0).to(tl.float64)
 base=b*XS0+h*XS1+s*XS2+2*pair*XS3
 a=tl.load(X+base,ok,0).to(tl.float64);v=tl.load(X+base+XS3,ok,0).to(tl.float64)
 tl.store(Y+2*j,(a*c-v*sn).to(tl.float32),ok);tl.store(Y+2*j+1,(a*sn+v*c).to(tl.float32),ok)

def rope_selected(value,positions,grid_sizes,freqs,frame_seqlen,indices):
 assert value.is_cuda and value.shape[-1]%2==0
 f=torch.view_as_real(freqs)
 assert f.stride(-1)==1 and f.dtype==torch.float64
 # LingBot keeps grid metadata on CPU. Copying three integers to CUDA here
 # synchronizes the stream thousands of times per video. Pass the host width
 # as a launch constant; retain device loading for CUDA metadata inputs.
 if grid_sizes.device.type == 'cpu':
  width=int(grid_sizes.reshape(-1,3)[0,2])
  grid=value
 else:
  width=0
  grid=grid_sizes.contiguous()
 out=torch.empty(value.shape,device=value.device,dtype=value.dtype)
 b,h,s,d=value.shape;n=b*h*s*(d//2)
 _rotate[(triton.cdiv(n,256),)](value,positions,f,indices,grid,out,n,h,s,d,int(frame_seqlen),freqs.shape[1],*value.stride(),*indices.stride(),f.stride(0),f.stride(1),positions.stride(0),width,256,enable_fp_fusion=False)
 return out
