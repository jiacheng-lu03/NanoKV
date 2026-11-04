"""Fuse selected-row INT4 unpack and BF16 token-scale application."""
import torch
import triton
import triton.language as tl


@triton.jit
def _read(P,S,I,Y,N:tl.constexpr,R:tl.constexpr,NS:tl.constexpr,NK:tl.constexpr,SELECT:tl.constexpr,BLOCK:tl.constexpr):
    t=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);ok=t<N
    r=t%R;s=t//R%NK;b=t//R//NK
    if SELECT:s=tl.load(I+s,ok,0)
    row=b*NS+s;source=row*R+r
    byte=tl.load(P+source//2,ok,0).to(tl.int32)
    code=((byte>>((source%2)*4))&15)-7
    scale=tl.load(S+row,ok,0).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+t,code.to(tl.float32)*scale,ok)


def read(packed,scales,batch,sequence,rank,index=None):
    count=sequence if index is None else index.numel()
    out=torch.empty((batch,count,rank),device=packed.device,dtype=torch.bfloat16)
    _read[(triton.cdiv(out.numel(),256),)](packed,scales,index,out,out.numel(),rank,sequence,count,index is not None,256)
    return out


@triton.jit
def _center_add_selected(Rows, Centers, Ids, Index,
                         COUNT:tl.constexpr,DIM:tl.constexpr,NS:tl.constexpr,
                         NK:tl.constexpr,NC:tl.constexpr,BLOCK:tl.constexpr):
    """Match rows.add_(gather(centers.to(rows.dtype), selected_ids))."""
    t=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    ok=t<COUNT
    channel=t%DIM
    selected=t//DIM%NK
    batch=t//DIM//NK
    source=tl.load(Index+selected,ok,0)
    center_id=tl.load(Ids+batch*NS+source,ok,0).to(tl.int32)
    center=tl.load(Centers+(batch*NC+center_id)*DIM+channel,ok,0)
    row=tl.load(Rows+t,ok,0)
    # The reference explicitly casts centers to the decoder output dtype before
    # addition. Keep that BF16 rounding point even if the stored table is FP32.
    center=center.to(row.dtype).to(tl.float32)
    tl.store(Rows+t,row.to(tl.float32)+center,ok)


def add_selected_centers_(rows,centers,ids,index):
    if not (rows.is_contiguous() and centers.is_contiguous() and ids.is_contiguous()):
        raise ValueError('selected center addition requires contiguous tensors')
    _center_add_selected[(triton.cdiv(rows.numel(),256),)](
        rows,centers,ids,index,rows.numel(),rows.shape[-1],ids.shape[1],
        index.numel(),centers.shape[1],256)
    return rows


@triton.jit
def _center_add_heads(Rows,Table,Ids,COUNT:tl.constexpr,H:tl.constexpr,
                      N:tl.constexpr,D:tl.constexpr,
                      RB:tl.constexpr,RH:tl.constexpr,RN:tl.constexpr,RD:tl.constexpr,
                      TB:tl.constexpr,TH:tl.constexpr,TC:tl.constexpr,TD:tl.constexpr,
                      IB:tl.constexpr,IN:tl.constexpr,BLOCK:tl.constexpr):
    ix=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    ok=ix<COUNT
    # Traverse channels and heads before tokens to coalesce the linear GEMM
    # output, which is viewed as BHND but physically stored BNHD.
    d=ix%D;h=ix//D%H;n=ix//D//H%N;b=ix//D//H//N
    center=tl.load(Ids+b*IB+n*IN,ok,0).to(tl.int32)
    ptr=Rows+b*RB+h*RH+n*RN+d*RD
    v=tl.load(ptr,ok,0)
    c=tl.load(Table+b*TB+h*TH+center*TC+d*TD,ok,0)
    tl.store(ptr,v.to(tl.float32)+c.to(tl.float32),ok)


def add_projected_centers_(rows,table,ids):
    """In-place equivalent of rows + table.gather(2, expanded token ids)."""
    b,h,n,d=rows.shape
    _center_add_heads[(triton.cdiv(rows.numel(),256),)](
        rows,table,ids,rows.numel(),h,n,d,*rows.stride(),*table.stride(),
        *ids.stride(),256)
    return rows


def main():
    from diffsynth.models.nanokv_codec import _pack_int4,_unpack_int4
    torch.manual_seed(11)
    for batch in (1,2):
        for n in (880,2640):
            codes=torch.randint(-7,8,(batch*n,384),device='cuda',dtype=torch.int8)
            packed=_pack_int4(codes);scales=torch.rand(batch*n,1,device='cuda',dtype=torch.bfloat16)
            old=(_unpack_int4(packed,codes.numel(),codes.shape).to(torch.bfloat16)*scales).reshape(batch,n,384)
            for index in (None,torch.arange(0,n,2,device='cuda')):
                result=read(packed,scales,batch,n,384,index)
                expected=old if index is None else old.index_select(1,index)
                assert torch.equal(result,expected)
    print('PASS selected/full INT4 unpack bitwise equal')


if __name__=='__main__':main()
