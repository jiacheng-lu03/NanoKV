"""Model-independent direct latent reads and globally normalized hybrid attention.

Persistent state remains the model's own INT4 codec payload. Readers are static
parameters. All reconstructed/read tensors here are call-local. This is a
correctness/training scaffold, NOT a port of the optimized HY Triton kernels.
"""
from functools import lru_cache
import hashlib
import atexit
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def enabled(prefix):
    return os.environ.get(prefix + "_ENABLE", "0").lower() in {"1", "true", "yes"}


@lru_cache(maxsize=16)
def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge(out_a, lse_a, out_b, lse_b):
    """Exact global-softmax merge for constructed logits; empty rows give zero."""
    if out_a.is_cuda and os.environ.get('MG35_CODEC_NATIVE_RUNTIME_OPT') in ('fast1','fast2','fast3','fast4','fast5'):
        from .fast_kernels import merge as fast_merge
        return fast_merge(out_a,lse_a,out_b,lse_b)
    lse = torch.logaddexp(lse_a, lse_b)
    origin = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
    wa, wb = (lse_a-origin).exp(), (lse_b-origin).exp()
    out = out_a.float()*wa[...,None] + out_b.float()*wb[...,None]
    return out.to(out_a.dtype), lse


def attention_lse(q, k, v, mask=None, *, scale=None, q_tile=128, k_tile=1024):
    """[B,H,Q,D]. Mask=True means visible. No full Q-by-history allocation."""
    scale = q.shape[-1] ** -.5 if scale is None else scale
    if k.shape[-2] == 0:
        return q.new_zeros(*q.shape[:-1],v.shape[-1]), torch.full(q.shape[:-1],-math.inf,device=q.device)
    if q.is_cuda and mask is None and q.dtype in (torch.float16,torch.bfloat16) and q.shape[-1]==v.shape[-1]:
        result = torch.ops.aten._scaled_dot_product_flash_attention.default(
            q.contiguous(), k.contiguous(), v.contiguous(), 0.0, False, False, scale=scale)
        return result[0], result[1]
    outs, lses = [], []
    if mask is not None and mask.dtype != torch.bool:
        raise ValueError("Only boolean visibility masks are supported")
    for qi in range(0,q.shape[-2],q_tile):
        qq=q[...,qi:qi+q_tile,:]
        out=qq.new_zeros(*qq.shape[:-1],v.shape[-1])
        lse=torch.full(qq.shape[:-1],-math.inf,device=q.device)
        for ki in range(0,k.shape[-2],k_tile):
            logits=(qq.float() @ k[...,ki:ki+k_tile,:].float().transpose(-1,-2))*scale
            if mask is not None:
                m = mask if mask.shape[-2] == 1 else mask[...,qi:qi+q_tile,:]
                logits=logits.masked_fill(~m[...,ki:ki+k_tile],-math.inf)
            ll=torch.logsumexp(logits,-1)
            origin=torch.where(torch.isfinite(ll),ll,torch.zeros_like(ll))
            probabilities=(logits-origin[...,None]).exp()
            oo=(probabilities @ v[...,ki:ki+k_tile,:].float()).to(q.dtype)
            out,lse=merge(out,lse,oo,ll)
        outs.append(out);lses.append(lse)
    return torch.cat(outs,-2),torch.cat(lses,-1)


def rope_selected(x, multiplier, indices):
    """Apply ORIGINAL frequencies to complete retained complex pairs.

    x [B,H,S,d], multiplier [S,D/2] or [S,1,D/2], indices [H,d].
    Never recompute temporal/spatial bands using d instead of native D.
    """
    plan=_POSITION_PLANS.get((str(indices.device),indices.data_ptr()))
    if x.is_cuda and plan is not None and os.environ.get('MG35_CODEC_NATIVE_RUNTIME_OPT') in ('fast1','fast2','fast3','fast4','fast5'):
        from .fast_kernels import rope as fast_rope
        return fast_rope(x,multiplier,plan[0])
    if multiplier.ndim == 3:
        multiplier=multiplier[:,0]
    if not torch.equal(indices[:,1::2],indices[:,::2]+1) or (indices[:,::2]%2).any():
        raise ValueError("RoPE mapping must retain complete original pairs")
    chosen=multiplier.to(x.device)[:,indices[:,::2]//2].permute(1,0,2)
    complex_x=torch.view_as_complex(x.double().reshape(*x.shape[:-1],-1,2).contiguous())
    return torch.view_as_real(complex_x*chosen[None]).flatten(-2).to(x.dtype)


def attention_grouped_mask(q,k,v,mask,query_groups,*,scale):
    """Identical visibility within a query group: gather visible keys then Flash.

    This preserves the mask, not a sparsification approximation. MG constructs
    visibility from current chunk IDs and a shared mosaic hole mask.
    """
    out=torch.zeros_like(q)
    lse=torch.full(q.shape[:-1],-math.inf,device=q.device,dtype=torch.float32)
    for group in torch.unique(query_groups):
        qi=torch.nonzero(query_groups==group,as_tuple=False).flatten()
        ki=torch.nonzero(mask[0,0,qi[0]],as_tuple=False).flatten()
        oo,ll=attention_lse(q.index_select(2,qi),k.index_select(2,ki),v.index_select(2,ki),scale=scale)
        out.index_copy_(2,qi,oo);lse.index_copy_(2,qi,ll)
    return out,lse


def camera_selected(x, matrix, indices, *, layout="full"):
    """MG full-layout PRoPE without lifting keys to native D128.

    Native layout is (p=4,c=4,j=4,r=2), NOT contiguous 4-vectors.
    Retain whole 8D (j,r) blocks. matrix is [B,S,4,4,4], already selected
    by the caller: Q=P_T, K/V=P_inv, output=P. Never infer role here.
    """
    if layout != "full" or x.shape[-1] % 8:
        raise ValueError("MG low PRoPE currently requires full layout and whole 8D blocks")
    blocks=indices.reshape(indices.shape[0],-1,8)
    plan=_POSITION_PLANS.get((str(indices.device),indices.data_ptr()))
    fast=x.is_cuda and plan is not None and os.environ.get('MG35_CODEC_NATIVE_RUNTIME_OPT') in ('fast1','fast2','fast3','fast4','fast5')
    if not fast and (not torch.equal(blocks,blocks[...,:1]+torch.arange(8,device=indices.device)) or (blocks[...,0]%8).any()):
        raise ValueError("Invalid MG camera blocks")
    block_id=blocks[...,0]//8 if not fast else None
    # Frozen einsum is bscij,bhstrpcj->bhstrpci: camera axis c is
    # the second (q) axis of (p,q,j,r), not the repeated p axis.
    cams=plan[1] if fast else block_id%4
    frames=int(matrix.shape[1])
    if x.shape[-2] % frames:
        raise ValueError("PRoPE frame count must divide the token sequence")
    if x.is_cuda and os.environ.get('MG35_CODEC_NATIVE_FUSED_CAMERA')=='1':
        from .camera_kernel import camera
        return camera(x,matrix,cams,mode='dot')
    selected=matrix[:,:,cams].permute(0,2,1,3,4,5)  # B,H,F,G,4,4
    values=x.reshape(x.shape[0],x.shape[1],frames,-1,blocks.shape[1],4,2)
    out=torch.einsum("bhfgij,bhftgjr->bhftgir",selected.to(x.dtype),values)
    return out.reshape(x.shape[0],x.shape[1],x.shape[-2],x.shape[-1])


def protected_tokens(frame_ids, recent, stride, sink=0):
    """Recent counts chronological cache frames, not absolute frame numbers."""
    if min(recent,stride,sink)<0:
        raise ValueError("Negative protection policy")
    # Consecutive frame runs preserve model ordering, including repeated camera IDs.
    _, inverse=torch.unique_consecutive(frame_ids,return_inverse=True)
    count=int(inverse.max())+1 if inverse.numel() else 0
    old=max(0,count-recent)
    keep=torch.arange(count,device=frame_ids.device)>=old
    keep[:min(sink,count)]=True
    if stride and old:
        keep[torch.arange(old-1,-1,-stride,device=frame_ids.device)]=True
    return keep[inverse]


class ReaderGroup(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.register_buffer("heads",state["heads"].long())
        self.register_buffer("indices",state["indices"].long())
        self.register_buffer("v_indices",state.get("v_indices",state["indices"]).long())
        for key in ("q_reduce","k_read","v_read","k_affine","v_affine","lift"):
            self.register_parameter(key,nn.Parameter(state[key].float()))

    def query(self,q):
        if getattr(self,'identity_query',False):return q
        return torch.einsum("bhnd,hde->bhne",q,self.q_reduce.to(q.dtype))

    def read(self, latent, centers, mean, *, target, dtype):
        weight=getattr(self,target+"_read").to(dtype)
        affine=getattr(self,target+"_affine").to(dtype)
        native_dim=self.q_reduce.shape[1]
        identity=getattr(self,'identity_'+target,False)
        cache=os.environ.get('MG35_CODEC_NATIVE_RUNTIME_OPT')=='fast5'
        try: key=(id(mean),mean._version,dtype)
        except RuntimeError: cache=False;key=None
        if cache and getattr(self,target+'_bias_key',None)==key:
            bias=getattr(self,target+'_cached_bias')
        else:
            mean_heads=mean.reshape(-1,native_dim).index_select(0,self.heads).to(dtype)
            bias=mean_heads if identity else torch.einsum("hd,hde->he",mean_heads,affine)
            if cache:
                self.register_buffer(target+'_cached_bias',bias.detach(),persistent=False)
                setattr(self,target+'_bias_key',key)
                # Retain the source to prevent id recycling; shared with codec.
                self.register_buffer(target+'_mean_source',mean,persistent=False)
        # Match frozen decoder: accumulate bias inside GEMM before BF16
        # rounding. Explicit 0 is retained only for legacy numerical audits.
        if os.environ.get('MG35_CODEC_NATIVE_FUSED_BIAS','1')=='1':
            output=F.linear(latent.to(dtype),weight.flatten(0,1),bias.flatten()).reshape(latent.shape[0],latent.shape[1],len(self.heads),weight.shape[1]).permute(0,2,1,3)
        else:
            output=torch.einsum("bnr,her->bhne",latent.to(dtype),weight)+bias[None,:,None]
        table,ids=centers
        if table is None:return output
        table=table.reshape(table.shape[0],table.shape[1],-1,native_dim).index_select(2,self.heads)
        projected=table.to(dtype).permute(0,2,1,3) if identity else torch.einsum("bchd,hde->bhce",table.to(dtype),affine)
        if output.is_cuda and os.environ.get('MG35_CODEC_NATIVE_LOW_CENTER_OPT','0')=='1':
            from .latent_kernel import add_projected_centers_
            return add_projected_centers_(output,projected,ids)
        assigned=projected.gather(2,ids[:,None,:,None].expand(-1,len(self.heads),-1,projected.shape[-1]))
        return output+assigned

    def lift_output(self,x):
        if getattr(self,'identity_lift',False):return x
        return torch.einsum("bhne,hed->bhnd",x,self.lift.to(x.dtype))


class ReaderLayer(nn.Module):
    def __init__(self,payload):
        super().__init__()
        self.groups=nn.ModuleList([ReaderGroup(s) for s in payload["groups"]])
        self.heads=int(payload["heads"]); self.native_dim=int(payload["native_dim"])
        self.rank=int(payload["rank"])
        all_heads=torch.cat([g.heads for g in self.groups]).tolist()
        if sorted(all_heads)!=list(range(self.heads)):
            raise ValueError("Reader groups must partition heads exactly once")
        for g in self.groups:
            h,d=len(g.heads),g.indices.shape[1]
            vd=g.v_indices.shape[1]
            expected={"q_reduce":(h,self.native_dim,d),"k_read":(h,d,self.rank),
                      "v_read":(h,vd,self.rank),"k_affine":(h,self.native_dim,d),
                      "v_affine":(h,self.native_dim,vd),"lift":(h,vd,self.native_dim)}
            for name,shape in expected.items():
                if tuple(getattr(g,name).shape)!=shape:
                    raise ValueError(f"Bad {name} shape; expected {shape}")


_BANKS={}
_POSITION_PLANS={}


def get_layer(prefix, model, layer, *, device, heads, native_dim, operator_path, sp_rank=0):
    """Load explicit model/rank-bound readers; unlisted layers stay native."""
    if not enabled(prefix): return None,None
    path=Path(os.environ[prefix+"_MANIFEST"]).resolve()
    key=(str(path),model,str(device),str(Path(operator_path).resolve()))
    if key not in _BANKS:
        manifest=json.loads(path.read_text())
        if manifest.get("format")!="codec_native_manifest_v1" or manifest["model"]!=model:
            raise ValueError("Wrong codec-native manifest/model")
        if manifest["codec_operator_sha256"]!=file_sha256(str(Path(operator_path).resolve())):
            raise ValueError("Reader is not bound to the deployed frozen codec operator")
        _BANKS[key]=(manifest,{})
    manifest,loaded=_BANKS[key]
    entry=manifest["layers"].get(f"{sp_rank}:{layer}")
    if entry is None:return None,manifest
    if not entry.get("calibrated",False) and os.environ.get(prefix+"_ALLOW_UNCALIBRATED")!="1":
        raise RuntimeError("Reader is initialization-only; fit and validate before deployment")
    identifier=(sp_rank,layer)
    if identifier not in loaded:
        artifact=(path.parent/entry["path"]).resolve()
        if file_sha256(str(artifact))!=entry["sha256"]:
            raise ValueError("Reader checkpoint checksum mismatch")
        payload=torch.load(artifact,map_location="cpu",weights_only=True)
        dtype=torch.bfloat16 if os.environ.get(prefix+"_BF16_WEIGHTS","0")=="1" else torch.float32
        loaded[identifier]=ReaderLayer(payload).to(device=device,dtype=dtype)
        if os.environ.get(prefix+'_RUNTIME_OPT') in ('fast1','fast2','fast3','fast4','fast5'):
            for group,cpu in zip(loaded[identifier].groups,payload['groups']):
                if os.environ.get(prefix+'_RUNTIME_OPT') in ('fast2','fast3','fast4','fast5'):
                    for flag,name in (('query','q_reduce'),('k','k_affine'),('v','v_affine'),('lift','lift')):
                        w=cpu[name]
                        exact=w.shape[-2]==w.shape[-1] and torch.equal(w,torch.eye(w.shape[-1]).expand_as(w))
                        setattr(group,'identity_'+flag,exact)
                for name in ('indices','v_indices'):
                    ix=cpu.get(name,cpu['indices']).long()
                    assert torch.equal(ix[:,1::2],ix[:,::2]+1) and not bool((ix[:,::2]%2).any())
                    blocks=ix.reshape(ix.shape[0],-1,8)
                    assert torch.equal(blocks,blocks[...,:1]+torch.arange(8)) and not bool((blocks[...,0]%8).any())
                    group.register_buffer(name+'_pairs',(ix[:,::2]//2).to(device))
                    group.register_buffer(name+'_cams',(blocks[...,0]//8%4).to(device))
                    on_device=getattr(group,name)
                    _POSITION_PLANS[(str(on_device.device),on_device.data_ptr())]=(getattr(group,name+'_pairs'),getattr(group,name+'_cams'))
    reader=loaded[identifier]
    if reader.heads!=heads or reader.native_dim!=native_dim:
        raise ValueError("Reader head/shard layout mismatch")
    return reader,manifest


def hybrid_attention(
    q_pre, q_native, native_segments, low_segments, reader, *,
    query_position, kv_position, output_position=lambda value, group: value,
):
    """Shared direct-read attention. Low segments are normalized codec rows.

    Native segments: (K,V,bool mask), [B,H,S,D]. Low segments: dictionaries
    with k/v latent, center rows, mean, position metadata. All inputs transient.
    """
    scale=reader.native_dim**-.5
    native_out=torch.zeros_like(q_native)
    native_lse=torch.full(q_native.shape[:-1],-math.inf,device=q_native.device)
    for key,value,mask in native_segments:
        out,lse=attention_lse(q_native,key,value,mask,scale=scale)
        native_out,native_lse=merge(native_out,native_lse,out,lse)
    result=torch.zeros_like(q_native)
    for group in reader.groups:
        head=group.heads
        query=query_position(group.query(q_pre.index_select(1,head)),group)
        out=native_out.index_select(1,head)
        lse=native_lse.index_select(1,head)
        for segment in low_segments:
            kk=group.read(*segment["k"],target="k",dtype=q_native.dtype)
            vv=group.read(*segment["v"],target="v",dtype=q_native.dtype)
            kk,vv=kv_position(kk,vv,segment,group)
            oo,ll=attention_lse(query,kk,vv,segment.get("mask"),scale=scale)
            # PRoPE output uses P (Q uses P_T, K/V use P_inv).  Applying the
            # low-space output transform before the learned lift preserves the
            # same linear coordinate role without materialising native K/V.
            oo=output_position(oo,group)
            out,lse=merge(out,lse,group.lift_output(oo),ll)
        result.index_copy_(1,head,out)
    return result


def resident_reader_bytes():
    seen=set();total=0
    for _,layers in _BANKS.values():
        for reader in layers.values():
            for tensor in list(reader.parameters())+list(reader.buffers()):
                storage=tensor.untyped_storage(); key=(str(tensor.device),storage.data_ptr())
                if key not in seen: seen.add(key);total+=storage.nbytes()
    if os.environ.get('MG35_CODEC_NATIVE_RUNTIME_OPT')=='fast5':
        from .metadata_cache import tensors
        for tensor in tensors():
            storage=tensor.untyped_storage();key=(str(tensor.device),storage.data_ptr())
            if key not in seen:seen.add(key);total+=storage.nbytes()
    return total

@atexit.register
def _save_reader_residency():
    dest=os.environ.get('MG35_CODEC_NATIVE_RESIDENCY_JSON')
    if not dest:return
    readers=[r for _,layers in _BANKS.values() for r in layers.values()]
    Path(dest).write_text(json.dumps(dict(scope='unique resident reader parameters and buffers; add to codec system_bytes',reader_bytes=resident_reader_bytes(),loaded_layers=len(readers),parameter_dtypes=sorted({str(p.dtype) for r in readers for p in r.parameters()})),indent=2))
