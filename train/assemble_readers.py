"""Assemble fitted readers with the fixed per-head allocation used in the paper."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
HY_TENSORS = ('k_main_read', 'q_main_reduce', 'k_main_anchor_reduce',
    'v_main_read', 'v_main_lift', 'k_prope_read', 'q_prope_mix',
    'v_prope_read', 'v_prope_mix', 'v_main_anchor_reduce',
    'v_prope_anchor_mix', 'log_residual_scale_main',
    'log_residual_scale_prope', 'main_indices')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def group_for_heads(state, heads):
    # Experts may already have multiple groups; preserve each group's basis.
    groups=[]
    remaining=set(heads)
    for group in state['groups']:
        ids=group['heads'].tolist()
        positions=[i for i,h in enumerate(ids) if h in remaining]
        if not positions: continue
        index=torch.tensor(positions,dtype=torch.long)
        groups.append({k:v.index_select(0,index).contiguous()
            if torch.is_tensor(v) and v.ndim and v.shape[0]==len(ids)
            else copy.deepcopy(v) for k,v in group.items()})
        remaining.difference_update(ids[i] for i in positions)
    if remaining: raise ValueError(f'Expert missing heads: {remaining}')
    return groups

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=('hy','lingbot','mg'),required=True)
    p.add_argument('--experts',type=Path,required=True,
        help='Root containing layer_XX/dWIDTH.pt uniform-width fitted readers')
    p.add_argument('--operator',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    dims=json.loads((ROOT/f'configs/{a.model}/head_dimensions.json').read_text())
    template=json.loads((ROOT/f'configs/{a.model}/fast_manifest.template.json').read_text())
    # Check all sources before creating the output directory.
    for layer,widths in dims.items():
        if a.model=='hy' and set(widths)=={128}: continue
        for width in set(widths):
            source=a.experts/f'layer_{int(layer):02d}'/f'd{width}.pt'
            if not source.is_file(): raise FileNotFoundError(source)
    digest=sha(a.operator)
    template['layers']={}
    template['operator']=str(a.operator.resolve())
    template['codec_operator_sha256']=digest
    target_dir=a.output/'readers';target_dir.mkdir(parents=True)
    for layer,widths in dims.items():
        if a.model=='hy' and set(widths)=={128}: continue
        states={d:torch.load(a.experts/f'layer_{int(layer):02d}'/f'd{d}.pt',
            map_location='cpu',weights_only=True) for d in sorted(set(widths))}
        if a.model=='hy':
            if len(states)==1:
                payload=next(iter(states.values()))
            else:
                groups=[]
                for dim,state in states.items():
                    heads=torch.tensor([h for h,d in enumerate(widths) if d==dim])
                    groups.append({'dim':dim,'head_indices':heads,'tensors':{
                        k:state['model'][k].index_select(0,heads).contiguous() for k in HY_TENSORS}})
                payload=dict(format_version=1,method='formal_z512_int4_c64_dictionary_learned_rope_grouped_head_width',
                    formal_rank=256,head_dim=128,num_heads=16,head_dims=widths,groups=groups)
        else:
            payload=copy.deepcopy(states.get(64,next(iter(states.values()))))
            payload.pop('width',None);payload['groups']=[]
            for dim,state in states.items():
                payload['groups'].extend(group_for_heads(state,[h for h,d in enumerate(widths) if d==dim]))
            payload['widths']=sorted(states)
        dest=target_dir/f'layer_{int(layer):02d}.pt'
        torch.save(payload,dest)
        key=layer if a.model=='hy' else f'0:{layer}'
        template['layers'][key]=str(dest.resolve()) if a.model=='hy' else dict(
            path=str(dest.relative_to(a.output)),sha256=sha(dest),calibrated=True,widths=sorted(states))
    (a.output/'manifest.json').write_text(json.dumps(template,indent=2)+'\n')

if __name__=='__main__':main()
