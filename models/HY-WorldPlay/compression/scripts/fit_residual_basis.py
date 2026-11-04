"""Fit shared-center residual PCA using the codec's training preprocessing."""
import argparse
from pathlib import Path
import torch
from residual_data import collect_smoothed
from sample_dataset import discover_atoms
from pca_utils import fit_pca

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layer',type=int,required=True)
    p.add_argument('--target',choices=['k','v'],required=True)
    p.add_argument('--rank',type=int,default=512)
    p.add_argument('--rows-per-atom',type=int,default=1024)
    p.add_argument('--seed',type=int,default=20260729)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    atoms=discover_atoms(a.data_root,'train')
    if not atoms: p.error('No training samples found')
    device=torch.device(a.device)
    _,residual,_,_=collect_smoothed(a.data_root,'train',atoms,a.layer,a.rows_per_atom,64,2,a.seed,device,'joint',a.target)
    mean,components,explained=fit_pca(residual,a.rank,device)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'mean':mean.cpu(),'components':components.cpu(),'explained':explained.cpu() if torch.is_tensor(explained) else explained},a.output)

if __name__=='__main__':main()
