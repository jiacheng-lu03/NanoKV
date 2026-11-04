"""Construct exact full-dimensional readers from a frozen packed codec."""
import argparse
from pathlib import Path
import torch
from codec_native.reader_training import decoder
from codec_native.reader_tools import initialize_reader_state

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    op=torch.load(a.operator,map_location='cpu',weights_only=True)
    for layer in range(30):
        state=initialize_reader_state(decoder(op,'k',layer),decoder(op,'v',layer),
            [(torch.arange(24),torch.arange(128).expand(24,-1))])
        directory=a.output/f'layer_{layer:02d}';directory.mkdir(parents=True)
        torch.save(state,directory/'d128.pt')

if __name__=='__main__':main()
