"""Validate paired MG collection files and write the fitting input manifest."""
import argparse
import json
from pathlib import Path
import torch


def build(roots, split, layers=30, rows=4096):
    roots = [Path(r).resolve() for r in roots]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError('Supply distinct nonempty collection roots')
    if layers < 1 or rows < 1:
        raise ValueError('layers and rows must be positive')
    for root in roots:
        for layer in range(layers):
            ids = None
            for kind in ('k', 'v', 'k_c64', 'v_c64'):
                path = root / kind / f'layer_{layer:02d}.pt'
                d = torch.load(path, map_location='cpu', weights_only=True)
                expected = dict(format='mg35_worldplay_pca_samples_v3', target=kind,
                                layer=layer, channels=3072, num_heads=24)
                residual = kind.endswith('_c64')
                expected.update(center_storage='bf16' if residual else 'none',
                                center_block_size=64 if residual else 0)
                if residual:
                    expected.update(c64_clusters=64, c64_iters=2,
                                    c64_seed_rule='20260804+layer')
                for k, v in expected.items():
                    if d.get(k) != v:
                        raise ValueError(f'{path}: {k}={d.get(k)!r}, expected {v!r}')
                if tuple(d['tensor_int8'].shape) != (rows, 24, 128):
                    raise ValueError(f'{path}: incomplete or incompatible sample shape')
                if tuple(d['sample_ids'].shape) != (rows, 2):
                    raise ValueError(f'{path}: invalid sample identifiers')
                if ids is not None and not torch.equal(ids, d['sample_ids']):
                    raise ValueError(f'{path}: K/V and raw/residual sample identifiers differ')
                ids = d['sample_ids']
    return dict(status='complete', split=split, trajectories=len(roots),
                roots=list(map(str, roots)), layers=layers, rows_per_root=rows,
                total_rows_per_layer_kind=rows * len(roots), center_storage='bf16',
                center_block_size=64, c64_clusters=64, c64_iters=2,
                c64_seed_rule='20260804+layer')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots', nargs='+', type=Path, required=True)
    p.add_argument('--split', choices=('train', 'val'), required=True)
    p.add_argument('--layers', type=int, default=30)
    p.add_argument('--rows-per-root', type=int, default=4096)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output exists; choose a new manifest path')
    data = build(a.roots, a.split, a.layers, a.rows_per_root)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(data, indent=2) + '\n')
    print(a.output)


if __name__ == '__main__':
    main()
