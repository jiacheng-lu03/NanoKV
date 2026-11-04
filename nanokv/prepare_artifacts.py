"""Bind a published reader manifest to local artifacts and verify its checksums."""
import argparse
import hashlib
import json
from pathlib import Path

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def bind(value,root):
    if isinstance(value,str):return value.replace('${ARTIFACTS}',str(root))
    if isinstance(value,list):return [bind(v,root) for v in value]
    if isinstance(value,dict):return {k:bind(v,root) for k,v in value.items()}
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['hy','lingbot','mg'],required=True)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    data=bind(json.loads((root/'configs'/a.model/'fast_manifest.template.json').read_text()),a.artifacts.resolve())
    checks=[(Path(data['operator']),data.get('codec_operator_sha256'))]
    for item in data['layers'].values():
        checks.append((Path(item if isinstance(item,str) else item['path']),None if isinstance(item,str) else item.get('sha256')))
    for path,expected in checks:
        if not path.is_file():raise FileNotFoundError(path)
        if expected and sha256(path)!=expected:raise ValueError(f'Checksum mismatch: {path}')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,indent=2)+'\n')
    print(a.output)

if __name__=='__main__':main()
