"""Bind a NanoKV inference profile to locally trained codec/reader artifacts."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=('hy', 'lingbot', 'mg'), required=True)
    p.add_argument('--artifacts', type=Path, required=True, help='Directory containing codec.pt')
    p.add_argument('--variant', choices=('nanokv', 'fast'), default='nanokv')
    p.add_argument('--manifest', type=Path, help='Assembled Fast reader manifest')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output already exists; choose a new profile path')
    codec = a.artifacts.resolve() / 'codec.pt'
    digest = hashlib.sha256()
    with codec.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    root = Path(__file__).resolve().parents[1]
    raw = (root / 'configs' / a.model / (a.variant + '.json')).read_text()
    profile = json.loads(raw.replace('${ARTIFACTS}', str(a.artifacts.resolve())))
    profile['env']['NANOKV_PORTABLE_CODEC_SHA256'] = digest.hexdigest()
    if a.variant == 'fast':
        manifest = (a.manifest or a.artifacts / 'manifest.json').resolve()
        data = json.loads(manifest.read_text())
        if data.get('codec_operator_sha256') != digest.hexdigest():
            p.error('Fast manifest is bound to a different codec')
        keys = [k for k in profile['env'] if k.endswith('_MANIFEST')]
        if len(keys) != 1:
            p.error('Expected one Fast manifest entry in the profile')
        profile['env'][keys[0]] = str(manifest)
    elif a.manifest:
        p.error('--manifest is only used with --variant fast')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(profile, indent=2) + '\n')
    print(a.output)


if __name__ == '__main__':
    main()
