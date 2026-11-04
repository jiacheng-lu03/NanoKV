"""Run a model-native implementation without importing its GPU dependencies."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'hy': 'HY-WorldPlay', 'lingbot': 'LingBot-World-2.0', 'mg': 'Matrix-Game-3.5'}
TRAIN = {
 'hy': {
  'collect': ('compression', 'scripts/collect_kv.py'),
  'fit_basis': ('compression', 'scripts/fit_residual_basis.py'),
  'fit_value': ('compression', 'scripts/train_value_codec.py'),
  'init_key': ('compression', 'scripts/initialize_key_decoder.py'),
  'fit_key': ('compression', 'scripts/train_key_alignment.py'),
  'export_key': ('compression', 'scripts/export_key_basis.py'),
  'export_value': ('compression', 'scripts/export_value_codec.py'),
  'pack_codec': ('compression', 'scripts/pack_codec.py'),
  'fit_fast': ('fast', 'scripts/train_attention_reader.py'),
  'collect_fast': ('fast', 'hyvideo/generate.py'),
 },
 'lingbot': {
  'collect': ('compression', 'generate.py'),
  'fit_basis': ('compression', 'scripts/fit_residual_basis.py'),
  'fit_value': ('compression', 'scripts/train_value_codec.py'),
  'fit_key': ('compression', 'scripts/train_key_alignment.py'),
  'merge_key': ('compression', 'scripts/merge_key_decoders.py'),
  'pack_codec': ('compression', 'scripts/pack_codec.py'),
  'init_fast': ('fast', 'scripts/initialize_attention_readers.py'),
  'fit_fast': ('fast', 'scripts/fit_attention_reader.py'),
  'collect_fast': ('fast', 'generate.py'),
 },
 'mg': {
  'collect': ('compression', 'infer_distilled.py'),
  'fit_basis': ('compression', 'scripts/fit_residual_basis.py'),
  'fit_value': ('compression', 'scripts/train_value_codec.py'),
  'pack_codec': ('compression', 'scripts/pack_codec.py'),
  'init_fast': ('fast', 'scripts/initialize_attention_readers.py'),
  'fit_fast': ('fast', 'scripts/train_attention_reader.py'),
  'collect_fast': ('fast', 'infer_distilled.py'),
 },
}
INFER = {m: {
 'generate': ('compression', 'hyvideo/generate.py' if m == 'hy' else 'generate.py' if m == 'lingbot' else 'infer_distilled.py'),
 'generate_fast': ('fast', 'hyvideo/generate.py' if m == 'hy' else 'generate.py' if m == 'lingbot' else 'infer_distilled.py'),
} for m in MODELS}

def resolve(group, model, stage):
    table = {'train': TRAIN, 'infer': INFER}[group]
    variant, script = table[model][stage]
    cwd = ROOT / 'models' / MODELS[model] / variant
    path = cwd / script
    if not path.is_file():
        raise FileNotFoundError(path)
    return cwd, path

def load_profile(path, env):
    data = json.loads(Path(path).read_text())
    def expand(value):
        def replace(match):
            name = match.group(1)
            if name not in env or not env[name]:
                raise ValueError(f'Profile requires environment variable {name}')
            return env[name]
        return re.sub(r'\$\{([A-Za-z_][A-Za-z_0-9]*)\}', replace, str(value))
    return {k: expand(v) for k,v in data.get('env',{}).items()}, [expand(v) for v in data.get('args',[])]

def main(group):
    table = {'train': TRAIN, 'infer': INFER}[group]
    parser = argparse.ArgumentParser(description=f'NanoKV {group}')
    parser.add_argument('model', choices=MODELS)
    parser.add_argument('stage', nargs='?')
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--env', action='append', default=[], metavar='KEY=VALUE',
                        help='Explicit environment override after applying a profile')
    # Everything after -- belongs to the native CLI, including --help.
    argv = sys.argv[1:]
    split = argv.index('--') if '--' in argv else len(argv)
    args = parser.parse_args(argv[:split])
    native = argv[split+1:]
    if args.stage is None:
        print('\n'.join(table[args.model])); return 0
    if args.stage not in table[args.model]:
        parser.error(f'Unknown stage {args.stage}; choose from {list(table[args.model])}')
    cwd, script = resolve(group, args.model, args.stage)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(cwd), str(script.parent), str(ROOT), env.get('PYTHONPATH','')])
    extra = []
    if args.profile:
        values, extra = load_profile(args.profile.resolve(), env)
        # Named profiles must not inherit an earlier experiment's switches.
        for key in list(env):
            if key.startswith(('HY_', 'LINGBOT_', 'MG35_', 'NANOKV_PORTABLE_')):
                del env[key]
        env.update(values)
    for assignment in args.env:
        key, separator, value = assignment.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', key):
            parser.error('--env requires KEY=VALUE')
        env[key] = value
    if env.get('QVG_ROOT'):
        env['PYTHONPATH'] += os.pathsep + str(Path(env['QVG_ROOT']).resolve())
    command = [sys.executable, str(script), *extra, *native]
    if args.dry_run:
        print(f'cwd: {cwd}\ncommand: {shlex.join(command)}'); return 0
    return subprocess.call(command, cwd=cwd, env=env)
