"""Restore selected models with SHA256 validation and a training-space reserve."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('models', nargs='*', default=['Qwen3.5-0.8B', 'Qwen3.5-4B'])
parser.add_argument('--dry-run', action='store_true')
parser.add_argument('--reserve-gib', type=float, default=12)
args = parser.parse_args()
if args.reserve_gib < 0:
    parser.error('reserve-gib must be nonnegative')
root = Path(os.environ.get('TAU3_FS_ROOT', '/root/autodl-fs/tau3_grpo_fix'))
target = Path(os.environ.get('TAU3_SCRATCH_ROOT', '/root/autodl-tmp/tau3')) / 'models'
target.mkdir(parents=True, exist_ok=True)
selected = []
required = 0
for name in dict.fromkeys(args.models):
    if Path(name).name != name or name.startswith('.'):
        parser.error(f'invalid model name: {name}')
    source = root / 'model_store' / name
    record = json.loads((source / 'tau3_source_revision.json').read_text())
    weights = record['weights']
    for entry in weights:
        rel = Path(entry['file'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError(f'invalid manifest path: {rel}')
        assert (source / rel).stat().st_size == entry['bytes'], rel
    # Changed files can require a complete temporary replacement under rsync.
    for path in source.rglob('*'):
        if path.is_file():
            dest = target / name / path.relative_to(source)
            if not dest.exists() or (dest.stat().st_size, dest.stat().st_mtime_ns) != (path.stat().st_size, path.stat().st_mtime_ns):
                required += path.stat().st_size
    selected.append((name, source, record))
free = shutil.disk_usage(target).free
reserve = int(args.reserve_gib * 1024**3)
print(json.dumps({'models': [m[0] for m in selected], 'additional_bytes_upper_bound': required,
                  'free_bytes': free, 'reserve_bytes': reserve, 'dry_run': args.dry_run}, indent=2), flush=True)
if required + reserve > free:
    raise SystemExit('Insufficient data-disk space; use models directly from file storage or select fewer models.')
if args.dry_run:
    raise SystemExit(0)
receipt = target / '.tau3_models_verified.json'
verified = json.loads(receipt.read_text()) if receipt.exists() else []
for name, source, record in selected:
    subprocess.run(['rsync', '-a', '--checksum', str(source) + '/', str(target / name) + '/'], check=True)
    for entry in record['weights']:
        with (target / name / entry['file']).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == entry['sha256'], entry['file']
    for asset in source.rglob('*'):
        if asset.is_file() and asset.suffix != '.safetensors':
            with asset.open('rb') as a, (target / name / asset.relative_to(source)).open('rb') as b:
                assert hashlib.file_digest(a, 'sha256').digest() == hashlib.file_digest(b, 'sha256').digest(), asset
    verified = [r for r in verified if r['model'] != record['model']]
    verified.append({'model': record['model'], 'revision': record['revision']})
    print('Verified', record['model'], flush=True)
partial = receipt.with_suffix('.tmp')
partial.write_text(json.dumps(verified, indent=2) + '\n')
partial.replace(receipt)
