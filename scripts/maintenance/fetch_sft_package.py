"""Install the frozen public SFT release with archive and per-file SHA256 checks."""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def verify(directory, files):
    for name, expected in files.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError(f'Missing or changed package file: {name}')


def install(archive, destination, release, identity):
    files = {'manifest.json': identity['manifest_sha256'], **identity['protected_files']}
    if digest(archive) != release['archive_sha256']:
        raise ValueError('Release archive SHA256 mismatch')
    if destination.exists():
        verify(destination, files)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = Path(release['package']).name
    with tempfile.TemporaryDirectory(prefix='.sft-install-', dir=destination.parent) as temporary:
        staging = Path(temporary) / prefix
        staging.mkdir()
        seen = set()
        with tarfile.open(archive, 'r:gz') as bundle:
            for member in bundle:
                name = member.name.removeprefix(prefix + '/')
                if (member.name != prefix + '/' + name or name not in files
                        or not member.isfile() or name in seen):
                    raise ValueError(f'Unexpected archive member: {member.name}')
                seen.add(name)
                with bundle.extractfile(member) as source, (staging / name).open('wb') as target:
                    shutil.copyfileobj(source, target)
        if seen != files.keys():
            raise ValueError('Release archive has missing files')
        verify(staging, files)
        if destination.exists():
            raise FileExistsError(f'Destination appeared during installation: {destination}')
        staging.rename(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use an already downloaded release archive')
    parser.add_argument('--data-root', type=Path,
                        default=Path(os.environ.get('TAU3_DATA_ROOT', ROOT / 'data')))
    parser.add_argument('--verify-only', action='store_true', help='Check installed files offline')
    args = parser.parse_args(argv)
    release = json.loads((ROOT / 'docs/sft500_dev150_release.json').read_text())
    identity = json.loads((ROOT / release['identity_file']).read_text())
    data_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
    destination = data_root / 'sft' / Path(release['package']).name
    files = {'manifest.json': identity['manifest_sha256'], **identity['protected_files']}
    if args.verify_only or destination.exists():
        verify(destination, files)
    elif args.archive:
        install(args.archive, destination, release, identity)
    else:
        with tempfile.TemporaryDirectory(prefix='tau3-sft-download-') as temporary:
            archive = Path(temporary) / release['archive_filename']
            with urllib.request.urlopen(release['download_url'], timeout=60) as response:
                with archive.open('wb') as target:
                    shutil.copyfileobj(response, target)
            install(archive, destination, release, identity)
    print(json.dumps({'status': 'verified', 'package': str(destination),
                      'files': len(files), 'manifest_sha256': identity['manifest_sha256']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
