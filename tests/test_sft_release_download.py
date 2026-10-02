"""A release must never install corrupt files or escape/overwrite the destination."""

import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'fetch_sft_package', Path(__file__).parents[1] / 'scripts/maintenance/fetch_sft_package.py'
)
FETCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCH)


def fixture_archive(tmp_path, names=None):
    contents = {'manifest.json': b'{}', 'train.jsonl': b'{"messages": []}\n'}
    archive = tmp_path / 'package.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for name in names or ['package/manifest.json', 'package/train.jsonl']:
            data = contents.get(name.removeprefix('package/'), b'bad')
            member = tarfile.TarInfo(name)
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    release = {'package': 'data/sft/package', 'archive_sha256': FETCH.digest(archive)}
    identity = {'manifest_sha256': hashlib.sha256(contents['manifest.json']).hexdigest(),
                'protected_files': {'train.jsonl': hashlib.sha256(contents['train.jsonl']).hexdigest()}}
    return archive, release, identity


def test_verified_install_and_existing_corruption_is_not_overwritten(tmp_path):
    archive, release, identity = fixture_archive(tmp_path)
    destination = tmp_path / 'installed/package'
    FETCH.install(archive, destination, release, identity)
    FETCH.install(archive, destination, release, identity)
    (destination / 'train.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='changed package file'):
        FETCH.install(archive, destination, release, identity)
    assert (destination / 'train.jsonl').read_text() == 'changed'


def test_bad_archive_hash_never_creates_destination(tmp_path):
    archive, release, identity = fixture_archive(tmp_path)
    archive.write_bytes(b'corrupt')
    destination = tmp_path / 'installed/package'
    with pytest.raises(ValueError, match='archive SHA256'):
        FETCH.install(archive, destination, release, identity)
    assert not destination.exists()


@pytest.mark.parametrize('bad', ['package/../../escaped', 'package/manifest.json', 'package/link'])
def test_unexpected_or_duplicate_members_leave_no_partial_install(tmp_path, bad):
    archive, release, identity = fixture_archive(
        tmp_path, ['package/manifest.json', 'package/train.jsonl', bad]
    )
    destination = tmp_path / 'installed/package'
    with pytest.raises(ValueError, match='Unexpected archive member'):
        FETCH.install(archive, destination, release, identity)
    assert not destination.exists()
    assert not (tmp_path / 'escaped').exists()


def test_member_file_hash_mismatch_leaves_no_partial_install(tmp_path):
    archive, release, identity = fixture_archive(tmp_path)
    identity['protected_files']['train.jsonl'] = '0' * 64
    destination = tmp_path / 'installed/package'
    with pytest.raises(ValueError, match='changed package file'):
        FETCH.install(archive, destination, release, identity)
    assert not destination.exists()


def test_symlink_member_is_rejected(tmp_path):
    archive, release, identity = fixture_archive(tmp_path)
    with tarfile.open(archive, 'w:gz') as bundle:
        member = tarfile.TarInfo('package/manifest.json')
        member.type = tarfile.SYMTYPE
        member.linkname = '../../escaped'
        bundle.addfile(member)
    release['archive_sha256'] = FETCH.digest(archive)
    destination = tmp_path / 'installed/package'
    with pytest.raises(ValueError, match='Unexpected archive member'):
        FETCH.install(archive, destination, release, identity)
    assert not destination.exists()
