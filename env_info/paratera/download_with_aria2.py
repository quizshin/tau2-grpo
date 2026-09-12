"""Resume pinned model weights with aria2; expose final files only after hashing.

Small metadata files must already match the manifest. Partial weights from the
standard downloader are adopted without copying. Run under run_guarded.py.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.parse import quote


def verified(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest() == entry["sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallel-files", type=int, default=2)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    pending = args.output / ".aria2"
    pending.mkdir(parents=True, exist_ok=True)
    lines, entries, results = [], [], []
    for entry in manifest["files"]:
        name = entry["path"]
        if Path(name).name != name or any(x in name for x in ("\n", "\r")):
            raise ValueError("This downloader requires flat model filenames")
        target = args.output / name
        if verified(target, entry):
            results.append({"path": name, "verified": True, "existing": True})
            print("Verified existing", name, flush=True)
            continue
        if target.exists():
            raise ValueError(f"Unexpected unverified final file: {name}")
        completed = pending / name
        if (not (pending / (name + ".aria2")).exists()) and verified(completed, entry):
            completed.rename(target)
            results.append({"path": name, "verified": True, "existing": True})
            print("Verified completed aria2 file", name, flush=True)
            continue
        if entry["size"] < 20_000_000:
            raise ValueError(f"Pinned metadata must be supplied first: {name}")
        part = target.with_name(name + ".part")
        if part.exists():
            if (pending / name).exists():
                raise ValueError(f"Two partial copies exist: {name}")
            part.rename(pending / name)
        lines += [f"https://modelscope.cn/models/{manifest['model']}/resolve/master/{quote(name)}",
                  f"  dir={pending}", f"  out={name}", f"  checksum=sha-256={entry['sha256']}"]
        entries.append(entry)
    if lines:
        inputs = pending / "download.txt"
        inputs.write_text("\n".join(lines) + "\n")
        subprocess.run(["aria2c", "--input-file=" + str(inputs), "--continue=true",
                        "--check-integrity=true", "--allow-overwrite=false", "--auto-file-renaming=false",
                        "--file-allocation=none", f"--max-concurrent-downloads={args.parallel_files}", "--split=8",
                        "--max-connection-per-server=8", "--min-split-size=64M",
                        "--auto-save-interval=10", "--max-tries=10", "--retry-wait=5",
                        "--timeout=45", "--connect-timeout=15", "--summary-interval=60",
                        "--show-console-readout=false", "--console-log-level=warn",
                        "--download-result=full"], check=True)
    for entry in entries:
        path = pending / entry["path"]
        if not verified(path, entry):
            raise ValueError(f"Pinned SHA-256 mismatch: {entry['path']}")
        path.rename(args.output / entry["path"])
        results.append({"path": entry["path"], "verified": True, "existing": False})
        print("Verified downloaded", entry["path"], flush=True)
    receipt = {"model": manifest["model"], "revision": manifest["revision"],
               "source": "modelscope (aria2 weights); pinned HF metadata",
               "verified_files": len(results), "files": results}
    (args.output / "tau3_source_revision.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
