"""Download an explicitly pinned simulator manifest with resumable hash verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests

from tau3_grpo.paths import CONFIG_ROOT


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fetch(entry: dict, manifest: dict, root: Path, source: str) -> dict:
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Manifest paths must remain within the model directory")
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == entry["size"] and digest(target) == entry["sha256"]:
            return {"path": str(relative), "verified": True, "existing": True}
        raise ValueError(f"Existing file differs from manifest: {relative}")
    part = target.with_name(target.name + ".part")
    repo = manifest["model"]
    revision = manifest["revision"]
    if source == "modelscope":
        # ModelScope's master mirror is accepted only if bytes match the pinned HF manifest.
        url = f"https://modelscope.cn/models/{repo}/resolve/master/{quote(str(relative))}"
    else:
        host = "huggingface.co" if source == "huggingface" else "hf-mirror.com"
        url = f"https://{host}/{repo}/resolve/{revision}/{quote(str(relative))}"
    for attempt in range(5):
        offset = part.stat().st_size if part.exists() else 0
        if offset == entry["size"]:
            break
        if offset > entry["size"]:
            raise ValueError(f"Partial file exceeds declared size: {relative}")
        try:
            with requests.get(url, headers={"Range": f"bytes={offset}-"} if offset else {},
                              stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Invalid server range response")
                written = offset if append else 0
                next_report = written + 512 * 1024 * 1024
                with part.open("ab" if append else "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        handle.write(chunk)
                        written += len(chunk)
                        if written > entry["size"]:
                            raise ValueError(f"Download exceeds declared size: {relative}")
                        if written >= next_report:
                            print(f"{relative}: {written}/{entry['size']} bytes", flush=True)
                            next_report = written + 512 * 1024 * 1024
            if part.stat().st_size == entry["size"]:
                break
        except requests.RequestException as exc:
            print(f"Retry {attempt + 1}/5 {relative}: {type(exc).__name__}", flush=True)
        if attempt < 4:
            time.sleep(min(2 ** attempt, 10))
    if not part.exists() or part.stat().st_size != entry["size"]:
        raise RuntimeError(f"Download incomplete: {relative}")
    if digest(part) != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch; retained partial file for inspection: {relative}")
    part.replace(target)
    print(f"Verified {relative}", flush=True)
    return {"path": str(relative), "verified": True, "existing": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CONFIG_ROOT / "simulator/qwen38_download.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=["modelscope", "hf-mirror", "huggingface"], default="modelscope")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda item: fetch(item, manifest, args.output, args.source), manifest["files"]))
    receipt = {"model": manifest["model"], "revision": manifest["revision"],
               "source": args.source, "verified_files": len(results), "files": results}
    (args.output / "tau3_source_revision.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
