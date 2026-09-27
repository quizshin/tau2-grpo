"""Inventory pinned vendor runtime trees and verify deployment drift offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from tau3_grpo.paths import CODE_ROOT

COMPONENTS = {
    "verl": {"revision": "bec9ef74768dd201881cd4e54cd0385e87caae27", "runtime": "verl/"},
    "tau2-bench": {"revision": "fc0055dc4e0a316c3f83133267fbd6faaa770992", "runtime": "src/tau2/"},
}
PACKAGING = {"setup.py", "setup.cfg", "pyproject.toml", "MANIFEST.in", "LICENSE", "requirements.txt"}


def included(name, runtime):
    return ((name.startswith(runtime) or name in PACKAGING or name.startswith("requirements/"))
            and not any(part.startswith(".") or part == "__pycache__" for part in Path(name).parts)
            and Path(name).name != "AGENTS.md"
            and not name.endswith((".pyc", ".pyo")))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def current_files(root, runtime):
    return {str(path.relative_to(root)): digest(path.read_bytes())
            for path in sorted(root.rglob("*")) if path.is_file()
            and included(str(path.relative_to(root)), runtime)}


def generate(archives, notes, root=CODE_ROOT):
    result = {"schema": "tau3_vendor_inventory_v1", "scope": "runtime_trees_and_packaging",
              "excludes": "upstream docs/AGENTS instructions, examples, test suites, datasets and generated caches",
              "components": {}}
    for component, pin in COMPONENTS.items():
        archive = Path(archives[component])
        upstream = {}
        with tarfile.open(archive) as tar:
            for member in tar:
                if not member.isfile() or "/" not in member.name:
                    continue
                prefix, name = member.name.split("/", 1)
                if not prefix.endswith("-" + pin["revision"]):
                    raise ValueError(f"Archive root does not match pinned revision: {prefix}")
                if included(name, pin["runtime"]):
                    upstream[name] = digest(tar.extractfile(member).read())
        current = current_files(root / component, pin["runtime"])
        patches = []
        for name in sorted(set(upstream) | set(current)):
            if upstream.get(name) == current.get(name):
                continue
            note = notes[f"{component}/{name}"]  # Unexplained changes fail closed.
            if not note.get("reason") or not note.get("tests"):
                raise ValueError(f"Patch lacks rationale/tests: {name}")
            for test in note["tests"]:
                if not (root / test).is_file():
                    raise ValueError(f"Patch test does not exist: {test}")
            patches.append({"path": name, "status": "added" if name not in upstream else
                            "absent" if name not in current else "modified",
                            "upstream_sha256": upstream.get(name), "local_sha256": current.get(name),
                            **note})
        result["components"][component] = {
            **pin, "upstream_archive_sha256": digest(archive.read_bytes()),
            "upstream_files": upstream, "local_files": current, "patches": patches,
        }
    return result


def verify(inventory, root=CODE_ROOT):
    errors = []
    for name, pin in COMPONENTS.items():
        component = inventory["components"][name]
        if any(component.get(key) != value for key, value in pin.items()):
            errors.append(f"{name}: pinned revision/scope differs")
        expected = component["local_files"]
        actual = current_files(root / name, pin["runtime"])
        errors += [f"{name}/{path}: source differs from inventory" for path in sorted(set(actual) | set(expected))
                   if actual.get(path) != expected.get(path)]
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=CODE_ROOT / "env_info/vendor_patches.json")
    parser.add_argument("--archive-dir", type=Path, help="Generate from <component>.tar.gz pinned archives")
    args = parser.parse_args(argv)
    if args.archive_dir:
        notes = json.loads((CODE_ROOT / "env_info/vendor_patch_notes.json").read_text())
        inventory = generate({name: args.archive_dir / f"{name}.tar.gz" for name in COMPONENTS}, notes)
        args.inventory.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    errors = verify(json.loads(args.inventory.read_text()))
    print(json.dumps({"errors": errors, "scope": "runtime_trees_and_packaging"}, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
