import json
from copy import deepcopy

from tau3_grpo.integrations.vendor_inventory import COMPONENTS, current_files, verify
from tau3_grpo.paths import CODE_ROOT


def test_vendor_runtime_matches_pinned_inventory():
    inventory = json.loads((CODE_ROOT / "env_info/vendor_patches.json").read_text())
    assert verify(inventory) == []
    for component in inventory["components"].values():
        for patch in component["patches"]:
            assert patch["reason"] and patch["validation_scope"]
            assert all((CODE_ROOT / path).is_file() for path in patch["tests"])


def test_new_vendor_patch_or_removed_source_requires_registration(tmp_path):
    inventory = {"components": {}}
    for name, pin in COMPONENTS.items():
        root = tmp_path / name
        path = root / pin["runtime"] / "test.py"
        path.parent.mkdir(parents=True)
        path.write_text("x = 1\n")
        inventory["components"][name] = {**pin, "local_files": current_files(root, pin["runtime"])}
    assert verify(inventory, tmp_path) == []
    path.write_text("x = 2\n")
    assert len(verify(inventory, tmp_path)) == 1
    changed = deepcopy(inventory)
    changed["components"]["verl"]["revision"] = "unapproved-upgrade"
    assert len(verify(changed, tmp_path)) == 2
