import json

import pytest

from tau3_grpo.integrations.boundary_checkpoint import complete_boundary
from tau3_grpo.training.rl.checkpoints import required_files, validate_checkpoint


def write_checkpoint(root, step):
    cp = root / f"global_step_{step}"
    for name in required_files(2):
        path = cp / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"state")
    complete_boundary(root, step, 2)
    return cp


@pytest.mark.parametrize("damage", ["missing", "zero", "traversal", "absolute", "symlink", "size", "world", "marker"])
def test_recovery_receipt_rejects_damage(tmp_path, damage):
    cp = write_checkpoint(tmp_path, 10)
    receipt = cp / "checkpoint-complete.json"
    info = json.loads(receipt.read_text())
    if damage == "missing":
        info["files"].pop("actor/optim_world_size_2_rank_1.pt")
    elif damage == "zero":
        (cp / "data.pt").write_bytes(b"")
        info["files"]["data.pt"] = 0
    elif damage in ("traversal", "absolute", "symlink"):
        outside = tmp_path / "outside"
        outside.write_bytes(b"state")
        if damage == "symlink":
            (cp / "data.pt").unlink()
            (cp / "data.pt").symlink_to(outside)
        else:
            info["files"]["../outside" if damage == "traversal" else str(outside)] = 5
    elif damage == "size":
        info["files"]["data.pt"] = True
    elif damage == "world":
        info["world_size"] = 4
    else:
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("20")
    receipt.write_text(json.dumps(info))
    with pytest.raises(ValueError):
        validate_checkpoint(tmp_path, 10, world_size=2)


def test_incomplete_new_boundary_keeps_previous(tmp_path):
    old = write_checkpoint(tmp_path, 10)
    cp = tmp_path / "global_step_20"
    cp.mkdir()
    (cp / "data.pt").write_bytes(b"state")
    with pytest.raises(ValueError, match="previous checkpoint retained"):
        complete_boundary(tmp_path, 20, 2)
    assert validate_checkpoint(tmp_path, 10, world_size=2) == old
    write_checkpoint(tmp_path, 20)
    assert not old.exists()
    assert validate_checkpoint(tmp_path, 20, world_size=2) == cp


def test_old_evaluation_import_is_same_implementation():
    from env_info.a800_20260912 import run_post_rl_eval
    from tau3_grpo.evaluation import controller
    assert run_post_rl_eval is controller
