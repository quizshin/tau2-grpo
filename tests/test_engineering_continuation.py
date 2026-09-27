"""Continuation must not silently restart training or accept partial restore."""
import importlib.util

import pytest

from tau3_grpo.paths import CODE_ROOT


@pytest.fixture
def controller(monkeypatch):
    folder = CODE_ROOT / "env_info/a800_20260919"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("engineering_continuation", folder / "architecture_continuation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_attempt_is_rejected_before_loading_plan(controller, tmp_path):
    with pytest.raises(ValueError, match="Existing attempt"):
        controller.execute(tmp_path, tmp_path / "missing", "b-tau_gigpo-df0", True, 4500)


def test_resume_overrides_preserve_original_command(controller):
    original = ["bash", "train.sh", "trainer.total_training_steps=2"]
    command = controller.resume_command({"command": original}, "/cp/global_step_2", 2)
    assert original == ["bash", "train.sh", "trainer.total_training_steps=2"]
    assert command[-3:] == ["trainer.total_training_steps=3", "trainer.resume_mode=resume_path",
                            "trainer.resume_from_path=/cp/global_step_2"]
    with pytest.raises(ValueError):
        controller.resume_command({"command": original}, "/cp/global_step_1", 1)


def restore_log():
    return "Setting global step to 2\n" + "\n".join(
        f"Loaded {kind} from /cp/actor/{prefix}_world_size_4_rank_{rank}.pt"
        for kind, prefix in (("model", "model"), ("optimizer", "optim"),
                             ("rng", "extra_state"), ("lr_scheduler", "extra_state"))
        for rank in range(4))


def test_all_restore_branches_are_required(controller):
    assert controller.restore_log_evidence(restore_log())["restored_step"] == 2
    with pytest.raises(ValueError, match="rng/rank2"):
        controller.restore_log_evidence(restore_log().replace("Loaded rng from /cp/actor/extra_state_world_size_4_rank_2.pt", ""))
    with pytest.raises(ValueError, match="dataloader"):
        controller.restore_log_evidence(restore_log() + "\nWarning: No dataloader state found")


def test_data_pointer_and_actual_next_tasks_must_agree(controller, tmp_path):
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    run = tmp_path / "run"
    (run / "global_step_3").mkdir(parents=True)
    (run / "rollouts").mkdir()
    def state(n):
        return {"_num_yielded": n, "_sampler_iter_yielded": n,
                "_sampler_iter_state": {"samples_yielded": n * 8}}

    torch.save(state(2), tmp_path / "prior-data.pt")
    torch.save(state(3), run / "global_step_3/data.pt")
    (run / "resolved-hydra.yaml").write_text("data:\n  shuffle: false\n")
    pq.write_table(pa.Table.from_pylist([{"extra_info": {"task_id": str(i)}} for i in range(32)]),
                   run / "train_schedule.parquet")
    rows = [{"trajectory_facts_json": json.dumps({"identity": {"task_id": str(i)}})}
            for i in range(16, 24) for _ in range(8)]
    output = run / "rollouts/3.jsonl"
    output.write_text("\n".join(map(json.dumps, rows)))
    assert controller.audit_resume_schedule(tmp_path, run)["next_schedule_batch_matches"]
    rows[0] = {"trajectory_facts_json": json.dumps({"identity": {"task_id": "0"}})}
    output.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="next eight"):
        controller.audit_resume_schedule(tmp_path, run)
    torch.save(state(2), run / "global_step_3/data.pt")
    with pytest.raises(ValueError, match="advance exactly"):
        controller.audit_resume_schedule(tmp_path, run)


def test_evaluation_rejects_existing_attempt_before_gpu_start(controller, tmp_path):
    folder = CODE_ROOT / "env_info/a800_20260919"
    spec = importlib.util.spec_from_file_location("engineering_eval", folder / "architecture_evaluation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="Existing evaluation"):
        module.prepare(tmp_path, tmp_path / "run", 3)
    (tmp_path / "budget.json").write_text("{}")
    with pytest.raises(ValueError, match="no automatic retry"):
        module.execute(tmp_path)


def test_port_preflight_rejects_live_listener_but_allows_closed_server(controller):
    import socket

    from architecture_acceptance import ensure_port_available

    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen()
    port = server.getsockname()[1]
    client = socket.socket()
    try:
        with pytest.raises(OSError):
            ensure_port_available(port)
        client.connect(("127.0.0.1", port))
        accepted, _ = server.accept()
        accepted.close()  # Server initiates close, leaving its side in TIME_WAIT.
        assert client.recv(1) == b""
    finally:
        client.close()
        server.close()
    ensure_port_available(port)
