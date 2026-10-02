import os
import subprocess
import sys
import time

import pytest

from tau3_grpo.training.services import launch_process, stop_process

pytestmark = pytest.mark.skipif(os.name != "posix", reason="controllers use POSIX sessions")


def wait_file(path):
    deadline = time.monotonic() + 5
    while not path.exists():
        if time.monotonic() >= deadline:
            pytest.fail(f"child did not create {path.name}")
        time.sleep(.01)


def test_launch_log_and_idempotent_stop(tmp_path):
    log, pid = tmp_path / "child.log", tmp_path / "child.pid"
    process = launch_process([sys.executable, "-c", "import time; print('ready',flush=True); time.sleep(60)"],
                             cwd=tmp_path, env=os.environ.copy(), log_path=log, pid_path=pid)
    try:
        assert int(pid.read_text()) == process.pid
        assert os.getpgid(process.pid) == process.pid
        stop_process(process, timeout=.2)
        stop_process(process, timeout=.2)
        assert process.poll() is not None
        with pytest.raises(FileExistsError):
            launch_process([sys.executable, "-c", "pass"], cwd=tmp_path,
                           env=os.environ.copy(), log_path=log)
    finally:
        stop_process(process, timeout=.2)


def test_foreign_process_cannot_be_stopped(tmp_path):
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with pytest.raises(ValueError, match="not launched"):
            stop_process(process, timeout=.1)
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_cleanup_reaches_child_after_leader_exits(tmp_path):
    child = ("import signal,time,pathlib,sys; "
             "signal.signal(signal.SIGTERM,lambda *_:(pathlib.Path('stopped').touch(),sys.exit(0))); "
             "pathlib.Path('ready').touch(); time.sleep(60)")
    parent = f"import subprocess,sys; subprocess.Popen([sys.executable,'-c',{child!r}])"
    process = launch_process([sys.executable, "-c", parent], cwd=tmp_path,
                             env=os.environ.copy(), log_path=tmp_path / "tree.log")
    try:
        wait_file(tmp_path / "ready")
        process.wait(timeout=5)
        stop_process(process, timeout=.2)
        wait_file(tmp_path / "stopped")
    finally:
        stop_process(process, timeout=.2)



def test_denied_group_probe_waits_until_group_disappears(monkeypatch):
    import signal
    from unittest.mock import Mock

    from tau3_grpo.training import services

    process = Mock(pid=12345)
    process.poll.side_effect = [None, None, 0]
    services._OWNED.add(process)
    services._GROUPS[process] = process.pid
    monkeypatch.setattr(services.os, "getpgid", lambda pid: pid)
    calls = []

    def killpg(group, sig):
        calls.append((group, sig))
        if len(calls) == 2:
            raise PermissionError("zombie group")
        if len(calls) == 3:
            raise ProcessLookupError("reaped")

    monkeypatch.setattr(services.os, "killpg", killpg)
    monkeypatch.setattr(services.time, "sleep", lambda _: None)
    stop_process(process, timeout=1)
    assert calls == [(12345, signal.SIGTERM), (12345, 0), (12345, 0)]
    assert process.poll.call_count == 3
    process.wait.assert_called_once_with(timeout=10)
    assert process not in services._GROUPS
    stop_process(process, timeout=1)
    assert len(calls) == 3


def test_denied_group_probe_does_not_hide_failed_force_stop(monkeypatch):
    import signal
    from unittest.mock import Mock

    from tau3_grpo.training import services

    process = Mock(pid=12345)
    process.poll.return_value = None
    services._OWNED.add(process)
    services._GROUPS[process] = process.pid
    monkeypatch.setattr(services.os, "getpgid", lambda pid: pid)
    clock = iter([0, 0, 0, 1])
    monkeypatch.setattr(services.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(services.time, "sleep", lambda _: None)
    calls = []

    def killpg(group, sig):
        calls.append((group, sig))
        if sig != signal.SIGTERM:
            raise PermissionError("group still inaccessible")

    monkeypatch.setattr(services.os, "killpg", killpg)
    with pytest.raises(PermissionError, match="inaccessible"):
        stop_process(process, timeout=.2)
    assert calls == [(12345, signal.SIGTERM), (12345, 0), (12345, signal.SIGKILL)]
    process.wait.assert_not_called()
    assert process in services._GROUPS
