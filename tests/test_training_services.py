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
