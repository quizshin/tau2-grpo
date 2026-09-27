"""Process lifecycle shared by formal training and evaluation controllers.

Only handles subprocesses created in their own session by this module. It does
not discover or terminate other jobs and does not start anything on import.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
import weakref
from pathlib import Path

_OWNED = weakref.WeakSet()
_GROUPS = weakref.WeakKeyDictionary()


def launch_process(command, *, cwd, env, log_path, pid_path=None):
    """Start an owned process group; refuse to overwrite its log."""
    with Path(log_path).open("x") as log:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    _OWNED.add(process)
    _GROUPS[process] = process.pid
    if pid_path is not None:
        try:
            Path(pid_path).write_text(str(process.pid))
        except BaseException:
            stop_process(process)
            raise
    return process


def stop_process(process, *, timeout=30):
    """Stop an owned group, including children surviving an exited leader.

    Call promptly from the controller's finally block. Never discover other jobs
    or accept a PID as proof of ownership. Repeated cleanup is harmless.
    """
    if process not in _OWNED:
        raise ValueError("refusing to stop a process not launched by this module")
    group = _GROUPS.get(process)
    if group is None:
        return
    try:
        if process.poll() is None and os.getpgid(process.pid) != group:
            raise ValueError("refusing to stop a process that is not an owned session leader")
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=timeout)
        _GROUPS.pop(process, None)
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process.poll()  # Reap the leader while waiting for the entire group.
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            break
        time.sleep(min(.05, max(0, deadline - time.monotonic())))
    else:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)
    _GROUPS.pop(process, None)
