"""Run a remote job with disk headroom below the user's 100 GB / 25 GB caps.

Measures entire mounted filesystems, not just the project directory. Stops only
its own process group, never deletes files, and preserves partial outputs.
Limits are decimal GB and intentionally leave 5 GB shared / 2 GB system margin.
"""

import argparse
import json
import os
import signal
import subprocess
import time


def usage(path):
    stat = os.statvfs(path)
    return (stat.f_blocks - stat.f_bfree) * stat.f_frsize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared", default="/root/shared-nvme")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    limits = {args.shared: 95_000_000_000, "/": 23_000_000_000}

    def snapshot():
        return {path: usage(path) for path in limits}

    initial = snapshot()
    print(json.dumps({"disk_start_bytes": initial, "stop_threshold_bytes": limits}), flush=True)
    if any(initial[path] >= limit for path, limit in limits.items()):
        print("Disk budget has insufficient headroom; job was not started.", flush=True)
        return 75
    process = subprocess.Popen(command, start_new_session=True)
    stopped = False
    try:
        while process.poll() is None:
            current = snapshot()
            if any(current[path] >= limit for path, limit in limits.items()):
                print(json.dumps({"disk_budget_stop": current, "pid": process.pid}), flush=True)
                stopped = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(1)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        raise
    print(json.dumps({"disk_end_bytes": snapshot(), "job_exit_code": process.returncode}), flush=True)
    return 75 if stopped else process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
