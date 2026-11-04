"""Record node/process counters alongside real training (no synchronization changes)."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

from diagnose_train_input import monitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Missing training command")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "settings.json").write_text(
        json.dumps(
            {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "hostname": os.uname().nodename,
                "unix_time": time.time(),
                "command": command,
                "interval_seconds": 10,
                "clock_ticks": os.sysconf("SC_CLK_TCK"),
                "note": "Process rchar includes cache hits; read_bytes is not an exact FSx network counter. "
                "Lustre RPC histograms are raw cumulative client counters; network is whole-node. "
                "CloudWatch FSx metrics unavailable with current IAM permissions.",
            },
            indent=2,
        )
    )

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    code = monitor(command, args.output, 10)
    (args.output / "exit.json").write_text(json.dumps({"returncode": code, "unix_time": time.time()}))
    sys.exit(code)


if __name__ == "__main__":
    main()
