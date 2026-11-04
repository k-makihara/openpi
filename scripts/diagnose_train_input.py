"""Long GPU input benchmark with timestamped /proc telemetry; no checkpoints."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read(path):
    try:
        return Path(path).read_text()
    except (OSError, UnicodeError):
        return None


def process_snapshot(pid):
    root = Path("/proc") / str(pid)
    raw = read(root / "stat")
    if raw is None:
        return None
    fields = raw[raw.rfind(")") + 2 :].split()
    status = read(root / "status") or ""
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "comm": raw[raw.find("(") + 1 : raw.rfind(")")],
        "state": fields[0],
        "cpu_ticks": int(fields[11]) + int(fields[12]),
        "start_ticks": int(fields[19]),
        "minor_faults": int(fields[7]),
        "major_faults": int(fields[9]),
        "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
        "io_delay_ticks": int(fields[39]) if len(fields) > 39 else None,
        "status": {
            k: v.strip()
            for k, v in (line.split(":", 1) for line in status.splitlines() if ":" in line)
            if k
            in {
                "VmRSS",
                "VmSwap",
                "RssAnon",
                "RssFile",
                "Threads",
                "Cpus_allowed_list",
                "voluntary_ctxt_switches",
                "nonvoluntary_ctxt_switches",
            }
        },
        "io": read(root / "io"),
        "wchan": read(root / "wchan"),
    }


def monitor(command, output, interval):
    """Monitor child descendants; flush records so stalls and crashes retain evidence."""
    previous = {}
    ticks = os.sysconf("SC_CLK_TCK")
    with (output / "train.log").open("w") as log, (output / "resources.jsonl").open("w", buffering=1) as stream:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                now, mono = time.time(), time.monotonic()
                processes = {}
                for path in Path("/proc").iterdir():
                    if path.name.isdigit():
                        with contextlib.suppress(OSError, ValueError, IndexError):
                            item = process_snapshot(int(path.name))
                            if item:
                                processes[item["pid"]] = item
                selected = {child.pid}
                while True:
                    expanded = selected | {pid for pid, item in processes.items() if item["ppid"] in selected}
                    if expanded == selected:
                        break
                    selected = expanded
                rows = []
                for pid in selected:
                    if pid not in processes:
                        continue
                    item = processes[pid]
                    old = previous.get((pid, item["start_ticks"]))
                    item["cpu_percent"] = (
                        (100 * (item["cpu_ticks"] - old[1]) / ticks / (mono - old[0])) if old else None
                    )
                    previous[(pid, item["start_ticks"])] = (mono, item["cpu_ticks"])
                    rows.append(item)
                record = {
                    "unix_time": now,
                    "monotonic": mono,
                    "processes": rows,
                    "host": {
                        name: read("/proc/" + name)
                        for name in ["stat", "meminfo", "vmstat", "pressure/cpu", "pressure/io", "pressure/memory"]
                    },
                }
                record["lustre"] = {
                    str(path): read(path)
                    for base in ["/proc/fs/lustre/llite", "/sys/fs/lustre/llite"]
                    for path in Path(base).glob("*/stats")
                }
                try:
                    rpc = subprocess.run(
                        ["/usr/sbin/lctl", "get_param", "osc.*.rpc_stats"],
                        capture_output=True,
                        text=True,
                        timeout=3,
                        check=False,
                    )
                    record["lustre_rpc"] = {"stdout": rpc.stdout, "stderr": rpc.stderr, "returncode": rpc.returncode}
                except (OSError, subprocess.TimeoutExpired) as exc:
                    record["lustre_rpc"] = {"error": str(exc)}
                record["network_interfaces"] = read("/proc/net/dev")
                stream.write(json.dumps(record) + "\n")
                if child.poll() is not None:
                    return child.returncode
                with contextlib.suppress(subprocess.TimeoutExpired):
                    child.wait(timeout=interval)
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                child.wait(timeout=10)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hz", type=int, choices=[10, 30], default=10)
    parser.add_argument("--steps", type=int, default=1200, help="Measured steps after warmup")
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.interval <= 0 or args.steps < 1 or args.warmup < 1:
        parser.error("interval, steps and warmup must be positive")
    if args.child:
        import benchmark_train_input as benchmark

        args.config = (
            f"pi05_umi_original_dual_arm_relative_state_hz{args.hz}_h50_bs512_100k_PickAndPlace_merged_gripper_224"
        )
        args.dataset = "/fsx/koshi.makihara/datasets_tasked/lerobot_v21_ph2_PickAndPlace_merged_gripper_224"
        args.batches, args.workers = [512], [32]
        args.downsample_anchor_frames = True
        args.excluded_episode_paths = []
        args.video_backend, args.table_query_mode = "pyav", "grouped"
        args.strict_samples = True
        args.record_variant_metrics = True
        args.result = args.output / "result.json"
        with (args.output / "events.jsonl").open("w", buffering=1) as stream:

            def event(**values):
                stream.write(json.dumps({"unix_time": time.time(), "monotonic": time.monotonic(), **values}) + "\n")

            args.diagnostic_event = event
            benchmark.run_case(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        **vars(args),
        "output": str(args.output),
        "hostname": os.uname().nodename,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "clock_ticks": os.sysconf("SC_CLK_TCK"),
        "io_delay_accounting": read("/proc/sys/kernel/task_delayacct"),
        "note": "CPU percent: 100 = one core. Host counters are cumulative; use interval deltas. "
        "RSS is shared-page inclusive; do not sum as physical memory. "
        "Zero io_delay does not rule out FSx waits. JAX is synchronized each step.",
    }
    (args.output / "settings.json").write_text(json.dumps(metadata, indent=2))
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--child",
        "--hz",
        str(args.hz),
        "--steps",
        str(args.steps),
        "--warmup",
        str(args.warmup),
        "--output",
        str(args.output),
    ]
    print(f"Diagnostics: {args.output}", flush=True)

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    code = monitor(command, args.output, args.interval)
    (args.output / "exit.json").write_text(json.dumps({"returncode": code, "unix_time": time.time()}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
