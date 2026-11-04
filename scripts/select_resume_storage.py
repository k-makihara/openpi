"""Select storage from the completed benchmark and stage an isolated local copy."""

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def choose(report, remaining_steps):
    results = report["results"]
    for name in ("fsx_before", "fsx_after", "local_nvme"):
        row = results[name]
        if row["status"] != "ok" or row["measure_steps"] < 512:
            raise ValueError(f"Incomplete benchmark: {name}")
        if not math.isfinite(row["seconds_per_step"]) or row["seconds_per_step"] <= 0:
            raise ValueError(f"Invalid timing: {name}")
    fsx = (results["fsx_before"]["seconds_per_step"] + results["fsx_after"]["seconds_per_step"]) / 2
    local = results["local_nvme"]["seconds_per_step"]
    copy = report["copy"]["seconds"]
    if not math.isfinite(copy) or copy < 0 or remaining_steps <= 0:
        raise ValueError("Invalid copy time or remaining steps")
    return {
        "selected": "local_nvme" if copy + remaining_steps * local < remaining_steps * fsx else "fsx",
        "fsx_mean_seconds_per_step": fsx,
        "local_seconds_per_step": local,
        "benchmark_copy_seconds": copy,
        "remaining_steps": remaining_steps,
        "estimated_fsx_seconds": remaining_steps * fsx,
        "estimated_local_seconds_including_copy": copy + remaining_steps * local,
    }


def manifest(root):
    return {str(p.relative_to(root)): p.stat().st_size for p in root.rglob("*") if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    latest = max(int(p.name) for p in args.checkpoints.iterdir() if p.is_dir() and p.name.isdecimal())
    decision = choose(report, 100000 - latest)
    decision.update(
        source=str(args.source),
        report=str(args.report),
        checkpoint_step=latest,
        note="PickPlace timings used as a proxy for Assembly; not a separate Assembly benchmark. "
        "FSx estimate averages the two runs; copy estimate uses the full PickPlace copy time. "
        "Checkpoint writes remain on FSx.",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.source
    if decision["selected"] == "local_nvme":
        base = Path("/opt/dlami/nvme")
        subprocess.run(["mountpoint", "-q", str(base)], check=True)
        device = subprocess.check_output(["findmnt", "-n", "-o", "SOURCE", "-T", str(base)], text=True).strip()
        if not device.startswith("/dev/"):
            raise ValueError(f"Not a local block device: {device}")
        original = manifest(args.source)
        size = sum(original.values())
        if shutil.disk_usage(base).free < size + 20 * 1024**3:
            raise ValueError("Insufficient local free space")
        root = Path(tempfile.mkdtemp(prefix=f"koshi-resume-{args.job_id}-", dir=base))
        # Preserve dataset basename to reuse existing normalization assets.
        target = root / args.source.name
        target.mkdir()
        (args.output / "local_root.txt").write_text(str(root) + "\n")
        started = time.monotonic()
        with (args.output / "rsync.log").open("w") as log:
            subprocess.run(
                ["rsync", "-aL", "--stats", str(args.source) + "/", str(target) + "/"],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if manifest(target) != original:
            raise ValueError("Local copy file paths/sizes differ")
        decision.update(actual_copy_seconds=time.monotonic() - started, copied_bytes=size)
    decision["dataset_path"] = str(target)
    (args.output / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
