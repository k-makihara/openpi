"""Short training throughput sweep; reuses train.py without saving checkpoints."""

import argparse
import contextlib
import csv
import dataclasses
import functools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def run_case(args, *, base_config=None):
    case_started = time.perf_counter()
    import jax
    import numpy as np
    import train

    from openpi.training import config as configs
    from openpi.training import data_loader
    from openpi.training import sharding

    base = base_config if base_config is not None else configs.get_config(args.config)
    data_config = dataclasses.replace(
        base.data,
        repo_id=args.dataset,
        downsample_anchor_frames=args.downsample_anchor_frames,
        excluded_episode_paths=tuple(args.excluded_episode_paths),
    )
    config = dataclasses.replace(
        base,
        batch_size=args.batches[0],
        num_workers=args.workers[0],
        data=data_config,
        exp_name="input_benchmark",
        resume=False,
        overwrite=False,
        wandb_enabled=False,
    )
    if config.batch_size % jax.device_count():
        raise ValueError("Batch size must be divisible by the visible GPU count")
    print(
        json.dumps(
            {
                "config": args.config,
                "dataset": args.dataset,
                "batch_size": config.batch_size,
                "workers": config.num_workers,
                "downsample_anchor_frames": args.downsample_anchor_frames,
                "excluded_episode_paths": args.excluded_episode_paths,
            }
        ),
        flush=True,
    )
    jax.config.update("jax_compilation_cache_dir", str(Path("~/.cache/jax").expanduser()))
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
    iterator = iter(loader)
    train_rng, init_rng = jax.random.split(jax.random.key(config.seed))
    state, state_sharding = train.init_train_state(config, init_rng, mesh, resume=False)
    step_fn = jax.jit(
        functools.partial(train.train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    setup_seconds = time.perf_counter() - case_started
    warmup_steps = max(args.warmup, config.num_workers)
    measure_steps = max(args.steps, config.num_workers)
    elapsed, waits, warmup_elapsed = [], [], []
    last_loss = None
    for index in range(warmup_steps + measure_steps):
        start = time.perf_counter()
        batch = next(iterator)
        # This includes host-to-device readiness, not only file I/O.
        jax.block_until_ready(batch)
        ready = time.perf_counter()
        with sharding.set_mesh(mesh):
            state, info = step_fn(train_rng, state, batch)
        jax.block_until_ready((state, info))
        duration = time.perf_counter() - start
        last_loss = float(info["loss"])
        if not np.isfinite(last_loss):
            raise ValueError(f"Nonfinite loss: {last_loss}")
        if index >= warmup_steps:
            elapsed.append(duration)
            waits.append(ready - start)
        else:
            warmup_elapsed.append(duration)
        phase = "warmup" if index < warmup_steps else "measure"
        print(f"{phase} step={index + 1} seconds={duration:.3f} loss={last_loss:.6f}", flush=True)
    elapsed_array = np.asarray(elapsed)
    wait_array = np.asarray(waits)
    result = {
        "batch_size": config.batch_size,
        "workers": config.num_workers,
        "status": "ok",
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "setup_seconds": setup_seconds,
        "warmup_seconds": float(np.sum(warmup_elapsed)),
        "seconds_per_step": float(np.mean(elapsed_array)),
        "samples_per_second": config.batch_size / float(np.mean(elapsed_array)),
        "data_wait_seconds": float(np.mean(wait_array)),
        "median_seconds": float(np.median(elapsed_array)),
        "step_p50_seconds": float(np.percentile(elapsed_array, 50)),
        "step_p95_seconds": float(np.percentile(elapsed_array, 95)),
        "data_wait_p50_seconds": float(np.percentile(wait_array, 50)),
        "data_wait_p95_seconds": float(np.percentile(wait_array, 95)),
        "last_loss": last_loss,
        "gpu_count": jax.device_count(),
        "downsample_anchor_frames": args.downsample_anchor_frames,
        "excluded_episode_paths": args.excluded_episode_paths,
    }
    args.result.write_text(json.dumps(result, indent=2) + "\n")


def sweep(args):
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "settings.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    columns = [
        "batch_size",
        "workers",
        "status",
        "warmup_steps",
        "measure_steps",
        "setup_seconds",
        "warmup_seconds",
        "seconds_per_step",
        "samples_per_second",
        "data_wait_seconds",
        "median_seconds",
        "step_p50_seconds",
        "step_p95_seconds",
        "data_wait_p50_seconds",
        "data_wait_p95_seconds",
        "last_loss",
        "gpu_count",
        "downsample_anchor_frames",
        "excluded_episode_paths",
    ]
    with (args.output / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        stream.flush()
        for batch in args.batches:
            for workers in args.workers:
                label = f"batch{batch}_workers{workers}"
                result_path = args.output / f"{label}.json"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--case",
                    "--config",
                    args.config,
                    "--dataset",
                    args.dataset,
                    "--batches",
                    str(batch),
                    "--workers",
                    str(workers),
                    "--warmup",
                    str(args.warmup),
                    "--steps",
                    str(args.steps),
                    "--result",
                    str(result_path),
                ]
                command.append(
                    "--downsample-anchor-frames"
                    if args.downsample_anchor_frames
                    else "--no-downsample-anchor-frames"
                )
                for episode_path in args.excluded_episode_paths:
                    command.extend(["--excluded-episode-path", episode_path])
                print(f"Starting {label}; log: {args.output / (label + '.log')}", flush=True)
                with (args.output / f"{label}.log").open("w") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = process.wait(timeout=args.timeout)
                        status = "ok" if code == 0 else f"failed:{code}"
                    except subprocess.TimeoutExpired:
                        status = "timeout"
                    finally:
                        # Also clean up DataLoader workers after failure or interruption.
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                row = {"batch_size": batch, "workers": workers, "status": status}
                if status == "ok" and result_path.exists():
                    row.update(json.loads(result_path.read_text()))
                elif status == "ok":
                    row["status"] = "missing_result"
                writer.writerow(row)
                stream.flush()
                print(json.dumps(row), flush=True)
    print(f"Results: {args.output / 'results.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="pi05_umi_original_dual_arm_relative_state_hz30_h32_bs1024_30k_assemblyecu_20260915"
    )
    parser.add_argument(
        "--dataset", default="/fsx/koshi.makihara/datasets/lerobot_v21_ph2_success_20260915_annex_fixed"
    )
    parser.add_argument("--batches", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--workers", type=int, nargs="+", default=[4, 8])
    parser.add_argument(
        "--downsample-anchor-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Match training by also applying action_sample_stride to anchor frames.",
    )
    parser.add_argument(
        "--excluded-episode-path",
        dest="excluded_episode_paths",
        action="append",
        default=[],
        help="Episode parquet path to exclude. Repeat for multiple episodes.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=1200, help="Maximum seconds per case, including initialization")
    parser.add_argument("--output", type=Path, default=Path("outputs") / time.strftime("input_benchmark_%Y%m%d_%H%M%S"))
    parser.add_argument("--case", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.batches) < 1 or min(args.workers) < 0 or min(args.warmup, args.steps, args.timeout) < 1:
        parser.error("Batches, warmup, steps and timeout must be positive; workers must be nonnegative")
    if args.case:
        run_case(args)
    else:
        sweep(args)


if __name__ == "__main__":
    main()
