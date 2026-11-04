# ruff: noqa: SLF001
"""Compare temporal table reads and video decoders using identical training samples.

Default: batch=256, workers=64, four legacy/grouped x pyav/torchcodec cases.
A separate CPU process first verifies exact table-mode equivalence on real samples
and reports image differences between decoders. Every timed case starts a fresh
model/loader with the same seed. Input errors fail a case instead of skipping data.
No checkpoints or W&B runs are created. Measurements synchronize JAX each step;
compare these cases to one another, not directly to asynchronous train.py timings.
"""

import argparse
import contextlib
import csv
import dataclasses
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types

import benchmark_train_input as benchmark

DEFAULT_CONFIG = "pi05_umi_original_dual_arm_relative_state_hz10_h50_bs512_100k_PickAndPlace_merged_gripper"
DEFAULT_DATASET = "/fsx/koshi.makihara/datasets_tasked/lerobot_v21_ph2_PickAndPlace_merged_gripper"


def validate(args):
    import jax
    import numpy as np
    import torch

    from openpi.training import data_loader as dl

    torch.set_num_threads(1)
    config = benchmark.make_config(args)
    dc = config.data.create(config.assets_dirs, config.model)
    dc = dl._configure_preprocessed_gripper_data(dataclasses.replace(dc, lerobot_table_query_mode="grouped"))
    dataset = benchmark.disable_sample_retries(dl.create_torch_dataset(dc, config.model.action_horizon, config.model))
    raw = dataset
    while hasattr(raw, "_dataset"):
        raw = raw._dataset
    if not isinstance(raw, dl.GroupedQueryLeRobotDataset):
        raise TypeError("Validation requires a LeRobot dataset with grouped temporal queries")
    # Norms don't affect table/decoder equivalence; include them whenever available.
    transformed = dl.transform_dataset(dataset, dc, skip_norm_stats=dc.norm_stats is None)
    indices = list(
        dict.fromkeys(
            [
                0,
                len(dataset) - 1,
                *np.random.default_rng(config.seed)
                .integers(
                    0,
                    len(dataset),
                    size=max(0, args.verify_samples - 2),
                )
                .tolist(),
            ]
        )
    )
    methods = {
        "legacy": dl.lerobot_dataset.LeRobotDataset._query_hf_dataset,
        "grouped": dl.GroupedQueryLeRobotDataset._query_hf_dataset,
    }
    image_max = 0.0
    image_sum = 0.0
    image_values = 0
    for index in indices:
        reference = None
        for backend in args.backends:
            raw.video_backend = backend
            raw._query_hf_dataset = types.MethodType(methods["legacy"], raw)
            legacy = transformed[index]
            raw._query_hf_dataset = types.MethodType(methods["grouped"], raw)
            grouped = transformed[index]
            assert jax.tree.structure(legacy) == jax.tree.structure(grouped)
            for a, b in zip(jax.tree.leaves(legacy), jax.tree.leaves(grouped), strict=True):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
            if reference is not None:
                assert legacy.keys() == reference.keys()
                for key in legacy:
                    if key == "image":
                        assert legacy[key].keys() == reference[key].keys()
                        for camera, image in legacy[key].items():
                            a, b = np.asarray(image), np.asarray(reference[key][camera])
                            assert a.shape == b.shape
                            assert a.dtype == b.dtype
                            difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
                            image_max = max(image_max, float(difference.max()))
                            image_sum += float(difference.sum())
                            image_values += difference.size
                    else:
                        assert jax.tree.structure(legacy[key]) == jax.tree.structure(reference[key])
                        for a, b in zip(jax.tree.leaves(legacy[key]), jax.tree.leaves(reference[key]), strict=True):
                            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
            reference = legacy
        print(f"Validated sample {index}: table modes exactly equal", flush=True)
    report = {
        "status": "ok",
        "sample_indices": indices,
        "backends": args.backends,
        "table_modes_exactly_equal": True,
        "backend_images_compared": len(args.backends) > 1,
        "backend_images_exactly_equal": image_max == 0 if len(args.backends) > 1 else None,
        "backend_image_max_abs_difference": image_max,
        "backend_image_mean_abs_difference": image_sum / image_values if image_values else None,
        "note": "Image differences are reported, not hidden; inspect them before choosing a decoder.",
    }
    args.result.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def execute(command, log_path, timeout, *, cpu=False):
    env = os.environ.copy()
    if cpu:
        env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            return "ok" if code == 0 else f"failed:{code}"
        except subprocess.TimeoutExpired:
            return "timeout"
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)
            # Remove any workers remaining in the child's process group before the next case.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def common_command(args):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        args.config,
        "--dataset",
        args.dataset,
        "--warmup",
        str(args.warmup),
        "--steps",
        str(args.steps),
    ]
    command.append("--downsample-anchor-frames" if args.downsample_anchor_frames else "--no-downsample-anchor-frames")
    if getattr(args, "norm_asset_id", None):
        command += ["--norm-asset-id", args.norm_asset_id]
    for path in args.excluded_episode_paths:
        command += ["--excluded-episode-path", path]
    return command


def sweep(args):
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "settings.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    if args.verify_samples:
        result = args.output / "validation.json"
        command = [
            *common_command(args),
            "--validate",
            "--verify-samples",
            str(args.verify_samples),
            "--backends",
            *args.backends,
            "--result",
            str(result),
        ]
        print(f"Checking table equivalence and video decoding: {args.output / 'validation.log'}", flush=True)
        status = execute(command, args.output / "validation.log", args.timeout, cpu=True)
        if status != "ok" or not result.exists():
            raise RuntimeError(f"Validation {status}; inspect {args.output / 'validation.log'}")
        validation = json.loads(result.read_text())
        if validation["backend_images_exactly_equal"] is False:
            print("WARNING: decoder images differ; inspect validation.json before choosing a decoder", flush=True)

    columns = [
        "repeat",
        "batch_size",
        "workers",
        "video_backend",
        "table_query_mode",
        "status",
        "warmup_steps",
        "measure_steps",
        "samples_per_second",
        "seconds_per_step",
        "data_wait_seconds",
        "compute_seconds",
        "data_wait_fraction",
        "step_p50_seconds",
        "step_p95_seconds",
        "step_max_seconds",
        "data_wait_max_seconds",
        "setup_seconds",
        "warmup_seconds",
        "last_loss",
    ]
    cases = list(itertools.product(args.batches, args.workers, args.backends, args.table_query_modes))
    failed = False
    with (args.output / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        stream.flush()
        for repeat in range(1, args.repeats + 1):
            # Reversing the second pass helps reveal order/filesystem-cache effects.
            for batch, workers, backend, mode in cases if repeat % 2 else reversed(cases):
                label = f"r{repeat}_batch{batch}_workers{workers}_{backend}_{mode}"
                result = args.output / f"{label}.json"
                command = [
                    *common_command(args),
                    "--case",
                    "--batches",
                    str(batch),
                    "--workers",
                    str(workers),
                    "--backends",
                    backend,
                    "--table-query-modes",
                    mode,
                    "--result",
                    str(result),
                ]
                print(f"Starting {label}; log: {args.output / (label + '.log')}", flush=True)
                status = execute(command, args.output / f"{label}.log", args.timeout)
                row = {
                    "repeat": repeat,
                    "batch_size": batch,
                    "workers": workers,
                    "video_backend": backend,
                    "table_query_mode": mode,
                    "status": status,
                }
                if status == "ok" and result.exists():
                    row.update(json.loads(result.read_text()))
                elif status == "ok":
                    row["status"] = "missing_result"
                failed |= row["status"] != "ok"
                writer.writerow(row)
                stream.flush()
                print(json.dumps(row), flush=True)
    print(f"Results: {args.output / 'results.csv'}", flush=True)
    if failed:
        raise SystemExit("Some cases failed or timed out; see results.csv and the corresponding logs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--norm-asset-id", help="Reuse existing state/action normalization assets for a resized dataset")
    parser.add_argument("--batches", type=int, nargs="+", default=[256])
    parser.add_argument("--workers", type=int, nargs="+", default=[64])
    parser.add_argument("--backends", choices=["pyav", "torchcodec"], nargs="+", default=["pyav", "torchcodec"])
    parser.add_argument("--table-query-modes", choices=["legacy", "grouped"], nargs="+", default=["legacy", "grouped"])
    parser.add_argument("--downsample-anchor-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--excluded-episode-path", dest="excluded_episode_paths", action="append", default=[])
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--verify-samples", type=int, default=16, help="Real-data equivalence checks; 0 disables")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat in alternating order to check cache effects")
    parser.add_argument("--timeout", type=int, default=28800, help="Seconds per condition, including initialization")
    parser.add_argument("--output", type=Path, default=Path("outputs") / time.strftime("input_variants_%Y%m%d_%H%M%S"))
    parser.add_argument("--case", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--validate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.batches) < 1 or min(args.workers) < 0 or min(args.warmup, args.steps, args.timeout, args.repeats) < 1:
        parser.error("Invalid batch size, workers, warmup, steps, timeout or repeats")
    if args.verify_samples != 0 and args.verify_samples < 2:
        parser.error("verify-samples must be 0 or at least 2")
    if args.steps < 4 * max(args.workers):
        parser.error("Use at least 4 * max(workers) measured steps to capture repeated input stalls")
    if (args.case or args.validate) and args.result is None:
        parser.error("Child processes require --result")
    args.video_backend = args.backends[0]
    args.table_query_mode = args.table_query_modes[0]
    args.strict_samples = True
    args.record_variant_metrics = True
    if args.validate:
        validate(args)
    elif args.case:
        benchmark.run_case(args)
    else:
        sweep(args)


if __name__ == "__main__":
    main()
