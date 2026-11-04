from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import pathlib
import re
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.policies.umi_original_policy import _GRIPPER_MAX_TARGET
from openpi.policies.umi_original_policy import _GRIPPER_MIN_TARGET
from openpi.policies.umi_original_policy import _GRIPPER_WRAP_OFFSET
from openpi.policies.umi_original_policy import _GRIPPER_WRAP_THRESHOLD
from openpi.policies.umi_original_policy import _forward_fill_zeros

_EPISODE_PATTERN = re.compile(r"episode_(\d+)\.parquet$")
_MARKER_RELATIVE_PATH = pathlib.Path("meta/gripper_transform.json")


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: pathlib.Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _copy_or_link(src: pathlib.Path, dst: pathlib.Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _validate_paths(src_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    if not (src_dir / "meta/info.json").is_file():
        raise FileNotFoundError(f"LeRobot meta/info.json not found under: {src_dir}")
    if (src_dir / _MARKER_RELATIVE_PATH).is_file():
        raise ValueError(f"Dataset is already marked as gripper-transformed: {src_dir}")
    if src_dir == out_dir or src_dir in out_dir.parents or out_dir in src_dir.parents:
        raise ValueError("Source and output directories must be separate, non-nested paths.")


def _prepare_output(src_dir: pathlib.Path, out_dir: pathlib.Path, *, overwrite: bool, copy_videos: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {out_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(out_dir)

    for src_path in src_dir.rglob("*"):
        if not src_path.is_file():
            continue
        relative_path = src_path.relative_to(src_dir)
        if relative_path.parts[0] == "data" and src_path.suffix == ".parquet":
            continue
        dst_path = out_dir / relative_path
        is_video = relative_path.parts[0] == "videos"
        _copy_or_link(src_path, dst_path, copy=copy_videos if is_video else True)


def _episode_index(path: pathlib.Path) -> int:
    match = _EPISODE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected episode parquet filename: {path}")
    return int(match.group(1))


def _read_sequence(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(table[column].to_pylist(), dtype=np.float64).reshape(-1)


def _set_sequence(table: pa.Table, column: str, values: np.ndarray) -> pa.Table:
    field_index = table.schema.get_field_index(column)
    target_type = table.schema.field(field_index).type
    array = pa.array(np.asarray(values).reshape(-1, 1).tolist(), type=target_type)
    return table.set_column(field_index, column, array)


def _column_stats(values: np.ndarray) -> dict[str, list[float | int]]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("Transformed gripper values still contain NaN or infinity.")
    return {
        "count": [int(values.size)],
        "max": [float(np.max(values))],
        "mean": [float(np.mean(values))],
        "min": [float(np.min(values))],
        "std": [float(np.std(values))],
    }


def _transform_side(table: pa.Table, side: str) -> tuple[pa.Table, dict[str, np.ndarray]]:
    state_column = f"observation.state.{side}_gripper"
    absolute_column = f"action.{side}_gripper.absolute"
    relative_column = f"action.{side}_gripper.relative"
    if state_column not in table.schema.names:
        return table, {}

    raw_state = _read_sequence(table, state_column)
    transformed_state = _forward_fill_zeros(raw_state)
    transformed = {state_column: transformed_state}

    if absolute_column in table.schema.names:
        raw_absolute = _read_sequence(table, absolute_column)
        expected_absolute = np.concatenate([raw_state[1:], raw_state[-1:]])
        if not np.allclose(raw_absolute, expected_absolute, rtol=0.0, atol=1e-8, equal_nan=True):
            raise ValueError(
                f"{absolute_column} is not the next-frame value of {state_column}; cannot safely reconstruct it."
            )
        transformed_absolute = np.concatenate([transformed_state[1:], transformed_state[-1:]])
        transformed[absolute_column] = transformed_absolute

        if relative_column in table.schema.names:
            raw_relative = _read_sequence(table, relative_column)
            if not np.allclose(raw_relative, raw_absolute - raw_state, rtol=0.0, atol=1e-8, equal_nan=True):
                raise ValueError(
                    f"{relative_column} is not {absolute_column} - {state_column}; cannot safely reconstruct it."
                )
            transformed[relative_column] = transformed_absolute - transformed_state
    elif relative_column in table.schema.names:
        raise ValueError(f"Found {relative_column} without required {absolute_column}.")

    for column, values in transformed.items():
        table = _set_sequence(table, column, values)
    return table, transformed


def _update_episode_stats(row: dict, transformed_columns: dict[str, np.ndarray]) -> dict:
    row = dict(row)
    stats = dict(row.get("stats", {}))
    for column, values in transformed_columns.items():
        stats[column] = _column_stats(values)
    row["stats"] = stats
    return row


def transform_dataset(args: argparse.Namespace) -> None:
    src_dir = args.dataset_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    _validate_paths(src_dir, out_dir)
    _prepare_output(
        src_dir,
        out_dir,
        overwrite=args.overwrite,
        copy_videos=args.copy_videos,
    )

    episode_stats = _read_jsonl(src_dir / "meta/episodes_stats.jsonl")
    stats_by_episode = {int(row["episode_index"]): row for row in episode_stats}
    transformed_stats: dict[int, dict] = {}
    aggregate: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    changed_frames = {"left": 0, "right": 0}

    parquet_files = sorted((src_dir / "data").rglob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files found under: {src_dir / 'data'}")

    for src_path in parquet_files:
        episode_index = _episode_index(src_path)
        table = pq.read_table(src_path)
        transformed_columns: dict[str, np.ndarray] = {}

        for side in ("left", "right"):
            state_column = f"observation.state.{side}_gripper"
            raw_state = _read_sequence(table, state_column) if state_column in table.schema.names else None
            table, side_columns = _transform_side(table, side)
            transformed_columns.update(side_columns)
            if raw_state is not None:
                transformed_state = side_columns[state_column]
                aggregate[side].append(transformed_state)
                changed_frames[side] += int(
                    np.count_nonzero(~np.isclose(raw_state, transformed_state, rtol=0.0, atol=0.0, equal_nan=True))
                )

        if not transformed_columns:
            raise ValueError(f"No supported gripper columns found in: {src_path}")

        dst_path = out_dir / src_path.relative_to(src_dir)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dst_path)

        if episode_index in stats_by_episode:
            transformed_stats[episode_index] = _update_episode_stats(
                stats_by_episode[episode_index],
                transformed_columns,
            )

    if episode_stats:
        missing_stats = set(stats_by_episode) - set(transformed_stats)
        if missing_stats:
            raise ValueError(f"Missing parquet files for episode stats: {sorted(missing_stats)[:10]}")
        _write_jsonl(
            out_dir / "meta/episodes_stats.jsonl",
            [transformed_stats[int(row["episode_index"])] for row in episode_stats],
        )

    sides = {}
    for side, sequences in aggregate.items():
        if not sequences:
            continue
        values = np.concatenate(sequences)
        sides[side] = {
            "frames": int(values.size),
            "changed_frames": changed_frames[side],
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    marker = {
        "transform": "openpi_umi_gripper_v2",
        "source_dataset": str(src_dir),
        "scope": "episode_per_side",
        "fill_epsilon": 1e-8,
        "low_branch_range": [0.0, _GRIPPER_WRAP_THRESHOLD],
        "low_branch_offset": _GRIPPER_WRAP_OFFSET,
        "maximum_target": _GRIPPER_MAX_TARGET,
        "minimum_target": _GRIPPER_MIN_TARGET,
        "sides": sides,
    }
    marker_path = out_dir / _MARKER_RELATIVE_PATH
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n")

    print(f"Source: {src_dir}")
    print(f"Output: {out_dir}")
    print(f"Episodes: {len(parquet_files)}")
    for side, stats in sides.items():
        print(
            f"{side.title()}: frames={stats['frames']} changed={stats['changed_frames']} "
            f"range={stats['min']:.6f}..{stats['max']:.6f}"
        )
    print(f"Marker: {marker_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a LeRobot dataset with episode-wise UMI gripper correction applied."
    )
    parser.add_argument("--dataset-dir", type=pathlib.Path, required=True, help="Source LeRobot dataset root.")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True, help="Output dataset root to create.")
    parser.add_argument("--copy-videos", action="store_true", help="Copy videos instead of hardlinking when possible.")
    parser.add_argument("--overwrite", action="store_true", help="Replace out-dir if it already exists.")
    transform_dataset(parser.parse_args())


if __name__ == "__main__":
    main()
