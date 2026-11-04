import argparse
import pathlib
import re

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from openpi.policies.umi_original_policy import _GRIPPER_MAX_TARGET
from openpi.policies.umi_original_policy import _GRIPPER_MIN_TARGET
from openpi.policies.umi_original_policy import _GRIPPER_WRAP_OFFSET
from openpi.policies.umi_original_policy import _GRIPPER_WRAP_THRESHOLD
from openpi.policies.umi_original_policy import _forward_fill_zeros

_EPISODE_PATTERN = re.compile(r"episode_(\d+)\.parquet$")


def _episode_files(dataset_dir: pathlib.Path) -> list[tuple[int, pathlib.Path]]:
    episodes = []
    for path in dataset_dir.glob("data/chunk-*/episode_*.parquet"):
        match = _EPISODE_PATTERN.search(path.name)
        if match:
            episodes.append((int(match.group(1)), path))
    if not episodes:
        raise FileNotFoundError(f"No episode parquet files found under {dataset_dir / 'data'}")
    return sorted(episodes)


def _read_gripper(path: pathlib.Path, side: str) -> np.ndarray:
    column = f"observation.state.{side}_gripper"
    table = pq.read_table(path, columns=[column])
    return np.asarray(table[column].to_pylist(), dtype=np.float32).reshape(-1)


def _changed_mask(raw: np.ndarray, filled: np.ndarray) -> np.ndarray:
    return (~np.isfinite(raw)) | (~np.isclose(raw, filled, rtol=0.0, atol=0.0, equal_nan=True))


def _plot_episode(
    episode: int,
    side: str,
    raw: np.ndarray,
    filled: np.ndarray,
    out_path: pathlib.Path,
) -> None:
    frame = np.arange(raw.size)
    changed = _changed_mask(raw, filled)

    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(frame, raw, color="#d95f59", linewidth=1.0, alpha=0.65, label="raw")
    ax.plot(frame, filled, color="#2369bd", linewidth=1.25, label="filled + shifted")
    if np.any(changed):
        ax.scatter(frame[changed], filled[changed], color="#e69822", s=14, zorder=3, label="changed point")
    ax.set_title(f"Episode {episode:06d} {side.title()} Gripper: Raw vs Transformed")
    ax.set_xlabel("frame")
    ax.set_ylabel("gripper value")
    ax.grid(visible=True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_overlay(
    episodes: list[tuple[int, np.ndarray, np.ndarray]],
    side: str,
    out_path: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for _, _, filled in episodes:
        normalized_frame = np.linspace(0.0, 1.0, filled.size)
        ax.plot(normalized_frame, filled, linewidth=0.7, alpha=0.18, color="#2369bd")
    ax.set_title(f"{side.title()} Gripper After Fill + Unwrap + Bound Alignment: All Episodes")
    ax.set_xlabel("normalized episode time")
    ax.set_ylabel("transformed gripper value")
    ax.grid(visible=True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_grid(
    episodes: list[tuple[int, np.ndarray, np.ndarray]],
    side: str,
    out_path: pathlib.Path,
) -> None:
    columns = 4
    rows = int(np.ceil(len(episodes) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(13.44, 1.6 * rows), squeeze=False)
    for ax, (episode, raw, filled) in zip(axes.reshape(-1), episodes, strict=False):
        frame = np.arange(raw.size)
        ax.plot(frame, raw, color="#d95f59", linewidth=0.55, alpha=0.55)
        ax.plot(frame, filled, color="#2369bd", linewidth=0.75)
        ax.set_title(f"ep {episode:06d}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(visible=True, alpha=0.2)
    for ax in axes.reshape(-1)[len(episodes) :]:
        ax.axis("off")
    fig.suptitle(f"{side.title()} Gripper: Raw (red) / Transformed (blue)", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.998))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_summary(
    dataset_dir: pathlib.Path,
    out_dir: pathlib.Path,
    side: str,
    episodes: list[tuple[int, np.ndarray, np.ndarray]],
) -> None:
    total_frames = sum(raw.size for _, raw, _ in episodes)
    raw_zeros = sum(np.count_nonzero(np.isfinite(raw) & (np.abs(raw) <= 1e-8)) for _, raw, _ in episodes)
    filled_zeros = sum(
        np.count_nonzero(np.isfinite(filled) & (np.abs(filled) <= 1e-8)) for _, _, filled in episodes
    )
    changed = sum(np.count_nonzero(_changed_mask(raw, filled)) for _, raw, filled in episodes)
    raw_values = np.concatenate([raw for _, raw, _ in episodes])
    filled_values = np.concatenate([filled for _, _, filled in episodes])
    raw_low_branch = np.count_nonzero(
        np.isfinite(raw_values) & (raw_values >= 0.0) & (raw_values <= _GRIPPER_WRAP_THRESHOLD)
    )
    filled_low_branch = np.count_nonzero(
        np.isfinite(filled_values) & (filled_values >= 0.0) & (filled_values <= _GRIPPER_WRAP_THRESHOLD)
    )
    episode_rows = []
    for episode, raw, filled in episodes:
        raw_zero_count = np.count_nonzero(np.isfinite(raw) & (np.abs(raw) <= 1e-8))
        filled_zero_count = np.count_nonzero(np.isfinite(filled) & (np.abs(filled) <= 1e-8))
        changed_count = np.count_nonzero(_changed_mask(raw, filled))
        if raw_zero_count or changed_count:
            episode_rows.append(f"{episode},{raw_zero_count},{filled_zero_count},{changed_count}")

    summary = [
        f"Dataset: {dataset_dir}",
        f"Output: {out_dir}",
        f"Side: {side}",
        f"Episodes: {len(episodes)}",
        f"Frames: {total_frames}",
        f"Raw min/max: {np.nanmin(raw_values)} / {np.nanmax(raw_values)}",
        f"Transformed min/max: {np.nanmin(filled_values)} / {np.nanmax(filled_values)}",
        f"Raw zero/dropout frames: {raw_zeros}",
        f"Transformed zero frames: {filled_zeros}",
        f"Raw low-branch frames [0, {_GRIPPER_WRAP_THRESHOLD}]: {raw_low_branch}",
        f"Transformed low-branch frames [0, {_GRIPPER_WRAP_THRESHOLD}]: {filled_low_branch}",
        f"Frames changed by fill/unwrap/bound alignment: {changed}",
        "Fill: leading zeros use the first nonzero value; later zeros/NaNs use the previous nonzero value.",
        f"Shift: values from 0 through {_GRIPPER_WRAP_THRESHOLD} add one rotation "
        f"({_GRIPPER_WRAP_OFFSET}).",
        f"Alignment: if the transformed maximum exceeds {_GRIPPER_MAX_TARGET}, subtract the excess "
        "from every finite value.",
        f"Alignment: if the transformed minimum is below {_GRIPPER_MIN_TARGET}, add the shortfall "
        "to every finite value.",
        "No clipping is applied.",
        f"Per-episode images: episode_XXXXXX_{side}_gripper_fill.png",
        f"Overview: overview_overlay_{side}_gripper_fill.png",
        f"Grid: overview_grid_{side}_gripper_fill.png",
        "",
        "episode,raw_zero_frames,after_fill_zero_frames,changed_frames",
        *episode_rows,
        "",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot raw and filled LeRobot gripper values for every episode.")
    parser.add_argument("--dataset-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    for episode, path in _episode_files(dataset_dir):
        raw = _read_gripper(path, args.side)
        filled = _forward_fill_zeros(raw)
        episodes.append((episode, raw, filled))
        _plot_episode(
            episode,
            args.side,
            raw,
            filled,
            out_dir / f"episode_{episode:06d}_{args.side}_gripper_fill.png",
        )

    _plot_overlay(episodes, args.side, out_dir / f"overview_overlay_{args.side}_gripper_fill.png")
    _plot_grid(episodes, args.side, out_dir / f"overview_grid_{args.side}_gripper_fill.png")
    _write_summary(dataset_dir, out_dir, args.side, episodes)
    print(f"Saved {args.side} gripper plots for {len(episodes)} episodes to {out_dir}")


if __name__ == "__main__":
    main()
