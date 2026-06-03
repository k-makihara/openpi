import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _read_episode_table(dataset_dir: str, episode_idx: int) -> pa.Table:
    episode_path = pathlib.Path(dataset_dir) / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {episode_path}")
    return pq.read_table(episode_path)


def _to_np_seq(table: pa.Table, key: str) -> np.ndarray:
    return np.asarray(table[key].to_pylist(), dtype=np.float32)


def _plot_dims(x: np.ndarray, y: np.ndarray, title: str, out_path: pathlib.Path, cols: int = 4) -> None:
    dim = y.shape[1]
    rows = int(np.ceil(dim / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for d in range(dim):
        ax = axes[d]
        ax.plot(x, y[:, d], linewidth=1.1)
        ax.set_title(f"dim {d}", fontsize=9)
        ax.grid(True, alpha=0.25)
    for d in range(dim, len(axes)):
        axes[d].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot raw state/action time-series from one LeRobot episode.")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--out-dir", type=str, default="./outputs/episode_plot")
    args = parser.parse_args()

    table = _read_episode_table(args.dataset_dir, args.episode)
    t = np.arange(table.num_rows, dtype=np.int32)

    left_pose = _to_np_seq(table, "observation.pose.left_left_finger_tip.absolute")
    right_pose = _to_np_seq(table, "observation.pose.right_right_finger_tip.absolute")
    left_gripper_state = _to_np_seq(table, "observation.state.left_gripper")
    right_gripper_state = _to_np_seq(table, "observation.state.right_gripper")

    left_action = _to_np_seq(table, "action.left_controller.relative")
    right_action = _to_np_seq(table, "action.right_controller.relative")
    left_gripper_action_abs = _to_np_seq(table, "action.left_gripper.absolute")
    right_gripper_action_abs = _to_np_seq(table, "action.right_gripper.absolute")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_dims(t, left_pose, f"Episode {args.episode:06d} Left Pose Absolute [xyz+quat]", out_dir / "left_pose_abs.png")
    _plot_dims(t, right_pose, f"Episode {args.episode:06d} Right Pose Absolute [xyz+quat]", out_dir / "right_pose_abs.png")
    _plot_dims(t, left_action, f"Episode {args.episode:06d} Left Action Relative [xyz+quat]", out_dir / "left_action_rel.png")
    _plot_dims(t, right_action, f"Episode {args.episode:06d} Right Action Relative [xyz+quat]", out_dir / "right_action_rel.png")

    plt.figure(figsize=(10, 4))
    plt.plot(t, left_gripper_state.reshape(-1), label="left_gripper_state")
    plt.plot(t, right_gripper_state.reshape(-1), label="right_gripper_state")
    plt.plot(t, left_gripper_action_abs.reshape(-1), label="left_gripper_action_abs")
    plt.plot(t, right_gripper_action_abs.reshape(-1), label="right_gripper_action_abs")
    plt.title(f"Episode {args.episode:06d} Gripper State/Action")
    plt.xlabel("timestep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "gripper_series.png", dpi=150)
    plt.close()

    print(f"[DONE] saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
