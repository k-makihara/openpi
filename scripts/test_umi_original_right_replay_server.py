#!/usr/bin/env python3

import argparse
import json
import pathlib
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from openpi_client import websocket_client_policy

from openpi.policies.umi_original_policy import UmiOriginalInputs


ACTION_DIM_NAMES = [
    "dx",
    "dy",
    "dz",
    "rot6d_0",
    "rot6d_1",
    "rot6d_2",
    "rot6d_3",
    "rot6d_4",
    "rot6d_5",
    "gripper_abs",
]


def _read_episode_table(dataset_dir: pathlib.Path, episode_idx: int):
    episode_path = dataset_dir / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {episode_path}")
    return pq.read_table(
        episode_path,
        columns=[
            "observation.pose.right_controller.absolute",
            "observation.state.right_gripper",
            "action.right_controller.relative",
            "action.right_gripper.absolute",
        ],
    )


def _to_np_seq(table, key: str) -> np.ndarray:
    return np.asarray(table[key].to_pylist(), dtype=np.float32)


class SequentialVideoReader:
    def __init__(self, path: pathlib.Path, *, resize_hw: tuple[int, int] | None = None):
        self.path = path
        self.resize_hw = resize_hw
        self._cv2 = None
        self._cap = None
        self._next_idx = 0
        self._last_frame = None

    def _ensure_open(self) -> None:
        if self._cap is not None:
            return
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing dependency: opencv-python / python3-opencv") from exc
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")

    def _reopen(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._next_idx = 0
        self._last_frame = None
        self._ensure_open()

    def get_frame(self, index: int) -> np.ndarray:
        if index < 0:
            raise IndexError(index)
        self._ensure_open()
        if index < self._next_idx:
            self._reopen()
        while self._next_idx <= index:
            ok, frame_bgr = self._cap.read()
            if not ok:
                raise IndexError(f"Video {self.path} ended before frame {index}")
            frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
            if self.resize_hw is not None:
                width, height = self.resize_hw[1], self.resize_hw[0]
                frame_rgb = self._cv2.resize(frame_rgb, (width, height), interpolation=self._cv2.INTER_AREA)
            self._last_frame = frame_rgb
            self._next_idx += 1
        return np.asarray(self._last_frame, dtype=np.uint8)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def _find_video(dataset_dir: pathlib.Path, key: str, episode_idx: int) -> pathlib.Path:
    path = dataset_dir / "videos" / "chunk-000" / key / f"episode_{episode_idx:06d}.mp4"
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    return path


def _save_matplotlib_plot(
    *,
    step_ids: np.ndarray,
    pred_first: np.ndarray,
    gt_first: np.ndarray,
    out_path: pathlib.Path,
) -> None:
    import matplotlib.pyplot as plt

    dim = pred_first.shape[1]
    cols = 2
    rows = int(np.ceil(dim / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 2.6 * rows), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for d in range(dim):
        ax = axes[d]
        ax.plot(step_ids, gt_first[:, d], label="gt", linewidth=1.3)
        ax.plot(step_ids, pred_first[:, d], label="pred", linewidth=1.1)
        ax.set_title(ACTION_DIM_NAMES[d])
        ax.grid(True, alpha=0.25)
    for d in range(dim, len(axes)):
        axes[d].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("UMI original right server replay test: first action in chunk")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_plotly_plot(
    *,
    step_ids: np.ndarray,
    pred_first: np.ndarray,
    gt_first: np.ndarray,
    out_path: pathlib.Path,
) -> bool:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ModuleNotFoundError:
        return False

    dim = pred_first.shape[1]
    rows = dim
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, subplot_titles=ACTION_DIM_NAMES)
    for d in range(dim):
        fig.add_trace(
            go.Scatter(x=step_ids, y=gt_first[:, d], name=f"{ACTION_DIM_NAMES[d]} gt", mode="lines"),
            row=d + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=step_ids, y=pred_first[:, d], name=f"{ACTION_DIM_NAMES[d]} pred", mode="lines"),
            row=d + 1,
            col=1,
        )
    fig.update_layout(
        height=max(900, 220 * rows),
        title="UMI original right server replay test: first action in chunk",
    )
    fig.write_html(out_path, include_plotlyjs=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay LeRobot UMI-original-right observations into an openpi policy server, "
            "then visualize predicted action time-series against dataset actions."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=pathlib.Path,
        default=pathlib.Path("/home/makihara/genoma_data/rosbags/lerobot_v21_ph2_success"),
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--horizon-steps", type=int, default=10)
    parser.add_argument("--prompt", type=str, default="handover object")
    parser.add_argument("--start-step", type=int, default=-1, help="Default = history_steps - 1")
    parser.add_argument("--end-step", type=int, default=-1, help="Inclusive end; default auto.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--resize", type=int, default=224, help="0 disables resize.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("./outputs/umi_original_right_server"))
    args = parser.parse_args()

    table = _read_episode_table(args.dataset_dir, args.episode)
    right_pose = _to_np_seq(table, "observation.pose.right_controller.absolute")
    right_gripper = _to_np_seq(table, "observation.state.right_gripper").reshape(-1)
    right_action_pose = _to_np_seq(table, "action.right_controller.relative")
    right_action_gripper = _to_np_seq(table, "action.right_gripper.absolute").reshape(-1)

    num_steps = right_pose.shape[0]
    hist = int(args.history_steps)
    requested_horizon = int(args.horizon_steps)
    start_t = hist - 1 if args.start_step < 0 else int(args.start_step)
    end_t = (num_steps - requested_horizon - 1) if args.end_step < 0 else int(args.end_step)
    if end_t < start_t:
        raise ValueError(f"Invalid step range: start={start_t}, end={end_t}, num_steps={num_steps}")

    resize_hw = None if args.resize <= 0 else (int(args.resize), int(args.resize))
    left_reader = SequentialVideoReader(
        _find_video(args.dataset_dir, "observation.image.left", args.episode),
        resize_hw=resize_hw,
    )
    right_reader = SequentialVideoReader(
        _find_video(args.dataset_dir, "observation.image.right", args.episode),
        resize_hw=resize_hw,
    )

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    server_metadata = client.get_server_metadata()
    to_model_space_cache: dict[int, UmiOriginalInputs] = {}
    inferred_horizon = None

    step_ids: list[int] = []
    pred_first_list: list[np.ndarray] = []
    gt_first_list: list[np.ndarray] = []
    pred_chunks: list[np.ndarray] = []
    gt_chunks: list[np.ndarray] = []
    pred_abs_pose_first: list[np.ndarray] = []
    pred_abs_gripper_first: list[float] = []

    try:
        for t in range(start_t, end_t + 1, max(1, int(args.stride))):
            if args.max_steps > 0 and len(step_ids) >= args.max_steps:
                break

            i0 = t - hist + 1
            i1 = t + 1
            obs = {
                "left_image": left_reader.get_frame(t),
                "right_image": right_reader.get_frame(t),
                "pose_seq": right_pose[i0:i1],
                "gripper_seq": right_gripper[i0:i1],
                "prompt": args.prompt,
            }
            pred = client.infer(obs)
            pred_actions = np.asarray(pred["actions"], dtype=np.float32)
            if pred_actions.ndim != 2 or pred_actions.shape[1] != len(ACTION_DIM_NAMES):
                raise ValueError(f"Unexpected predicted action shape at t={t}: {pred_actions.shape}")

            horizon = int(pred_actions.shape[0])
            if inferred_horizon is None:
                inferred_horizon = horizon
                print(f"[INFO] inferred server action horizon = {inferred_horizon}", flush=True)
            if horizon not in to_model_space_cache:
                to_model_space_cache[horizon] = UmiOriginalInputs(
                    history_steps=hist,
                    action_horizon_steps=horizon,
                )
            if (t + horizon) > num_steps:
                print(
                    f"[INFO] stopping at t={t} because dataset tail is shorter than server horizon={horizon}",
                    flush=True,
                )
                break

            gt_obs = {
                **obs,
                "action_pose_seq": right_action_pose[t : t + horizon],
                "action_gripper_seq": right_action_gripper[t : t + horizon],
            }
            gt_actions = np.asarray(to_model_space_cache[horizon](gt_obs)["actions"], dtype=np.float32)
            if pred_actions.shape != gt_actions.shape:
                raise ValueError(f"Shape mismatch at t={t}: pred={pred_actions.shape}, gt={gt_actions.shape}")

            step_ids.append(t)
            pred_first_list.append(pred_actions[0].copy())
            gt_first_list.append(gt_actions[0].copy())
            pred_chunks.append(pred_actions.copy())
            gt_chunks.append(gt_actions.copy())

            right_arm = pred.get("right_arm", {})
            pred_abs_pose = np.asarray(right_arm.get("pose", np.zeros((horizon, 7), dtype=np.float32)), dtype=np.float32)
            pred_abs_gripper = np.asarray(right_arm.get("gripper", np.zeros((horizon,), dtype=np.float32)), dtype=np.float32)
            pred_abs_pose_first.append(pred_abs_pose[0].copy())
            pred_abs_gripper_first.append(float(pred_abs_gripper[0]))

            if len(step_ids) == 1 or len(step_ids) % 20 == 0:
                mse = float(np.mean((pred_actions - gt_actions) ** 2))
                print(f"[INFO] step={len(step_ids)} t={t} mse={mse:.8f}", flush=True)
    finally:
        left_reader.close()
        right_reader.close()

    step_ids_np = np.asarray(step_ids, dtype=np.int32)
    pred_first = np.stack(pred_first_list, axis=0)
    gt_first = np.stack(gt_first_list, axis=0)
    pred_chunks_np = np.stack(pred_chunks, axis=0)
    gt_chunks_np = np.stack(gt_chunks, axis=0)
    pred_abs_pose_first_np = np.stack(pred_abs_pose_first, axis=0)
    pred_abs_gripper_first_np = np.asarray(pred_abs_gripper_first, dtype=np.float32)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{args.episode:06d}_umi_original_right_server"
    npz_path = out_dir / f"{stem}.npz"
    summary_path = out_dir / f"{stem}.json"
    html_path = out_dir / f"{stem}.html"
    png_path = out_dir / f"{stem}.png"

    np.savez_compressed(
        npz_path,
        step_ids=step_ids_np,
        pred_first=pred_first,
        gt_first=gt_first,
        pred_chunks=pred_chunks_np,
        gt_chunks=gt_chunks_np,
        pred_abs_pose_first=pred_abs_pose_first_np,
        pred_abs_gripper_first=pred_abs_gripper_first_np,
    )

    plotly_ok = _save_plotly_plot(step_ids=step_ids_np, pred_first=pred_first, gt_first=gt_first, out_path=html_path)
    _save_matplotlib_plot(step_ids=step_ids_np, pred_first=pred_first, gt_first=gt_first, out_path=png_path)

    mse_all = float(np.mean((pred_chunks_np - gt_chunks_np) ** 2))
    mae_first = np.mean(np.abs(pred_first - gt_first), axis=0)
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "episode": int(args.episode),
        "host": args.host,
        "port": int(args.port),
        "history_steps": hist,
        "requested_horizon_steps": requested_horizon,
        "inferred_server_horizon_steps": inferred_horizon,
        "num_queries": int(len(step_ids)),
        "mse_all": mse_all,
        "mae_first_by_dim": {name: float(mae_first[i]) for i, name in enumerate(ACTION_DIM_NAMES)},
        "server_metadata": server_metadata,
        "artifacts": {
            "npz": str(npz_path),
            "html": str(html_path) if plotly_ok else None,
            "png": str(png_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[DONE] queries={len(step_ids)} mse_all={mse_all:.8f}")
    if plotly_ok:
        print(f"[DONE] html={html_path}")
    print(f"[DONE] png={png_path}")
    print(f"[DONE] npz={npz_path}")
    print(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()
