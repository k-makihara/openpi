#!/usr/bin/env python3

import argparse
import json
import pathlib
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from openpi_client import websocket_client_policy

from openpi.policies.umi_original_policy import (
    UmiOriginalInputs,
    UmiOriginalRightPrevAndCurrentThirdSlotInputs,
    UmiOriginalRightThirdSlotInputs,
)


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


def _quat_angle_deg(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> float:
    q1 = np.asarray(q1_xyzw, dtype=np.float64)
    q2 = np.asarray(q2_xyzw, dtype=np.float64)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    dot = float(np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _read_episode_table(dataset_dir: pathlib.Path, episode_idx: int):
    episode_path = dataset_dir / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {episode_path}")
    return pq.read_table(
        episode_path,
        columns=[
            "observation.pose.right_controller.absolute",
            "observation.state.right_gripper",
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
    pred_chunks: np.ndarray,
    gt_chunks: np.ndarray,
    out_path: pathlib.Path,
) -> None:
    import matplotlib.pyplot as plt

    dim = pred_chunks.shape[2]
    cols = 2
    rows = int(np.ceil(dim / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 2.6 * rows), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    horizon = pred_chunks.shape[1]
    for d in range(dim):
        ax = axes[d]
        gt_label_used = False
        pred_label_used = False
        marker_label_used = False
        for q_idx, t in enumerate(step_ids):
            x = t + np.arange(horizon, dtype=np.int32)
            gt_y = gt_chunks[q_idx, :, d]
            pred_y = pred_chunks[q_idx, :, d]
            ax.plot(
                x,
                gt_y,
                color="tab:blue",
                linewidth=0.9,
                alpha=0.35,
                label="gt chunk" if not gt_label_used else None,
            )
            ax.plot(
                x,
                pred_y,
                color="tab:orange",
                linewidth=0.9,
                alpha=0.35,
                label="pred chunk" if not pred_label_used else None,
            )
            ax.scatter(
                [t],
                [gt_y[0]],
                color="tab:red",
                s=14,
                zorder=4,
                label="inference step" if not marker_label_used else None,
            )
            gt_label_used = True
            pred_label_used = True
            marker_label_used = True
        ax.set_title(ACTION_DIM_NAMES[d])
        ax.grid(True, alpha=0.25)
    for d in range(dim, len(axes)):
        axes[d].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("UMI original right server replay test: all chunk actions with inference steps")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _make_right_image_input(
    *,
    right_reader: SequentialVideoReader,
    t: int,
    image_layout: str,
    for_server: bool,
) -> np.ndarray:
    current_image = right_reader.get_frame(t)
    if (not for_server) and image_layout == "prev_current_third_slot":
        prev_t = max(0, t - 1)
        prev_image = right_reader.get_frame(prev_t)
        return np.stack([prev_image, current_image], axis=0)
    return current_image


def _make_model_input_transform(image_layout: str, history_steps: int, action_horizon_steps: int):
    if image_layout == "third_slot_only":
        return UmiOriginalRightThirdSlotInputs(
            history_steps=history_steps,
            action_horizon_steps=action_horizon_steps,
            action_pose_target="relative",
        )
    if image_layout == "prev_current_third_slot":
        return UmiOriginalRightPrevAndCurrentThirdSlotInputs(
            history_steps=history_steps,
            action_horizon_steps=action_horizon_steps,
            action_pose_target="relative",
        )
    return UmiOriginalInputs(
        history_steps=history_steps,
        action_horizon_steps=action_horizon_steps,
        action_pose_target="relative",
    )


def _resolve_hist_and_horizon(server_metadata: dict, args) -> tuple[int, int]:
    hist = int(args.history_steps) if int(args.history_steps) > 0 else int(
        server_metadata.get("umi_history_steps", server_metadata.get("history_steps", 4))
    )
    horizon = int(args.horizon_steps) if int(args.horizon_steps) > 0 else int(
        server_metadata.get("umi_action_horizon_steps", server_metadata.get("action_horizon_steps", 10))
    )
    return hist, horizon


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay raw LeRobot UMI observations into an OpenPI policy server and "
            "compare predicted relative actions against GT action targets."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=pathlib.Path,
        default=pathlib.Path("/home/makihara/genoma_data/rosbags/lerobot_v21_ph2_success"),
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--history-steps", type=int, default=-1)
    parser.add_argument("--horizon-steps", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default="pick the object and place it in the box")
    parser.add_argument("--start-step", type=int, default=-1, help="Default = history_steps - 1")
    parser.add_argument("--end-step", type=int, default=-1, help="Inclusive end; default auto.")
    parser.add_argument("--stride", type=int, default=0, help="0 means use action horizon as the replay stride.")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resize", type=int, default=224, help="0 disables resize.")
    parser.add_argument("--trace-jsonl", action="store_true")
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("./outputs/umi_original_right_server"))
    args = parser.parse_args()

    table = _read_episode_table(args.dataset_dir, args.episode)
    right_pose = _to_np_seq(table, "observation.pose.right_controller.absolute")
    right_gripper = _to_np_seq(table, "observation.state.right_gripper").reshape(-1)
    num_steps = right_pose.shape[0]

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
    hist, requested_horizon = _resolve_hist_and_horizon(server_metadata, args)
    downsample_step = int(server_metadata.get("downsample_step", 1))
    image_layout = str(server_metadata.get("umi_image_layout", "double_current"))
    replay_stride = int(args.stride) if int(args.stride) > 0 else int(requested_horizon)
    print(
        f"[INFO] resolved history_steps={hist} action_horizon_steps={requested_horizon} "
        f"downsample_step={downsample_step} image_layout={image_layout} "
        f"replay_stride={replay_stride}",
        flush=True,
    )

    start_t = ((hist - 1) * downsample_step) if args.start_step < 0 else int(args.start_step)
    end_t = (num_steps - 1 - (requested_horizon - 1) * downsample_step) if args.end_step < 0 else int(args.end_step)
    if end_t < start_t:
        raise ValueError(f"Invalid step range: start={start_t}, end={end_t}, num_steps={num_steps}")

    to_model_space_cache: dict[int, Any] = {}
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{args.episode:06d}_umi_original_right_server"
    trace_path = out_dir / f"{stem}.trace.jsonl"
    trace_fp = trace_path.open("w", encoding="utf-8") if args.trace_jsonl else None

    step_ids: list[int] = []
    pred_chunks: list[np.ndarray] = []
    gt_chunks: list[np.ndarray] = []
    pred_abs_pose_first: list[np.ndarray] = []
    gt_abs_pose_first: list[np.ndarray] = []

    try:
        for t in range(start_t, end_t + 1, max(1, replay_stride)):
            if args.max_steps > 0 and len(step_ids) >= args.max_steps:
                break

            obs_steps = [t - (hist - 1 - idx) * downsample_step for idx in range(hist)]
            obs = {
                "left_image": left_reader.get_frame(t),
                "right_image": _make_right_image_input(
                    right_reader=right_reader,
                    t=t,
                    image_layout=image_layout,
                    for_server=True,
                ),
                "pose_seq": right_pose[obs_steps],
                "gripper_seq": right_gripper[obs_steps],
                "prompt": args.prompt,
            }
            pred = client.infer(obs)
            pred_actions = np.asarray(pred["actions"], dtype=np.float32)
            if pred_actions.ndim != 2 or pred_actions.shape[1] != len(ACTION_DIM_NAMES):
                raise ValueError(f"Unexpected predicted action shape at t={t}: {pred_actions.shape}")

            horizon = int(pred_actions.shape[0])
            if horizon not in to_model_space_cache:
                to_model_space_cache[horizon] = _make_model_input_transform(
                    image_layout=image_layout,
                    history_steps=hist,
                    action_horizon_steps=horizon,
                )
            pred_steps = [t + idx * downsample_step for idx in range(horizon)]
            if pred_steps[-1] >= num_steps:
                print(f"[INFO] stopping at t={t} because dataset tail is shorter than server horizon={horizon}", flush=True)
                break

            gt_pose_steps = obs_steps + pred_steps
            gt_obs = {
                "left_image": obs["left_image"],
                "right_image": _make_right_image_input(
                    right_reader=right_reader,
                    t=t,
                    image_layout=image_layout,
                    for_server=False,
                ),
                "pose_seq": right_pose[gt_pose_steps],
                "gripper_seq": right_gripper[gt_pose_steps],
                "prompt": args.prompt,
            }
            model_inputs = to_model_space_cache[horizon](gt_obs)
            gt_actions = np.asarray(model_inputs["actions"], dtype=np.float32)
            if pred_actions.shape != gt_actions.shape:
                raise ValueError(f"Shape mismatch at t={t}: pred={pred_actions.shape}, gt={gt_actions.shape}")

            gt_future_pose = right_pose[pred_steps]
            right_arm = pred.get("right_arm", {})
            pred_abs_pose = np.asarray(right_arm.get("pose", np.zeros((horizon, 7), dtype=np.float32)), dtype=np.float32)

            step_ids.append(t)
            pred_chunks.append(pred_actions.copy())
            gt_chunks.append(gt_actions.copy())
            pred_abs_pose_first.append(pred_abs_pose[0].copy())
            gt_abs_pose_first.append(gt_future_pose[0].copy())

            if trace_fp is not None:
                row = {
                    "t": int(t),
                    "obs_steps": obs_steps,
                    "pred_steps": pred_steps,
                    "first_action_pred": _to_serializable(pred_actions[0]),
                    "first_action_gt": _to_serializable(gt_actions[0]),
                    "first_abs_pose_pred": _to_serializable(pred_abs_pose[0]),
                    "first_abs_pose_gt": _to_serializable(gt_future_pose[0]),
                    "first_abs_pos_err_m": float(np.linalg.norm(pred_abs_pose[0, :3] - gt_future_pose[0, :3])),
                    "first_abs_rot_err_deg": _quat_angle_deg(pred_abs_pose[0, 3:7], gt_future_pose[0, 3:7]),
                    "policy_timing": _to_serializable(pred.get("policy_timing", {})),
                }
                trace_fp.write(json.dumps(row, ensure_ascii=False) + "\n")

            if len(step_ids) == 1 or len(step_ids) % 20 == 0:
                mse = float(np.mean((pred_actions - gt_actions) ** 2))
                print(f"[INFO] step={len(step_ids)} t={t} mse={mse:.8f}", flush=True)
    finally:
        if trace_fp is not None:
            trace_fp.close()
        left_reader.close()
        right_reader.close()

    step_ids_np = np.asarray(step_ids, dtype=np.int32)
    pred_chunks_np = np.stack(pred_chunks, axis=0)
    gt_chunks_np = np.stack(gt_chunks, axis=0)
    pred_abs_pose_first_np = np.stack(pred_abs_pose_first, axis=0)
    gt_abs_pose_first_np = np.stack(gt_abs_pose_first, axis=0)

    npz_path = out_dir / f"{stem}.npz"
    summary_path = out_dir / f"{stem}.json"
    png_path = out_dir / f"{stem}.png"

    np.savez_compressed(
        npz_path,
        step_ids=step_ids_np,
        pred_chunks=pred_chunks_np,
        gt_chunks=gt_chunks_np,
        pred_abs_pose_first=pred_abs_pose_first_np,
        gt_abs_pose_first=gt_abs_pose_first_np,
    )
    _save_matplotlib_plot(step_ids=step_ids_np, pred_chunks=pred_chunks_np, gt_chunks=gt_chunks_np, out_path=png_path)

    mae_first = np.mean(np.abs(pred_chunks_np[:, 0, :] - gt_chunks_np[:, 0, :]), axis=0)
    abs_pos_err = np.linalg.norm(pred_abs_pose_first_np[:, :3] - gt_abs_pose_first_np[:, :3], axis=1)
    abs_rot_err = [
        _quat_angle_deg(pred_abs_pose_first_np[i, 3:7], gt_abs_pose_first_np[i, 3:7])
        for i in range(pred_abs_pose_first_np.shape[0])
    ]
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "episode": int(args.episode),
        "host": args.host,
        "port": int(args.port),
        "history_steps": hist,
        "requested_horizon_steps": requested_horizon,
        "downsample_step": downsample_step,
        "num_queries": int(len(step_ids)),
        "mse_all": float(np.mean((pred_chunks_np - gt_chunks_np) ** 2)),
        "mae_first_by_dim": {name: float(mae_first[i]) for i, name in enumerate(ACTION_DIM_NAMES)},
        "mean_first_abs_pos_err_m": float(np.mean(abs_pos_err)),
        "mean_first_abs_rot_err_deg": float(np.mean(abs_rot_err)),
        "server_metadata": _to_serializable(server_metadata),
        "artifacts": {
            "npz": str(npz_path),
            "png": str(png_path),
            "trace_jsonl": str(trace_path) if args.trace_jsonl else None,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[DONE] queries={len(step_ids)} mse_all={summary['mse_all']:.8f}")
    print(f"[DONE] mean_first_abs_pos_err_m={summary['mean_first_abs_pos_err_m']:.6f}")
    print(f"[DONE] mean_first_abs_rot_err_deg={summary['mean_first_abs_rot_err_deg']:.6f}")
    print(f"[DONE] png={png_path}")
    print(f"[DONE] npz={npz_path}")
    if args.trace_jsonl:
        print(f"[DONE] trace={trace_path}")
    print(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()
