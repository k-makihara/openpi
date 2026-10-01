#!/usr/bin/env python3

import argparse
import json
import pathlib
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from openpi_client import websocket_client_policy
import scipy.spatial.transform as st

from openpi.policies.umi_original_policy import UmiOriginalDualArmRelativeStateInputs


LEFT_ACTION_DIM_NAMES = [
    "left_dx",
    "left_dy",
    "left_dz",
    "left_rot6d_0",
    "left_rot6d_1",
    "left_rot6d_2",
    "left_rot6d_3",
    "left_rot6d_4",
    "left_rot6d_5",
    "left_gripper_abs",
]
RIGHT_ACTION_DIM_NAMES = [
    "right_dx",
    "right_dy",
    "right_dz",
    "right_rot6d_0",
    "right_rot6d_1",
    "right_rot6d_2",
    "right_rot6d_3",
    "right_rot6d_4",
    "right_rot6d_5",
    "right_gripper_abs",
]
ACTION_DIM_NAMES = LEFT_ACTION_DIM_NAMES + RIGHT_ACTION_DIM_NAMES


def _quat_angle_deg(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> float:
    q1 = np.asarray(q1_xyzw, dtype=np.float64)
    q2 = np.asarray(q2_xyzw, dtype=np.float64)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    dot = float(np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float32).reshape(6)
    a1 = np.asarray([rot6d[0], rot6d[2], rot6d[4]], dtype=np.float32)
    a2 = np.asarray([rot6d[1], rot6d[3], rot6d[5]], dtype=np.float32)
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2_orth = a2 - np.dot(b1, a2) * b1
    b2 = a2_orth / (np.linalg.norm(a2_orth) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def _pose7_to_transform(pose7: np.ndarray) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float32).reshape(7)
    tf = np.eye(4, dtype=np.float32)
    tf[:3, :3] = st.Rotation.from_quat(pose7[3:7]).as_matrix().astype(np.float32)
    tf[:3, 3] = pose7[:3]
    return tf


def _pose9_to_transform(pose9: np.ndarray) -> np.ndarray:
    pose9 = np.asarray(pose9, dtype=np.float32).reshape(9)
    tf = np.eye(4, dtype=np.float32)
    tf[:3, :3] = _rot6d_to_rotmat(pose9[3:9])
    tf[:3, 3] = pose9[:3]
    return tf


def _transform_to_pose7(tf: np.ndarray) -> np.ndarray:
    tf = np.asarray(tf, dtype=np.float32).reshape(4, 4)
    pos = tf[:3, 3].astype(np.float32)
    quat = st.Rotation.from_matrix(tf[:3, :3]).as_quat().astype(np.float32)
    return np.concatenate([pos, quat], axis=0)


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
    episode_path = dataset_dir / "data" / "chunk-000" / "episode_{:06d}.parquet".format(episode_idx)
    if not episode_path.exists():
        raise FileNotFoundError("Episode parquet not found: {}".format(episode_path))
    return pq.read_table(
        episode_path,
        columns=[
            "observation.pose.left_controller.absolute",
            "observation.pose.right_controller.absolute",
            "observation.state.left_gripper",
            "observation.state.right_gripper",
        ],
    )


def _to_np_seq(table, key: str) -> np.ndarray:
    return np.asarray(table[key].to_pylist(), dtype=np.float32)


class SequentialVideoReader:
    def __init__(self, path: pathlib.Path, resize_hw=None):
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
            raise RuntimeError("Failed to open video: {}".format(self.path))

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
                raise IndexError("Video {} ended before frame {}".format(self.path, index))
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
    path = dataset_dir / "videos" / "chunk-000" / key / "episode_{:06d}.mp4".format(episode_idx)
    if not path.exists():
        raise FileNotFoundError("Video not found: {}".format(path))
    return path


def _resolve_hist_and_horizon(server_metadata: dict, args) -> tuple:
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
            "Replay raw dual-arm LeRobot UMI observations into an OpenPI policy server "
            "and compare predicted relative actions against GT action targets."
        )
    )
    parser.add_argument("--dataset-dir", type=pathlib.Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--history-steps", type=int, default=-1)
    parser.add_argument("--horizon-steps", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default="handover object")
    parser.add_argument("--start-step", type=int, default=-1, help="Default = history_steps - 1")
    parser.add_argument("--end-step", type=int, default=-1, help="Inclusive end; default auto.")
    parser.add_argument("--stride", type=int, default=0, help="0 means use action horizon as replay stride.")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resize", type=int, default=224, help="0 disables resize.")
    parser.add_argument("--trace-jsonl", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("./outputs/umi_original_dual_arm_server"),
    )
    args = parser.parse_args()

    table = _read_episode_table(args.dataset_dir, args.episode)
    left_pose = _to_np_seq(table, "observation.pose.left_controller.absolute")
    right_pose = _to_np_seq(table, "observation.pose.right_controller.absolute")
    left_gripper = _to_np_seq(table, "observation.state.left_gripper").reshape(-1)
    right_gripper = _to_np_seq(table, "observation.state.right_gripper").reshape(-1)
    num_steps = left_pose.shape[0]

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
    replay_stride = int(args.stride) if int(args.stride) > 0 else int(requested_horizon)
    print(
        "[INFO] resolved history_steps={} action_horizon_steps={} downsample_step={} replay_stride={}".format(
            hist, requested_horizon, downsample_step, replay_stride
        ),
        flush=True,
    )

    start_t = ((hist - 1) * downsample_step) if args.start_step < 0 else int(args.start_step)
    end_t = (num_steps - 1 - (requested_horizon - 1) * downsample_step) if args.end_step < 0 else int(args.end_step)
    if end_t < start_t:
        raise ValueError("Invalid step range: start={}, end={}, num_steps={}".format(start_t, end_t, num_steps))

    transform_cache = {}
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "episode_{:06d}_umi_original_dual_arm_server".format(args.episode)
    trace_path = out_dir / "{}.trace.jsonl".format(stem)
    trace_fp = trace_path.open("w", encoding="utf-8") if args.trace_jsonl else None

    step_ids = []
    pred_chunks = []
    gt_chunks = []
    pred_left_abs_pose_first = []
    pred_right_abs_pose_first = []
    gt_left_abs_pose_first = []
    gt_right_abs_pose_first = []
    gt_recon_left_abs_pose_first = []
    gt_recon_right_abs_pose_first = []

    try:
        for t in range(start_t, end_t + 1, max(1, replay_stride)):
            if args.max_steps > 0 and len(step_ids) >= args.max_steps:
                break

            obs_steps = [t - (hist - 1 - idx) * downsample_step for idx in range(hist)]
            obs = {
                "left_image": left_reader.get_frame(t),
                "right_image": right_reader.get_frame(t),
                "left_pose_seq": left_pose[obs_steps],
                "right_pose_seq": right_pose[obs_steps],
                "left_gripper_seq": left_gripper[obs_steps],
                "right_gripper_seq": right_gripper[obs_steps],
                "prompt": args.prompt,
            }
            pred = client.infer(obs)
            pred_actions = np.asarray(pred["actions"], dtype=np.float32)
            if pred_actions.ndim != 2 or pred_actions.shape[1] != len(ACTION_DIM_NAMES):
                raise ValueError("Unexpected predicted action shape at t={}: {}".format(t, pred_actions.shape))

            horizon = int(pred_actions.shape[0])
            if horizon not in transform_cache:
                transform_cache[horizon] = UmiOriginalDualArmRelativeStateInputs(
                    history_steps=hist,
                    action_horizon_steps=horizon,
                    action_pose_target="relative",
                )
            pred_steps = [t + idx * downsample_step for idx in range(horizon)]
            if pred_steps[-1] >= num_steps:
                print(
                    "[INFO] stopping at t={} because dataset tail is shorter than server horizon={}".format(t, horizon),
                    flush=True,
                )
                break

            gt_pose_steps = obs_steps + pred_steps
            gt_obs = {
                "left_image": obs["left_image"],
                "right_image": obs["right_image"],
                "left_pose_seq": left_pose[gt_pose_steps],
                "right_pose_seq": right_pose[gt_pose_steps],
                "left_gripper_seq": left_gripper[gt_pose_steps],
                "right_gripper_seq": right_gripper[gt_pose_steps],
                "prompt": args.prompt,
            }
            model_inputs = transform_cache[horizon](gt_obs)
            gt_actions = np.asarray(model_inputs["actions"], dtype=np.float32)
            if pred_actions.shape != gt_actions.shape:
                raise ValueError("Shape mismatch at t={}: pred={} gt={}".format(t, pred_actions.shape, gt_actions.shape))

            left_arm = pred.get("left_arm", {})
            right_arm = pred.get("right_arm", {})
            pred_left_abs_pose = np.asarray(left_arm.get("pose", np.zeros((horizon, 7), dtype=np.float32)), dtype=np.float32)
            pred_right_abs_pose = np.asarray(right_arm.get("pose", np.zeros((horizon, 7), dtype=np.float32)), dtype=np.float32)
            gt_left_future_pose = left_pose[pred_steps]
            gt_right_future_pose = right_pose[pred_steps]
            gt_left_action = gt_actions[:, :10]
            gt_right_action = gt_actions[:, 10:20]
            left_current_tf = _pose7_to_transform(left_pose[obs_steps][-1])
            right_current_tf = _pose7_to_transform(right_pose[obs_steps][-1])
            gt_recon_left_abs_pose = []
            gt_recon_right_abs_pose = []
            for left_action_row, right_action_row in zip(gt_left_action, gt_right_action):
                left_abs_tf = left_current_tf @ _pose9_to_transform(left_action_row[:9])
                right_abs_tf = right_current_tf @ _pose9_to_transform(right_action_row[:9])
                gt_recon_left_abs_pose.append(_transform_to_pose7(left_abs_tf))
                gt_recon_right_abs_pose.append(_transform_to_pose7(right_abs_tf))
            gt_recon_left_abs_pose = np.asarray(gt_recon_left_abs_pose, dtype=np.float32)
            gt_recon_right_abs_pose = np.asarray(gt_recon_right_abs_pose, dtype=np.float32)

            step_ids.append(t)
            pred_chunks.append(pred_actions.copy())
            gt_chunks.append(gt_actions.copy())
            pred_left_abs_pose_first.append(pred_left_abs_pose[0].copy())
            pred_right_abs_pose_first.append(pred_right_abs_pose[0].copy())
            gt_left_abs_pose_first.append(gt_left_future_pose[0].copy())
            gt_right_abs_pose_first.append(gt_right_future_pose[0].copy())
            gt_recon_left_abs_pose_first.append(gt_recon_left_abs_pose[0].copy())
            gt_recon_right_abs_pose_first.append(gt_recon_right_abs_pose[0].copy())

            if trace_fp is not None:
                row = {
                    "t": int(t),
                    "obs_steps": obs_steps,
                    "pred_steps": pred_steps,
                    "first_action_pred": _to_serializable(pred_actions[0]),
                    "first_action_gt": _to_serializable(gt_actions[0]),
                    "first_left_abs_pose_pred": _to_serializable(pred_left_abs_pose[0]),
                    "first_left_abs_pose_gt": _to_serializable(gt_left_future_pose[0]),
                    "first_left_abs_pose_gt_reconstructed": _to_serializable(gt_recon_left_abs_pose[0]),
                    "first_right_abs_pose_pred": _to_serializable(pred_right_abs_pose[0]),
                    "first_right_abs_pose_gt": _to_serializable(gt_right_future_pose[0]),
                    "first_right_abs_pose_gt_reconstructed": _to_serializable(gt_recon_right_abs_pose[0]),
                    "first_left_abs_pos_err_m": float(np.linalg.norm(pred_left_abs_pose[0, :3] - gt_left_future_pose[0, :3])),
                    "first_left_abs_rot_err_deg": _quat_angle_deg(pred_left_abs_pose[0, 3:7], gt_left_future_pose[0, 3:7]),
                    "first_left_gt_recon_abs_pos_err_m": float(np.linalg.norm(gt_recon_left_abs_pose[0, :3] - gt_left_future_pose[0, :3])),
                    "first_left_gt_recon_abs_rot_err_deg": _quat_angle_deg(gt_recon_left_abs_pose[0, 3:7], gt_left_future_pose[0, 3:7]),
                    "first_right_abs_pos_err_m": float(np.linalg.norm(pred_right_abs_pose[0, :3] - gt_right_future_pose[0, :3])),
                    "first_right_abs_rot_err_deg": _quat_angle_deg(pred_right_abs_pose[0, 3:7], gt_right_future_pose[0, 3:7]),
                    "first_right_gt_recon_abs_pos_err_m": float(np.linalg.norm(gt_recon_right_abs_pose[0, :3] - gt_right_future_pose[0, :3])),
                    "first_right_gt_recon_abs_rot_err_deg": _quat_angle_deg(gt_recon_right_abs_pose[0, 3:7], gt_right_future_pose[0, 3:7]),
                    "policy_timing": _to_serializable(pred.get("policy_timing", {})),
                }
                trace_fp.write(json.dumps(row, ensure_ascii=False) + "\n")

            if len(step_ids) == 1 or len(step_ids) % 20 == 0:
                mse = float(np.mean((pred_actions - gt_actions) ** 2))
                print("[INFO] step={} t={} mse={:.8f}".format(len(step_ids), t, mse), flush=True)
    finally:
        if trace_fp is not None:
            trace_fp.close()
        left_reader.close()
        right_reader.close()

    step_ids_np = np.asarray(step_ids, dtype=np.int32)
    pred_chunks_np = np.stack(pred_chunks, axis=0)
    gt_chunks_np = np.stack(gt_chunks, axis=0)
    pred_left_abs_pose_first_np = np.stack(pred_left_abs_pose_first, axis=0)
    pred_right_abs_pose_first_np = np.stack(pred_right_abs_pose_first, axis=0)
    gt_left_abs_pose_first_np = np.stack(gt_left_abs_pose_first, axis=0)
    gt_right_abs_pose_first_np = np.stack(gt_right_abs_pose_first, axis=0)
    gt_recon_left_abs_pose_first_np = np.stack(gt_recon_left_abs_pose_first, axis=0)
    gt_recon_right_abs_pose_first_np = np.stack(gt_recon_right_abs_pose_first, axis=0)

    npz_path = out_dir / "{}.npz".format(stem)
    summary_path = out_dir / "{}.json".format(stem)
    np.savez_compressed(
        npz_path,
        step_ids=step_ids_np,
        pred_chunks=pred_chunks_np,
        gt_chunks=gt_chunks_np,
        pred_left_abs_pose_first=pred_left_abs_pose_first_np,
        pred_right_abs_pose_first=pred_right_abs_pose_first_np,
        gt_left_abs_pose_first=gt_left_abs_pose_first_np,
        gt_right_abs_pose_first=gt_right_abs_pose_first_np,
        gt_recon_left_abs_pose_first=gt_recon_left_abs_pose_first_np,
        gt_recon_right_abs_pose_first=gt_recon_right_abs_pose_first_np,
    )

    mae_first = np.mean(np.abs(pred_chunks_np[:, 0, :] - gt_chunks_np[:, 0, :]), axis=0)
    left_abs_pos_err = np.linalg.norm(pred_left_abs_pose_first_np[:, :3] - gt_left_abs_pose_first_np[:, :3], axis=1)
    right_abs_pos_err = np.linalg.norm(pred_right_abs_pose_first_np[:, :3] - gt_right_abs_pose_first_np[:, :3], axis=1)
    left_gt_recon_abs_pos_err = np.linalg.norm(
        gt_recon_left_abs_pose_first_np[:, :3] - gt_left_abs_pose_first_np[:, :3], axis=1
    )
    right_gt_recon_abs_pos_err = np.linalg.norm(
        gt_recon_right_abs_pose_first_np[:, :3] - gt_right_abs_pose_first_np[:, :3], axis=1
    )
    left_abs_rot_err = [
        _quat_angle_deg(pred_left_abs_pose_first_np[i, 3:7], gt_left_abs_pose_first_np[i, 3:7])
        for i in range(pred_left_abs_pose_first_np.shape[0])
    ]
    right_abs_rot_err = [
        _quat_angle_deg(pred_right_abs_pose_first_np[i, 3:7], gt_right_abs_pose_first_np[i, 3:7])
        for i in range(pred_right_abs_pose_first_np.shape[0])
    ]
    left_gt_recon_abs_rot_err = [
        _quat_angle_deg(gt_recon_left_abs_pose_first_np[i, 3:7], gt_left_abs_pose_first_np[i, 3:7])
        for i in range(gt_recon_left_abs_pose_first_np.shape[0])
    ]
    right_gt_recon_abs_rot_err = [
        _quat_angle_deg(gt_recon_right_abs_pose_first_np[i, 3:7], gt_right_abs_pose_first_np[i, 3:7])
        for i in range(gt_recon_right_abs_pose_first_np.shape[0])
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
        "mean_first_left_abs_pos_err_m": float(np.mean(left_abs_pos_err)),
        "mean_first_left_abs_rot_err_deg": float(np.mean(left_abs_rot_err)),
        "mean_first_right_abs_pos_err_m": float(np.mean(right_abs_pos_err)),
        "mean_first_right_abs_rot_err_deg": float(np.mean(right_abs_rot_err)),
        "mean_first_left_gt_recon_abs_pos_err_m": float(np.mean(left_gt_recon_abs_pos_err)),
        "mean_first_left_gt_recon_abs_rot_err_deg": float(np.mean(left_gt_recon_abs_rot_err)),
        "mean_first_right_gt_recon_abs_pos_err_m": float(np.mean(right_gt_recon_abs_pos_err)),
        "mean_first_right_gt_recon_abs_rot_err_deg": float(np.mean(right_gt_recon_abs_rot_err)),
        "server_metadata": _to_serializable(server_metadata),
        "artifacts": {
            "npz": str(npz_path),
            "trace_jsonl": str(trace_path) if args.trace_jsonl else None,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[DONE] queries={} mse_all={:.8f}".format(len(step_ids), summary["mse_all"]))
    print("[DONE] mean_first_left_abs_pos_err_m={:.6f}".format(summary["mean_first_left_abs_pos_err_m"]))
    print("[DONE] mean_first_left_abs_rot_err_deg={:.6f}".format(summary["mean_first_left_abs_rot_err_deg"]))
    print("[DONE] mean_first_right_abs_pos_err_m={:.6f}".format(summary["mean_first_right_abs_pos_err_m"]))
    print("[DONE] mean_first_right_abs_rot_err_deg={:.6f}".format(summary["mean_first_right_abs_rot_err_deg"]))
    print("[DONE] mean_first_left_gt_recon_abs_pos_err_m={:.6f}".format(summary["mean_first_left_gt_recon_abs_pos_err_m"]))
    print("[DONE] mean_first_left_gt_recon_abs_rot_err_deg={:.6f}".format(summary["mean_first_left_gt_recon_abs_rot_err_deg"]))
    print("[DONE] mean_first_right_gt_recon_abs_pos_err_m={:.6f}".format(summary["mean_first_right_gt_recon_abs_pos_err_m"]))
    print("[DONE] mean_first_right_gt_recon_abs_rot_err_deg={:.6f}".format(summary["mean_first_right_gt_recon_abs_rot_err_deg"]))
    print("[DONE] npz={}".format(npz_path))
    if args.trace_jsonl:
        print("[DONE] trace={}".format(trace_path))
    print("[DONE] summary={}".format(summary_path))


if __name__ == "__main__":
    main()
