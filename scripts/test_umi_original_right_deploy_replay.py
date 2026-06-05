#!/usr/bin/env python3

import argparse
import csv
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq
import scipy.interpolate as si
import scipy.spatial.transform as st
from openpi_client import websocket_client_policy


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

    def get_frame(self, index: int) -> np.ndarray:
        if index < 0:
            raise IndexError(index)
        self._ensure_open()
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


def _history_indices(t: int, history_steps: int, downsample_step: int) -> np.ndarray:
    return np.asarray(
        [t - downsample_step * (history_steps - 1 - i) for i in range(history_steps)],
        dtype=np.int32,
    )


def quat_normalize_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return q / max(np.linalg.norm(q), 1e-9)


def pose7_to_pose6(pose7: np.ndarray) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(7)
    rot = st.Rotation.from_quat(pose7[3:7])
    return np.concatenate([pose7[:3], rot.as_rotvec()], axis=0)


def pose6_to_pose7(pose6: np.ndarray) -> np.ndarray:
    pose6 = np.asarray(pose6, dtype=np.float64).reshape(6)
    quat = st.Rotation.from_rotvec(pose6[3:6]).as_quat()
    return np.concatenate([pose6[:3], quat], axis=0)


def pose7_to_transform(pose7: np.ndarray) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(7)
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = st.Rotation.from_quat(quat_normalize_xyzw(pose7[3:7])).as_matrix()
    tf[:3, 3] = pose7[:3]
    return tf


def pose_matrix_to_pose7(pose: np.ndarray) -> np.ndarray:
    quat = st.Rotation.from_matrix(np.asarray(pose[:3, :3], dtype=np.float64)).as_quat()
    return np.concatenate([pose[:3, 3], quat], axis=0).astype(np.float32)


def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    a1 = np.asarray(rot6d[:3], dtype=np.float64)
    a2 = np.asarray(rot6d[3:6], dtype=np.float64)
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2_orth = a2 - np.dot(b1, a2) * b1
    b2 = a2_orth / (np.linalg.norm(a2_orth) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def apply_dataset_relative_pose9(current_pose7: np.ndarray, rel_pose9: np.ndarray) -> np.ndarray:
    current_tf = pose7_to_transform(np.asarray(current_pose7, dtype=np.float64).reshape(7))
    rel_pose9 = np.asarray(rel_pose9, dtype=np.float64).reshape(9)
    next_tf = np.eye(4, dtype=np.float64)
    next_tf[:3, 3] = current_tf[:3, 3] + rel_pose9[:3]
    next_tf[:3, :3] = current_tf[:3, :3] @ rot6d_to_rotmat(rel_pose9[3:9])
    return pose_matrix_to_pose7(next_tf)


def reconstruct_pose_chunk_from_actions(base_pose7: np.ndarray, raw_actions: np.ndarray) -> np.ndarray:
    raw_actions = np.asarray(raw_actions, dtype=np.float64)
    if raw_actions.ndim != 2 or raw_actions.shape[1] < 9:
        raise ValueError(f"Unexpected raw action chunk shape for pose reconstruction: {raw_actions.shape}")
    poses = []
    current_pose7 = np.asarray(base_pose7, dtype=np.float64).reshape(7)
    for action in raw_actions:
        current_pose7 = apply_dataset_relative_pose9(current_pose7, action[:9])
        poses.append(current_pose7.astype(np.float32).copy())
    return np.asarray(poses, dtype=np.float32)


def blend_pose7(prev_pose7: np.ndarray, next_pose7: np.ndarray, *, pos_alpha: float, rot_alpha: float) -> np.ndarray:
    prev_pose7 = np.asarray(prev_pose7, dtype=np.float64).reshape(7)
    next_pose7 = np.asarray(next_pose7, dtype=np.float64).reshape(7)
    pos_alpha = float(np.clip(pos_alpha, 0.0, 1.0))
    rot_alpha = float(np.clip(rot_alpha, 0.0, 1.0))
    pos = (1.0 - pos_alpha) * prev_pose7[:3] + pos_alpha * next_pose7[:3]
    prev_rot = st.Rotation.from_quat(prev_pose7[3:7])
    next_rot = st.Rotation.from_quat(next_pose7[3:7])
    slerp = st.Slerp([0.0, 1.0], st.Rotation.concatenate([prev_rot, next_rot]))
    quat = slerp([rot_alpha]).as_quat()[0]
    return np.concatenate([pos, quat], axis=0).astype(np.float32)


class PoseTrajectoryInterpolator:
    def __init__(self, times: np.ndarray, poses: np.ndarray):
        self._times = np.asarray(times, dtype=np.float64).reshape(-1)
        self._poses = np.asarray(poses, dtype=np.float64).reshape(len(times), 6)
        self.single_step = len(times) == 1
        if not self.single_step:
            self.pos_interp = si.interp1d(self._times, self._poses[:, :3], axis=0, assume_sorted=True)
            self.rot_interp = st.Slerp(self._times, st.Rotation.from_rotvec(self._poses[:, 3:6]))

    def __call__(self, t) -> np.ndarray:
        is_single = np.isscalar(t)
        t = np.array([t], dtype=np.float64) if is_single else np.asarray(t, dtype=np.float64)
        if self.single_step:
            out = np.repeat(self._poses[:1], len(t), axis=0)
        else:
            tt = np.clip(t, self._times[0], self._times[-1])
            out = np.zeros((len(tt), 6), dtype=np.float64)
            out[:, :3] = self.pos_interp(tt)
            out[:, 3:6] = self.rot_interp(tt).as_rotvec()
        return out[0] if is_single else out


class TimedActionChunk:
    def __init__(self, *, poses: np.ndarray, grippers: np.ndarray, raw_actions: np.ndarray, start_time: float, dt: float):
        pose7_seq = np.asarray(poses, dtype=np.float64).reshape(len(poses), 7)
        self.pose7_seq = pose7_seq
        self.pose6_seq = np.asarray([pose7_to_pose6(p) for p in pose7_seq], dtype=np.float64)
        self.grippers = np.asarray(grippers, dtype=np.float64).reshape(-1)
        self.raw_actions = np.asarray(raw_actions, dtype=np.float64)
        self.start_time = float(start_time)
        self.dt = float(dt)
        self.times = self.start_time + np.arange(len(self.pose6_seq), dtype=np.float64) * self.dt
        self.pose_interp = PoseTrajectoryInterpolator(self.times, self.pose6_seq)
        self.gripper_interp = si.interp1d(
            self.times,
            self.grippers,
            axis=0,
            assume_sorted=True,
            fill_value=(self.grippers[0], self.grippers[-1]),
            bounds_error=False,
        )

    @property
    def horizon(self) -> int:
        return int(self.pose7_seq.shape[0])


def sample_chunk_plan(chunk: TimedActionChunk, query_time: float, *, interpolate: bool):
    if chunk.horizon == 1 or chunk.dt <= 1e-9:
        return chunk.pose7_seq[0], float(chunk.grippers[0]), chunk.raw_actions[0], 0.0
    phase = max(0.0, (query_time - chunk.start_time) / chunk.dt)
    if phase >= (chunk.horizon - 1):
        return chunk.pose7_seq[-1], float(chunk.grippers[-1]), chunk.raw_actions[-1], float(chunk.horizon - 1)
    if not interpolate:
        i0 = int(np.floor(phase))
        return chunk.pose7_seq[i0], float(chunk.grippers[i0]), chunk.raw_actions[i0], phase
    pose6 = chunk.pose_interp(query_time)
    pose7 = pose6_to_pose7(pose6)
    grip = float(chunk.gripper_interp(query_time))
    i0 = int(np.floor(phase))
    i1 = min(i0 + 1, chunk.horizon - 1)
    alpha = float(phase - i0)
    raw = (1.0 - alpha) * chunk.raw_actions[i0] + alpha * chunk.raw_actions[i1]
    return pose7, grip, raw, phase


def normalize_server_gripper_value(gripper_value: float) -> float:
    v = float(gripper_value)
    if 0.0 <= v <= 1.0:
        return v
    raw_close = 5.41
    raw_open = 6.28
    if abs(v) < 1e-9:
        v = raw_open
    return float(np.clip((v - raw_close) / max(raw_open - raw_close, 1e-9), 0.0, 1.0))


def dump_replan_snapshot(
    dump_dir: pathlib.Path | None,
    *,
    replan_index: int,
    prompt: str,
    obs: dict,
    raw_actions_chunk: np.ndarray,
    pose_chunk: np.ndarray,
    gripper_chunk: np.ndarray,
) -> None:
    if dump_dir is None:
        return
    dump_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dump_dir / f"replan_{replan_index:04d}.npz",
        prompt=np.asarray(prompt),
        left_image=np.asarray(obs["left_image"], dtype=np.uint8),
        right_image=np.asarray(obs["right_image"], dtype=np.uint8),
        pose_seq=np.asarray(obs["pose_seq"], dtype=np.float32),
        gripper_seq=np.asarray(obs["gripper_seq"], dtype=np.float32),
        raw_actions=np.asarray(raw_actions_chunk, dtype=np.float32),
        pose_chunk=np.asarray(pose_chunk, dtype=np.float32),
        gripper_chunk=np.asarray(gripper_chunk, dtype=np.float32),
    )


def apply_gripper_offset_deg(gripper_abs: float, offset_deg: float) -> float:
    if abs(offset_deg) < 1e-9:
        return float(np.clip(gripper_abs, 0.0, 1.0))
    raw_close = 5.41
    raw_open = 6.28
    raw = raw_close + float(np.clip(gripper_abs, 0.0, 1.0)) * (raw_open - raw_close)
    raw += np.deg2rad(float(offset_deg))
    norm = (raw - raw_close) / max(raw_open - raw_close, 1e-9)
    return float(np.clip(norm, 0.0, 1.0))


def pose_error_metrics(current_pose: np.ndarray, target_pose: np.ndarray) -> tuple[float, float]:
    pos_err = float(np.linalg.norm(target_pose[:3, 3] - current_pose[:3, 3]))
    cur_rot = st.Rotation.from_matrix(current_pose[:3, :3])
    tgt_rot = st.Rotation.from_matrix(target_pose[:3, :3])
    rot_err = (tgt_rot * cur_rot.inv()).magnitude()
    return pos_err, rot_err


def pose_error_twist_base(current_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    delta_pos = target_pose[:3, 3] - current_pose[:3, 3]
    delta_rot = (st.Rotation.from_matrix(target_pose[:3, :3]) * st.Rotation.from_matrix(current_pose[:3, :3]).inv()).as_rotvec()
    return np.concatenate([delta_pos, delta_rot], axis=0)


def limit_twist_by_speed(
    twist: np.ndarray,
    *,
    dt: float,
    max_pos_speed: float,
    max_rot_speed: float,
    max_xyz_step: float,
    max_rpy_step: float,
) -> np.ndarray:
    out = np.asarray(twist, dtype=np.float64).copy()
    dt = max(float(dt), 1e-6)
    pos = out[:3]
    rot = out[3:]
    pos_step_limit = float(max_pos_speed) * dt if max_pos_speed > 0 else np.inf
    rot_step_limit = float(max_rot_speed) * dt if max_rot_speed > 0 else np.inf
    pos_norm = float(np.linalg.norm(pos))
    if np.isfinite(pos_step_limit) and pos_norm > pos_step_limit and pos_norm > 1e-9:
        pos *= pos_step_limit / pos_norm
    rot_norm = float(np.linalg.norm(rot))
    if np.isfinite(rot_step_limit) and rot_norm > rot_step_limit and rot_norm > 1e-9:
        rot *= rot_step_limit / rot_norm
    if max_xyz_step > 0:
        pos[:] = np.clip(pos, -max_xyz_step, max_xyz_step)
    if max_rpy_step > 0:
        rot[:] = np.clip(rot, -max_rpy_step, max_rpy_step)
    out[:3] = pos
    out[3:] = rot
    return out


def apply_delta_to_pose(current_pose: np.ndarray, delta_pose: np.ndarray) -> np.ndarray:
    current_pose = np.asarray(current_pose, dtype=np.float64)
    delta_pose = np.asarray(delta_pose, dtype=np.float64).reshape(6)
    next_pose = current_pose.copy()
    next_pose[:3, 3] = current_pose[:3, 3] + delta_pose[:3]
    next_rot = st.Rotation.from_rotvec(delta_pose[3:6]) * st.Rotation.from_matrix(current_pose[:3, :3])
    next_pose[:3, :3] = next_rot.as_matrix()
    return next_pose


def _save_plot(step_ids: np.ndarray, rows: np.ndarray, out_path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    axes[0].plot(step_ids, rows[:, 0], label="raw_dx")
    axes[0].plot(step_ids, rows[:, 1], label="raw_dy")
    axes[0].plot(step_ids, rows[:, 2], label="raw_dz")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[0].set_title("Raw action translation")

    axes[1].plot(step_ids, rows[:, 6], label="current_z")
    axes[1].plot(step_ids, rows[:, 7], label="target_z")
    axes[1].plot(step_ids, rows[:, 8], label="filtered_target_z")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    axes[1].set_title("Deploy replay Z trace")

    axes[2].plot(step_ids, rows[:, 3], label="raw_gripper")
    axes[2].plot(step_ids, rows[:, 4], label="target_gripper_norm")
    axes[2].plot(step_ids, rows[:, 5], label="filtered_gripper_norm")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()
    axes[2].set_title("Gripper trace")

    axes[3].plot(step_ids, rows[:, 9], label="pos_err_mm")
    axes[3].plot(step_ids, rows[:, 10], label="rot_err_deg")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend()
    axes[3].set_title("Executor error")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay dataset observations through the current deploy executor logic.")
    parser.add_argument("--dataset-dir", type=pathlib.Path, default=pathlib.Path("/home/makihara/genoma_data/rosbags/lerobot_v21_ph2_success"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--prompt", default="handover object")
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--downsample-step", type=int, default=3)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--policy-step-hz", type=float, default=10.0)
    parser.add_argument(
        "--action-speed-scale",
        type=float,
        default=1.0,
        help="Execution speed scale for the predicted action chunk. 1.0 keeps the nominal speed, 0.5 stretches the chunk to half speed, 2.0 runs it twice as fast.",
    )
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument("--latency-match-ms", type=float, default=80.0)
    parser.add_argument("--disable-interpolation", action="store_true")
    parser.add_argument("--max-pos-speed", type=float, default=0.04)
    parser.add_argument("--max-rot-speed", type=float, default=0.35)
    parser.add_argument("--max-xyz-step", type=float, default=0.01)
    parser.add_argument("--max-rpy-step", type=float, default=0.08)
    parser.add_argument("--target-pos-alpha", type=float, default=0.30)
    parser.add_argument("--target-rot-alpha", type=float, default=0.25)
    parser.add_argument("--gripper-alpha", type=float, default=0.4)
    parser.add_argument("--position-deadband-mm", type=float, default=3.0)
    parser.add_argument("--rotation-deadband-deg", type=float, default=2.0)
    parser.add_argument(
        "--disable-rotation",
        action="store_true",
        help="Ignore policy/output rotation and keep the current EE orientation while tracking translation.",
    )
    parser.add_argument("--gripper-offset-deg", type=float, default=0.0)
    parser.add_argument("--resize", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("./outputs/umi_original_right_deploy_replay"))
    parser.add_argument(
        "--debug-dump-dir",
        type=pathlib.Path,
        default=None,
        help="Optional directory to dump replan snapshots (obs/raw_actions/chunks) as NPZ files.",
    )
    args = parser.parse_args()

    table = _read_episode_table(args.dataset_dir, args.episode)
    right_pose = _to_np_seq(table, "observation.pose.right_controller.absolute")
    right_gripper = _to_np_seq(table, "observation.state.right_gripper").reshape(-1)
    num_steps = right_pose.shape[0]
    hist = max(1, int(args.history_steps))
    downsample_step = max(1, int(args.downsample_step))
    start_t = (hist - 1) * downsample_step
    end_t = num_steps - 1

    resize_hw = None if args.resize <= 0 else (int(args.resize), int(args.resize))
    left_reader = SequentialVideoReader(_find_video(args.dataset_dir, "observation.image.left", args.episode), resize_hw=resize_hw)
    right_reader = SequentialVideoReader(_find_video(args.dataset_dir, "observation.image.right", args.episode), resize_hw=resize_hw)

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    server_metadata = client.get_server_metadata()
    print(f"[INFO] server metadata: {server_metadata}")

    current_pose = pose7_to_transform(right_pose[start_t])
    current_gripper = normalize_server_gripper_value(float(right_gripper[start_t]))
    filtered_target_pose7 = right_pose[start_t].copy()
    filtered_target_gripper = float(current_gripper)
    chunk = None
    replan_count = 0
    nominal_policy_dt = 1.0 / max(float(args.policy_step_hz), 1e-6)
    action_speed_scale = max(float(args.action_speed_scale), 1e-6)
    policy_dt = nominal_policy_dt / action_speed_scale
    control_dt = 1.0 / max(float(args.hz), 1e-6)
    latency_match_s = max(0.0, float(args.latency_match_ms) / 1000.0)

    print(
        f"[INFO] policy_dt={policy_dt:.4f}s nominal_policy_dt={nominal_policy_dt:.4f}s "
        f"action_speed_scale={action_speed_scale:.3f} control_dt={control_dt:.4f}s"
    )

    rows = []
    try:
        for n, t in enumerate(range(start_t, end_t + 1)):
            if args.max_steps > 0 and len(rows) >= args.max_steps:
                break
            sim_t = n * control_dt
            hist_idx = _history_indices(t, hist, downsample_step)
            obs = {
                "left_image": left_reader.get_frame(t),
                "right_image": right_reader.get_frame(t),
                "pose_seq": right_pose[hist_idx],
                "gripper_seq": right_gripper[hist_idx],
                "prompt": args.prompt,
            }

            need_replan = chunk is None
            if chunk is not None:
                steps_used = max(0.0, (sim_t - chunk.start_time) / max(chunk.dt, 1e-9))
                if steps_used >= max(1, int(args.replan_steps)):
                    need_replan = True

            if need_replan:
                pred = client.infer(obs)
                right_arm = pred.get("right_arm", {})
                gripper_chunk = np.asarray(right_arm.get("gripper"), dtype=np.float32).reshape(-1)
                raw_actions_chunk = np.asarray(pred.get("actions", np.zeros((1, 10), dtype=np.float32)), dtype=np.float32)
                if raw_actions_chunk.ndim != 2 or raw_actions_chunk.shape[1] < 10:
                    raise ValueError(f"Unexpected raw action chunk shape: {raw_actions_chunk.shape}")
                if gripper_chunk.shape[0] != raw_actions_chunk.shape[0]:
                    raise ValueError(
                        f"Unexpected gripper chunk length {gripper_chunk.shape[0]} for raw action chunk {raw_actions_chunk.shape[0]}"
                    )
                pose_chunk = reconstruct_pose_chunk_from_actions(pose_matrix_to_pose7(current_pose), raw_actions_chunk)
                dump_replan_snapshot(
                    args.debug_dump_dir,
                    replan_index=replan_count + 1,
                    prompt=args.prompt,
                    obs=obs,
                    raw_actions_chunk=raw_actions_chunk,
                    pose_chunk=pose_chunk,
                    gripper_chunk=gripper_chunk,
                )
                chunk = TimedActionChunk(
                    poses=pose_chunk,
                    grippers=gripper_chunk,
                    raw_actions=raw_actions_chunk,
                    start_time=sim_t,
                    dt=policy_dt,
                )
                replan_count += 1

            target_pose7, target_gripper, raw_action, chunk_phase = sample_chunk_plan(
                chunk,
                sim_t + latency_match_s,
                interpolate=not bool(args.disable_interpolation),
            )
            current_pose7 = pose_matrix_to_pose7(current_pose)
            raw_gripper = float(raw_action[9]) if raw_action.shape[0] >= 10 else float("nan")
            target_gripper = normalize_server_gripper_value(target_gripper)
            target_gripper = apply_gripper_offset_deg(target_gripper, args.gripper_offset_deg)
            filtered_target_pose7 = blend_pose7(
                filtered_target_pose7,
                target_pose7,
                pos_alpha=args.target_pos_alpha,
                rot_alpha=args.target_rot_alpha,
            )
            if args.disable_rotation:
                filtered_target_pose7[3:7] = current_pose7[3:7]
            filtered_target_gripper = (1.0 - float(np.clip(args.gripper_alpha, 0.0, 1.0))) * filtered_target_gripper + float(
                np.clip(args.gripper_alpha, 0.0, 1.0)
            ) * float(target_gripper)

            target_pose = pose7_to_transform(filtered_target_pose7)
            pos_err_norm, rot_err_norm = pose_error_metrics(current_pose, target_pose)
            if (
                pos_err_norm < (float(args.position_deadband_mm) / 1000.0)
                and rot_err_norm < np.deg2rad(float(args.rotation_deadband_deg))
            ):
                target_pose = current_pose.copy()

            delta = limit_twist_by_speed(
                pose_error_twist_base(current_pose, target_pose),
                dt=control_dt,
                max_pos_speed=args.max_pos_speed,
                max_rot_speed=args.max_rot_speed,
                max_xyz_step=args.max_xyz_step,
                max_rpy_step=args.max_rpy_step,
            )
            current_pose = apply_delta_to_pose(current_pose, delta)
            current_gripper = filtered_target_gripper
            current_pose7 = pose_matrix_to_pose7(current_pose)
            target_pose7_logged = pose_matrix_to_pose7(target_pose)

            rows.append(
                [
                    t,
                    sim_t,
                    replan_count,
                    chunk_phase,
                    float(raw_action[0]),
                    float(raw_action[1]),
                    float(raw_action[2]),
                    raw_gripper,
                    float(target_gripper),
                    float(filtered_target_gripper),
                    float(current_pose7[2]),
                    float(target_pose7_logged[2]),
                    float(filtered_target_pose7[2]),
                    float(pos_err_norm * 1000.0),
                    float(np.rad2deg(rot_err_norm)),
                ]
            )
            if len(rows) == 1 or len(rows) % 20 == 0:
                print(
                    f"[INFO] step={len(rows)} t={t} "
                    f"raw_d=({raw_action[0]:+.4f},{raw_action[1]:+.4f},{raw_action[2]:+.4f}) "
                    f"grip={target_gripper:.3f}/{filtered_target_gripper:.3f} "
                    f"z={current_pose7[2]:+.4f}->{target_pose7_logged[2]:+.4f} "
                    f"perr={pos_err_norm*1000.0:.1f}mm rerr={np.rad2deg(rot_err_norm):.1f}deg"
                )
    finally:
        left_reader.close()
        right_reader.close()

    rows_np = np.asarray(rows, dtype=np.float64)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{args.episode:06d}_umi_original_right_deploy_replay"
    csv_path = out_dir / f"{stem}.csv"
    npz_path = out_dir / f"{stem}.npz"
    png_path = out_dir / f"{stem}.png"
    summary_path = out_dir / f"{stem}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset_step",
                "sim_t_sec",
                "replan_count",
                "chunk_phase",
                "raw_dx",
                "raw_dy",
                "raw_dz",
                "raw_gripper",
                "target_gripper",
                "filtered_gripper",
                "current_z",
                "target_z",
                "target_filtered_z",
                "pos_err_mm",
                "rot_err_deg",
            ]
        )
        writer.writerows(rows)

    np.savez_compressed(npz_path, rows=rows_np)
    _save_plot(rows_np[:, 0].astype(np.int32), rows_np[:, 4:15], png_path)

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "episode": int(args.episode),
        "host": args.host,
        "port": int(args.port),
        "prompt": args.prompt,
        "history_steps": hist,
        "downsample_step": downsample_step,
        "rows": int(rows_np.shape[0]),
        "mean_pos_err_mm": float(np.mean(rows_np[:, 13])) if rows_np.size else None,
        "mean_rot_err_deg": float(np.mean(rows_np[:, 14])) if rows_np.size else None,
        "server_metadata": server_metadata,
        "artifacts": {
            "csv": str(csv_path),
            "npz": str(npz_path),
            "png": str(png_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] csv={csv_path}")
    print(f"[DONE] npz={npz_path}")
    print(f"[DONE] png={png_path}")
    print(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()
