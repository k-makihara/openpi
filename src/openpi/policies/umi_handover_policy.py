import dataclasses

import einops
import numpy as np

from openpi import transforms


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _quat_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-8)
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _pose7_to_pos_rot6d_batch(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim == 1:
        pose = pose[None, :]
    if pose.ndim != 2 or pose.shape[-1] != 7:
        raise ValueError(f"Expected pose shape [T,7] or [7], got {pose.shape}")

    out = []
    for p in pose:
        rotmat = _quat_to_rotmat(p[3:7])
        rot6d = rotmat[:, :2].reshape(-1)
        out.append(np.concatenate([p[:3], rot6d], axis=0))
    return np.asarray(out, dtype=np.float32)


def _forward_fill_zeros(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1).copy()
    last = 0.0
    for i in range(x.shape[0]):
        if abs(float(x[i])) <= eps:
            x[i] = last
        else:
            last = float(x[i])
    return x


@dataclasses.dataclass(frozen=True)
class UmiHandoverInputs(transforms.DataTransformFn):
    """UMI-style transform.

    State:
      past poses (including t) are represented relative to current pose at t.
    Action:
      future poses are represented relative to current pose at t.
    """

    history_steps: int = 4
    action_horizon_steps: int = 10

    def __call__(self, data: dict) -> dict:
        left_image = _parse_image(data["left_image"])
        right_image = _parse_image(data["right_image"])

        left_pose_seq = np.asarray(data["left_pose_seq"], dtype=np.float32)
        right_pose_seq = np.asarray(data["right_pose_seq"], dtype=np.float32)
        left_gripper_seq = np.asarray(data["left_gripper_seq"], dtype=np.float32)
        right_gripper_seq = np.asarray(data["right_gripper_seq"], dtype=np.float32)

        hist_len = self.history_steps
        fut_len = self.action_horizon_steps
        left_hist = left_pose_seq[:hist_len]
        right_hist = right_pose_seq[:hist_len]
        left_future = left_pose_seq[hist_len : hist_len + fut_len]
        right_future = right_pose_seq[hist_len : hist_len + fut_len]

        # Current pose at t is the last entry in history.
        left_t = left_hist[-1]
        right_t = right_hist[-1]

        left_hist_posrot = _pose7_to_pos_rot6d_batch(left_hist)      # [Th, 9]
        right_hist_posrot = _pose7_to_pos_rot6d_batch(right_hist)    # [Th, 9]
        left_t_posrot = _pose7_to_pos_rot6d_batch(left_t)[0]         # [9]
        right_t_posrot = _pose7_to_pos_rot6d_batch(right_t)[0]       # [9]

        # State: past relative-to-t for pose, gripper absolute.
        left_hist_rel = left_hist_posrot - left_t_posrot[None, :]
        right_hist_rel = right_hist_posrot - right_t_posrot[None, :]

        left_gripper_hist = _forward_fill_zeros(left_gripper_seq[:hist_len])
        right_gripper_hist = _forward_fill_zeros(right_gripper_seq[:hist_len])
        left_state_seq = np.concatenate([left_hist_rel, left_gripper_hist[:, None]], axis=-1)   # [Th, 10]
        right_state_seq = np.concatenate([right_hist_rel, right_gripper_hist[:, None]], axis=-1) # [Th, 10]
        state = np.concatenate([left_state_seq, right_state_seq], axis=-1).reshape(-1)  # [Th * 20]

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": left_image,
                "left_wrist_0_rgb": right_image,
                "right_wrist_0_rgb": np.zeros_like(left_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if left_future.size > 0:
            left_future_posrot = _pose7_to_pos_rot6d_batch(left_future)     # [H, 9]
            right_future_posrot = _pose7_to_pos_rot6d_batch(right_future)   # [H, 9]
            left_future_rel = left_future_posrot - left_t_posrot[None, :]
            right_future_rel = right_future_posrot - right_t_posrot[None, :]

            left_gripper_future = _forward_fill_zeros(left_gripper_seq[hist_len : hist_len + fut_len])
            right_gripper_future = _forward_fill_zeros(right_gripper_seq[hist_len : hist_len + fut_len])
            # Keep gripper targets in absolute space (do not convert to relative deltas).
            left_gripper_abs = left_gripper_future[:, None]
            right_gripper_abs = right_gripper_future[:, None]

            actions = np.concatenate(
                [left_future_rel, right_future_rel, left_gripper_abs, right_gripper_abs],
                axis=-1,
            )  # [H, 20]
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UmiHandoverOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :20])}
