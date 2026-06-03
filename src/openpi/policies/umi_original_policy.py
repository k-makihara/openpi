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


def _rotmat_to_rot6d(rotmat: np.ndarray) -> np.ndarray:
    return rotmat[:, :2].reshape(-1).astype(np.float32)


def _pose7_to_transform(pose7: np.ndarray) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float32)
    if pose7.shape != (7,):
        raise ValueError(f"Expected pose shape (7,), got {pose7.shape}")
    tf = np.eye(4, dtype=np.float32)
    tf[:3, :3] = _quat_to_rotmat(pose7[3:7])
    tf[:3, 3] = pose7[:3]
    return tf


def _invert_transform(tf: np.ndarray) -> np.ndarray:
    rot = tf[:3, :3]
    pos = tf[:3, 3]
    inv = np.eye(4, dtype=np.float32)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -(rot.T @ pos)
    return inv


def _relative_pose9d(anchor_pose7: np.ndarray, pose7: np.ndarray) -> np.ndarray:
    rel = _invert_transform(_pose7_to_transform(anchor_pose7)) @ _pose7_to_transform(pose7)
    return np.concatenate([rel[:3, 3], _rotmat_to_rot6d(rel[:3, :3])], axis=0).astype(np.float32)


def _pose7_seq_to_rot6d_seq(pose_seq: np.ndarray) -> np.ndarray:
    pose_seq = np.asarray(pose_seq, dtype=np.float32)
    if pose_seq.ndim == 1:
        pose_seq = pose_seq[None, :]
    return np.asarray(
        [np.concatenate([pose[:3], _rotmat_to_rot6d(_quat_to_rotmat(pose[3:7]))], axis=0) for pose in pose_seq],
        dtype=np.float32,
    )


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
class UmiOriginalInputs(transforms.DataTransformFn):
    """Single-arm UMI transform using current-pose-relative history and stepwise relative actions."""

    history_steps: int = 4
    action_horizon_steps: int = 10

    def __call__(self, data: dict) -> dict:
        right_image = _parse_image(data["right_image"])
        pose_seq = np.asarray(data["pose_seq"], dtype=np.float32)
        gripper_seq = np.asarray(data["gripper_seq"], dtype=np.float32).reshape(-1)
        action_pose_seq = np.asarray(data["action_pose_seq"], dtype=np.float32)
        action_gripper_seq = np.asarray(data["action_gripper_seq"], dtype=np.float32).reshape(-1)

        hist_len = self.history_steps
        action_len = self.action_horizon_steps
        hist_pose_seq = pose_seq[:hist_len]
        current_pose = hist_pose_seq[-1]

        hist_rel = np.asarray([_relative_pose9d(current_pose, pose) for pose in hist_pose_seq], dtype=np.float32)
        gripper_hist = _forward_fill_zeros(gripper_seq[:hist_len])[:, None]
        state = np.concatenate([hist_rel, gripper_hist], axis=-1).reshape(-1)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": right_image,
                "left_wrist_0_rgb": right_image,
                "right_wrist_0_rgb": np.zeros_like(right_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if action_pose_seq.size > 0:
            action_pose_rot6d = _pose7_seq_to_rot6d_seq(action_pose_seq[:action_len])
            action_gripper_abs = _forward_fill_zeros(action_gripper_seq[:action_len])[:, None]
            inputs["actions"] = np.concatenate([action_pose_rot6d, action_gripper_abs], axis=-1)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UmiOriginalOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :10])}


@dataclasses.dataclass(frozen=True)
class UmiOriginalBimanualInputs(transforms.DataTransformFn):
    """Bimanual UMI transform using current-pose-relative history and stepwise relative actions."""

    history_steps: int = 4
    action_horizon_steps: int = 10

    def __call__(self, data: dict) -> dict:
        left_image = _parse_image(data["left_image"])
        right_image = _parse_image(data["right_image"])

        left_pose_seq = np.asarray(data["left_pose_seq"], dtype=np.float32)
        right_pose_seq = np.asarray(data["right_pose_seq"], dtype=np.float32)
        left_gripper_seq = np.asarray(data["left_gripper_seq"], dtype=np.float32).reshape(-1)
        right_gripper_seq = np.asarray(data["right_gripper_seq"], dtype=np.float32).reshape(-1)

        left_action_pose_seq = np.asarray(data["left_action_pose_seq"], dtype=np.float32)
        right_action_pose_seq = np.asarray(data["right_action_pose_seq"], dtype=np.float32)
        left_action_gripper_seq = np.asarray(data["left_action_gripper_seq"], dtype=np.float32).reshape(-1)
        right_action_gripper_seq = np.asarray(data["right_action_gripper_seq"], dtype=np.float32).reshape(-1)

        hist_len = self.history_steps
        action_len = self.action_horizon_steps

        left_hist_pose_seq = left_pose_seq[:hist_len]
        right_hist_pose_seq = right_pose_seq[:hist_len]
        left_current_pose = left_hist_pose_seq[-1]
        right_current_pose = right_hist_pose_seq[-1]

        left_hist_rel = np.asarray(
            [_relative_pose9d(left_current_pose, pose) for pose in left_hist_pose_seq], dtype=np.float32
        )
        right_hist_rel = np.asarray(
            [_relative_pose9d(right_current_pose, pose) for pose in right_hist_pose_seq], dtype=np.float32
        )
        left_gripper_hist = _forward_fill_zeros(left_gripper_seq[:hist_len])[:, None]
        right_gripper_hist = _forward_fill_zeros(right_gripper_seq[:hist_len])[:, None]
        left_state_seq = np.concatenate([left_hist_rel, left_gripper_hist], axis=-1)
        right_state_seq = np.concatenate([right_hist_rel, right_gripper_hist], axis=-1)
        state = np.concatenate([left_state_seq, right_state_seq], axis=-1).reshape(-1)

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

        if left_action_pose_seq.size > 0:
            left_action_pose_rot6d = _pose7_seq_to_rot6d_seq(left_action_pose_seq[:action_len])
            right_action_pose_rot6d = _pose7_seq_to_rot6d_seq(right_action_pose_seq[:action_len])
            left_action_gripper_abs = _forward_fill_zeros(left_action_gripper_seq[:action_len])[:, None]
            right_action_gripper_abs = _forward_fill_zeros(right_action_gripper_seq[:action_len])[:, None]
            inputs["actions"] = np.concatenate(
                [
                    left_action_pose_rot6d,
                    right_action_pose_rot6d,
                    left_action_gripper_abs,
                    right_action_gripper_abs,
                ],
                axis=-1,
            )

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UmiOriginalBimanualOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :20])}
