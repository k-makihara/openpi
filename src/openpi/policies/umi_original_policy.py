import dataclasses
from typing import Literal

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


def _parse_image_sequence(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim == 3:
        return _parse_image(images)[None, ...]
    if images.ndim == 4 and images.shape[1] == 3:
        images = einops.rearrange(images, "t c h w -> t h w c")
    if np.issubdtype(images.dtype, np.floating):
        images = (255 * images).astype(np.uint8)
    return images


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


def _delta_pose9d_seq_from_absolute(current_pose7: np.ndarray, future_pose_seq_abs: np.ndarray) -> np.ndarray:
    prev_pose7 = np.asarray(current_pose7, dtype=np.float32)
    deltas = []
    for pose7 in np.asarray(future_pose_seq_abs, dtype=np.float32):
        deltas.append(_relative_pose9d(prev_pose7, pose7))
        prev_pose7 = pose7
    return np.asarray(deltas, dtype=np.float32)


def _forward_fill_zeros(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1).copy()
    last = 0.0
    for i in range(x.shape[0]):
        if abs(float(x[i])) <= eps:
            x[i] = last
        else:
            last = float(x[i])
    return x


def _ensure_sequence_length(x: np.ndarray, expected_min_len: int, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.shape[0] < expected_min_len:
        raise ValueError(f"Expected {name} to have at least {expected_min_len} steps, got {x.shape[0]}")
    return x


def _select_action_target(
    *,
    current_pose: np.ndarray,
    future_pose_seq_abs: np.ndarray,
    future_gripper_seq_abs: np.ndarray,
    action_len: int,
    action_pose_target: Literal["delta", "relative"],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    future_pose_seq_abs = np.asarray(future_pose_seq_abs[:action_len], dtype=np.float32)
    future_gripper_abs = _forward_fill_zeros(future_gripper_seq_abs[:action_len])[:, None]

    delta_pose9d = _delta_pose9d_seq_from_absolute(current_pose, future_pose_seq_abs)
    delta_actions = np.concatenate([delta_pose9d, future_gripper_abs], axis=-1)

    future_pose_rel = np.asarray(
        [_relative_pose9d(current_pose, pose) for pose in future_pose_seq_abs],
        dtype=np.float32,
    )
    relative_actions = np.concatenate([future_pose_rel, future_gripper_abs], axis=-1)

    actions = delta_actions if action_pose_target == "delta" else relative_actions
    aux = {
        "actions_delta": delta_actions,
        "actions_relative": relative_actions,
    }
    return actions, aux


@dataclasses.dataclass(frozen=True)
class UmiOriginalInputs(transforms.DataTransformFn):
    """Single-arm UMI transform using current-pose-relative history and stepwise relative actions."""

    history_steps: int = 4
    action_horizon_steps: int = 10
    action_pose_target: Literal["delta", "relative"] = "delta"

    def __call__(self, data: dict) -> dict:
        right_image = _parse_image(data["right_image"])
        pose_seq = _ensure_sequence_length(
            np.asarray(data["pose_seq"], dtype=np.float32),
            self.history_steps + self.action_horizon_steps,
            "pose_seq",
        )
        gripper_seq = _ensure_sequence_length(
            np.asarray(data["gripper_seq"], dtype=np.float32).reshape(-1),
            self.history_steps + self.action_horizon_steps,
            "gripper_seq",
        )

        hist_len = self.history_steps
        action_len = self.action_horizon_steps
        hist_pose_seq = pose_seq[:hist_len]
        current_pose = hist_pose_seq[-1]
        future_pose_seq_abs = pose_seq[hist_len : hist_len + action_len]
        future_gripper_seq_abs = gripper_seq[hist_len : hist_len + action_len]

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

        if future_pose_seq_abs.size > 0:
            actions, aux = _select_action_target(
                current_pose=current_pose,
                future_pose_seq_abs=future_pose_seq_abs,
                future_gripper_seq_abs=future_gripper_seq_abs,
                action_len=action_len,
                action_pose_target=self.action_pose_target,
            )
            inputs["actions"] = actions
            inputs.update(aux)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UmiOriginalOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :10])}


@dataclasses.dataclass(frozen=True)
class UmiOriginalRightThirdSlotInputs(transforms.DataTransformFn):
    """Single-arm UMI transform that routes the right camera only into the third image slot."""

    history_steps: int = 4
    action_horizon_steps: int = 10
    action_pose_target: Literal["delta", "relative"] = "delta"

    def __call__(self, data: dict) -> dict:
        right_image = _parse_image(data["right_image"])
        pose_seq = _ensure_sequence_length(
            np.asarray(data["pose_seq"], dtype=np.float32),
            self.history_steps + self.action_horizon_steps,
            "pose_seq",
        )
        gripper_seq = _ensure_sequence_length(
            np.asarray(data["gripper_seq"], dtype=np.float32).reshape(-1),
            self.history_steps + self.action_horizon_steps,
            "gripper_seq",
        )

        hist_len = self.history_steps
        action_len = self.action_horizon_steps
        hist_pose_seq = pose_seq[:hist_len]
        current_pose = hist_pose_seq[-1]
        future_pose_seq_abs = pose_seq[hist_len : hist_len + action_len]
        future_gripper_seq_abs = gripper_seq[hist_len : hist_len + action_len]

        hist_rel = np.asarray([_relative_pose9d(current_pose, pose) for pose in hist_pose_seq], dtype=np.float32)
        gripper_hist = _forward_fill_zeros(gripper_seq[:hist_len])[:, None]
        state = np.concatenate([hist_rel, gripper_hist], axis=-1).reshape(-1)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": np.zeros_like(right_image),
                "left_wrist_0_rgb": np.zeros_like(right_image),
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if future_pose_seq_abs.size > 0:
            actions, aux = _select_action_target(
                current_pose=current_pose,
                future_pose_seq_abs=future_pose_seq_abs,
                future_gripper_seq_abs=future_gripper_seq_abs,
                action_len=action_len,
                action_pose_target=self.action_pose_target,
            )
            inputs["actions"] = actions
            inputs.update(aux)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UmiOriginalRightPrevAndCurrentThirdSlotInputs(transforms.DataTransformFn):
    """Single-arm UMI transform with previous right image in slot 2 and current right image in slot 3."""

    history_steps: int = 4
    action_horizon_steps: int = 10
    action_pose_target: Literal["delta", "relative"] = "delta"

    def __call__(self, data: dict) -> dict:
        right_image_seq = _parse_image_sequence(data["right_image"])
        prev_right_image = right_image_seq[0]
        current_right_image = right_image_seq[-1]

        pose_seq = _ensure_sequence_length(
            np.asarray(data["pose_seq"], dtype=np.float32),
            self.history_steps + self.action_horizon_steps,
            "pose_seq",
        )
        gripper_seq = _ensure_sequence_length(
            np.asarray(data["gripper_seq"], dtype=np.float32).reshape(-1),
            self.history_steps + self.action_horizon_steps,
            "gripper_seq",
        )

        hist_len = self.history_steps
        action_len = self.action_horizon_steps
        hist_pose_seq = pose_seq[:hist_len]
        current_pose = hist_pose_seq[-1]
        future_pose_seq_abs = pose_seq[hist_len : hist_len + action_len]
        future_gripper_seq_abs = gripper_seq[hist_len : hist_len + action_len]

        hist_rel = np.asarray([_relative_pose9d(current_pose, pose) for pose in hist_pose_seq], dtype=np.float32)
        gripper_hist = _forward_fill_zeros(gripper_seq[:hist_len])[:, None]
        state = np.concatenate([hist_rel, gripper_hist], axis=-1).reshape(-1)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": np.zeros_like(current_right_image),
                "left_wrist_0_rgb": prev_right_image,
                "right_wrist_0_rgb": current_right_image,
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if future_pose_seq_abs.size > 0:
            actions, aux = _select_action_target(
                current_pose=current_pose,
                future_pose_seq_abs=future_pose_seq_abs,
                future_gripper_seq_abs=future_gripper_seq_abs,
                action_len=action_len,
                action_pose_target=self.action_pose_target,
            )
            inputs["actions"] = actions
            inputs.update(aux)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UmiOriginalBimanualInputs(transforms.DataTransformFn):
    """Bimanual UMI transform using current-pose-relative history and stepwise relative actions."""

    history_steps: int = 4
    action_horizon_steps: int = 10
    action_pose_target: Literal["delta", "relative"] = "delta"

    def __call__(self, data: dict) -> dict:
        left_image = _parse_image(data["left_image"])
        right_image = _parse_image(data["right_image"])

        left_pose_seq = _ensure_sequence_length(
            np.asarray(data["left_pose_seq"], dtype=np.float32),
            self.history_steps + self.action_horizon_steps,
            "left_pose_seq",
        )
        right_pose_seq = _ensure_sequence_length(
            np.asarray(data["right_pose_seq"], dtype=np.float32),
            self.history_steps + self.action_horizon_steps,
            "right_pose_seq",
        )
        left_gripper_seq = _ensure_sequence_length(
            np.asarray(data["left_gripper_seq"], dtype=np.float32).reshape(-1),
            self.history_steps + self.action_horizon_steps,
            "left_gripper_seq",
        )
        right_gripper_seq = _ensure_sequence_length(
            np.asarray(data["right_gripper_seq"], dtype=np.float32).reshape(-1),
            self.history_steps + self.action_horizon_steps,
            "right_gripper_seq",
        )

        hist_len = self.history_steps
        action_len = self.action_horizon_steps

        left_hist_pose_seq = left_pose_seq[:hist_len]
        right_hist_pose_seq = right_pose_seq[:hist_len]
        left_current_pose = left_hist_pose_seq[-1]
        right_current_pose = right_hist_pose_seq[-1]
        left_future_pose_seq_abs = left_pose_seq[hist_len : hist_len + action_len]
        right_future_pose_seq_abs = right_pose_seq[hist_len : hist_len + action_len]
        left_future_gripper_seq_abs = left_gripper_seq[hist_len : hist_len + action_len]
        right_future_gripper_seq_abs = right_gripper_seq[hist_len : hist_len + action_len]

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

        if left_future_pose_seq_abs.size > 0:
            left_actions, left_aux = _select_action_target(
                current_pose=left_current_pose,
                future_pose_seq_abs=left_future_pose_seq_abs,
                future_gripper_seq_abs=left_future_gripper_seq_abs,
                action_len=action_len,
                action_pose_target=self.action_pose_target,
            )
            right_actions, right_aux = _select_action_target(
                current_pose=right_current_pose,
                future_pose_seq_abs=right_future_pose_seq_abs,
                future_gripper_seq_abs=right_future_gripper_seq_abs,
                action_len=action_len,
                action_pose_target=self.action_pose_target,
            )
            inputs["actions"] = np.concatenate([left_actions, right_actions], axis=-1)
            inputs["actions_delta"] = np.concatenate([left_aux["actions_delta"], right_aux["actions_delta"]], axis=-1)
            inputs["actions_relative"] = np.concatenate(
                [left_aux["actions_relative"], right_aux["actions_relative"]],
                axis=-1,
            )

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UmiOriginalBimanualOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :20])}


@dataclasses.dataclass(frozen=True)
class UmiOriginalDualArmRelativeStateInputs(transforms.DataTransformFn):
    """Bimanual UMI transform with inter-arm relative end-effector pose appended to state."""

    history_steps: int = 4
    action_horizon_steps: int = 10
    action_pose_target: Literal["delta", "relative"] = "delta"

    def __call__(self, data: dict) -> dict:
        left_image = _parse_image(data["left_image"])
        right_image = _parse_image(data["right_image"])

        left_pose_seq = _ensure_sequence_length(
            np.asarray(data["left_pose_seq"], dtype=np.float32),
            self.history_steps + self.action_horizon_steps,
            "left_pose_seq",
        )
        right_pose_seq = _ensure_sequence_length(
            np.asarray(data["right_pose_seq"], dtype=np.float32),
            self.history_steps + self.action_horizon_steps,
            "right_pose_seq",
        )
        left_gripper_seq = _ensure_sequence_length(
            np.asarray(data["left_gripper_seq"], dtype=np.float32).reshape(-1),
            self.history_steps + self.action_horizon_steps,
            "left_gripper_seq",
        )
        right_gripper_seq = _ensure_sequence_length(
            np.asarray(data["right_gripper_seq"], dtype=np.float32).reshape(-1),
            self.history_steps + self.action_horizon_steps,
            "right_gripper_seq",
        )

        hist_len = self.history_steps
        action_len = self.action_horizon_steps

        left_hist_pose_seq = left_pose_seq[:hist_len]
        right_hist_pose_seq = right_pose_seq[:hist_len]
        left_current_pose = left_hist_pose_seq[-1]
        right_current_pose = right_hist_pose_seq[-1]
        left_future_pose_seq_abs = left_pose_seq[hist_len : hist_len + action_len]
        right_future_pose_seq_abs = right_pose_seq[hist_len : hist_len + action_len]
        left_future_gripper_seq_abs = left_gripper_seq[hist_len : hist_len + action_len]
        right_future_gripper_seq_abs = right_gripper_seq[hist_len : hist_len + action_len]

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
        left_wrt_right = _relative_pose9d(right_current_pose, left_current_pose)
        right_wrt_left = _relative_pose9d(left_current_pose, right_current_pose)
        state = np.concatenate(
            [left_state_seq.reshape(-1), right_state_seq.reshape(-1), left_wrt_right, right_wrt_left],
            axis=0,
        ).astype(np.float32)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": np.zeros_like(left_image),
                "left_wrist_0_rgb": left_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if left_future_pose_seq_abs.size > 0:
            left_actions, left_aux = _select_action_target(
                current_pose=left_current_pose,
                future_pose_seq_abs=left_future_pose_seq_abs,
                future_gripper_seq_abs=left_future_gripper_seq_abs,
                action_len=action_len,
                action_pose_target=self.action_pose_target,
            )
            right_actions, right_aux = _select_action_target(
                current_pose=right_current_pose,
                future_pose_seq_abs=right_future_pose_seq_abs,
                future_gripper_seq_abs=right_future_gripper_seq_abs,
                action_len=action_len,
                action_pose_target=self.action_pose_target,
            )
            inputs["actions"] = np.concatenate([left_actions, right_actions], axis=-1)
            inputs["actions_delta"] = np.concatenate([left_aux["actions_delta"], right_aux["actions_delta"]], axis=-1)
            inputs["actions_relative"] = np.concatenate(
                [left_aux["actions_relative"], right_aux["actions_relative"]],
                axis=-1,
            )

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UmiOriginalDualArmRelativeStateOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :20])}
