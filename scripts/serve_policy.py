import dataclasses
import enum
import logging
import socket

import numpy as np
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies import umi_handover_policy as _umi_handover_policy
from openpi.policies import umi_original_policy as _umi_original_policy
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # If true, use UMI handover absolute-state input and post-process outputs into per-arm absolute poses.
    umi_handover_absolute_io: bool = False
    # If true, use right-arm UMI original absolute-state input and post-process outputs into absolute poses.
    umi_original_right_io: bool = False


def _rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    a1 = np.asarray(rot6d[:3], dtype=np.float32)
    a2 = np.asarray(rot6d[3:6], dtype=np.float32)
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2_orth = a2 - np.dot(b1, a2) * b1
    b2 = a2_orth / (np.linalg.norm(a2_orth) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def _rotmat_to_quat_xyzw(rotmat: np.ndarray) -> np.ndarray:
    m = np.asarray(rotmat, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float32)
    return q / (np.linalg.norm(q) + 1e-8)


def _pose_rel9_to_abs_pose7(rel9: np.ndarray, base_pose7: np.ndarray) -> np.ndarray:
    base_pos_rot6d = _umi_handover_policy._pose7_to_pos_rot6d_batch(base_pose7)[0]
    abs_pos_rot6d = np.asarray(rel9, dtype=np.float32) + base_pos_rot6d
    pos = abs_pos_rot6d[:3]
    quat = _rotmat_to_quat_xyzw(_rot6d_to_rotmat(abs_pos_rot6d[3:9]))
    return np.concatenate([pos, quat], axis=0).astype(np.float32)


def _pose7_to_transform(pose7: np.ndarray) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float32)
    tf = np.eye(4, dtype=np.float32)
    tf[:3, :3] = _umi_original_policy._quat_to_rotmat(pose7[3:7])
    tf[:3, 3] = pose7[:3]
    return tf


def _transform_to_pose7(tf: np.ndarray) -> np.ndarray:
    pos = tf[:3, 3].astype(np.float32)
    quat = _rotmat_to_quat_xyzw(tf[:3, :3])
    return np.concatenate([pos, quat], axis=0).astype(np.float32)


def _pose9_to_transform(pose9: np.ndarray) -> np.ndarray:
    pose9 = np.asarray(pose9, dtype=np.float32)
    tf = np.eye(4, dtype=np.float32)
    tf[:3, :3] = _rot6d_to_rotmat(pose9[3:9])
    tf[:3, 3] = pose9[:3]
    return tf


class UmiAbsoluteIOPolicy(_policy.BasePolicy):
    """Wrapper policy for UMI handover that accepts absolute state sequences and returns absolute arm-wise outputs."""

    def __init__(self, policy: _policy.BasePolicy, *, history_steps: int = 4, action_horizon_steps: int = 16):
        self._policy = policy
        self._to_inputs = _umi_handover_policy.UmiHandoverInputs(
            history_steps=history_steps, action_horizon_steps=action_horizon_steps
        )
        self._metadata = getattr(policy, "metadata", {})

    def infer(self, obs: dict) -> dict:
        left_pose_seq = np.asarray(obs["left_pose_seq"], dtype=np.float32)
        right_pose_seq = np.asarray(obs["right_pose_seq"], dtype=np.float32)

        model_inputs = self._to_inputs(obs)
        raw = self._policy.infer(model_inputs)
        actions = np.asarray(raw["actions"], dtype=np.float32)

        left_base_pose = left_pose_seq[self._to_inputs.history_steps - 1]
        right_base_pose = right_pose_seq[self._to_inputs.history_steps - 1]
        left_abs_pose = np.stack([_pose_rel9_to_abs_pose7(a[:9], left_base_pose) for a in actions], axis=0)
        right_abs_pose = np.stack([_pose_rel9_to_abs_pose7(a[9:18], right_base_pose) for a in actions], axis=0)
        # Gripper channels are trained/served in absolute space.
        left_abs_gripper = actions[:, 18]
        right_abs_gripper = actions[:, 19]

        return {
            "left_arm": {
                "pose": left_abs_pose,  # [H,7] xyz + quaternion(xyzw)
                "gripper": left_abs_gripper.astype(np.float32),  # [H]
            },
            "right_arm": {
                "pose": right_abs_pose,  # [H,7] xyz + quaternion(xyzw)
                "gripper": right_abs_gripper.astype(np.float32),  # [H]
            },
            "actions": actions,  # original model output (relative 20D)
            "policy_timing": raw.get("policy_timing", {}),
        }

    @property
    def metadata(self) -> dict:
        return self._metadata


class UmiOriginalRightIOPolicy(_policy.BasePolicy):
    """Wrapper for right-arm UMI original models.

    Inputs are absolute pose7/gripper histories.
    Outputs include raw relative actions and integrated absolute targets.
    """

    def __init__(self, policy: _policy.BasePolicy):
        self._policy = policy
        self._metadata = getattr(policy, "metadata", {})

    def infer(self, obs: dict) -> dict:
        right_image = np.asarray(obs["right_image"], dtype=np.uint8)
        right_pose_seq = np.asarray(obs.get("right_pose_seq", obs.get("pose_seq")), dtype=np.float32)
        right_gripper_seq = np.asarray(obs.get("right_gripper_seq", obs.get("gripper_seq")), dtype=np.float32).reshape(-1)

        raw_obs = {
            "left_image": np.asarray(obs.get("left_image", right_image), dtype=np.uint8),
            "right_image": right_image,
            "pose_seq": right_pose_seq,
            "gripper_seq": right_gripper_seq,
        }
        if "prompt" in obs:
            raw_obs["prompt"] = obs["prompt"]

        raw = self._policy.infer(raw_obs)
        actions = np.asarray(raw["actions"], dtype=np.float32)

        current_pose = right_pose_seq[-1]
        current_gripper = _umi_original_policy._forward_fill_zeros(right_gripper_seq)[-1]

        current_tf = _pose7_to_transform(current_pose)
        abs_pose_seq = []
        abs_gripper_seq = []
        for action in actions:
            delta_tf = _pose9_to_transform(action[:9])
            current_tf = current_tf @ delta_tf
            abs_pose_seq.append(_transform_to_pose7(current_tf))
            current_gripper = float(action[9])
            abs_gripper_seq.append(current_gripper)

        return {
            "right_arm": {
                "pose": np.asarray(abs_pose_seq, dtype=np.float32),
                "gripper": np.asarray(abs_gripper_seq, dtype=np.float32),
            },
            "actions": actions,
            "policy_timing": raw.get("policy_timing", {}),
        }

    @property
    def metadata(self) -> dict:
        return self._metadata


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    if args.umi_handover_absolute_io:
        policy = UmiAbsoluteIOPolicy(policy, history_steps=4, action_horizon_steps=16)
    if args.umi_original_right_io:
        policy = UmiOriginalRightIOPolicy(policy)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
