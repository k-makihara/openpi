import dataclasses
import enum
import logging
import socket

import numpy as np
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
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


class UmiActionMode(enum.Enum):
    AUTO = "auto"
    DELTA = "delta"
    RELATIVE = "relative"


class UmiImageLayout(enum.Enum):
    AUTO = "auto"
    DOUBLE_CURRENT = "double_current"
    THIRD_SLOT_ONLY = "third_slot_only"
    PREV_CURRENT_THIRD_SLOT = "prev_current_third_slot"


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

    # If true, use right-arm UMI original absolute-state input and post-process outputs into absolute poses.
    umi_original_right_io: bool = False
    # How to interpret predicted pose actions when converting back to absolute poses.
    umi_action_mode: UmiActionMode = UmiActionMode.AUTO
    # How input images should be prepared for UMI original right-arm models.
    umi_image_layout: UmiImageLayout = UmiImageLayout.AUTO


@dataclasses.dataclass(frozen=True)
class UmiServeSpec:
    action_mode: UmiActionMode
    image_layout: UmiImageLayout


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


class UmiOriginalRightIOPolicy(_policy.BasePolicy):
    """Wrapper for right-arm UMI original models.

    Inputs are absolute pose7/gripper histories.
    Outputs include raw relative actions and integrated absolute targets.
    """

    def __init__(self, policy: _policy.BasePolicy, *, serve_spec: UmiServeSpec):
        self._policy = policy
        self._metadata = getattr(policy, "metadata", {})
        self._serve_spec = serve_spec
        self._prev_right_image: np.ndarray | None = None

    def _prepare_right_image(self, obs: dict) -> np.ndarray:
        layout = self._serve_spec.image_layout
        if "right_image_seq" in obs:
            return np.asarray(obs["right_image_seq"], dtype=np.uint8)
        if "right_image_history" in obs:
            return np.asarray(obs["right_image_history"], dtype=np.uint8)

        right_image = np.asarray(obs["right_image"], dtype=np.uint8)
        if layout != UmiImageLayout.PREV_CURRENT_THIRD_SLOT:
            return right_image

        if "right_image_prev" in obs:
            prev_image = np.asarray(obs["right_image_prev"], dtype=np.uint8)
        elif self._prev_right_image is not None:
            prev_image = self._prev_right_image
        else:
            prev_image = right_image

        self._prev_right_image = right_image.copy()
        return np.stack([prev_image, right_image], axis=0)

    def infer(self, obs: dict) -> dict:
        right_image = self._prepare_right_image(obs)
        right_pose_seq = np.asarray(obs.get("right_pose_seq", obs.get("pose_seq")), dtype=np.float32)
        right_gripper_seq = np.asarray(obs.get("right_gripper_seq", obs.get("gripper_seq")), dtype=np.float32).reshape(-1)

        raw_obs = {
            "left_image": np.asarray(obs.get("left_image", obs["right_image"]), dtype=np.uint8),
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
        anchor_tf = current_tf.copy()
        abs_pose_seq = []
        abs_gripper_seq = []
        for action in actions:
            target_tf = _pose9_to_transform(action[:9])
            if self._serve_spec.action_mode == UmiActionMode.DELTA:
                current_tf = current_tf @ target_tf
                abs_tf = current_tf
            else:
                abs_tf = anchor_tf @ target_tf
            abs_pose_seq.append(_transform_to_pose7(abs_tf))
            current_gripper = float(action[9])
            abs_gripper_seq.append(current_gripper)

        return {
            "right_arm": {
                "pose": np.asarray(abs_pose_seq, dtype=np.float32),
                "gripper": np.asarray(abs_gripper_seq, dtype=np.float32),
            },
            "actions": actions,
            "umi_action_mode": self._serve_spec.action_mode.value,
            "umi_image_layout": self._serve_spec.image_layout.value,
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


def _resolve_umi_serve_spec(args: Args) -> UmiServeSpec:
    action_mode = args.umi_action_mode
    image_layout = args.umi_image_layout

    train_config = None
    if isinstance(args.policy, Checkpoint):
        train_config = _config.get_config(args.policy.config)
        data_config = train_config.data

        if action_mode == UmiActionMode.AUTO:
            action_pose_target = getattr(data_config, "action_pose_target", "delta")
            action_mode = UmiActionMode.RELATIVE if action_pose_target == "relative" else UmiActionMode.DELTA

        if image_layout == UmiImageLayout.AUTO:
            if isinstance(data_config, _config.LeRobotUmiOriginalRightPrevCurrentThirdSlotDataConfig):
                image_layout = UmiImageLayout.PREV_CURRENT_THIRD_SLOT
            elif isinstance(data_config, _config.LeRobotUmiOriginalRightThirdSlotDataConfig):
                image_layout = UmiImageLayout.THIRD_SLOT_ONLY
            else:
                image_layout = UmiImageLayout.DOUBLE_CURRENT

    if action_mode == UmiActionMode.AUTO:
        action_mode = UmiActionMode.DELTA
    if image_layout == UmiImageLayout.AUTO:
        image_layout = UmiImageLayout.DOUBLE_CURRENT

    logging.info(
        "Resolved UMI serve spec: action_mode=%s image_layout=%s%s",
        action_mode.value,
        image_layout.value,
        f" from config={args.policy.config}" if isinstance(args.policy, Checkpoint) else "",
    )
    return UmiServeSpec(action_mode=action_mode, image_layout=image_layout)


def main(args: Args) -> None:
    policy = create_policy(args)
    if args.umi_original_right_io:
        policy = UmiOriginalRightIOPolicy(policy, serve_spec=_resolve_umi_serve_spec(args))
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
