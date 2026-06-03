import argparse

import numpy as np
from openpi_client import websocket_client_policy


def _normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    return q / (np.linalg.norm(q) + 1e-8)


def _make_dummy_pose_seq(length: int, base_xyz: np.ndarray) -> np.ndarray:
    poses = []
    for i in range(length):
        xyz = base_xyz + np.array([0.002 * i, 0.0, 0.0], dtype=np.float32)
        quat = _normalize_quat_xyzw(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        poses.append(np.concatenate([xyz, quat], axis=0))
    return np.asarray(poses, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--horizon-steps", type=int, default=16)
    parser.add_argument("--prompt", type=str, default="handover object")
    args = parser.parse_args()

    total_steps = args.history_steps + args.horizon_steps
    h, w = 224, 224

    left_image = np.zeros((h, w, 3), dtype=np.uint8)
    right_image = np.zeros((h, w, 3), dtype=np.uint8)

    obs = {
        "left_image": left_image,
        "right_image": right_image,
        # Absolute pose7 input: [x, y, z, qx, qy, qz, qw]
        "left_pose_seq": _make_dummy_pose_seq(total_steps, np.array([0.3, 0.2, 0.4], dtype=np.float32)),
        "right_pose_seq": _make_dummy_pose_seq(total_steps, np.array([0.3, -0.2, 0.4], dtype=np.float32)),
        # Absolute gripper sequence
        "left_gripper_seq": np.full((total_steps,), 0.4, dtype=np.float32),
        "right_gripper_seq": np.full((total_steps,), 0.6, dtype=np.float32),
        "prompt": args.prompt,
    }

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    out = client.infer(obs)

    print("keys:", list(out.keys()))
    print("left_arm.pose shape:", np.asarray(out["left_arm"]["pose"]).shape)
    print("left_arm.gripper shape:", np.asarray(out["left_arm"]["gripper"]).shape)
    print("right_arm.pose shape:", np.asarray(out["right_arm"]["pose"]).shape)
    print("right_arm.gripper shape:", np.asarray(out["right_arm"]["gripper"]).shape)


if __name__ == "__main__":
    main()
