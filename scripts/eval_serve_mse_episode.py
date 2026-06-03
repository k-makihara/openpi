import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from openpi_client import websocket_client_policy

from openpi.policies.umi_handover_policy import UmiHandoverInputs
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _train_config


def _read_episode_table(dataset_dir: str, episode_idx: int) -> pa.Table:
    episode_path = pathlib.Path(dataset_dir) / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {episode_path}")
    return pq.read_table(episode_path)


def _to_np_seq(table: pa.Table, key: str) -> np.ndarray:
    # list<element: ...> per-row -> [T, D]
    return np.asarray(table[key].to_pylist(), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate serve_policy MSE on one LeRobot UMI handover episode.")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy-config", type=str, default=None, help="Local inference config name (no server mode).")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Local checkpoint dir (no server mode).")
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--horizon-steps", type=int, default=16)
    parser.add_argument("--prompt", type=str, default="handover object")
    parser.add_argument("--out-dir", type=str, default="./outputs")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--plot-first-horizon-only",
        action="store_true",
        help="If set, plot pred vs gt per action-dimension using the first action in each horizon (k=0).",
    )
    args = parser.parse_args()

    table = _read_episode_table(args.dataset_dir, args.episode)
    left_pose = _to_np_seq(table, "observation.pose.left_left_finger_tip.absolute")
    right_pose = _to_np_seq(table, "observation.pose.right_right_finger_tip.absolute")
    left_gripper = _to_np_seq(table, "observation.state.left_gripper").reshape(-1)
    right_gripper = _to_np_seq(table, "observation.state.right_gripper").reshape(-1)
    num_steps = left_pose.shape[0]

    window = args.history_steps + args.horizon_steps
    start_t = args.history_steps - 1
    end_t = num_steps - args.horizon_steps - 1
    if end_t < start_t:
        raise ValueError(f"Episode too short: {num_steps=}, need at least {window} steps.")

    to_model_space = UmiHandoverInputs(history_steps=args.history_steps, action_horizon_steps=args.horizon_steps)

    local_policy = None
    client = None
    if (args.policy_config is None) != (args.checkpoint_dir is None):
        raise ValueError("Set both --policy-config and --checkpoint-dir for local inference, or neither for server mode.")
    if args.policy_config is not None:
        cfg = _train_config.get_config(args.policy_config)
        local_policy = _policy_config.create_trained_policy(cfg, args.checkpoint_dir, default_prompt=args.prompt)
        print(f"[INFO] local policy mode: config={args.policy_config}, ckpt={args.checkpoint_dir}", flush=True)
    else:
        client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
        print(f"[INFO] websocket mode: host={args.host}, port={args.port}", flush=True)

    dummy_left_img = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_right_img = np.zeros((224, 224, 3), dtype=np.uint8)

    mse_ts: list[float] = []
    step_ids: list[int] = []
    pred_first_list: list[np.ndarray] = []
    gt_first_list: list[np.ndarray] = []
    total_points = end_t - start_t + 1
    print(f"[INFO] episode_len={num_steps}, eval_points={total_points}, host={args.host}, port={args.port}", flush=True)

    for t in range(start_t, end_t + 1):
        i0 = t - args.history_steps + 1
        i1 = t + args.horizon_steps + 1

        obs = {
            "left_image": dummy_left_img,
            "right_image": dummy_right_img,
            "left_pose_seq": left_pose[i0:i1],
            "right_pose_seq": right_pose[i0:i1],
            "left_gripper_seq": left_gripper[i0:i1],
            "right_gripper_seq": right_gripper[i0:i1],
            "prompt": args.prompt,
        }

        gt_actions = np.asarray(to_model_space(obs)["actions"], dtype=np.float32)  # [H, 20]
        if local_policy is not None:
            pred = local_policy.infer(obs)
        else:
            pred = client.infer(obs)
        pred_actions = np.asarray(pred["actions"], dtype=np.float32)  # [H, 20]
        if pred_actions.shape != gt_actions.shape:
            raise ValueError(f"Shape mismatch at t={t}: pred {pred_actions.shape}, gt {gt_actions.shape}")

        mse_t = float(np.mean((pred_actions - gt_actions) ** 2))
        mse_ts.append(mse_t)
        step_ids.append(t)
        pred_first_list.append(pred_actions[0].copy())
        gt_first_list.append(gt_actions[0].copy())
        done = len(step_ids)
        if done == 1 or done % max(1, args.log_every) == 0 or done == total_points:
            print(f"[INFO] step {done}/{total_points} t={t} mse={mse_t:.8f}", flush=True)

    final_mse = float(np.mean(np.asarray(mse_ts, dtype=np.float32)))

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"mse_episode_{args.episode:06d}.png"
    txt_path = out_dir / f"mse_episode_{args.episode:06d}.txt"

    plt.figure(figsize=(8, 4))
    plt.plot(step_ids, mse_ts)
    plt.xlabel("Episode timestep t")
    plt.ylabel("MSE (mean over horizon x action_dim)")
    plt.title(f"Serve vs GT Action MSE (episode {args.episode:06d})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    dim_fig_path = out_dir / f"pred_vs_gt_dims_episode_{args.episode:06d}.png"
    if args.plot_first_horizon_only:
        pred_first = np.stack(pred_first_list, axis=0)  # [N, D]
        gt_first = np.stack(gt_first_list, axis=0)      # [N, D]
        dim = pred_first.shape[1]
        cols = 4
        rows = int(np.ceil(dim / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows), sharex=True)
        axes = np.asarray(axes).reshape(-1)
        x = np.asarray(step_ids)
        for d in range(dim):
            ax = axes[d]
            ax.plot(x, gt_first[:, d], label="gt", linewidth=1.3)
            ax.plot(x, pred_first[:, d], label="pred", linewidth=1.1)
            diff = np.abs(pred_first[:, d] - gt_first[:, d])
            ax.fill_between(x, gt_first[:, d], pred_first[:, d], alpha=0.15)
            ax.set_title(f"dim {d} | mean|err|={diff.mean():.4f}", fontsize=9)
            ax.grid(True, alpha=0.25)
        for d in range(dim, len(axes)):
            axes[d].axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle(f"Pred vs GT per-dimension (first horizon action), episode {args.episode:06d}")
        fig.tight_layout()
        fig.savefig(dim_fig_path, dpi=150)
        plt.close(fig)

    txt_path.write_text(
        "\n".join(
            [
                f"dataset_dir: {args.dataset_dir}",
                f"episode: {args.episode}",
                f"host: {args.host}",
                f"port: {args.port}",
                f"history_steps: {args.history_steps}",
                f"horizon_steps: {args.horizon_steps}",
                f"num_eval_points: {len(mse_ts)}",
                f"final_mse: {final_mse:.10f}",
                f"plot: {fig_path}",
                f"dim_plot: {dim_fig_path if args.plot_first_horizon_only else 'disabled'}",
            ]
        )
        + "\n"
    )

    print(f"[DONE] final_mse={final_mse:.10f}")
    print(f"[DONE] plot={fig_path}")
    print(f"[DONE] summary={txt_path}")


if __name__ == "__main__":
    main()
