"""GPU storage comparison case with explicit, identical data settings."""

import argparse
import dataclasses
from pathlib import Path

from benchmark_train_input import run_case

from openpi.training import config as configs
from openpi.training import data_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--steps", type=int, default=512)
    args = parser.parse_args()
    if args.steps < 128 or args.warmup < 32:
        parser.error("Require warmup >= 32 and measured steps >= 128")
    args.batches = [512]
    args.workers = [32]
    args.downsample_anchor_frames = False
    args.excluded_episode_paths = []
    base = configs.get_config(args.config)
    base = dataclasses.replace(
        base,
        data=dataclasses.replace(
            base.data,
            assets=dataclasses.replace(base.data.assets, asset_id=args.norm_asset_id),
            lerobot_table_query_mode="grouped",
            lerobot_video_backend="pyav",
        ),
    )
    # Fail on corrupt samples instead of silently changing the sampled workload.
    data_loader.RetryOnErrorDataset.__getitem__ = lambda self, index: self._dataset[index]
    run_case(args, base_config=base)


if __name__ == "__main__":
    main()
