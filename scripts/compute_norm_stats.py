"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses
import json

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class SelectStatsKeys(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        # Do not collate or transfer unused images/prompts to the accelerator.
        return {key: x[key] for key in ("state", "actions")}


def _get_config_with_data_overrides(
    config_name: str,
    data_repo_id: str | None,
    downsample_anchor_frames: bool | None,
) -> _config.TrainConfig:
    config = _config.get_config(config_name)
    data_overrides = {}
    if data_repo_id is not None:
        data_overrides["repo_id"] = data_repo_id
    if downsample_anchor_frames is not None:
        data_overrides["downsample_anchor_frames"] = downsample_anchor_frames
    if data_overrides:
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, **data_overrides))
    return config


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
    *,
    skip_video_decode: bool = True,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    # Match training: already-corrected gripper values must not be corrected again.
    data_config = _data_loader._configure_preprocessed_gripper_data(data_config)  # noqa: SLF001
    dataset = _data_loader.create_torch_dataset(
        data_config, action_horizon, model_config, skip_video_decode=skip_video_decode
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            SelectStatsKeys(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    max_frames: int | None = None,
    data_repo_id: str | None = None,
    downsample_anchor_frames: bool | None = None,
    *,
    skip_video_decode: bool = True,
):
    """Compute stats; skip video decoding by default for LeRobot datasets.

    Set --no-skip-video-decode for custom state/action transforms that depend on images.
    RLDS loading is unchanged.
    """
    config = _get_config_with_data_overrides(config_name, data_repo_id, downsample_anchor_frames)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        print(
            f"Stats loader: video_decode={'off' if skip_video_decode else 'on'}, "
            f"temporal_reads={'direct' if skip_video_decode else 'select'}, "
            f"anchor_sampling={'strided' if data_config.downsample_anchor_frames else 'all'}, "
            f"num_workers={config.num_workers}",
            flush=True,
        )
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            max_frames,
            skip_video_decode=skip_video_decode,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    if data_config.asset_id is None:
        raise ValueError("Data config must have an asset_id to save normalization statistics.")
    output_path = config.assets_dirs / data_config.asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    (output_path / "norm_stats_metadata.json").write_text(
        json.dumps(
            {
                "downsample_anchor_frames": data_config.downsample_anchor_frames,
                "action_sample_stride": data_config.action_sample_stride,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    tyro.cli(main)
