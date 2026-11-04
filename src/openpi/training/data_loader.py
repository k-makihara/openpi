from collections.abc import Iterator, Sequence
import bisect
import dataclasses
import json
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.policies.umi_original_policy as _umi_original_policy
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


_UMI_GRIPPER_INPUT_TRANSFORMS = (
    _umi_original_policy.UmiOriginalInputs,
    _umi_original_policy.UmiOriginalRightThirdSlotInputs,
    _umi_original_policy.UmiOriginalRightPrevAndCurrentThirdSlotInputs,
    _umi_original_policy.UmiOriginalBimanualInputs,
    _umi_original_policy.UmiOriginalDualArmRelativeStateInputs,
)
_GRIPPER_TRANSFORM_MARKER = pathlib.Path("meta/gripper_transform.json")
_LEROBOT_EXCLUDED_EPISODE_PATHS: set[str] = set()


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class RetryOnErrorDataset(Dataset[T_co]):
    """Retries neighboring indices when a sample is corrupted/unreadable."""

    def __init__(self, dataset: Dataset[T_co], *, max_retries: int = 32):
        self._dataset = dataset
        self._max_retries = max_retries

    def __getitem__(self, index: SupportsIndex) -> T_co:
        base = index.__index__()
        n = len(self._dataset)
        for retry in range(self._max_retries):
            idx = (base + retry) % n
            try:
                return self._dataset[idx]
            except (AssertionError, RuntimeError, ValueError) as exc:
                if retry == 0:
                    logging.warning("Skipping unreadable sample at index %d: %s", idx, exc)
                continue
        raise RuntimeError(f"Failed to fetch a valid sample after {self._max_retries} retries from index {base}.")

    def __len__(self) -> int:
        return len(self._dataset)


class EpisodeAnchorStrideDataset(Dataset[T_co]):
    """Subsample anchor frames within each episode without changing temporal queries."""

    def __init__(self, dataset: Dataset[T_co], episode_data_index: dict[str, torch.Tensor], stride: int):
        if stride < 1:
            raise ValueError(f"Anchor sample stride must be positive, got {stride}.")
        self._dataset = dataset
        self._stride = stride
        self._episode_starts = tuple(int(index) for index in episode_data_index["from"])
        episode_ends = tuple(int(index) for index in episode_data_index["to"])
        self._sample_ends = []
        sample_count = 0
        for start, end in zip(self._episode_starts, episode_ends, strict=True):
            sample_count += (end - start + stride - 1) // stride
            self._sample_ends.append(sample_count)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        sample_index = index.__index__()
        if sample_index < 0:
            sample_index += len(self)
        if sample_index < 0 or sample_index >= len(self):
            raise IndexError(sample_index)

        episode_position = bisect.bisect_right(self._sample_ends, sample_index)
        previous_end = self._sample_ends[episode_position - 1] if episode_position > 0 else 0
        source_index = self._episode_starts[episode_position] + (sample_index - previous_end) * self._stride
        return self._dataset[source_index]

    def __len__(self) -> int:
        return self._sample_ends[-1] if self._sample_ends else 0


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


_LEROBOT_STATS_PATCHED = False


def _patch_lerobot_aggregate_stats() -> None:
    """Patch LeRobot aggregate_stats to tolerate malformed episodes_stats in local datasets.

    Some locally generated v2.1 datasets may contain per-episode image stats with shapes that
    do not satisfy strict assertions in lerobot. In that case we fall back to empty stats,
    which is sufficient for openpi because normalization stats are computed/loaded separately.
    """
    global _LEROBOT_STATS_PATCHED
    if _LEROBOT_STATS_PATCHED:
        return

    original = lerobot_dataset.aggregate_stats
    original_get_episode_paths = lerobot_dataset.LeRobotDataset.get_episodes_file_paths

    def _safe_aggregate_stats(stats_list):
        try:
            return original(stats_list)
        except ValueError as exc:
            logging.warning("Falling back to empty lerobot metadata stats due to malformed episodes_stats: %s", exc)
            return {}

    lerobot_dataset.aggregate_stats = _safe_aggregate_stats

    def _safe_get_episodes_file_paths(self):
        # Keep only files that actually exist locally. This avoids forcing HF fallback
        # when dataset metadata contains optional/missing video streams.
        fpaths = original_get_episode_paths(self)
        return [
            fp
            for fp in fpaths
            if (self.root / fp).is_file() and not _is_episode_path_excluded(self.root, fp, _LEROBOT_EXCLUDED_EPISODE_PATHS)
        ]

    lerobot_dataset.LeRobotDataset.get_episodes_file_paths = _safe_get_episodes_file_paths
    _LEROBOT_STATS_PATCHED = True


def _episode_path_variants(dataset_root: pathlib.Path, episode_path: str | pathlib.Path) -> set[str]:
    path = pathlib.Path(episode_path).expanduser()
    variants = {path.as_posix()}
    if path.is_absolute():
        resolved = path.resolve()
        variants.add(resolved.as_posix())
        try:
            variants.add(resolved.relative_to(dataset_root.resolve()).as_posix())
        except ValueError:
            pass
    else:
        resolved = (dataset_root / path).resolve()
        variants.add(resolved.as_posix())
        variants.add((dataset_root / path).as_posix())
    return variants


def _is_episode_path_excluded(
    dataset_root: pathlib.Path,
    episode_path: str | pathlib.Path,
    excluded_episode_paths: set[str],
) -> bool:
    return bool(_episode_path_variants(dataset_root, episode_path) & excluded_episode_paths)


def _configure_lerobot_episode_exclusions(dataset_root: pathlib.Path, excluded_episode_paths: Sequence[str]) -> None:
    global _LEROBOT_EXCLUDED_EPISODE_PATHS
    _LEROBOT_EXCLUDED_EPISODE_PATHS = set()
    for episode_path in excluded_episode_paths:
        _LEROBOT_EXCLUDED_EPISODE_PATHS.update(_episode_path_variants(dataset_root, episode_path))


def _filtered_episode_indices(
    dataset_root: pathlib.Path,
    dataset_meta: lerobot_dataset.LeRobotDatasetMetadata,
    excluded_episode_paths: Sequence[str],
) -> list[int]:
    available_episode_indices = sorted(dataset_meta.episodes)
    if not excluded_episode_paths:
        return available_episode_indices

    requested_paths = [
        (episode_path, _episode_path_variants(dataset_root, episode_path))
        for episode_path in excluded_episode_paths
    ]
    matched_requests: set[int] = set()
    excluded_episode_indices = []
    retained_episode_indices = []
    for episode_index in available_episode_indices:
        episode_variants = _episode_path_variants(dataset_root, dataset_meta.get_data_file_path(episode_index))
        matching_requests = {
            request_index
            for request_index, (_, requested_variants) in enumerate(requested_paths)
            if episode_variants & requested_variants
        }
        if matching_requests:
            matched_requests.update(matching_requests)
            excluded_episode_indices.append(episode_index)
        else:
            retained_episode_indices.append(episode_index)

    unmatched_paths = [path for index, (path, _) in enumerate(requested_paths) if index not in matched_requests]
    if unmatched_paths:
        raise ValueError(f"Excluded episode path(s) did not match dataset metadata: {unmatched_paths}")

    logging.warning(
        "Excluding LeRobot episode indices from training: %s",
        ", ".join(map(str, excluded_episode_indices)),
    )
    return retained_episode_indices


class EpisodeSubsetLeRobotDataset(lerobot_dataset.LeRobotDataset):
    """LeRobot dataset that supports non-contiguous episode selections."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._video_timestamp_offsets: dict[int, float] = {}
        self._episode_position_by_index = (
            {episode_index: position for position, episode_index in enumerate(self.episodes)}
            if self.episodes is not None
            else None
        )

    def _get_query_indices(self, idx: int, ep_idx: int):
        if self._episode_position_by_index is not None:
            ep_idx = self._episode_position_by_index[ep_idx]
        return super()._get_query_indices(idx, ep_idx)

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        if ep_idx not in self._video_timestamp_offsets:
            episode_position = (
                self._episode_position_by_index[ep_idx]
                if self._episode_position_by_index is not None
                else ep_idx
            )
            episode_start = int(self.episode_data_index["from"][episode_position])
            self._video_timestamp_offsets[ep_idx] = float(self.hf_dataset[episode_start]["timestamp"])

        offset = self._video_timestamp_offsets[ep_idx]
        normalized_timestamps = {
            key: [max(0.0, timestamp - offset) for timestamp in timestamps]
            for key, timestamps in query_timestamps.items()
        }
        return super()._query_videos(normalized_timestamps, ep_idx)


class GroupedQueryLeRobotDataset(EpisodeSubsetLeRobotDataset):
    """Fetch columns sharing temporal indices together and reuse column views."""

    def _query_columns(self, keys: tuple[str, ...], indices: list[int]) -> dict:
        # select() constructs a new Dataset and copies its metadata on every
        # query. Cache column views, then gather rows directly instead. This also
        # avoids converting unrelated columns to tensors for temporal queries.
        if not hasattr(self, "_stats_column_views"):
            self._stats_column_views = {}
        if keys not in self._stats_column_views:
            self._stats_column_views[keys] = self.hf_dataset.select_columns(list(keys))
        return self._stats_column_views[keys][indices]

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        grouped: dict[tuple[int, ...], list[str]] = {}
        for key, indices in query_indices.items():
            if key not in self.meta.video_keys:
                grouped.setdefault(tuple(indices), []).append(key)
        result = {}
        for indices, keys in grouped.items():
            rows = self._query_columns(tuple(keys), list(indices))
            result.update({key: torch.stack(rows[key]) for key in keys})
        return result


class VideoFreeLeRobotDataset(GroupedQueryLeRobotDataset):
    """Stats-only dataset: preserve temporal queries but replace videos with tiny placeholders.

    Use only when state/action transforms do not depend on image contents. Keeping
    camera keys and temporal dimensions lets the normal repack/UMI transforms run.
    This class is module-level so spawned data-loader workers can pickle it.
    """

    def _get_query_timestamps(
        self, current_ts: float, query_indices: dict[str, list[int]] | None = None
    ) -> dict[str, list[float]]:
        # Only the number of frames matters for placeholder videos. Preserve
        # temporal dimensions without fetching timestamps from the dataset.
        return {
            key: [current_ts] * (len(query_indices[key]) if query_indices is not None and key in query_indices else 1)
            for key in self.meta.video_keys
        }

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        return {
            key: torch.zeros((len(timestamps), 3, 1, 1), dtype=torch.float32).squeeze(0)
            for key, timestamps in query_timestamps.items()
        }


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    skip_video_decode: bool = False,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        _configure_lerobot_episode_exclusions(pathlib.Path("."), ())
        return FakeDataset(model_config, num_samples=1024)

    _patch_lerobot_aggregate_stats()
    dataset_kwargs = {}
    _configure_lerobot_episode_exclusions(pathlib.Path("."), ())
    local_repo_id = repo_id
    if repo_id is not None and (repo_id.startswith("/") or pathlib.Path(repo_id).exists()):
        repo_path = pathlib.Path(repo_id).expanduser().resolve()
        # Fail fast for broken local datasets where parquet shards exist but are empty.
        data_dir = repo_path / "data"
        if data_dir.is_dir():
            parquet_files = list(data_dir.rglob("*.parquet"))
            if parquet_files and not any(p.stat().st_size > 0 for p in parquet_files):
                raise ValueError(
                    f"Local LeRobot dataset appears corrupted: all {len(parquet_files)} parquet files are 0 bytes under "
                    f"'{data_dir}'. Re-download/regenerate the dataset and retry."
                )
        local_repo_id = repo_path.name
        dataset_kwargs["root"] = str(repo_path)
        _configure_lerobot_episode_exclusions(repo_path, data_config.excluded_episode_paths)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(local_repo_id, **dataset_kwargs)
    if repo_id is not None and (repo_id.startswith("/") or pathlib.Path(repo_id).exists()):
        retained_episode_indices = _filtered_episode_indices(
            pathlib.Path(dataset_kwargs["root"]), dataset_meta, data_config.excluded_episode_paths
        )
        if retained_episode_indices:
            dataset_kwargs["episodes"] = retained_episode_indices
        else:
            raise ValueError(
                "All episodes were excluded from the dataset. Check data.excluded_episode_paths and try again."
            )
    stride = max(1, int(data_config.action_sample_stride))
    delta_timestamps = data_config.delta_timestamps_overrides
    if delta_timestamps is None:
        delta_timestamps = {
            key: [(t * stride) / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        }
    if skip_video_decode:
        dataset_class = VideoFreeLeRobotDataset
    elif data_config.lerobot_table_query_mode == "grouped":
        dataset_class = GroupedQueryLeRobotDataset
    elif data_config.lerobot_table_query_mode == "legacy":
        dataset_class = EpisodeSubsetLeRobotDataset
    else:
        raise ValueError(f"Unknown table query mode: {data_config.lerobot_table_query_mode}")
    logging.info(
        "LeRobot dataset class: %s (table_query_mode=%s, skip_video_decode=%s)",
        dataset_class.__name__, data_config.lerobot_table_query_mode, skip_video_decode,
    )
    dataset = dataset_class(
        local_repo_id,
        delta_timestamps=delta_timestamps,
        tolerance_s=float(data_config.lerobot_tolerance_s),
        video_backend=data_config.lerobot_video_backend,
        **dataset_kwargs,
    )

    # Some local datasets include video features in metadata without corresponding files.
    # Disable such streams so __getitem__ does not attempt to decode missing videos.
    for key, ft in dataset.meta.features.items():
        if skip_video_decode or ft.get("dtype") != "video":
            continue
        any_found = any((dataset.root / dataset.meta.get_video_file_path(ep, key)).is_file() for ep in dataset.meta.episodes)
        if not any_found:
            logging.warning("Disabling missing video stream '%s' from dataset metadata.", key)
            ft["dtype"] = "unsupported"

    if data_config.downsample_anchor_frames and stride > 1:
        original_size = len(dataset)
        dataset = EpisodeAnchorStrideDataset(dataset, dataset.episode_data_index, stride)
        logging.info(
            "Downsampled LeRobot anchor frames by %d: %d -> %d samples.",
            stride,
            original_size,
            len(dataset),
        )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return RetryOnErrorDataset(dataset)


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def _configure_preprocessed_gripper_data(data_config: _config.DataConfig) -> _config.DataConfig:
    repo_id = data_config.repo_id
    if repo_id is None:
        return data_config

    repo_path = pathlib.Path(repo_id).expanduser()
    if not repo_path.exists():
        return data_config

    marker_path = repo_path.resolve() / _GRIPPER_TRANSFORM_MARKER
    if not marker_path.is_file():
        return data_config

    marker = json.loads(marker_path.read_text())
    if marker.get("transform") != "openpi_umi_gripper_v2":
        raise ValueError(f"Unsupported gripper transform marker: {marker_path}")

    transformed_inputs = []
    found_umi_transform = False
    for transform in data_config.data_transforms.inputs:
        configured_transform = transform
        if isinstance(transform, _UMI_GRIPPER_INPUT_TRANSFORMS):
            configured_transform = dataclasses.replace(transform, gripper_preprocessed=True)
            found_umi_transform = True
        transformed_inputs.append(configured_transform)

    if not found_umi_transform:
        raise ValueError(f"Dataset is marked as UMI gripper-preprocessed but config has no UMI transform: {repo_path}")

    logging.info("Using preprocessed gripper values from dataset marker: %s", marker_path)
    return dataclasses.replace(
        data_config,
        data_transforms=dataclasses.replace(data_config.data_transforms, inputs=tuple(transformed_inputs)),
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    data_config = _configure_preprocessed_gripper_data(data_config)
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
