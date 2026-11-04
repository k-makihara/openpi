import dataclasses
import pathlib
import types

import jax
import pytest
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_episode_path_exclusion_matching():
    dataset_root = pathlib.Path("/tmp/dataset")
    excluded = {"episodes/0001.parquet", "/tmp/dataset/episodes/0002.parquet"}

    assert _data_loader._is_episode_path_excluded(dataset_root, "episodes/0001.parquet", excluded)
    assert _data_loader._is_episode_path_excluded(dataset_root, "/tmp/dataset/episodes/0002.parquet", excluded)
    assert not _data_loader._is_episode_path_excluded(dataset_root, "episodes/0003.parquet", excluded)


def test_filtered_episode_indices_excludes_matching_path():
    dataset_root = pathlib.Path("/tmp/dataset")
    dataset_meta = types.SimpleNamespace(
        episodes={0: {}, 1: {}, 2: {}},
        get_data_file_path=lambda episode_index: pathlib.Path(f"data/episode_{episode_index:06d}.parquet"),
    )

    retained = _data_loader._filtered_episode_indices(
        dataset_root,
        dataset_meta,
        [str(dataset_root / "data/episode_000001.parquet")],
    )

    assert retained == [0, 2]


def test_filtered_episode_indices_rejects_unmatched_path():
    dataset_root = pathlib.Path("/tmp/dataset")
    dataset_meta = types.SimpleNamespace(
        episodes={0: {}},
        get_data_file_path=lambda episode_index: pathlib.Path(f"data/episode_{episode_index:06d}.parquet"),
    )

    with pytest.raises(ValueError, match="did not match dataset metadata"):
        _data_loader._filtered_episode_indices(
            dataset_root,
            dataset_meta,
            [str(dataset_root / "data/episode_000001.parquet")],
        )


def test_episode_subset_remaps_sparse_episode_index_for_temporal_query():
    dataset = _data_loader.EpisodeSubsetLeRobotDataset.__new__(_data_loader.EpisodeSubsetLeRobotDataset)
    dataset._episode_position_by_index = {2166: 0, 2168: 1}
    dataset.episode_data_index = {
        "from": torch.tensor([0, 10]),
        "to": torch.tensor([10, 20]),
    }
    dataset.delta_indices = {"observation.state": [-1, 0, 1]}

    query_indices, padding = dataset._get_query_indices(idx=10, ep_idx=2168)

    assert query_indices == {"observation.state": [10, 10, 11]}
    assert padding["observation.state_is_pad"].tolist() == [True, False, False]


def test_episode_subset_normalizes_video_timestamps(monkeypatch):
    dataset = _data_loader.EpisodeSubsetLeRobotDataset.__new__(_data_loader.EpisodeSubsetLeRobotDataset)
    dataset._episode_position_by_index = {42: 0}
    dataset._video_timestamp_offsets = {}
    dataset.episode_data_index = {"from": torch.tensor([0]), "to": torch.tensor([2])}
    dataset.hf_dataset = [{"timestamp": torch.tensor(0.1)}, {"timestamp": torch.tensor(0.2)}]
    captured = {}

    def query_videos(_dataset, query_timestamps, ep_idx):
        captured["query_timestamps"] = query_timestamps
        captured["ep_idx"] = ep_idx
        return {}

    monkeypatch.setattr(_data_loader.lerobot_dataset.LeRobotDataset, "_query_videos", query_videos)

    dataset._query_videos({"camera": [0.1, 0.2]}, 42)

    assert captured["ep_idx"] == 42
    assert captured["query_timestamps"]["camera"] == pytest.approx([0.0, 0.1])
    assert dataset._video_timestamp_offsets == {42: pytest.approx(0.1)}


def test_episode_anchor_stride_downsamples_each_episode_independently():
    dataset = list(range(12))
    episode_data_index = {
        "from": torch.tensor([0, 5]),
        "to": torch.tensor([5, 12]),
    }

    strided = _data_loader.EpisodeAnchorStrideDataset(dataset, episode_data_index, stride=3)

    assert len(strided) == 5
    assert [strided[index] for index in range(len(strided))] == [0, 3, 5, 8, 11]


def test_episode_anchor_stride_supports_negative_indices():
    dataset = list(range(5))
    episode_data_index = {
        "from": torch.tensor([0]),
        "to": torch.tensor([5]),
    }

    strided = _data_loader.EpisodeAnchorStrideDataset(dataset, episode_data_index, stride=2)

    assert strided[-1] == 4
    with pytest.raises(IndexError):
        _ = strided[3]
