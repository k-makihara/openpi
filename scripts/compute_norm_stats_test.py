# ruff: noqa: SLF001
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openpi.policies import umi_original_policy as umi
from openpi.training import config
from openpi.training import data_loader
from scripts import compute_norm_stats as stats_script


def test_video_free_getitem_never_decodes(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Video decoding must not run")

    monkeypatch.setattr(data_loader.lerobot_dataset, "decode_video_frames", fail)
    dataset = object.__new__(data_loader.VideoFreeLeRobotDataset)
    dataset.hf_dataset = [
        {
            "episode_index": torch.tensor(0),
            "timestamp": torch.tensor(0.0),
            "task_index": torch.tensor(0),
            "state": torch.tensor([2.0]),
        }
    ]
    dataset.meta = SimpleNamespace(video_keys=["camera"], tasks={0: "task"})
    dataset.delta_indices = None
    dataset.image_transforms = None
    dataset = pickle.loads(pickle.dumps(dataset))
    item = dataset[0]
    assert item["camera"].shape == (3, 1, 1)
    assert item["state"].item() == 2.0
    assert dataset._query_videos({"camera": [0.0, 0.1]}, 0)["camera"].shape == (2, 3, 1, 1)


@pytest.mark.parametrize("downsample_anchor_frames", [False, True])
def test_get_config_with_anchor_sampling_override(downsample_anchor_frames):
    train_config = stats_script._get_config_with_data_overrides(
        "debug",
        "/tmp/dataset",
        downsample_anchor_frames,
    )

    assert train_config.data.repo_id == "/tmp/dataset"
    assert train_config.data.downsample_anchor_frames is downsample_anchor_frames


@pytest.mark.parametrize("preprocessed", [False, True])
@pytest.mark.parametrize("target", ["delta", "relative"])
def test_umi_stats_identical_without_video(preprocessed, target):
    transform = umi.UmiOriginalDualArmRelativeStateInputs(
        history_steps=2,
        action_horizon_steps=2,
        action_pose_target=target,
        gripper_preprocessed=preprocessed,
    )
    poses = np.array([[i * 0.1, 0, 0, 0, 0, 0, 1] for i in range(4)], dtype=np.float32)
    sample = {
        "left_image": np.ones((3, 8, 8), dtype=np.float32),
        "right_image": np.ones((3, 8, 8), dtype=np.float32),
        "left_pose_seq": poses,
        "right_pose_seq": poses + np.array([1, 0, 0, 0, 0, 0, 0]),
        "left_gripper_seq": np.array([5.5, 5.6, 5.7, 5.8]),
        "right_gripper_seq": np.array([5.6, 5.7, 5.8, 5.9]),
    }
    dataset = object.__new__(data_loader.VideoFreeLeRobotDataset)
    placeholders = dataset._query_videos({"left_image": [0.0], "right_image": [0.0]}, 0)
    expected = stats_script.SelectStatsKeys()(transform(dict(sample)))
    actual = stats_script.SelectStatsKeys()(transform({**sample, **placeholders}))
    assert set(actual) == {"state", "actions"}
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_stats_respects_preprocessed_gripper_marker(tmp_path, monkeypatch):
    from openpi.models import pi0_config
    from openpi.training import config

    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/gripper_transform.json").write_text('{"transform":"openpi_umi_gripper_v2"}')
    model = pi0_config.Pi0Config()
    dc = config.LeRobotUmiOriginalDualArmRelativeStateDataConfig(repo_id=str(tmp_path)).create(tmp_path, model)
    captured = {}

    def create(data_config, *args, **kwargs):
        captured["preprocessed"] = data_config.data_transforms.inputs[0].gripper_preprocessed
        captured.update(kwargs)
        return [None] * 32

    monkeypatch.setattr(data_loader, "create_torch_dataset", create)
    monkeypatch.setattr(data_loader, "TorchDataLoader", lambda *args, **kwargs: None)
    stats_script.create_torch_dataloader(dc, 10, 32, model, 0)
    assert captured == {"preprocessed": True, "skip_video_decode": True}
    stats_script.create_torch_dataloader(dc, 10, 32, model, 0, skip_video_decode=False)
    assert captured["skip_video_decode"] is False


def test_temporal_queries_match_without_select(monkeypatch):
    from datasets import Dataset
    from lerobot.common.datasets.utils import hf_transform_to_torch

    hf = Dataset.from_dict(
        {
            "pose": [[float(i), 0.0] for i in range(6)],
            "gripper": [[float(i)] for i in range(6)],
            "other": [[float(i + 10)] for i in range(6)],
            "timestamp": [i / 60 for i in range(6)],
        }
    )
    hf.set_transform(hf_transform_to_torch)
    dataset = object.__new__(data_loader.VideoFreeLeRobotDataset)
    dataset.hf_dataset = hf
    dataset.meta = SimpleNamespace(video_keys=["camera"])
    queries = {"pose": [0, 0, 3, 5, 5], "gripper": [0, 0, 3, 5, 5], "other": [5, 2, 0], "camera": [0, 5]}
    expected = data_loader.lerobot_dataset.LeRobotDataset._query_hf_dataset(dataset, queries)

    def fail(*args, **kwargs):
        raise AssertionError("Temporal reads must not construct Dataset.select views")

    monkeypatch.setattr(Dataset, "select", fail)
    actual = dataset._query_hf_dataset(queries)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    # Cached column views also remain usable after worker serialization.
    restored = pickle.loads(pickle.dumps(dataset))
    for key, values in restored._query_hf_dataset(queries).items():
        torch.testing.assert_close(values, expected[key], rtol=0, atol=0)
    assert len(dataset._stats_column_views) == 2
    timestamps = dataset._get_query_timestamps(0.0, queries)
    assert dataset._query_videos(timestamps, 0)["camera"].shape == (2, 3, 1, 1)
