# ruff: noqa: SLF001
"""Regression checks for temporal reads, including duplicate/padded frame indices."""

import multiprocessing
import types

import datasets
from lerobot.common.datasets.utils import hf_transform_to_torch
import pytest
import torch

from openpi.training import data_loader as dl


def make_dataset():
    dataset = dl.GroupedQueryLeRobotDataset.__new__(dl.GroupedQueryLeRobotDataset)
    dataset.meta = types.SimpleNamespace(video_keys=["camera"])
    dataset.hf_dataset = datasets.Dataset.from_dict(
        {
            "left": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "right": [[7.0], [8.0], [9.0]],
            "gripper": [[0.0], [0.5], [1.0]],
            "camera": [0, 1, 2],
        }
    ).with_transform(hf_transform_to_torch)
    return dataset


def test_grouped_queries_match_legacy_with_repeated_and_distinct_indices(monkeypatch):
    dataset = make_dataset()
    queries = {"left": [2, 0, 0], "right": [2, 0, 0], "gripper": [1, 2], "camera": [0]}
    expected = dl.lerobot_dataset.LeRobotDataset._query_hf_dataset(dataset, queries)
    calls = []
    original = datasets.Dataset.select_columns

    def record(self, names, **kwargs):
        calls.append(tuple(names))
        return original(self, names, **kwargs)

    monkeypatch.setattr(datasets.Dataset, "select_columns", record)
    for _ in range(2):
        actual = dataset._query_hf_dataset(queries)
        assert actual.keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert calls == [("left", "right"), ("gripper",)]
    assert dataset._query_hf_dataset({"camera": [1]}) == {}


def child_query(dataset):
    result = dataset._query_hf_dataset({"left": [2, 0, 0], "right": [2, 0, 0]})
    return {key: value.tolist() for key, value in result.items()}


def test_grouped_dataset_and_cached_views_work_after_spawn():
    dataset = make_dataset()
    expected = child_query(dataset)  # Warm the view cache before pickling to a worker.
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        assert pool.apply(child_query, (dataset,)) == expected


@pytest.mark.parametrize(
    ("mode", "skip", "expected"),
    [
        ("grouped", False, dl.GroupedQueryLeRobotDataset),
        ("legacy", False, dl.EpisodeSubsetLeRobotDataset),
        ("grouped", True, dl.VideoFreeLeRobotDataset),
    ],
)
def test_factory_uses_requested_dataset_class(monkeypatch, mode, skip, expected):
    from openpi.training import config

    meta = types.SimpleNamespace(fps=60, features={})
    monkeypatch.setattr(dl, "_patch_lerobot_aggregate_stats", lambda: None)
    monkeypatch.setattr(dl.lerobot_dataset, "LeRobotDatasetMetadata", lambda *a, **kw: meta)
    monkeypatch.setattr(dl.EpisodeSubsetLeRobotDataset, "__init__", lambda self, *a, **kw: setattr(self, "meta", meta))
    result = dl.create_torch_dataset(
        config.DataConfig(repo_id="test/nonexistent", lerobot_table_query_mode=mode),
        50,
        None,
        skip_video_decode=skip,
    )
    assert type(result._dataset) is expected
