# ruff: noqa: SLF001
import csv
import json
import types

import benchmark_input_variants as variants
import benchmark_train_input as benchmark

from openpi.training import config
from openpi.training import data_loader


def test_variant_config_preserves_base_options_and_selects_mode():
    args = types.SimpleNamespace(
        config=variants.DEFAULT_CONFIG,
        dataset=variants.DEFAULT_DATASET,
        downsample_anchor_frames=True,
        excluded_episode_paths=[],
        batches=[256],
        workers=[64],
        video_backend="torchcodec",
        table_query_mode="grouped",
    )
    actual = benchmark.make_config(args)
    assert actual.data.base_config.lerobot_video_backend == "torchcodec"
    assert actual.data.base_config.lerobot_table_query_mode == "grouped"
    assert actual.data.base_config.lerobot_tolerance_s == 0.05
    assert actual.batch_size == 256
    assert actual.seed == config.get_config(variants.DEFAULT_CONFIG).seed
    assert config.get_config(variants.DEFAULT_CONFIG).data.base_config.lerobot_table_query_mode == "legacy"


def test_retry_removal_preserves_transforms_and_fails_on_bad_sample():
    import pytest

    class BrokenDataset:
        def __getitem__(self, _index):
            raise ValueError("bad sample")

    source = BrokenDataset()
    wrapped = data_loader.TransformedDataset(data_loader.RetryOnErrorDataset(source), [])
    assert benchmark.disable_sample_retries(wrapped) is wrapped
    assert wrapped._dataset is source
    with pytest.raises(ValueError, match="bad sample"):
        wrapped[0]


def test_sweep_runs_four_cases_with_validation(tmp_path, monkeypatch):
    args = types.SimpleNamespace(
        output=tmp_path / "results",
        verify_samples=4,
        backends=["pyav", "torchcodec"],
        table_query_modes=["legacy", "grouped"],
        config=variants.DEFAULT_CONFIG,
        dataset=variants.DEFAULT_DATASET,
        warmup=128,
        steps=256,
        timeout=100,
        batches=[256],
        workers=[64],
        repeats=1,
        downsample_anchor_frames=True,
        excluded_episode_paths=["bad.parquet"],
    )
    commands = []

    def execute(command, _log_path, _timeout, *, cpu=False):
        commands.append(command)
        result = variants.Path(command[command.index("--result") + 1])
        if "--validate" in command:
            assert cpu
            result.write_text(json.dumps({"backend_images_exactly_equal": True}))
        else:
            assert not cpu
            assert command[command.index("--steps") + 1] == "256"
            assert command[command.index("--warmup") + 1] == "128"
            assert "bad.parquet" in command
            result.write_text(json.dumps({"status": "ok", "samples_per_second": 123.0}))
        return "ok"

    monkeypatch.setattr(variants, "execute", execute)
    variants.sweep(args)
    assert len(commands) == 5
    with (args.output / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {(r["video_backend"], r["table_query_mode"]) for r in rows} == {
        ("pyav", "legacy"),
        ("pyav", "grouped"),
        ("torchcodec", "legacy"),
        ("torchcodec", "grouped"),
    }
