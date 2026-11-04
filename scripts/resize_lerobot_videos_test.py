import json
from pathlib import Path
import subprocess
import sys

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def test_exclude_missing_episode_keeps_data_and_metadata_consistent(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    (source / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 60,
        "chunks_size": 1000,
        "total_episodes": 3,
        "total_frames": 6,
        "total_videos": 3,
        "total_chunks": 1,
        "splits": {"train": "0:3"},
        "features": {"camera": {"dtype": "video", "shape": [24, 32, 3]}},
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    }
    (source / "meta/info.json").write_text(json.dumps(info))
    (source / "meta/episodes.jsonl").write_text(
        "".join(json.dumps({"episode_index": i, "length": 2}) + "\n" for i in range(3))
    )
    (source / "meta/episodes_stats.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": i, "stats": {"episode_index": {}, "index": {}, "camera": {}}}) + "\n"
            for i in range(3)
        )
    )
    (source / "meta/gripper_transform.json").write_text('{"transform": "openpi_umi_gripper_v2"}')
    (source / "meta/stats.json").write_text('{"index": "stale"}')
    for index in (0, 2):
        parquet = source / f"data/chunk-000/episode_{index:06d}.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "episode_index": [index, index],
                    "index": [index * 2, index * 2 + 1],
                    "state": [[float(index), 1.0], [float(index), 2.0]],
                }
            ),
            parquet,
        )
        video = source / f"videos/chunk-000/camera/episode_{index:06d}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(video), mode="w") as writer:
            stream = writer.add_stream("mpeg4", rate=60)
            stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
            for _ in range(2):
                frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), 50 + index, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    writer.mux(packet)
            for packet in stream.encode():
                writer.mux(packet)
    command = [
        sys.executable,
        str(Path(__file__).with_name("resize_lerobot_videos.py")),
        "--source",
        str(source),
        "--output",
        str(output),
        "--workers",
        "2",
    ]
    failed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert failed.returncode != 0
    assert not output.exists()
    command += ["--exclude-episode-index", "1"]
    subprocess.run([*command, "--dry-run"], capture_output=True, text=True, check=True)
    assert not output.exists()
    subprocess.run(command, capture_output=True, text=True, check=True)
    actual = json.loads((output / "meta/info.json").read_text())
    assert actual["total_episodes"] == 2
    assert actual["total_frames"] == 4
    assert actual["total_videos"] == 2
    assert actual["splits"] == {"train": "0:2"}
    assert actual["features"]["camera"]["shape"] == [224, 224, 3]
    table = pq.read_table(output / "data/chunk-000/episode_000001.parquet")
    assert table["episode_index"].to_pylist() == [1, 1]
    assert table["index"].to_pylist() == [2, 3]
    assert table["state"].to_pylist() == [[2.0, 1.0], [2.0, 2.0]]
    assert not (output / "data/chunk-000/episode_000002.parquet").exists()
    assert (output / "videos/chunk-000/camera/episode_000001.mp4").exists()
    assert (source / "meta/gripper_transform.json").read_bytes() == (
        output / "meta/gripper_transform.json"
    ).read_bytes()
    assert not (output / "meta/stats.json").exists()
    rows = [json.loads(line) for line in (output / "meta/episodes_stats.jsonl").read_text().splitlines()]
    assert [r["episode_index"] for r in rows] == [0, 1]
    assert rows[1]["stats"]["episode_index"]["mean"] == [1.0]
    assert rows[1]["stats"]["index"]["mean"] == [2.5]
    assert all("camera" not in row["stats"] for row in rows)
    assert json.loads((output / "meta/resize_episode_mapping.json").read_text())["source_to_output"] == {"0": 0, "2": 1}
    (output / "meta/info.json").unlink()  # Simulate interruption before final publication.
    resumed = subprocess.run([*command, "--resume"], capture_output=True, text=True, check=True)
    assert resumed.stdout.count("reused") == 2
