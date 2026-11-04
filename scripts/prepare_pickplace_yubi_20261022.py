"""Curate original 224 PickPlace with the recorded current yubi-local snapshot."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path("/fsx/koshi.makihara/openpi")
SOURCE = Path("/fsx/koshi.makihara/datasets_tasked/lerobot_v21_ph2_PickAndPlace_merged_gripper_224")
OUTPUT = SOURCE.with_name(SOURCE.name + "_yubi_20261022_gop12")
SNAPSHOT = ROOT / "pipelines/pickplace_20261022"


def verify(job):
    import av

    path, expected = job
    with av.open(str(path)) as reader:
        stream = reader.streams.video[0]
        stream.codec_context.thread_count = 1
        assert (stream.width, stream.height, stream.average_rate, stream.codec_context.format.name) == (
            224,
            224,
            60,
            "yuv420p",
        )
        count = 0
        for count, frame in enumerate(reader.decode(video=0), 1):
            assert frame.pts is not None
            assert bool(frame.key_frame) == ((count - 1) % 12 == 0), path
            assert abs(float(frame.pts * frame.time_base) - (count - 1) / 60) < 1e-6, path
        assert count == expected, (path, count, expected)
    return expected


def jobs(root, info):
    for line in (root / "meta/episodes.jsonl").read_text().splitlines():
        row = json.loads(line)
        index = row["episode_index"]
        args = {"episode_index": index, "episode_chunk": index // info["chunks_size"]}
        parquet = root / info["data_path"].format(**args)
        assert parquet.is_file(), parquet
        for key, feature in info["features"].items():
            if feature["dtype"] == "video":
                path = root / info["video_path"].format(**args, video_key=key)
                assert path.is_file(), path
                yield path, row["length"]


def main():
    marker = OUTPUT / "meta/pipeline_complete.json"
    if marker.exists():
        assert json.loads(marker.read_text())["all_video_gop_and_timestamps_verified"]
        print("Already complete", OUTPUT, flush=True)
        return
    assert shutil.disk_usage(SOURCE.parent).free > 200 * 1024**3, "Need 200GiB free before curation"
    info = json.loads((SOURCE / "meta/info.json").read_text())
    assert info["fps"] == 60
    inputs = list(jobs(SOURCE, info))
    assert len(inputs) == info["total_videos"]
    print("Source preflight:", info["total_episodes"], info["total_frames"], len(inputs), flush=True)
    subprocess.run(
        [sys.executable, str(SNAPSHOT / "yubi-local/yubi.py"), "build", str(SOURCE), str(OUTPUT), "--workers", "8"],
        check=True,
    )
    report = json.loads((OUTPUT / "meta/build_manifest.json").read_text())
    assert report["state"] == "complete"
    settings = report["settings"]
    assert settings["exact_freeze_min_frames"] == 6
    assert settings["min_segment_frames"] == 60
    assert settings["freeze_min_frames"] == 31
    assert settings["video_gop"] == 12
    assert settings["trim_first_frames"] == settings["trim_last_frames"] == 0
    info = json.loads((OUTPUT / "meta/info.json").read_text())
    videos = list(jobs(OUTPUT, info))
    assert len(videos) == info["total_videos"]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for n, _ in enumerate(pool.map(verify, videos, chunksize=1), 1):
            if n % 100 == 0 or n == len(videos):
                print(f"Verified videos {n}/{len(videos)}", flush=True)
    shutil.copy2(SNAPSHOT / "tool_snapshot.json", OUTPUT / "meta/curation_tool_snapshot.json")
    marker.write_text(
        json.dumps(
            {
                "state": "complete",
                "source": str(SOURCE),
                "settings": settings,
                "all_video_gop_and_timestamps_verified": True,
                "episodes": info["total_episodes"],
                "frames": info["total_frames"],
                "videos": len(videos),
            },
            indent=2,
        )
        + "\n"
    )
    print("READY", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
