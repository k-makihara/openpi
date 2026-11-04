"""Preflight and prepare five-source AssemblyECU training dataset without source mutations."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path("/fsx/koshi.makihara/openpi")
ROOT = Path("/fsx/koshi.makihara/datasets")
WORK = Path("/fsx/koshi.makihara/datasets_tasked/assemblyecu_20261017_build")
FINAL = Path("/fsx/koshi.makihara/datasets_tasked/lerobot_v21_assemblyecu_5sources_yubi_20261017_224_gop12")
NAMES = [
    "20260917_AssemblyECU",
    "20260918_AssemblyECU",
    "20260924_AssemblyECU",
    "20261002_failurerecover",
    "20261008_AssemblyECU",
]


def rows(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def preflight():
    from merge_lerobot_task_datasets import _normalize_missing_video_features
    from merge_lerobot_task_datasets import _validate_compatible_infos

    reference = Path("/fsx/koshi.makihara/datasets_tasked/lerobot_v21_ph2_success_20260917_20260918_AssemblyECU_fixed")
    reference_rows = rows(reference / "meta/episodes.jsonl")
    reference_tasks = (reference / "tasks.txt").read_text().splitlines()
    assert len(reference_rows) == len(reference_tasks)
    mapping = {
        (r["source_dataset"], r["source_episode_index"]): (r, task)
        for r, task in zip(reference_rows, reference_tasks, strict=True)
    }
    sources, infos, tasks = [], [], []
    for name in NAMES:
        source = ROOT / ("lerobot_v21_ph2_success_" + name)
        info = json.loads((source / "meta/info.json").read_text())
        episodes = rows(source / "meta/episodes.jsonl")
        assert len(episodes) == info["total_episodes"]
        if (source / "tasks.txt").exists():
            labels = (source / "tasks.txt").read_text().splitlines()
        else:
            labels = []
            for r in episodes:
                old, label = mapping[(source.name, r["episode_index"])]
                assert old["uuid"] == r["uuid"]
                assert old["length"] == r["length"]
                labels.append(label)
        assert len(labels) == len(episodes)
        assert all(t.strip() for t in labels)
        _normalize_missing_video_features(source, info)
        for r in episodes:
            args = {"episode_chunk": r["episode_index"] // info["chunks_size"], "episode_index": r["episode_index"]}
            paths = [source / info["data_path"].format(**args)]
            paths.extend(
                source / info["video_path"].format(**args, video_key=k)
                for k, v in info["features"].items()
                if v["dtype"] == "video"
            )
            for p in paths:
                if not p.is_file() or not p.stat().st_size:
                    raise ValueError(f"Missing or empty input: {p}")
        sources.append(source)
        infos.append(info)
        tasks.append(labels)
    _validate_compatible_infos(sources, infos)
    return sources, infos, tasks


def run(command):
    print("RUN", command, flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    sources, infos, tasks = preflight()
    print("Preflight OK:", sum(i["total_episodes"] for i in infos), "episodes", flush=True)
    if args.check_only:
        return
    WORK.mkdir(parents=True, exist_ok=True)
    views = []
    for source, labels in zip(sources, tasks, strict=True):
        view = WORK / "inputs" / source.name
        view.mkdir(parents=True, exist_ok=True)
        for d in ["meta", "data", "videos"]:
            if not (view / d).exists():
                (view / d).symlink_to(source / d, target_is_directory=True)
        (view / "tasks.txt").write_text("\n".join(labels) + "\n")
        views.append(view)
    (WORK / "sources.json").write_text(
        json.dumps(
            {
                "sources": [str(s) for s in sources],
                "task_note": "Existing tasks.txt preserved; 0917/0918 labels mapped from fixed dataset by source episode and UUID",
            },
            indent=2,
        )
    )
    merged = WORK / "merged"
    if not (WORK / "merge_complete.json").exists():
        if merged.exists():
            raise RuntimeError("Incomplete merge exists; inspect before retry")
        command = [sys.executable, str(REPO / "scripts/merge_lerobot_task_datasets.py"), "--out-dir", str(merged)]
        for v in views:
            command += ["--dataset-dir", str(v)]
        run(command)
        (WORK / "merge_complete.json").write_text(json.dumps({"episodes": sum(i["total_episodes"] for i in infos)}))
    resized = WORK / "merged_224_lossless"
    run(
        [
            sys.executable,
            str(REPO / "scripts/resize_lerobot_videos.py"),
            "--source",
            str(merged),
            "--output",
            str(resized),
            "--workers",
            "8",
            "--gop",
            "12",
            "--resume",
        ]
    )
    run(
        [
            sys.executable,
            str(REPO / "pipelines/assemblyecu_20261017/yubi-local/yubi.py"),
            "build",
            str(resized),
            str(FINAL),
            "--workers",
            "8",
        ]
    )
    manifest = json.loads((FINAL / "meta/build_manifest.json").read_text())
    assert manifest["state"] == "complete"
    assert manifest["settings"]["video_gop"] == 12
    info = json.loads((FINAL / "meta/info.json").read_text())
    for feature in info["features"].values():
        if feature["dtype"] == "video":
            assert feature["shape"] == [224, 224, 3]
            assert feature["info"]["video.codec"] == "h264"
            assert feature["info"]["video.pix_fmt"] == "yuv420p"
    # Validate keyframe placement and timestamps on every completed video.
    import av

    for p in (FINAL / "videos").rglob("*.mp4"):
        with av.open(str(p)) as video:
            video.streams.video[0].codec_context.thread_count = 1
            count = 0
            for count, f in enumerate(video.decode(video=0), 1):
                assert bool(f.key_frame) == ((count - 1) % 12 == 0), str(p)
                assert abs(float(f.pts * f.time_base) - (count - 1) / 60) < 1e-6, str(p)
            assert count == video.streams.video[0].frames, str(p)
    shutil.copy2(REPO / "pipelines/assemblyecu_20261017/tool_snapshot.json", FINAL / "meta/curation_tool_snapshot.json")
    (FINAL / "meta/pipeline_complete.json").write_text(
        json.dumps(
            {
                "state": "complete",
                "sources": [str(s) for s in sources],
                "settings": manifest["settings"],
                "all_video_gop_and_timestamps_verified": True,
            },
            indent=2,
        )
    )
    print("READY", FINAL, manifest["total_episodes"], manifest["total_frames"], flush=True)


if __name__ == "__main__":
    main()
