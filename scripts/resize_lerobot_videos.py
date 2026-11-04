"""Create a separate LeRobot v2.1 dataset with pre-resized, lossless RGB videos.

Uses the same PIL bilinear resize + black padding as OpenPI's ResizeImages.
Keeps all retained frames (no temporal downsampling). State/action values are preserved.
When excluding episodes, output episode IDs and global row indices are renumbered
contiguously; meta/resize_episode_mapping.json records the mapping.
Output meta/info.json is published only after every video finishes successfully.
Run again with --resume after an interruption; do not modify the source meanwhile.

Example:
    python scripts/resize_lerobot_videos.py --source /path/to/dataset \
        --output /path/to/dataset_224 --workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil

import av
from openpi_client import image_tools


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def signature(path: Path) -> dict:
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def resize_video(job: tuple) -> dict:
    source, target, height, width, gop, fps, expected_frames, resume = job
    source, target = Path(source), Path(target)
    receipt = target.with_suffix(".resize.json")
    source_signature = signature(source)
    if resume and target.is_file() and receipt.is_file():
        saved = json.loads(receipt.read_text())
        if saved["source"] == source_signature and saved["output"] == signature(target):
            return {"path": str(target), "skipped": True, "frames": saved["frames"]}

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".partial.mp4")
    count = 0
    try:
        with av.open(str(source)) as reader:
            stream = reader.streams.video[0]
            stream.thread_type = "SLICE"
            stream.codec_context.thread_count = 1
            if abs(float(stream.average_rate) - fps) > 1e-6:
                raise ValueError(f"FPS mismatch: {source}: {stream.average_rate} != {fps}")
            with av.open(str(temporary), mode="w", format="mp4", options={"movflags": "+faststart"}) as writer:
                encoded = writer.add_stream("libx264rgb", rate=stream.average_rate)
                encoded.width, encoded.height = width, height
                encoded.pix_fmt = "rgb24"
                encoded.time_base = stream.time_base
                encoded.codec_context.time_base = stream.time_base
                encoded.codec_context.thread_count = 1
                encoded.codec_context.gop_size = gop
                encoded.codec_context.max_b_frames = 0
                encoded.options = {
                    "crf": "0",
                    "preset": "fast",
                    "x264-params": f"keyint={gop}:min-keyint={gop}:scenecut=0:bframes=0",
                }
                for frame in reader.decode(stream):
                    if frame.pts is None:
                        raise ValueError(f"Missing frame timestamp: {source}")
                    # This script intentionally supports the zero-based CFR videos used by this dataset.
                    if abs(float(frame.pts * frame.time_base) - count / fps) > 1e-5:
                        raise ValueError(f"Nonzero-based or variable frame timing: {source}, frame {count}")
                    rgb = frame.to_ndarray(format="rgb24")
                    resized = image_tools.resize_with_pad(rgb, height, width)
                    output_frame = av.VideoFrame.from_ndarray(resized, format="rgb24")
                    output_frame.pts = frame.pts
                    output_frame.time_base = frame.time_base
                    for packet in encoded.encode(output_frame):
                        writer.mux(packet)
                    count += 1
                for packet in encoded.encode():
                    writer.mux(packet)
        if count != expected_frames:
            raise ValueError(f"Frame count mismatch: {source}: {count} != {expected_frames}")
        with av.open(str(temporary)) as check:
            result = check.streams.video[0]
            if result.frames != count or (result.width, result.height) != (width, height):
                raise ValueError(f"Invalid encoded video: {temporary}")
            if abs(float(result.average_rate) - fps) > 1e-6:
                raise ValueError(f"Output FPS mismatch: {temporary}")
            pixel_format = result.codec_context.format.name
        temporary.replace(target)
        write_json(
            receipt,
            {
                "source": source_signature,
                "output": signature(target),
                "frames": count,
                "pixel_format": pixel_format,
            },
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"path": str(target), "skipped": False, "frames": count}


def copy_metadata(
    source: Path, output: Path, video_keys: list[str], info: dict, episodes: list[dict], excluded: list[int]
) -> None:
    if not excluded:
        shutil.copytree(source / "data", output / "data", dirs_exist_ok=True)
    shutil.copytree(source / "meta", output / "meta", dirs_exist_ok=True, ignore=shutil.ignore_patterns("info.json"))
    if excluded:
        import pyarrow as pa
        import pyarrow.parquet as pq

        mapping = {episode["episode_index"]: index for index, episode in enumerate(episodes)}
        offsets = {}
        offset = 0
        for episode in episodes:
            old = episode["episode_index"]
            new = mapping[old]
            offsets[old] = offset
            src = source / episode_path(info, "data_path", old)
            dst = output / episode_path(info, "data_path", new)
            table = pq.read_table(src)
            if table.num_rows != episode["length"]:
                raise ValueError(f"Parquet frame count mismatch: {src}")
            for key, values in (
                ("episode_index", [new] * table.num_rows),
                ("index", range(offset, offset + table.num_rows)),
            ):
                if key in table.schema.names:
                    column = pa.array(values, type=table.schema.field(key).type)
                    table = table.set_column(table.schema.get_field_index(key), table.schema.field(key), column)
            dst.parent.mkdir(parents=True, exist_ok=True)
            temporary = dst.with_suffix(".parquet.tmp")
            pq.write_table(table, temporary)
            temporary.replace(dst)
            offset += table.num_rows
        with (output / "meta/episodes.jsonl").open("w") as writer:
            for episode in episodes:
                writer.write(json.dumps({**episode, "episode_index": mapping[episode["episode_index"]]}) + "\n")
        stats_file = output / "meta/episodes_stats.jsonl"
        if stats_file.is_file():
            rows = [json.loads(line) for line in stats_file.read_text().splitlines() if line.strip()]
            lengths = {episode["episode_index"]: episode["length"] for episode in episodes}
            with stats_file.open("w") as writer:
                for row in rows:
                    old = row["episode_index"]
                    if old not in mapping:
                        continue
                    row["episode_index"] = mapping[old]
                    n, start = lengths[old], offsets[old]
                    for key, minimum, maximum, mean, std in (
                        ("episode_index", mapping[old], mapping[old], float(mapping[old]), 0.0),
                        ("index", start, start + n - 1, start + (n - 1) / 2, ((n * n - 1) / 12) ** 0.5),
                    ):
                        if key in row["stats"]:
                            row["stats"][key] = {
                                "count": [n],
                                "min": [minimum],
                                "max": [maximum],
                                "mean": [mean],
                                "std": [std],
                            }
                    writer.write(json.dumps(row) + "\n")
        # v2.1 aggregates retained per-episode statistics when loading.
        (output / "meta/stats.json").unlink(missing_ok=True)
        write_json(
            output / "meta/resize_episode_mapping.json",
            {
                "source_to_output": mapping,
                "excluded_source_episode_indices": excluded,
            },
        )
        tasks_text = output / "meta/tasks.txt"
        source_tasks_text = source / "meta/tasks.txt"
        if source_tasks_text.is_file():
            lines = source_tasks_text.read_text().splitlines()
            tasks_text.write_text("\n".join(lines[episode["episode_index"]] for episode in episodes) + "\n")
    # Original image statistics no longer describe resized/padded images.
    # Keep state/action statistics; OpenPI does not use video statistics for normalization.
    stats_path = output / "meta/stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        for key in video_keys:
            stats.pop(key, None)
        write_json(stats_path, stats)
    episodes_stats = output / "meta/episodes_stats.jsonl"
    if episodes_stats.is_file():
        temporary = episodes_stats.with_suffix(".jsonl.tmp")
        with episodes_stats.open() as reader, temporary.open("w") as writer:
            for line in reader:
                if not line.strip():
                    continue
                row = json.loads(line)
                for key in video_keys:
                    row["stats"].pop(key, None)
                writer.write(json.dumps(row) + "\n")
        temporary.replace(episodes_stats)


def episode_path(info: dict, key: str, index: int, video_key: str = "") -> Path:
    path = Path(info[key].format(episode_chunk=index // info["chunks_size"], episode_index=index, video_key=video_key))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe dataset path: {path}")
    return path


def subset_info(info: dict, episodes: list[dict], video_keys: list[str]) -> dict:
    result = dict(info)
    result.update(
        total_episodes=len(episodes),
        total_frames=sum(e["length"] for e in episodes),
        total_videos=len(episodes) * len(video_keys),
        total_chunks=(len(episodes) + info["chunks_size"] - 1) // info["chunks_size"],
    )
    splits = {}
    for name, interval in info["splits"].items():
        start, end = map(int, interval.split(":"))
        indices = [i for i, e in enumerate(episodes) if start <= e["episode_index"] < end]
        if indices:
            splits[name] = f"{indices[0]}:{indices[-1] + 1}"
    result["splits"] = splits
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--gop", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--exclude-episode-index",
        type=int,
        action="append",
        default=[],
        help="Source episode ID to exclude entirely; repeat to exclude multiple episodes",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate paths and report counts without writing output"
    )
    args = parser.parse_args()
    if min(args.workers, args.height, args.width, args.gop) < 1:
        parser.error("workers, height, width and gop must be positive")
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error("Source and output must be separate, non-nested directories")
    info_bytes = (source / "meta/info.json").read_bytes()
    episodes_bytes = (source / "meta/episodes.jsonl").read_bytes()
    info = json.loads(info_bytes)
    if info["codebase_version"] != "v2.1":
        parser.error("This script supports LeRobot v2.1 datasets only")
    av.codec.Codec("libx264rgb", "w")  # Fail before copying data if encoder is unavailable.
    video_keys = [key for key, value in info["features"].items() if value["dtype"] == "video"]
    episodes = [json.loads(line) for line in episodes_bytes.splitlines() if line.strip()]
    original_info = info
    excluded = sorted(set(args.exclude_episode_index))
    unknown = set(excluded) - {e["episode_index"] for e in episodes}
    if unknown:
        parser.error(f"Unknown source episode indices: {sorted(unknown)}")
    if excluded:
        episodes = sorted((e for e in episodes if e["episode_index"] not in excluded), key=lambda e: e["episode_index"])
        if not episodes:
            parser.error("Cannot exclude every episode")
        info = subset_info(info, episodes, video_keys)
    settings = {
        "source": str(source),
        "excluded_episode_indices": excluded,
        "height": args.height,
        "width": args.width,
        "gop": args.gop,
        "codec": "libx264rgb",
        "crf": 0,
        "resize": "openpi_client.image_tools.resize_with_pad",
        "source_metadata_sha256": hashlib.sha256(info_bytes + episodes_bytes).hexdigest(),
    }
    marker = output / ".resize-settings.json"
    jobs = []
    for position, episode in enumerate(episodes):
        index = episode["episode_index"]
        output_index = position if excluded else index
        parquet = source / episode_path(original_info, "data_path", index)
        if not parquet.is_file():
            raise ValueError(f"Missing source parquet: {parquet}")
        for key in video_keys:
            relative = Path(
                info["video_path"].format(
                    episode_chunk=index // info["chunks_size"],
                    episode_index=index,
                    video_key=key,
                )
            )
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "videos":
                raise ValueError(f"Unsupported video path: {relative}")
            if relative.suffix != ".mp4" or not (source / relative).is_file():
                raise ValueError(f"Missing or non-MP4 source video: {source / relative}")
            jobs.append(
                (
                    str(source / relative),
                    str(output / episode_path(info, "video_path", output_index, key)),
                    args.height,
                    args.width,
                    args.gop,
                    info["fps"],
                    episode["length"],
                    args.resume,
                )
            )
    if not jobs or len(jobs) != info["total_videos"]:
        raise ValueError(f"Video count mismatch: {len(jobs)} != {info['total_videos']}")
    print(f"Plan: {len(episodes)} episodes, {len(jobs)} videos; excluded source IDs: {excluded}", flush=True)
    if args.dry_run:
        return
    if output.exists():
        if not args.resume or not marker.is_file() or json.loads(marker.read_text()) != settings:
            parser.error("Output exists. Use --resume with the same source and conversion settings")
        if (output / "meta/info.json").is_file():
            print(f"Already complete: {output}", flush=True)
            return
    else:
        output.mkdir(parents=True)
        write_json(marker, settings)
    print(f"Copying state/action data and metadata to {output}", flush=True)
    copy_metadata(source, output, video_keys, original_info, episodes, excluded)
    print(f"Converting {len(jobs)} videos with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for number, result in enumerate(pool.map(resize_video, jobs, chunksize=1), start=1):
            print(
                f"[{number}/{len(jobs)}] {'reused' if result['skipped'] else 'converted'} "
                f"{result['path']} ({result['frames']} frames)",
                flush=True,
            )
    pixel_format = json.loads(Path(jobs[0][1]).with_suffix(".resize.json").read_text())["pixel_format"]
    for key in video_keys:
        feature = info["features"][key]
        feature["shape"] = [args.height, args.width, 3]
        feature.setdefault("info", {}).update({"video.codec": "h264", "video.pix_fmt": pixel_format})
    write_json(output / "meta/resize_transform.json", {**settings, "video_statistics": "removed"})
    write_json(output / "meta/info.json", info)
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
