"""Create a separate H.264/YUV420 CRF18 dataset with a short GOP; verifies every frame timestamp."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import shutil
import subprocess

import av
import numpy as np


def signature(path):
    s = path.stat()
    return [s.st_size, s.st_mtime_ns]


def convert(job):
    source, target, gop, ffmpeg = job
    receipt = target.with_suffix(".recompress.json")
    if receipt.exists() and target.exists():
        saved = json.loads(receipt.read_text())
        if saved["source_signature"] == signature(source) and saved["output_signature"] == signature(target):
            return saved
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < source.stat().st_size * 2 + 50 * 2**30:
        raise RuntimeError("Insufficient free space: preserve 50 GiB headroom")
    tmp = target.with_suffix(".partial.mp4")
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-threads",
            "1",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            "-x264-params",
            f"keyint={gop}:min-keyint={gop}:scenecut=0:bframes=0",
            "-fps_mode",
            "passthrough",
            "-movflags",
            "+faststart",
            str(tmp),
        ],
        check=True,
    )
    count = 0
    absolute_error = 0
    values = 0
    with av.open(str(source)) as a, av.open(str(tmp)) as b:
        for container in [a, b]:
            container.streams.video[0].codec_context.thread_count = 1
        for x, y in itertools.zip_longest(a.decode(video=0), b.decode(video=0)):
            if x is None or y is None:
                raise ValueError(f"Frame count mismatch: {source}")
            if x.pts * x.time_base != y.pts * y.time_base:
                raise ValueError(f"Timestamp mismatch: {source} frame {count}")
            before = x.to_ndarray(format="rgb24")
            after = y.to_ndarray(format="rgb24")
            if before.shape != after.shape:
                raise ValueError(f"Shape mismatch: {source} frame {count}")
            absolute_error += int(np.abs(before.astype(np.int16) - after.astype(np.int16)).sum())
            values += before.size
            if bool(y.key_frame) != (count % gop == 0):
                raise ValueError(f"Unexpected keyframe: {source} frame {count}")
            count += 1
    tmp.replace(target)
    saved = {
        "source": str(source),
        "output": str(target),
        "source_signature": signature(source),
        "output_signature": signature(target),
        "frames": count,
        "rgb_mean_absolute_error": absolute_error / values,
        "rgb_exact": absolute_error == 0,
        "timestamps_exact": True,
        "gop": gop,
    }
    receipt.write_text(json.dumps(saved) + "\n")
    return saved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--gop", type=int, default=12)
    p.add_argument("--limit", type=int, help="Pilot only: output is NOT published as a complete dataset")
    p.add_argument("--ffmpeg", default="/fsx/koshi.makihara/miniforge3/envs/airoapi/bin/ffmpeg")
    args = p.parse_args()
    if args.workers < 1 or args.gop < 1 or (args.limit is not None and args.limit < 1):
        p.error("workers, gop and limit must be positive")
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents:
        p.error("output must be separate from source")
    settings = {
        "source": str(source),
        "gop": args.gop,
        "codec": "libx264",
        "crf": 18,
        "preset": "fast",
        "limit": args.limit,
    }
    marker = output / "recompression.json"
    if output.exists() and (not marker.exists() or json.loads(marker.read_text())["settings"] != settings):
        p.error("Existing output has different settings or no marker")
    output.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"settings": settings, "state": "running"}, indent=2))
    videos = sorted((source / "videos").rglob("*.mp4"))
    if not videos:
        raise ValueError("No source videos")
    if args.limit:
        videos = videos[: args.limit]
    if not args.limit:
        required = sum(v.stat().st_size for v in videos) + 50 * 2**30
        if shutil.disk_usage(output).free < required:
            raise RuntimeError("Full copy requires source-video size plus 50 GiB free space; pilot with --limit first")
        for directory in ["data", "meta"]:
            shutil.copytree(
                source / directory,
                output / directory,
                dirs_exist_ok=True,
                ignore=lambda path, names: ["info.json"] if Path(path) == source / "meta" else [],
            )
        for path in source.iterdir():
            if path.is_file() and path.name != "recompression.json":
                shutil.copy2(path, output / path.name)
    jobs = [(v, output / v.relative_to(source), args.gop, args.ffmpeg) for v in videos]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(convert, jobs), 1):
            results.append(result)
            print(f"{i}/{len(jobs)} verified: {result['source']}", flush=True)
    before = sum(r["source_signature"][0] for r in results)
    after = sum(r["output_signature"][0] for r in results)
    report = {
        "settings": settings,
        "state": "pilot_complete" if args.limit else "complete",
        "videos": len(results),
        "source_video_bytes": before,
        "output_video_bytes": after,
        "reduction_percent": 100 * (1 - after / before),
        "all_rgb_exact": all(r["rgb_exact"] for r in results),
        "lossy_reencode": True,
        "all_timestamps_exact": True,
    }
    if not args.limit:
        # Publish only after all videos passed timestamp, shape and GOP validation.
        shutil.copy2(source / "meta/info.json", output / "meta/info.json")
    marker.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
