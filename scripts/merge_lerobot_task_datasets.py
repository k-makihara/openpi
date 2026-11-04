from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
from datetime import datetime, timezone

from build_lerobot_task_subset import _copy_or_link
from build_lerobot_task_subset import _episode_file
from build_lerobot_task_subset import _load_exclude_tasks
from build_lerobot_task_subset import _load_include_tasks
from build_lerobot_task_subset import _ordered_unique
from build_lerobot_task_subset import _read_jsonl
from build_lerobot_task_subset import _read_tasks_txt
from build_lerobot_task_subset import _rewrite_episode_parquet
from build_lerobot_task_subset import _video_file
from build_lerobot_task_subset import _video_keys
from build_lerobot_task_subset import _write_jsonl


def _load_info(dataset_dir: pathlib.Path) -> dict:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot meta/info.json not found under: {dataset_dir}")
    return json.loads(info_path.read_text())


def _normalize_missing_video_features(dataset_dir: pathlib.Path, info: dict) -> None:
    episodes = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    episode_indices = [int(row["episode_index"]) for row in episodes]
    for video_key in _video_keys(info):
        if any((dataset_dir / _video_file(info, episode_index, video_key)).is_file() for episode_index in episode_indices):
            continue
        print(f"Warning: treating missing video feature as unsupported: {dataset_dir} [{video_key}]")
        info["features"][video_key]["dtype"] = "unsupported"


def _validate_compatible_infos(dataset_dirs: list[pathlib.Path], infos: list[dict]) -> None:
    first = infos[0]
    for dataset_dir, info in zip(dataset_dirs[1:], infos[1:], strict=False):
        for key in ["fps", "data_path", "video_path", "features"]:
            if info.get(key) != first.get(key):
                raise ValueError(f"Dataset {dataset_dir} is not compatible: info[{key!r}] differs.")


def _source_episode_tasks(dataset_dir: pathlib.Path, *, default_task: str | None = None) -> list[tuple[int, str]]:
    episodes = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    episodes_by_index = {int(row["episode_index"]): row for row in episodes}
    if default_task is not None:
        return [(episode_index, default_task) for episode_index in sorted(episodes_by_index)]

    tasks_path = dataset_dir / "tasks.txt"
    if not tasks_path.is_file():
        raise FileNotFoundError(
            f"Missing tasks.txt: {tasks_path}. Create one task line per episode, "
            'or use --task-mode single --task "Your instruction" to label every episode.'
        )
    tasks_txt = _read_tasks_txt(tasks_path)
    if len(tasks_txt) != len(episodes_by_index):
        raise ValueError(
            f"{dataset_dir}/tasks.txt has {len(tasks_txt)} lines but episodes.jsonl has "
            f"{len(episodes_by_index)} episodes."
        )
    return [(episode_index, tasks_txt[episode_index]) for episode_index in sorted(episodes_by_index)]


def _collect_dataset_dirs(
    dataset_dirs: list[pathlib.Path], dataset_roots: list[pathlib.Path], out_dir: pathlib.Path
) -> list[pathlib.Path]:
    sources = [path.resolve() for path in dataset_dirs]
    for root in dataset_roots:
        root = root.resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Dataset search root is not a directory: {root}")
        discovered = sorted({
            path.parent.parent.resolve()
            for path in root.rglob("meta/info.json")
            if path.is_file()
        })
        discovered = [path for path in discovered if path != out_dir and out_dir not in path.parents]
        if not discovered:
            raise ValueError(f"No LeRobot datasets (meta/info.json) found under: {root}")
        sources.extend(discovered)
    sources = list(dict.fromkeys(sources))
    if not sources:
        raise ValueError("Specify at least one --dataset-dir or --dataset-root.")
    for source in sources:
        if source == out_dir or source in out_dir.parents or out_dir in source.parents:
            raise ValueError(f"Source and output directories must not overlap: {source}, {out_dir}")
    return sources


def merge_datasets(args: argparse.Namespace) -> None:
    out_dir = args.out_dir.resolve()
    dataset_dirs = _collect_dataset_dirs(args.dataset_dir or [], getattr(args, "dataset_root", None) or [], out_dir)
    default_task = args.default_task.strip() if args.default_task is not None else None
    if args.default_task is not None and not default_task:
        raise ValueError("--default-task must not be empty.")
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {out_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(out_dir)

    infos = [_load_info(dataset_dir) for dataset_dir in dataset_dirs]
    for dataset_dir, info in zip(dataset_dirs, infos, strict=True):
        _normalize_missing_video_features(dataset_dir, info)
    _validate_compatible_infos(dataset_dirs, infos)
    base_info = infos[0]

    include_tasks = _load_include_tasks(args)
    exclude_tasks = _load_exclude_tasks(args)
    include_regex = re.compile(args.include_regex) if args.include_regex else None
    exclude_regex = re.compile(args.exclude_regex) if args.exclude_regex else None
    candidates: list[tuple[pathlib.Path, dict, int, str]] = []
    all_tasks_for_index: list[str] = []

    for dataset_dir, info in zip(dataset_dirs, infos, strict=False):
        for old_episode_index, task in _source_episode_tasks(dataset_dir, default_task=default_task):
            if include_tasks is not None and task not in include_tasks:
                continue
            if include_regex is not None and include_regex.search(task) is None:
                continue
            if task in exclude_tasks:
                continue
            if exclude_regex is not None and exclude_regex.search(task) is not None:
                continue
            candidates.append((dataset_dir, info, old_episode_index, task))
            all_tasks_for_index.append(task)

    if not candidates:
        raise ValueError("No episodes matched the requested task filters.")

    selected_tasks = _ordered_unique(all_tasks_for_index)
    task_to_new_index = {task: idx for idx, task in enumerate(selected_tasks)}

    new_info = dict(base_info)
    new_info["total_episodes"] = len(candidates)
    new_info["total_tasks"] = len(selected_tasks)
    new_info["total_chunks"] = (len(candidates) + int(new_info.get("chunks_size", 1000)) - 1) // int(
        new_info.get("chunks_size", 1000)
    )
    new_info["splits"] = {"train": f"0:{len(candidates)}"}

    new_episodes: list[dict] = []
    new_episode_stats: list[dict] = []
    copied_video_counts = dict.fromkeys(_video_keys(base_info), 0)
    global_index = 0

    source_episode_rows: dict[pathlib.Path, dict[int, dict]] = {}
    source_stats_rows: dict[pathlib.Path, dict[int, dict]] = {}
    for dataset_dir in dataset_dirs:
        source_episode_rows[dataset_dir] = {
            int(row["episode_index"]): row for row in _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
        }
        source_stats_rows[dataset_dir] = {
            int(row["episode_index"]): row for row in _read_jsonl(dataset_dir / "meta" / "episodes_stats.jsonl")
        }

    for new_episode_index, (dataset_dir, info, old_episode_index, task) in enumerate(candidates):
        task_index = task_to_new_index[task]
        episode = dict(source_episode_rows[dataset_dir][old_episode_index])
        episode["episode_index"] = new_episode_index
        episode["tasks"] = [task]
        episode["short_horizon_task"] = [task]
        episode["task_index"] = task_index
        episode["source_dataset"] = dataset_dir.name
        episode["source_episode_index"] = old_episode_index

        src_parquet = dataset_dir / _episode_file(info, old_episode_index)
        dst_parquet = out_dir / _episode_file(new_info, new_episode_index)
        if not src_parquet.is_file():
            raise FileNotFoundError(f"Missing parquet for episode {old_episode_index}: {src_parquet}")
        frame_count = _rewrite_episode_parquet(
            src_parquet,
            dst_parquet,
            new_episode_index=new_episode_index,
            new_task_index=task_index,
            global_index_start=global_index,
        )
        episode["length"] = frame_count
        global_index += frame_count

        if old_episode_index in source_stats_rows[dataset_dir]:
            stats_row = dict(source_stats_rows[dataset_dir][old_episode_index])
            stats_row["episode_index"] = new_episode_index
            new_episode_stats.append(stats_row)
        new_episodes.append(episode)

        for video_key in _video_keys(info):
            src_video = dataset_dir / _video_file(info, old_episode_index, video_key)
            dst_video = out_dir / _video_file(new_info, new_episode_index, video_key)
            if src_video.is_file():
                _copy_or_link(src_video, dst_video, copy=args.copy_videos)
                copied_video_counts[video_key] += 1

    new_info["total_frames"] = global_index
    for video_key, count in copied_video_counts.items():
        if count == 0:
            new_info["features"][video_key]["dtype"] = "unsupported"
    if "total_videos" in new_info:
        active_video_keys = [key for key in _video_keys(new_info) if copied_video_counts.get(key, 0) > 0]
        new_info["total_videos"] = len(candidates) * len(active_video_keys)

    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta" / "info.json").write_text(json.dumps(new_info, ensure_ascii=False, indent=4) + "\n")
    _write_jsonl(
        out_dir / "meta" / "tasks.jsonl", [{"task_index": i, "task": task} for i, task in enumerate(selected_tasks)]
    )
    _write_jsonl(out_dir / "meta" / "episodes.jsonl", new_episodes)
    _write_jsonl(out_dir / "meta" / "episodes_stats.jsonl", new_episode_stats)
    (out_dir / "tasks.txt").write_text("\n".join(task for _, _, _, task in candidates) + "\n")

    source_records = {str(path): {"dataset_dir": str(path), "episodes": []} for path in dataset_dirs}
    for episode, (dataset_dir, _, old_episode_index, task) in zip(new_episodes, candidates, strict=True):
        source_records[str(dataset_dir)]["episodes"].append({
            "source_episode_index": old_episode_index,
            "output_episode_index": episode["episode_index"],
            "task": task,
            "task_index": episode["task_index"],
            "frames": episode["length"],
        })
    for record in source_records.values():
        record["selected_episodes"] = len(record["episodes"])
        record["selected_frames"] = sum(episode["frames"] for episode in record["episodes"])
    manifest = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_dir),
        "requested_dataset_dirs": [str(path.resolve()) for path in args.dataset_dir or []],
        "requested_dataset_roots": [str(path.resolve()) for path in getattr(args, "dataset_root", None) or []],
        "task_mode": "single" if default_task is not None else "tasks-txt",
        "task": default_task,
        "filters": {
            "include_tasks": sorted(include_tasks) if include_tasks is not None else None,
            "exclude_tasks": sorted(exclude_tasks),
            "include_regex": args.include_regex,
            "exclude_regex": args.exclude_regex,
        },
        "copy_videos": args.copy_videos,
        "total_episodes": len(candidates),
        "total_frames": global_index,
        "sources": list(source_records.values()),
    }
    manifest_path = out_dir / "merge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Output: {out_dir}")
    print(f"Merge manifest: {manifest_path}")
    print(f"Sources: {len(dataset_dirs)}")
    for dataset_dir in dataset_dirs:
        count = sum(1 for candidate in candidates if candidate[0] == dataset_dir)
        print(f"  - {dataset_dir}: {count} episodes")
    print(f"Episodes: {len(candidates)}")
    print(f"Frames: {global_index}")
    print(f"Tasks: {len(selected_tasks)}")
    for task in selected_tasks:
        count = sum(1 for _, _, _, selected_task in candidates if selected_task == task)
        print(f"  [{task_to_new_index[task]}] {count} episodes: {task}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple compatible LeRobot v2.1 datasets, using each dataset's tasks.txt "
            "or a shared default task as the episode task labels."
        )
    )
    parser.add_argument("--dataset-dir", type=pathlib.Path, action="append", help="Source dataset root. Repeatable.")
    parser.add_argument(
        "--dataset-root", type=pathlib.Path, action="append",
        help="Recursively find datasets containing meta/info.json, in path order. Repeatable; "
        "can be combined with --dataset-dir. Duplicate sources and the output subtree are excluded.",
    )
    parser.add_argument("--out-dir", type=pathlib.Path, required=True, help="Output dataset root to create.")
    parser.add_argument(
        "--task-mode",
        choices=("tasks-txt", "single"),
        help="tasks-txt: read each source's tasks.txt; single: assign --task to all episodes. "
        "If omitted, use single when --task/--default-task is supplied, otherwise tasks-txt.",
    )
    parser.add_argument(
        "--task",
        "--default-task",
        dest="default_task",
        help="Assign this task to every episode instead of reading tasks.txt from each source dataset.",
    )
    parser.add_argument(
        "--include-task",
        action="append",
        help="Exact task name to keep. Can be passed multiple times. If omitted, all tasks are kept.",
    )
    parser.add_argument("--include-tasks-file", type=pathlib.Path, help="Text file of exact task names to keep.")
    parser.add_argument("--include-regex", help="Regex filter for task names to keep.")
    parser.add_argument(
        "--exclude-task", action="append", help="Exact task name to drop. Can be passed multiple times."
    )
    parser.add_argument("--exclude-tasks-file", type=pathlib.Path, help="Text file of exact task names to drop.")
    parser.add_argument("--exclude-regex", help="Regex filter for task names to drop.")
    parser.add_argument("--copy-videos", action="store_true", help="Copy videos instead of hardlinking when possible.")
    parser.add_argument("--overwrite", action="store_true", help="Replace out-dir if it already exists.")
    args = parser.parse_args()
    if not args.dataset_dir and not args.dataset_root:
        parser.error("Specify at least one --dataset-dir or --dataset-root.")
    task_mode = args.task_mode or ("single" if args.default_task is not None else "tasks-txt")
    if task_mode == "single" and (args.default_task is None or not args.default_task.strip()):
        parser.error('--task-mode single requires a non-empty --task "Your instruction".')
    if task_mode == "tasks-txt" and args.default_task is not None:
        parser.error("--task/--default-task cannot be combined with --task-mode tasks-txt.")
    merge_datasets(args)


if __name__ == "__main__":
    main()
