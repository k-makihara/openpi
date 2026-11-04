from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
from collections.abc import Iterable


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: pathlib.Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _read_tasks_txt(path: pathlib.Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines()]


def _ordered_unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _load_include_tasks(args: argparse.Namespace) -> set[str] | None:
    include_tasks: list[str] = []
    for task in args.include_task or []:
        include_tasks.append(task.strip())
    if args.include_tasks_file is not None:
        include_tasks.extend(_read_tasks_txt(args.include_tasks_file))
    include_tasks = [task for task in include_tasks if task]
    return set(include_tasks) if include_tasks else None


def _load_exclude_tasks(args: argparse.Namespace) -> set[str]:
    exclude_tasks: list[str] = []
    for task in args.exclude_task or []:
        exclude_tasks.append(task.strip())
    if args.exclude_tasks_file is not None:
        exclude_tasks.extend(_read_tasks_txt(args.exclude_tasks_file))
    return {task for task in exclude_tasks if task}


def _copy_or_link(src: pathlib.Path, dst: pathlib.Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _format_path(
    pattern: str,
    *,
    episode_index: int,
    chunks_size: int,
    video_key: str | None = None,
) -> pathlib.Path:
    episode_chunk = episode_index // chunks_size
    kwargs = {"episode_chunk": episode_chunk, "episode_index": episode_index}
    if video_key is not None:
        kwargs["video_key"] = video_key
    return pathlib.Path(pattern.format(**kwargs))


def _rewrite_episode_parquet(
    src: pathlib.Path,
    dst: pathlib.Path,
    *,
    new_episode_index: int,
    new_task_index: int,
    global_index_start: int,
) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(src)
    num_rows = table.num_rows

    def set_column(name: str, values: list[int]) -> None:
        nonlocal table
        target_type = table.schema.field(name).type if name in table.schema.names else pa.int64()
        column = pa.array(values, type=target_type)
        if name in table.schema.names:
            table = table.set_column(table.schema.get_field_index(name), name, column)
        else:
            table = table.append_column(name, column)

    set_column("episode_index", [new_episode_index] * num_rows)
    set_column("task_index", [new_task_index] * num_rows)
    if "short_horizon_task_index" in table.schema.names:
        set_column("short_horizon_task_index", [new_task_index] * num_rows)
    if "index" in table.schema.names:
        set_column("index", list(range(global_index_start, global_index_start + num_rows)))

    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)
    return num_rows


def _chunks_size(info: dict) -> int:
    return int(info.get("chunks_size", 1000))


def _episode_file(info: dict, episode_index: int) -> pathlib.Path:
    return _format_path(info["data_path"], episode_index=episode_index, chunks_size=_chunks_size(info))


def _video_keys(info: dict) -> list[str]:
    return [key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "video"]


def _video_file(info: dict, episode_index: int, video_key: str) -> pathlib.Path:
    return _format_path(
        info["video_path"],
        episode_index=episode_index,
        chunks_size=_chunks_size(info),
        video_key=video_key,
    )


def _select_episodes(
    episode_tasks: list[tuple[int, str]],
    *,
    include_tasks: set[str] | None,
    include_regex: re.Pattern[str] | None,
    exclude_tasks: set[str],
    exclude_regex: re.Pattern[str] | None,
) -> list[tuple[int, str]]:
    selected: list[tuple[int, str]] = []
    for episode_index, task in episode_tasks:
        if include_tasks is not None and task not in include_tasks:
            continue
        if include_regex is not None and include_regex.search(task) is None:
            continue
        if task in exclude_tasks:
            continue
        if exclude_regex is not None and exclude_regex.search(task) is not None:
            continue
        selected.append((episode_index, task))
    return selected


def build_subset(args: argparse.Namespace) -> None:
    src_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve()
    if not (src_dir / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"LeRobot meta/info.json not found under: {src_dir}")
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {out_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(out_dir)

    info = json.loads((src_dir / "meta" / "info.json").read_text())
    episodes = _read_jsonl(src_dir / "meta" / "episodes.jsonl")
    episodes_by_index = {int(row["episode_index"]): row for row in episodes}
    episode_stats = _read_jsonl(src_dir / "meta" / "episodes_stats.jsonl")
    episode_stats_by_index = {int(row["episode_index"]): row for row in episode_stats}
    tasks_txt = _read_tasks_txt(src_dir / "tasks.txt")

    if len(tasks_txt) != len(episodes_by_index):
        raise ValueError(
            f"tasks.txt has {len(tasks_txt)} lines but episodes.jsonl has {len(episodes_by_index)} episodes."
        )

    episode_tasks = [(episode_index, tasks_txt[episode_index]) for episode_index in sorted(episodes_by_index)]
    include_regex = re.compile(args.include_regex) if args.include_regex else None
    exclude_regex = re.compile(args.exclude_regex) if args.exclude_regex else None
    selected = _select_episodes(
        episode_tasks,
        include_tasks=_load_include_tasks(args),
        include_regex=include_regex,
        exclude_tasks=_load_exclude_tasks(args),
        exclude_regex=exclude_regex,
    )
    if not selected:
        raise ValueError("No episodes matched the requested task filters.")

    selected_tasks = _ordered_unique(task for _, task in selected)
    task_to_new_index = {task: idx for idx, task in enumerate(selected_tasks)}

    new_info = dict(info)
    new_info["total_episodes"] = len(selected)
    new_info["total_tasks"] = len(selected_tasks)
    new_info["total_chunks"] = (len(selected) + int(new_info.get("chunks_size", 1000)) - 1) // int(
        new_info.get("chunks_size", 1000)
    )
    new_info["splits"] = {"train": f"0:{len(selected)}"}
    if "total_videos" in new_info:
        new_info["total_videos"] = len(selected) * len(_video_keys(new_info))

    new_episodes: list[dict] = []
    new_episode_stats: list[dict] = []
    global_index = 0
    copied_video_counts = {video_key: 0 for video_key in _video_keys(info)}

    for new_episode_index, (old_episode_index, task) in enumerate(selected):
        task_index = task_to_new_index[task]
        old_episode = dict(episodes_by_index[old_episode_index])
        old_episode["episode_index"] = new_episode_index
        old_episode["tasks"] = [task]
        old_episode["short_horizon_task"] = [task]
        old_episode["task_index"] = task_index

        src_parquet = src_dir / _episode_file(info, old_episode_index)
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
        old_episode["length"] = frame_count
        global_index += frame_count

        if old_episode_index in episode_stats_by_index:
            stats_row = dict(episode_stats_by_index[old_episode_index])
            stats_row["episode_index"] = new_episode_index
            new_episode_stats.append(stats_row)
        new_episodes.append(old_episode)

        for video_key in _video_keys(info):
            src_video = src_dir / _video_file(info, old_episode_index, video_key)
            dst_video = out_dir / _video_file(new_info, new_episode_index, video_key)
            if src_video.is_file():
                _copy_or_link(src_video, dst_video, copy=args.copy_videos)
                copied_video_counts[video_key] += 1

    new_info["total_frames"] = global_index
    for video_key, count in copied_video_counts.items():
        if count == 0:
            new_info["features"][video_key]["dtype"] = "unsupported"
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta" / "info.json").write_text(json.dumps(new_info, ensure_ascii=False, indent=4) + "\n")
    _write_jsonl(out_dir / "meta" / "tasks.jsonl", [{"task_index": i, "task": task} for i, task in enumerate(selected_tasks)])
    _write_jsonl(out_dir / "meta" / "episodes.jsonl", new_episodes)
    _write_jsonl(out_dir / "meta" / "episodes_stats.jsonl", new_episode_stats)
    (out_dir / "tasks.txt").write_text("\n".join(task for _, task in selected) + "\n")

    print(f"Source: {src_dir}")
    print(f"Output: {out_dir}")
    print(f"Episodes: {len(selected)} / {len(episodes_by_index)}")
    print(f"Frames: {global_index}")
    print(f"Tasks: {len(selected_tasks)}")
    for task in selected_tasks:
        count = sum(1 for _, selected_task in selected if selected_task == task)
        print(f"  [{task_to_new_index[task]}] {count} episodes: {task}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a LeRobot v2.1 dataset whose episode task labels come from tasks.txt, "
            "optionally keeping only selected task names."
        )
    )
    parser.add_argument("--dataset-dir", type=pathlib.Path, required=True, help="Source LeRobot dataset root.")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True, help="Output dataset root to create.")
    parser.add_argument(
        "--include-task",
        action="append",
        help="Exact task name to keep. Can be passed multiple times. If omitted, all tasks are kept.",
    )
    parser.add_argument(
        "--include-tasks-file",
        type=pathlib.Path,
        help="Text file containing exact task names to keep, one per line.",
    )
    parser.add_argument("--include-regex", help="Regex filter for task names to keep.")
    parser.add_argument(
        "--exclude-task",
        action="append",
        help="Exact task name to drop. Can be passed multiple times.",
    )
    parser.add_argument(
        "--exclude-tasks-file",
        type=pathlib.Path,
        help="Text file containing exact task names to drop, one per line.",
    )
    parser.add_argument("--exclude-regex", help="Regex filter for task names to drop.")
    parser.add_argument("--copy-videos", action="store_true", help="Copy videos instead of hardlinking when possible.")
    parser.add_argument("--overwrite", action="store_true", help="Replace out-dir if it already exists.")
    args = parser.parse_args()
    build_subset(args)


if __name__ == "__main__":
    main()
