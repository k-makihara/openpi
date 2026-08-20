from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections.abc import Iterable


_EPISODE_FILE_RE = re.compile(r"episode_(\d{6})\.parquet$")


def _parse_assignments(raw_assignments: Iterable[str]) -> list[tuple[int, int, int]]:
    assignments: list[tuple[int, int, int]] = []
    for raw in raw_assignments:
        for chunk in raw.split(","):
            spec = chunk.strip()
            if not spec:
                continue
            try:
                episode_range, task_index_text = spec.split(":", maxsplit=1)
                start_text, end_text = episode_range.split("-", maxsplit=1)
                start_episode = int(start_text)
                end_episode = int(end_text)
                task_index = int(task_index_text)
            except ValueError as exc:
                raise ValueError(f"Invalid assignment '{spec}'. Expected format 'start-end:task_index'.") from exc

            if start_episode < 0 or end_episode < 0 or task_index < 0:
                raise ValueError(f"Assignment '{spec}' must use non-negative integers.")
            if start_episode > end_episode:
                raise ValueError(f"Assignment '{spec}' has start > end.")

            assignments.append((start_episode, end_episode, task_index))

    assignments.sort(key=lambda item: (item[0], item[1], item[2]))
    for prev, curr in zip(assignments, assignments[1:], strict=False):
        if curr[0] <= prev[1]:
            raise ValueError(f"Assignments overlap: {prev} and {curr}")
    if not assignments:
        raise ValueError("No valid assignments were provided.")
    return assignments


def _find_episode_files(dataset_dir: pathlib.Path) -> dict[int, pathlib.Path]:
    episode_files: dict[int, pathlib.Path] = {}
    for episode_path in sorted((dataset_dir / "data").rglob("episode_*.parquet")):
        match = _EPISODE_FILE_RE.search(episode_path.name)
        if not match:
            continue
        episode_index = int(match.group(1))
        episode_files[episode_index] = episode_path
    if not episode_files:
        raise FileNotFoundError(f"No episode parquet files found under: {dataset_dir / 'data'}")
    return episode_files


def _lookup_task_index(episode_index: int, assignments: list[tuple[int, int, int]]) -> int | None:
    for start_episode, end_episode, task_index in assignments:
        if start_episode <= episode_index <= end_episode:
            return task_index
    return None


def _rewrite_task_index_column(episode_path: pathlib.Path, task_index: int) -> bool:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(episode_path)
    target_type = table.schema.field("task_index").type if "task_index" in table.schema.names else pa.int64()
    new_column = pa.array([task_index] * table.num_rows, type=target_type)

    if "task_index" in table.schema.names:
        column_index = table.schema.get_field_index("task_index")
        existing = table.column(column_index)
        if existing.combine_chunks().equals(new_column):
            return False
        updated_table = table.set_column(column_index, "task_index", new_column)
    else:
        updated_table = table.append_column("task_index", new_column)

    tmp_path = episode_path.with_suffix(f"{episode_path.suffix}.tmp")
    pq.write_table(updated_table, tmp_path)
    tmp_path.replace(episode_path)
    return True


def _load_task_mapping(tasks_path: pathlib.Path) -> dict[int, str]:
    if not tasks_path.exists():
        return {}

    task_mapping: dict[int, str] = {}
    for line_number, line in enumerate(tasks_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        try:
            task_mapping[int(record["task_index"])] = str(record["task"])
        except KeyError as exc:
            raise KeyError(f"Missing required key in {tasks_path}:{line_number}") from exc
    return task_mapping


def _sync_info_json(
    dataset_dir: pathlib.Path,
    *,
    assigned_task_indices: set[int],
    tasks_path: pathlib.Path,
) -> bool:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        return False

    info = json.loads(info_path.read_text())
    task_mapping = _load_task_mapping(tasks_path)
    desired_total_tasks = len(task_mapping) if task_mapping else len(assigned_task_indices)

    if info.get("total_tasks") == desired_total_tasks:
        return False

    info["total_tasks"] = desired_total_tasks
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Update LeRobot parquet episodes so each episode range uses a specific task_index, "
            "and optionally sync meta/info.json total_tasks."
        )
    )
    parser.add_argument("--dataset-dir", type=pathlib.Path, required=True, help="Path to the LeRobot dataset root.")
    parser.add_argument(
        "--assign",
        action="append",
        required=True,
        help="Episode-to-task mapping in the form '0-49:1'. Can be passed multiple times or comma-separated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned updates without rewriting parquet files or info.json.",
    )
    parser.add_argument(
        "--require-full-coverage",
        action="store_true",
        help="Fail unless every discovered episode parquet is covered by one of the assignments.",
    )
    parser.add_argument(
        "--skip-info-sync",
        action="store_true",
        help="Do not update meta/info.json total_tasks.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    assignments = _parse_assignments(args.assign)
    episode_files = _find_episode_files(dataset_dir)

    planned_updates: list[tuple[int, pathlib.Path, int]] = []
    uncovered_episodes: list[int] = []
    for episode_index, episode_path in episode_files.items():
        task_index = _lookup_task_index(episode_index, assignments)
        if task_index is None:
            uncovered_episodes.append(episode_index)
            continue
        planned_updates.append((episode_index, episode_path, task_index))

    if args.require_full_coverage and uncovered_episodes:
        preview = ", ".join(map(str, uncovered_episodes[:10]))
        suffix = "..." if len(uncovered_episodes) > 10 else ""
        raise ValueError(f"Found uncovered episodes: {preview}{suffix}")

    print(f"Dataset: {dataset_dir}")
    print("Assignments:")
    for start_episode, end_episode, task_index in assignments:
        print(f"  - episodes {start_episode}-{end_episode} -> task_index {task_index}")
    print(f"Discovered parquet episodes: {len(episode_files)}")
    print(f"Planned parquet updates: {len(planned_updates)}")
    if uncovered_episodes:
        preview = ", ".join(map(str, uncovered_episodes[:10]))
        suffix = "..." if len(uncovered_episodes) > 10 else ""
        print(f"Uncovered episodes left unchanged: {len(uncovered_episodes)} ({preview}{suffix})")

    info_updated = False
    rewritten_count = 0
    if not args.dry_run:
        for _, episode_path, task_index in planned_updates:
            rewritten_count += int(_rewrite_task_index_column(episode_path, task_index))

        if not args.skip_info_sync:
            info_updated = _sync_info_json(
                dataset_dir,
                assigned_task_indices={task_index for _, _, task_index in planned_updates},
                tasks_path=dataset_dir / "meta" / "tasks.jsonl",
            )

    print(f"Parquet files rewritten: {rewritten_count}" if not args.dry_run else "Dry run only; no parquet files rewritten.")
    if args.skip_info_sync:
        print("Skipped meta/info.json sync.")
    elif args.dry_run:
        print("Dry run only; meta/info.json not updated.")
    else:
        print("Updated meta/info.json total_tasks." if info_updated else "meta/info.json total_tasks already matched.")


if __name__ == "__main__":
    main()
