from __future__ import annotations

import argparse
import json
import pathlib


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _rewrite_parquet_task_index(path: pathlib.Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path)

    def set_zero_column(name: str) -> None:
        nonlocal table
        if name not in table.schema.names:
            return
        target_type = table.schema.field(name).type
        column = pa.array([0] * table.num_rows, type=target_type)
        table = table.set_column(table.schema.get_field_index(name), name, column)

    set_zero_column("task_index")
    set_zero_column("short_horizon_task_index")

    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    pq.write_table(table, tmp_path)
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Set every episode in a local LeRobot dataset to one task instruction.")
    parser.add_argument("--dataset-dir", type=pathlib.Path, required=True, help="LeRobot dataset root to update in place.")
    parser.add_argument("--task", required=True, help="Task instruction to use for every episode.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not (dataset_dir / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"LeRobot meta/info.json not found under: {dataset_dir}")

    info_path = dataset_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_tasks"] = 1
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=4) + "\n")

    _write_jsonl(dataset_dir / "meta" / "tasks.jsonl", [{"task_index": 0, "task": args.task}])

    episodes = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
    for episode in episodes:
        episode["tasks"] = [args.task]
        episode["short_horizon_task"] = [args.task]
        episode["task_index"] = 0
    _write_jsonl(dataset_dir / "meta" / "episodes.jsonl", episodes)

    (dataset_dir / "tasks.txt").write_text("\n".join([args.task] * len(episodes)) + "\n")

    parquet_files = sorted((dataset_dir / "data").rglob("episode_*.parquet"))
    for parquet_path in parquet_files:
        _rewrite_parquet_task_index(parquet_path)

    print(f"Updated: {dataset_dir}")
    print(f"Episodes: {len(episodes)}")
    print(f"Parquet files: {len(parquet_files)}")
    print(f"Task: {args.task}")


if __name__ == "__main__":
    main()
