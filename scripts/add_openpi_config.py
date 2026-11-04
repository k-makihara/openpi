from __future__ import annotations

import argparse
import pathlib
import re


DEFAULT_CONFIG_PY = pathlib.Path("/fsx/koshi.makihara/openpi/src/openpi/training/config.py")


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)
    return starts


def _offset(line_starts: list[int], line: int, col: int) -> int:
    return line_starts[line - 1] + col


def _matching_paren(text: str, open_index: int) -> int:
    depth = 0
    index = open_index
    quote: str | None = None
    triple = False
    escape = False
    comment = False

    while index < len(text):
        char = text[index]
        nxt = text[index : index + 3]

        if comment:
            if char == "\n":
                comment = False
            index += 1
            continue

        if quote is not None:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif triple and nxt == quote * 3:
                quote = None
                triple = False
                index += 2
            elif not triple and char == quote:
                quote = None
            index += 1
            continue

        if char == "#":
            comment = True
        elif nxt in ("'''", '"""'):
            quote = char
            triple = True
            index += 2
        elif char in ("'", '"'):
            quote = char
            triple = False
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1

    raise ValueError(f"No matching ')' found for offset {open_index}")


def _find_train_config_blocks(text: str) -> dict[str, tuple[int, int]]:
    blocks: dict[str, tuple[int, int]] = {}
    for match in re.finditer(r"(?m)^    TrainConfig\(", text):
        start = match.start()
        open_index = text.index("(", match.start())
        close_index = _matching_paren(text, open_index)
        end = close_index + 1
        if end < len(text) and text[end] == ",":
            end += 1
        if end < len(text) and text[end] == "\n":
            end += 1
        block = text[start:end]
        name_match = re.search(r'name\s*=\s*"([^"]+)"', block)
        if not name_match:
            continue
        blocks[name_match.group(1)] = (start, end)
    return blocks


def _replace_first_string_kwarg(block: str, key: str, value: str) -> str:
    pattern = rf'({re.escape(key)}\s*=\s*)"[^"]*"'
    replacement = rf'\1"{value}"'
    updated, count = re.subn(pattern, replacement, block, count=1)
    if count != 1:
        raise ValueError(f"Could not find string kwarg '{key}=...' in base config.")
    return updated


def _replace_or_insert_top_level_kwarg(block: str, key: str, value: str) -> str:
    pattern = rf"(?m)^(        {re.escape(key)}\s*=\s*).*,\s*$"
    replacement = rf"\1{value},"
    updated, count = re.subn(pattern, replacement, block, count=1)
    if count:
        return updated

    last_close = block.rfind("    ),")
    if last_close == -1:
        raise ValueError("Could not find the end of the TrainConfig block.")
    return block[:last_close] + f"        {key}={value},\n" + block[last_close:]


def _assert_config_name_available(blocks: dict[str, tuple[int, int]], name: str) -> None:
    if name in blocks:
        raise ValueError(f"Config '{name}' already exists in config.py.")


def build_new_config(
    text: str,
    *,
    base_config: str,
    new_config: str,
    repo_id: str,
    insert_after: str | None,
    num_train_steps: int | None,
    batch_size: int | None,
    wandb_enabled: bool | None,
) -> str:
    blocks = _find_train_config_blocks(text)
    if base_config not in blocks:
        raise ValueError(f"Base config '{base_config}' was not found.")
    _assert_config_name_available(blocks, new_config)

    base_start, base_end = blocks[base_config]
    new_block = text[base_start:base_end]
    new_block = _replace_first_string_kwarg(new_block, "name", new_config)
    new_block = _replace_first_string_kwarg(new_block, "repo_id", repo_id)

    if num_train_steps is not None:
        new_block = _replace_or_insert_top_level_kwarg(new_block, "num_train_steps", str(num_train_steps))
    if batch_size is not None:
        new_block = _replace_or_insert_top_level_kwarg(new_block, "batch_size", str(batch_size))
    if wandb_enabled is not None:
        new_block = _replace_or_insert_top_level_kwarg(new_block, "wandb_enabled", str(wandb_enabled))

    target_name = insert_after or base_config
    if target_name not in blocks:
        raise ValueError(f"Insert-after config '{target_name}' was not found.")
    _, insert_at = blocks[target_name]
    return text[:insert_at] + new_block + text[insert_at:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append a new OpenPI TrainConfig to src/openpi/training/config.py by cloning an existing config."
    )
    parser.add_argument("--config-py", type=pathlib.Path, default=DEFAULT_CONFIG_PY)
    parser.add_argument("--base-config", required=True, help="Existing config name to clone.")
    parser.add_argument("--new-config", required=True, help="New config name to create.")
    parser.add_argument("--repo-id", required=True, help="Dataset path/repo_id for the new config.")
    parser.add_argument("--insert-after", help="Config name to insert after. Defaults to --base-config.")
    parser.add_argument("--num-train-steps", type=int, help="Override top-level num_train_steps.")
    parser.add_argument("--batch-size", type=int, help="Override top-level batch_size.")
    wandb = parser.add_mutually_exclusive_group()
    wandb.add_argument("--wandb-enabled", dest="wandb_enabled", action="store_true")
    wandb.add_argument("--no-wandb-enabled", dest="wandb_enabled", action="store_false")
    parser.set_defaults(wandb_enabled=None)
    parser.add_argument("--write", action="store_true", help="Write config.py. Without this, prints the updated file.")
    args = parser.parse_args()

    text = args.config_py.read_text()
    updated = build_new_config(
        text,
        base_config=args.base_config,
        new_config=args.new_config,
        repo_id=args.repo_id,
        insert_after=args.insert_after,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        wandb_enabled=args.wandb_enabled,
    )
    if args.write:
        args.config_py.write_text(updated)
        print(f"Added config '{args.new_config}' to {args.config_py}")
    else:
        print(updated, end="")


if __name__ == "__main__":
    main()
