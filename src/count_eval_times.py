#!/usr/bin/env python3
from pathlib import Path
from statistics import mean

TARGET_DIRS = [
    "outputs/edu/contextual/Gender",
    "outputs/edu/contextual/Race",
    "outputs/edu/societal/Gender Identity",
    "outputs/edu/societal/Sexual Orientation",

    "outputs/hiring/contextual/Gender",
    "outputs/hiring/contextual/Race",
    "outputs/hiring/societal/Gender Identity",
    "outputs/hiring/societal/Sexual Orientation",

    "outputs/loan/contextual/Gender",
    "outputs/loan/contextual/Race",
    "outputs/loan/societal/Gender Identity",
    "outputs/loan/societal/Sexual Orientation",
]

IGNORE_10_DIRS = {
    Path("outputs/edu/contextual/Gender"),
    Path("outputs/edu/contextual/Race"),
    Path("outputs/hiring/contextual/Gender"),
    Path("outputs/hiring/contextual/Race"),
}


def count_lines(file_path: Path) -> int:
    """Read-only line count. This function never modifies the file."""
    line_count = 0
    last_byte = b""

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            line_count += chunk.count(b"\n")
            last_byte = chunk[-1:]

    if last_byte and last_byte != b"\n":
        line_count += 1

    return line_count


def should_ignore_file(file_path: Path, directory: Path) -> bool:
    filename = file_path.name

    # Ignore all JSONL files whose file name contains "_rd".
    if "_rd" in filename:
        return True

    # Additionally ignore files containing "_10_" only in the four specified dirs.
    if directory in IGNORE_10_DIRS and "_10_" in filename:
        return True

    return False


def main():
    all_file_stats = []

    for dir_str in TARGET_DIRS:
        directory = Path(dir_str)

        if not directory.exists():
            print(f"[Warning] Directory does not exist: {directory}")
            continue

        jsonl_files = sorted(
            p for p in directory.glob("*.jsonl")
            if not should_ignore_file(p, directory)
        )

        for file_path in jsonl_files:
            n_lines = count_lines(file_path)
            all_file_stats.append((file_path, n_lines))

    total_files = len(all_file_stats)
    total_lines = sum(n_lines for _, n_lines in all_file_stats)
    avg_lines = mean(n_lines for _, n_lines in all_file_stats) if total_files > 0 else 0

    print("=" * 80)
    print("Per-file line counts")
    print("=" * 80)

    for file_path, n_lines in all_file_stats:
        print(f"{n_lines:10d}  {file_path}")

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total JSONL files:        {total_files}")
    print(f"Total line numbers:       {total_lines}")
    print(f"Average lines per file:   {avg_lines:.2f}")


if __name__ == "__main__":
    main()