#!/usr/bin/env python3
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

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
    "outputs/edu/contextual/Gender",
    "outputs/edu/contextual/Race",
    "outputs/hiring/contextual/Gender",
    "outputs/hiring/contextual/Race",
}

OUTPUT_ZIP = "counted_evaluation_outputs.zip"


def should_ignore_file(file_path: Path, dir_str: str) -> bool:
    filename = file_path.name

    # Ignore all JSONL files whose file name contains "_rd".
    if "_rd" in filename:
        return True

    # Additionally ignore files containing "_10_" only in the four specified dirs.
    if dir_str in IGNORE_10_DIRS and "_10_" in filename:
        return True

    return False


def collect_files(project_root: Path):
    selected_files = []

    for dir_str in TARGET_DIRS:
        directory = project_root / dir_str

        if not directory.exists():
            print(f"[Warning] Directory does not exist: {directory}")
            continue

        jsonl_files = sorted(
            p for p in directory.glob("*.jsonl")
            if not should_ignore_file(p, dir_str)
        )

        selected_files.extend(jsonl_files)

    return selected_files


def main():
    project_root = Path.cwd()
    output_zip_path = project_root / OUTPUT_ZIP

    selected_files = collect_files(project_root)

    if not selected_files:
        print("No files found to compress.")
        return

    with ZipFile(output_zip_path, mode="w", compression=ZIP_DEFLATED) as zipf:
        for file_path in selected_files:
            # Keep directory structure relative to the project root.
            arcname = file_path.relative_to(project_root)
            zipf.write(file_path, arcname=arcname)

    print("=" * 80)
    print("Compression finished")
    print("=" * 80)
    print(f"Output ZIP file:     {output_zip_path}")
    print(f"Total files zipped:  {len(selected_files)}")
    print()
    print("Original files were not deleted or modified.")


if __name__ == "__main__":
    main()