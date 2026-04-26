#!/usr/bin/env python3
"""Convert reports between CSV and Parquet formats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.entropy_knn.report_io import SUPPORTED_TABULAR_SUFFIXES, convert_tabular_report

sys.path.insert(0, str(Path(__file__).parent))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert tabular reports between CSV and Parquet")
    parser.add_argument("input_path", help="Input file or directory")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output file or directory. If omitted, the script swaps the extension for files or creates a sibling folder for directories.",
    )
    parser.add_argument("--to", choices=["csv", "parquet"], default=None, help="Force the target format")
    return parser.parse_args()


def _target_suffix(input_path: Path, forced_format: str | None) -> str:
    if forced_format is not None:
        return f".{forced_format}"
    if input_path.suffix.lower() == ".csv":
        return ".parquet"
    if input_path.suffix.lower() == ".parquet":
        return ".csv"
    raise ValueError(f"Unsupported input format: {input_path.suffix}")


def _convert_file(input_path: Path, output_path: Path | None, forced_format: str | None) -> Path:
    target_suffix = _target_suffix(input_path, forced_format)
    target_path = output_path or input_path.with_suffix(target_suffix)
    return convert_tabular_report(input_path, target_path)


def _convert_directory(input_dir: Path, output_dir: Path | None, forced_format: str | None) -> list[Path]:
    target_dir = output_dir or input_dir.parent / f"{input_dir.name}_converted"
    written_paths: list[Path] = []

    for source_path in sorted(input_dir.rglob("*")):
        if not source_path.is_file() or source_path.suffix.lower() not in SUPPORTED_TABULAR_SUFFIXES:
            continue

        relative_path = source_path.relative_to(input_dir)
        target_suffix = _target_suffix(source_path, forced_format)
        destination = (target_dir / relative_path).with_suffix(target_suffix)
        written_paths.append(convert_tabular_report(source_path, destination))

    return written_paths


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else None

    if input_path.is_dir():
        written_paths = _convert_directory(input_path, output_path, args.to)
        print(f"Converted {len(written_paths)} files")
        for path in written_paths:
            print(path)
    else:
        written_path = _convert_file(input_path, output_path, args.to)
        print(written_path)


if __name__ == "__main__":
    main()
