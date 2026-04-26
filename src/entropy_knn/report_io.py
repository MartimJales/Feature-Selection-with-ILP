"""Small helpers for reading, writing and converting tabular reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SUPPORTED_TABULAR_SUFFIXES = {".csv", ".parquet"}


def read_tabular_report(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet report into a DataFrame."""
    report_path = Path(path)
    suffix = report_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(report_path)
    if suffix == ".parquet":
        return pd.read_parquet(report_path)

    raise ValueError(f"Unsupported tabular format: {report_path.suffix}")


def write_tabular_report(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame as CSV or Parquet based on the file suffix."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = report_path.suffix.lower()

    if suffix == ".csv":
        df.to_csv(report_path, index=False)
        return report_path

    if suffix == ".parquet":
        df.to_parquet(report_path, index=False, compression="zstd")
        return report_path

    raise ValueError(f"Unsupported tabular format: {report_path.suffix}")


def convert_tabular_report(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Convert a CSV/Parquet file into another CSV/Parquet file."""
    source_path = Path(input_path)
    if output_path is None:
        output_path = source_path.with_suffix(_other_suffix(source_path.suffix.lower()))

    target_path = Path(output_path)
    df = read_tabular_report(source_path)
    return write_tabular_report(df, target_path)


def _other_suffix(current_suffix: str) -> str:
    if current_suffix == ".csv":
        return ".parquet"
    if current_suffix == ".parquet":
        return ".csv"
    raise ValueError(f"Unsupported tabular format: {current_suffix}")
