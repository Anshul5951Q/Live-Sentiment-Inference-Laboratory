"""Atomic file I/O operations for JSON, CSV, and Parquet."""
import json
from pathlib import Path
from typing import Any, Dict, List, Union
import pandas as pd


def save_json(data: Union[Dict, List], file_path: Union[str, Path], indent: int = 2) -> Path:
    """Save dictionary or list as JSON atomically."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    temp_path.replace(path)
    return path


def load_json(file_path: Union[str, Path]) -> Union[Dict, List]:
    """Load JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataframe(df: pd.DataFrame, file_path: Union[str, Path]) -> Path:
    """Save DataFrame to Parquet or CSV based on extension."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported dataframe format: {path.suffix}")
    return path


def load_dataframe(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load DataFrame from Parquet or CSV."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported dataframe format: {path.suffix}")

