"""Artifact IO: local-first persistence.

All experiment outputs go under ``runs/<model_id>/<stage>/``. Helpers cover
the formats the pipeline uses: JSON, JSONL (append-friendly for sweep cells),
NumPy ``.npz`` (vectors / activations) and Parquet (score tables).

The ``mirror`` parameter on write helpers is accepted but ignored (kept for
backward compatibility with existing call sites).
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Local filesystem
# ---------------------------------------------------------------------------


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, path: str, mirror: bool = True) -> str:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def load_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(record: dict, path: str, mirror: bool = True) -> None:
    """Append one record. Used for per-cell sweep results (resumable)."""
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(records: Iterable[dict], path: str, mirror: bool = True) -> str:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def save_npz(path: str, mirror: bool = True, **arrays: np.ndarray) -> str:
    ensure_dir(os.path.dirname(path))
    np.savez(path, **arrays)
    return path


def load_npz(path: str) -> dict[str, np.ndarray] | None:
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def save_parquet(df, path: str, mirror: bool = True) -> str:
    """Save a pandas DataFrame to parquet (falls back to CSV on failure)."""
    ensure_dir(os.path.dirname(path))
    try:
        df.to_parquet(path, index=False)
        out = path
    except Exception as e:  # pragma: no cover - environment dependent
        alt = os.path.splitext(path)[0] + ".csv"
        print(f"[io] parquet failed ({e}); writing CSV {alt}")
        df.to_csv(alt, index=False)
        out = alt
    return out
