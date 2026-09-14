"""File description: JSONL persistence helpers for diagnostic outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluate import _json_safe


def save_jsonl(record: dict[str, Any], path: Path) -> None:
    """Append one JSON-safe record to a JSONL file.

    Args:
        record: JSON-serializable payload.
        path: Destination path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(
            json.dumps(_json_safe(record), sort_keys=True, allow_nan=False) + "\n"
        )


def save_jsonl_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Append multiple JSON-safe diagnostic rows to a JSONL file.

    Args:
        rows: Per-case diagnostic rows.
        path: Destination path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(
                json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n"
            )
