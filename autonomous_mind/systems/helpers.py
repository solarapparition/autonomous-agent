"""Helpers for systems."""

from pathlib import Path
from typing import Literal, Sequence

from autonomous_mind.helpers import timestamp_to_filename
from autonomous_mind.schema import Item


def save_items(items: Sequence[Item], directory: Path) -> Literal[True]:
    """Save an event to disk."""
    for item in items:
        event_str = repr(item)
        event_file = directory / f"{timestamp_to_filename(item.timestamp)}.yaml"
        event_file.write_text(event_str, encoding="utf-8")
    return True
